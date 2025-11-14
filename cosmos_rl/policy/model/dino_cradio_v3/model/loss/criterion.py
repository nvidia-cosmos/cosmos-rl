"""Criterion Loss functions."""

import torch
import torch.nn.functional as F
from torch import nn

from ..model_utils import sigmoid_focal_loss
from ..deformable_detr import box_ops
from ..deformable_detr.utils import accuracy
from ...dist_utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """This class computes the loss for DINO.

    The process happens in two steps:
        1) Compute hungarian assignment between ground truth boxes and the outputs of the model
        2) Supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, focal_alpha, losses):
        """Create the criterion.

        Args:
            num_classes (int): number of object categories, omitting the special no-object category
            matcher (nn.Module): module able to compute a matching between targets and proposals
            losses (list[str]): list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha (float): alpha in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss).

        Args:
            outputs (dict[torch.Tensor]): computed outputs
            targets List[dict]): target annotations.
                targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            classification loss
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=self.focal_alpha,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            outputs (dict[torch.Tensor]): computed outputs
            targets (List[dict]): target annotations
                targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
                target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            bbox loss and giou loss
        """
        assert "pred_boxes" in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx].float()
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        ).float()

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes),
            )
        )
        losses["loss_giou"] = loss_giou.sum() / num_boxes

        # calculate the x,y and h,w loss
        with torch.no_grad():
            losses["loss_xy"] = loss_bbox[..., :2].sum() / num_boxes
            losses["loss_hw"] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def _get_src_permutation_idx(self, indices):
        """Permute predictions following indices.

        Args:
            indices (list): matching indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """Permute targets following indices.

        Args:
            indices (list): matching indices.
        """
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Args:
            loss (str): name of the loss to get
            outputs (dict[torch.Tensor]): computed outputs
            targets (List[dict]): target annotations
            indices (list): matching indices
            num_boxes (int): number of bounding boxes

        Returns:
            the loss value given the loss name
        """
        loss_map = {
            "labels": self.loss_labels,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """Performs the loss computation.

        Args:
            outputs (dict[torch.Tensor]): dict of tensors, see the output specification of the model for the format
            targets (List[dict]): list of dicts, such that len(targets) == batch_size.
                    The expected keys in each dict depends on the losses applied, see each loss' doc

        Returns:
            losses (dict): Dictionary of computed losses
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        device = next(iter(outputs.values())).device
        indices = self.matcher(outputs_without_aux, targets)

        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}

        # prepare for dn loss
        dn_meta = outputs["dn_meta"]

        if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)

            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.range(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (
                        torch.tensor(range(scalar)) * single_pad
                    ).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
            l_dict = {}
            for loss in self.losses:
                kwargs = {}
                if "labels" in loss:
                    kwargs = {"log": False}
                l_dict.update(
                    self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                        **kwargs,
                    )
                )

            l_dict = {f"{k}_dn": v for k, v in l_dict.items()}
            losses.update(l_dict)
        else:
            l_dict = dict()
            l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
            l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
            losses.update(l_dict)

        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for idx, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

                if self.training and dn_meta and "output_known_lbs_bboxes" in dn_meta:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][idx]
                    l_dict = {}
                    for loss in self.losses:
                        kwargs = {}
                        if "labels" in loss:
                            kwargs = {"log": False}

                        l_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                                **kwargs,
                            )
                        )

                    l_dict = {k + f"_dn_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict["loss_bbox_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_giou_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_ce_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_xy_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict["loss_hw_dn"] = torch.as_tensor(0.0).to("cuda")
                    l_dict = {k + f"_{idx}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if "interm_outputs" in outputs:
            interm_outputs = outputs["interm_outputs"]
            indices = self.matcher(interm_outputs, targets)
            if return_indices:
                indices_list.append(indices)
            for loss in self.losses:
                kwargs = {}
                if loss == "labels":
                    # Logging is enabled only for the last layer
                    kwargs = {"log": False}
                l_dict = self.get_loss(
                    loss, interm_outputs, targets, indices, num_boxes, **kwargs
                )
                l_dict = {f"{k}_interm": v for k, v in l_dict.items()}
                losses.update(l_dict)

        # enc output loss
        if "enc_outputs" in outputs:
            for i, enc_outputs in enumerate(outputs["enc_outputs"]):
                indices = self.matcher(enc_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}
                    l_dict = self.get_loss(
                        loss, enc_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_enc_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses

    def prep_for_dn(self, dn_meta):
        """Prepare for de-noising.

        Args:
            dn_meta (dict): meta information about de-noising.
        """
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups
