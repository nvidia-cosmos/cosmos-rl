"""DINO model."""

import copy
import math
import warnings
import torch
import torch.nn.functional as F
from torch import nn

from cosmos_rl.utils.logging import logger
from .dn_components import prepare_for_cdn, dn_post_process
from .model_utils import MLP
from .deformable_detr.utils import tensor_from_tensor_list, inverse_sigmoid


class DINO(nn.Module):
    """This is the Cross-Attention Detector module that performs object detection"""

    def __init__(
        self,
        backbone,
        position_embedding,
        transformer,
        num_classes,
        num_queries,
        aux_loss=False,
        random_refpoints_xy=False,
        fix_refpoints_hw=-1,
        num_feature_levels=1,
        nheads=8,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        two_stage_add_query_num=0,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        two_stage_class_embed_share=True,
        two_stage_bbox_embed_share=True,
        decoder_sa_type="sa",
        num_patterns=0,
        dn_number=100,
        dn_box_noise_scale=0.4,
        dn_label_noise_ratio=0.5,
        dn_labelbook_size=100,
        export=False,
    ):
        """Initializes the model.

        Args:
            backbone (torch.Tensor): torch module of the backbone to be used. See backbone.py.
            transformer (torch.Tensor): torch module of the transformer architecture. See deformable_transformer.py.
            num_classes (int): number of object classes.
            num_queries (int): number of object queries, ie detection slot. This is the maximal number of objects.
                         DINO can detect in a single image. For COCO, we recommend 300 queries.
            aux_loss (bool): True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            random_refpoints_xy (bool): whether to randomly initialize reference point embedding.
            fix_refpoints_hw (int): -1(default): learn w and h for each box seperately
                               >0 : given fixed number
                               -2 : learn a shared w and h
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            nheads (int): number of heads.
            two_stage_type (str): type of two stage in DINO.
            two_stage_add_query_num (int): size of the target embedding.
            dec_pred_class_embed_share (bool): whether to share embedding for decoder classification prediction.
            dec_pred_bbox_embed_share (bool): whether to share embedding for decoder bounding box prediction.
            two_stage_bbox_embed_share (bool): whether to share embedding for two stage bounding box.
            two_stage_class_embed_share (bool): whether to share embedding for two stage classification.
            decoder_layer_noise (bool): a flag to add random perturbation to decoder query.
            num_patterns (int): number of patterns in encoder-decoder.
            dn_number (bool): the number of de-noising queries in DINO
            dn_box_noise_scale (float): the scale of noise applied to boxes during contrastive de-noising.
                                        If this value is 0, noise is not applied.
            dn_label_noise_ratio (float): the scale of noise applied to labels during contrastive de-noising.
                                          If this value is 0, noise is not applied.
            dn_labelbook_size (int): de-nosing labelbook size. should be same as number of classes
            export (bool): flag to indicate if the current model is being used for ONNX export.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim = transformer.d_model
        self.num_feature_levels = num_feature_levels
        self.nheads = nheads
        self.label_enc = nn.Embedding(dn_labelbook_size + 1, hidden_dim)

        # setting query dim
        self.query_dim = 4
        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.num_patterns = num_patterns
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_labelbook_size = dn_labelbook_size

        self.backbone = backbone

        # prepare input projection layers
        self.input_proj = self.prepare_channel_mapper(
            num_feature_levels, hidden_dim, two_stage_type
        )

        self.position_embedding = position_embedding
        self.export = export

        self.aux_loss = aux_loss

        if self.export:
            warnings.warn("Setting aux_loss to be False for export")
            self.aux_loss = False

        self.box_pred_damping = None

        self.iter_update = True

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(hidden_dim, num_classes)
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        if two_stage_type != "no":
            if two_stage_bbox_embed_share:
                if not (dec_pred_class_embed_share and dec_pred_bbox_embed_share):
                    raise ValueError(
                        "two_stage_bbox_embed_share was set to true but "
                        f"dec_pred_class_embed_share was set to {dec_pred_class_embed_share} "
                        f"dec_pred_bbox_embed_share was set to {dec_pred_bbox_embed_share}"
                    )
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        if decoder_sa_type == "ca_label":
            self.label_embedding = nn.Embedding(num_classes, hidden_dim)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

        self._reset_parameters()

    def prepare_channel_mapper(self, num_feature_levels, hidden_dim, two_stage_type):
        """Create Channel Mapper style for DETR-based model.

        Args:
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            two_stage_type (str): type of two stage in DINO.
            hidden_dim (int): size of the hidden dimension.

        Returns:
            nn.ModuleList of input projection.
        """
        if num_feature_levels > 1:
            num_backbone_outs = len(self.backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = self.backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            return nn.ModuleList(input_proj_list)

        assert (
            two_stage_type == "no"
        ), "two_stage_type should be no if num_feature_levels=1 !!!"
        return nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.backbone.num_channels[-1], hidden_dim, kernel_size=1
                    ),
                    nn.GroupNorm(32, hidden_dim),
                )
            ]
        )

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        """Initialize reference points"""
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)
        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            logger.info("fix_refpoints_hw: {}".format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:]
            )
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            logger.info("learn a shared h and w")
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError(
                "Unknown fix_refpoints_hw {}".format(self.fix_refpoints_hw)
            )

    def forward(self, samples, targets=None):
        """Forward function of DINO Model

        Args:
            samples (torch.Tensor): batched images, of shape [batch_size x 3 x H x W]
            targets (dict): batched annotations

        Returns:
            pred_logits (torch.Tensor): the classification logits (including no-object) for all queries. Shape= [batch_size x num_queries x (num_classes + 1)]
            pred_boxes (torch.Tensor): The normalized boxes coordinates for all queries, represented as(center_x, center_y, height, width)
        """
        if not isinstance(samples, torch.Tensor):
            samples = tensor_from_tensor_list(samples)

        features = self.backbone(samples)

        srcs = []
        masks = []
        for level, feat in enumerate(features):
            src = feat[0]
            mask = feat[1].float()[:, 0].bool()
            srcs.append(self.input_proj[level](src))
            masks.append(mask)

        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for li in range(_len_srcs, self.num_feature_levels):
                if li == _len_srcs:
                    src = self.input_proj[li](features[-1][0])
                else:
                    src = self.input_proj[li](srcs[-1])
                srcs.append(src)

                if self.export:
                    m = torch.zeros(
                        (src.shape[0], 1, src.shape[2], src.shape[3]),
                        dtype=src.dtype,
                        device=src.device,
                    )
                else:
                    m = samples[:, 3:4]
                mask = F.interpolate(m.float(), size=src.shape[-2:]).to(torch.bool)
                masks.append(mask.float()[:, 0].bool())

        # build positional embedding
        pos = []
        for mask in masks:
            if self.export:
                N, H, W = mask.shape
                tensor_shape = torch.tensor([N, H, W], device=src.device)
                pos.append(self.position_embedding(tensor_shape, src.device))
            else:
                not_mask = ~mask
                pos.append(self.position_embedding(not_mask, src.device))

        if self.dn_number > 0 or targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta = prepare_for_cdn(
                dn_args=(
                    targets,
                    self.dn_number,
                    self.dn_label_noise_ratio,
                    self.dn_box_noise_scale,
                ),
                training=self.training,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                label_enc=self.label_enc,
            )
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, input_query_bbox, pos, input_query_label, attn_mask
        )
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(
            reference[:-1], self.bbox_embed, hs
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        outputs_class = torch.stack(
            [
                layer_cls_embed(layer_hs)
                for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
            ]
        )
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list = dn_post_process(
                outputs_class,
                outputs_coord_list,
                dn_meta,
                self.aux_loss,
                self._set_aux_loss,
            )
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord_list[-1]}
        if not self.export:
            out["srcs"] = srcs
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord_list)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])

            if not self.export:
                out["interm_outputs"] = {
                    "pred_logits": interm_class,
                    "pred_boxes": interm_coord,
                }
                out["interm_outputs_for_matching_pre"] = {
                    "pred_logits": interm_class,
                    "pred_boxes": init_box_proposal,
                }

            # prepare enc outputs
            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for (
                    layer_box_embed,
                    layer_class_embed,
                    layer_hs_enc,
                    layer_ref_enc,
                ) in zip(
                    self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1]
                ):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = (
                        layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    )
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                if not self.export:
                    out["enc_outputs"] = [
                        {"pred_logits": a, "pred_boxes": b}
                        for a, b in zip(enc_outputs_class, enc_outputs_coord)
                    ]

        if not self.export:
            out["dn_meta"] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        """This is a workaround to make torchscript happy, as torchscript
        doesn't support dictionary with non-homogeneous values, such
        as a dict having both a Tensor and a list.
        """
        return [
            {"pred_logits": a, "pred_boxes": b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
