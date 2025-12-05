import copy

import torch.nn as nn

from .matcher import HungarianMatcher
from .criterion import SetCriterion


class DINOLoss(nn.Module):
    def __init__(self, model_config, dataset_config):
        self.matcher = HungarianMatcher(
            cost_class=model_config["cls_loss_coef"],
            cost_bbox=model_config["bbox_loss_coef"],
            cost_giou=model_config["giou_loss_coef"],
        )
        weight_dict = {
            "loss_ce": model_config["cls_loss_coef"],
            "loss_bbox": model_config["bbox_loss_coef"],
            "loss_giou": model_config["giou_loss_coef"],
        }
        clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

        # for de-noising training
        if model_config["use_dn"]:
            weight_dict["loss_ce_dn"] = model_config["cls_loss_coef"]
            weight_dict["loss_bbox_dn"] = model_config["bbox_loss_coef"]
            weight_dict["loss_giou_dn"] = model_config["giou_loss_coef"]
        clean_weight_dict = copy.deepcopy(weight_dict)

        if model_config["aux_loss"]:
            aux_weight_dict = {}
            for i in range(model_config["dec_layers"] - 1):
                aux_weight_dict.update(
                    {k + f"_{i}": v for k, v in clean_weight_dict.items()}
                )
            weight_dict.update(aux_weight_dict)

        if model_config["two_stage_type"] != "no":
            interm_weight_dict = {}
            _coeff_weight_dict = {
                "loss_ce": 1.0,
                "loss_bbox": 1.0 if not model_config["no_interm_box_loss"] else 0.0,
                "loss_giou": 1.0 if not model_config["no_interm_box_loss"] else 0.0,
            }
            interm_weight_dict.update(
                {
                    f"{k}_interm": v
                    * model_config["interm_loss_coef"]
                    * _coeff_weight_dict[k]
                    for k, v in clean_weight_dict_wo_dn.items()
                }
            )
            weight_dict.update(interm_weight_dict)

        self.weight_dict = copy.deepcopy(weight_dict)

        self.criterion = SetCriterion(
            dataset_config["num_classes"],
            matcher=self.matcher,
            losses=model_config["loss_types"],
            focal_alpha=model_config["focal_alpha"],
        )

    def forward(self, outputs, targets):
        loss_dict = self.criterion(outputs, targets)

        losses = sum(
            loss_dict[k] * self.weight_dict[k]
            for k in loss_dict.keys()
            if k in self.weight_dict
        )

        return losses
