"""The build nn module model."""

import torch.nn as nn

from .backbone import Backbone, Joiner
from .deformable_transformer import DeformableTransformer
from .dino import DINO

from .position_encoding import PositionEmbeddingSineHW, PositionEmbeddingSineHWExport


class DINOModel(nn.Module):
    """DINO model module."""

    def __init__(
        self,
        num_classes=4,
        hidden_dim=256,
        pretrained_backbone_path=None,
        backbone="resnet_50",
        train_backbone=True,
        num_feature_levels=2,
        nheads=8,
        enc_layers=6,
        dec_layers=6,
        dim_feedforward=1024,
        dec_n_points=4,
        enc_n_points=4,
        num_queries=300,
        aux_loss=True,
        dilation=False,
        dropout_ratio=0.3,
        export=False,
        activation_checkpoint=True,
        return_interm_indices=[1, 2, 3, 4],
        pre_norm=False,
        num_patterns=0,
        decoder_layer_noise=False,
        dln_xy_noise=0.2,
        dln_hw_noise=0.2,
        add_channel_attention=False,
        random_refpoints_xy=False,
        two_stage_type="standard",
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        decoder_sa_type="sa",
        embed_init_tgt=True,
        use_detached_boxes_dec_out=False,
        fix_refpoints_hw=-1,
        dec_pred_class_embed_share=False,
        dec_pred_bbox_embed_share=False,
        two_stage_bbox_embed_share=False,
        two_stage_class_embed_share=False,
        use_dn=True,
        dn_number=100,
        dn_box_noise_scale=1.0,
        dn_label_noise_ratio=0.5,
        pe_temperatureH=20,
        pe_temperatureW=20,
        lsj_resolution=None,
    ):
        """Initialize DINO Model.

        Args:
            num_classes (int): number of classes for the model.
            hidden_dim (int): size of the hidden dimension.
            pretrained_backbone_path (str): pretrained backbone path.
                                            If not provided, train from scratch.
            backbone (str): type of backbone architecture.
            train_backbone (bool): whether to train backbone or not.
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            nheads (int): number of heads.
            enc_layers (int): number of encoder layers.
            dec_layers (int): number of decoder layers.
            dim_feedforward (int): dimension of the feedforward layer.
            dec_n_points (int): number of reference points in the decoder.
            enc_n_points (int): number of reference points in the encoder.
            num_queries (int): number of queries to be used in D-DETR encoder-decoder.
            aux_loss (bool): flag to indicate if auxiliary loss is used.
            dilation (bool): flag to indicate if dilation is used (only for ResNet).
            dropout_ratio (float): probability for the dropout layer.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
            return_interm_indices (list): indices of feature level to use.
            pre_norm (bool): whether to add LayerNorm before the encoder.
            num_patterns (int): number of patterns in encoder-decoder.
            decoder_layer_noise (bool): a flag to add random perturbation to decoder query.
            dln_xy_noise (float): scale of noise applied along xy dimension during random perturbation.
            dln_hw_noise (float): scale of noise applied along wh dimension during random perturbation.
            add_channel_attention (bool): whether to add channel attention.
            random_refpoints_xy (bool): whether to randomly initialize reference point embedding.
            two_stage_type (str): type of two stage in DINO.
            two_stage_pat_embed (int): size of the patch embedding for the second stage.
            two_stage_add_query_num (int): size of the target embedding.
            two_stage_learn_wh (bool): add embedding for learnable w and h.
            two_stage_keep_all_tokens (bool): whether to keep all tokens in the second stage.
            decoder_sa_type (str): type of self-attention in the decoder.
            embed_init_tgt (bool): whether to add target embedding.
            use_detached_boxes_dec_out (bool): use detached box decoder output in the reference points.
            fix_refpoints_hw (int): If this value is -1, width and height are learned seperately for each box.
                                    If this value is -2, a shared w and h are learned.
                                    A value greater than 0 specifies learning with a fixed number.
            dec_pred_class_embed_share (bool): whether to share embedding for decoder classification prediction.
            dec_pred_bbox_embed_share (bool): whether to share embedding for decoder bounding box prediction.
            two_stage_bbox_embed_share (bool): whether to share embedding for two stage bounding box.
            two_stage_class_embed_share (bool): whether to share embedding for two stage classification.
            use_dn (bool): a flag specifying whether to enbable contrastive de-noising training in DINO.
            dn_number (bool): the number of de-noising queries in DINO
            dn_box_noise_scale (float): the scale of noise applied to boxes during contrastive de-noising.
                                        If this value is 0, noise is not applied.
            dn_label_noise_ratio (float): the scale of noise applied to labels during contrastive de-noising.
                                          If this value is 0, noise is not applied.
            pe_temperatureH (int): the temperature applied to the height dimension of Positional Sine Embedding.
            pe_temperatureW (int): the temperature applied to the width dimension of Positional Sine Embedding.
            lsj_resolution (int): LSJ resolution taht is only applicable for ViT backbones.
        """
        super(__class__, self).__init__()  # pylint:disable=undefined-variable

        # TODO: Update position_embedding in the build stage
        # build positional encoding. only support PositionEmbeddingSine
        if export:
            position_embedding = PositionEmbeddingSineHWExport(
                hidden_dim // 2,
                temperatureH=pe_temperatureH,
                temperatureW=pe_temperatureW,
                normalize=True,
            )
        else:
            position_embedding = PositionEmbeddingSineHW(
                hidden_dim // 2,
                temperatureH=pe_temperatureH,
                temperatureW=pe_temperatureW,
                normalize=True,
            )

        # build backbone
        if num_feature_levels != len(return_interm_indices):
            raise ValueError(
                f"num_feature_levels: {num_feature_levels} does not match the size of "
                f"return_interm_indices: {return_interm_indices}"
            )

        # sanity check for ViT backbones
        if backbone.startswith("vit") and lsj_resolution is None:
            raise ValueError(
                f"{backbone} requires dataset.augmentation.fixed_random_crop to be set. "
                "Please set dataset.augmentation.fixed_random_crop in the spec file."
            )

        # Index 4 is not part of the backbone but taken from index 3 with conv 3x3 stride 2
        return_interm_indices = [r for r in return_interm_indices if r != 4]
        backbone_only = Backbone(
            backbone,
            pretrained_backbone_path,
            train_backbone,
            lsj_resolution,
            return_interm_indices,
            dilation,
            export,
            activation_checkpoint,
        )

        # Keep joiner for backward compatibility
        joined_backbone = Joiner(backbone_only)

        decoder_query_perturber = None
        if decoder_layer_noise:
            from .model_utils import RandomBoxPerturber

            decoder_query_perturber = RandomBoxPerturber(
                x_noise_scale=dln_xy_noise,
                y_noise_scale=dln_xy_noise,
                w_noise_scale=dln_hw_noise,
                h_noise_scale=dln_hw_noise,
            )

        # build tranformer
        transformer = DeformableTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            export=export,
            activation_checkpoint=activation_checkpoint,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_ratio,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            enc_n_points=enc_n_points,
            dec_n_points=dec_n_points,
            num_queries=num_queries,
            normalize_before=pre_norm,
            num_patterns=num_patterns,
            modulate_hw_attn=True,
            deformable_decoder=True,
            decoder_query_perturber=decoder_query_perturber,
            add_channel_attention=add_channel_attention,
            random_refpoints_xy=random_refpoints_xy,
            # two stage
            two_stage_type=two_stage_type,  # ['no', 'standard', 'early']
            two_stage_pat_embed=two_stage_pat_embed,
            two_stage_add_query_num=two_stage_add_query_num,
            two_stage_learn_wh=two_stage_learn_wh,
            two_stage_keep_all_tokens=two_stage_keep_all_tokens,
            dec_layer_number=None,
            rm_self_attn_layers=None,
            key_aware_type=None,
            layer_share_type=None,
            rm_detach=None,
            decoder_sa_type=decoder_sa_type,
            module_seq=["sa", "ca", "ffn"],
            embed_init_tgt=embed_init_tgt,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        )

        # build deformable detr model
        self.model = DINO(
            joined_backbone,
            position_embedding,
            transformer,
            num_classes=num_classes,
            num_queries=num_queries,
            aux_loss=aux_loss,
            export=export,
            random_refpoints_xy=random_refpoints_xy,
            fix_refpoints_hw=fix_refpoints_hw,
            num_feature_levels=num_feature_levels,
            nheads=nheads,
            dec_pred_class_embed_share=dec_pred_class_embed_share,
            dec_pred_bbox_embed_share=dec_pred_bbox_embed_share,
            # two stage
            two_stage_type=two_stage_type,
            # box_share
            two_stage_bbox_embed_share=two_stage_bbox_embed_share,
            two_stage_class_embed_share=two_stage_class_embed_share,
            decoder_sa_type=decoder_sa_type,
            num_patterns=num_patterns,
            dn_number=dn_number if use_dn else 0,
            dn_box_noise_scale=dn_box_noise_scale,
            dn_label_noise_ratio=dn_label_noise_ratio,
            dn_labelbook_size=num_classes,
        )

    def forward(self, x, targets=None):
        """model forward function"""
        x = self.model(x, targets)
        return x
