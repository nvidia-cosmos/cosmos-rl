"""Deformable Transformer functions."""

import math
import random
import copy
from typing import Optional

import torch
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint

from cosmos_rl.utils.logging import logger

from .model_utils import (
    gen_encoder_output_proposals,
    MLP,
    _get_activation_fn,
    gen_sineembed_for_position,
)
from .deformable_detr.utils import inverse_sigmoid
from .deformable_detr.model.ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    """Deformable Transfromer module"""

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_queries=300,
        export=False,
        activation_checkpoint=True,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        num_patterns=0,
        modulate_hw_attn=False,
        # for deformable encoder
        deformable_decoder=False,
        num_feature_levels=1,
        enc_n_points=4,
        dec_n_points=4,
        # init query
        decoder_query_perturber=None,
        add_channel_attention=False,
        random_refpoints_xy=False,
        # two stage
        two_stage_type="no",  # ['no', 'standard']
        two_stage_pat_embed=0,
        two_stage_add_query_num=0,
        two_stage_learn_wh=False,
        two_stage_keep_all_tokens=False,
        # evo of #anchors
        dec_layer_number=None,
        rm_self_attn_layers=None,
        key_aware_type=None,
        # layer share
        layer_share_type=None,
        # for detach
        rm_detach=None,
        decoder_sa_type="ca",
        module_seq=["sa", "ca", "ffn"],
        # for dn
        embed_init_tgt=False,
        use_detached_boxes_dec_out=False,
    ):
        """Initialize Encoder-Decoder Class for DINO.

        Args:
            d_model (int): size of the hidden dimension.
            nheads (int): number of heads.
            num_queries (int): number of queries to be used in D-DETR encoder-decoder.
            export (bool): flag to indicate if the current model is being used for ONNX export.
            activation_checkpoint (bool): flag to indicate if activation checkpointing is used.
            num_encoder_layers (int): number of encoder layers.
            num_decoder_layers (int): number of decoder layers.
            dim_feedforward (int): dimension of the feedforward layer.
            dropout (float): probability for the dropout layer.
            activation (str): type of activation.
            normalize_before (bool): whether to add LayerNorm before the encoder.
            return_intermediate_dec (bool): whether to return intermediate decoder.
            num_patterns (int): number of patterns in encoder-decoder.
            modulate_hw_attn (bool): whether to apply modulated HW attentions.
            deformable_attention (bool): whether to apply deformable attention.
            num_feature_levels (int): Number of levels to extract from the backbone feature maps.
            enc_n_points (int): number of reference points in the encoder.
            dec_n_points (int): number of reference points in the decoder.
            decoder_query_perturber (class): RandomBoxPertuber.
            add_channel_attention (bool): whether to add channel attention.
            random_refpoints_xy (bool): whether to randomly initialize reference point embedding.
            two_stage_type (str): type of two stage in DINO.
            two_stage_pat_embed (int): size of the patch embedding for the second stage.
            two_stage_add_query_num (int): size of the target embedding.
            two_stage_learn_wh (bool): add embedding for learnable w and h.
            two_stage_keep_all_tokens (bool): whether to keep all tokens in the second stage.
            dec_layer_number (int): number of decoder layers.
            rm_self_attn_layers (bool): remove self-attention in decoder.
            key_aware_type (str): type of key_aware in cross-attention.
            layer_share_type (str): type of layer sharing.
            rm_detach (list): list of names to remove detach.
            decoder_sa_type (str): type of self-attention in the decoder.
            module_seq (list): sequence of modules in the forward function.
            embed_init_tgt (bool): whether to add target embedding.
            use_detached_boxes_dec_out (bool): use detached box decoder output in the reference points.
        """
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.deformable_decoder = deformable_decoder
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        self.num_queries = num_queries
        self.random_refpoints_xy = random_refpoints_xy
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.export = export

        assert layer_share_type in [None, "encoder", "decoder", "both"]

        enc_layer_share = layer_share_type in ["encoder", "both"]
        dec_layer_share = layer_share_type in ["decoder", "both"]

        assert layer_share_type is None

        self.decoder_sa_type = decoder_sa_type
        supported_decoder_types = ["sa", "ca_label", "ca_content"]
        if decoder_sa_type not in supported_decoder_types:
            raise NotImplementedError(
                f"Decoder type {decoder_sa_type} unsupported. Please set the decoder type to any one of {supported_decoder_types}"
            )

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
            add_channel_attention=add_channel_attention,
            export=export,
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            encoder_norm,
            d_model=d_model,
            num_queries=num_queries,
            enc_layer_share=enc_layer_share,
            two_stage_type=two_stage_type,
            export=export,
            activation_checkpoint=activation_checkpoint,
        )

        # choose decoder layer type
        if deformable_decoder:
            decoder_layer = DeformableTransformerDecoderLayer(
                d_model,
                dim_feedforward,
                dropout,
                activation,
                num_feature_levels,
                nhead,
                dec_n_points,
                key_aware_type=key_aware_type,
                decoder_sa_type=decoder_sa_type,
                module_seq=module_seq,
                export=export,
            )
        else:
            raise NotImplementedError

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            export=export,
            activation_checkpoint=activation_checkpoint,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=4,
            modulate_hw_attn=modulate_hw_attn,
            num_feature_levels=num_feature_levels,
            deformable_decoder=deformable_decoder,
            decoder_query_perturber=decoder_query_perturber,
            dec_layer_number=dec_layer_number,
            dec_layer_share=dec_layer_share,
            use_detached_boxes_dec_out=use_detached_boxes_dec_out,
        )

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
        if not isinstance(num_patterns, int):
            try:
                num_patterns = int(num_patterns)
            except Exception:
                logger.warning(
                    "num_patterns should be int but {}".format(type(num_patterns))
                )
                num_patterns = 0
        self.num_patterns = num_patterns

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(
                    torch.Tensor(num_feature_levels, d_model)
                )
            else:
                self.level_embed = None

        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        # for two stage
        self.two_stage_type = two_stage_type
        self.two_stage_pat_embed = two_stage_pat_embed
        self.two_stage_add_query_num = two_stage_add_query_num
        self.two_stage_learn_wh = two_stage_learn_wh
        assert two_stage_type in [
            "no",
            "standard",
        ], f"unknown param {two_stage_type} of two_stage_type"
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_pat_embed > 0:
                self.pat_embed_for_2stage = nn.Parameter(
                    torch.Tensor(two_stage_pat_embed, d_model)
                )
                nn.init.normal_(self.pat_embed_for_2stage)

            if two_stage_add_query_num > 0:
                self.tgt_embed = nn.Embedding(self.two_stage_add_query_num, d_model)

            if two_stage_learn_wh:
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        self.enc_out_class_embed = None
        self.enc_out_bbox_embed = None

        # evolution of anchors
        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            if self.two_stage_type != "no" or num_patterns == 0:
                assert (
                    dec_layer_number[0] == num_queries
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries})"
            else:
                assert (
                    dec_layer_number[0] == num_queries * num_patterns
                ), f"dec_layer_number[0]({dec_layer_number[0]}) != num_queries({num_queries}) * num_patterns({num_patterns})"

        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            logger.info(
                "Removing the self-attn in {} decoder layers".format(
                    rm_self_attn_layers
                )
            )
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ["enc_ref", "enc_tgt", "dec"] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        """Reset parmaeters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(
                self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05))
            )

    def get_valid_ratio(self, mask):
        """Compute the valid ratio from given mask"""
        _, H, W = mask.shape
        temp_mask = mask.bool()
        valid_H = torch.sum((~temp_mask).float()[:, :, 0], 1)
        valid_W = torch.sum((~temp_mask).float()[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def init_ref_points(self, use_num_queries):
        """Initialize reference points"""
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

        if self.random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def forward(self, srcs, masks, refpoint_embed, pos_embeds, tgt, attn_mask=None):
        """Encoder-Decoder forward function.

        Args:
            srcs (torch.Tensor): List of multi features [bs, ci, hi, wi].
            masks (torch.Tensor): List of multi masks [bs, hi, wi].
            refpoint_embed (torch.Tensor): [bs, num_dn, 4]. None in infer.
            pos_embeds (torch.Tensor): List of multi pos embeds [bs, ci, hi, wi].
            tgt (torch.Tensor): [bs, num_dn, d_model]. None in infer.

        Returns:
            hs (torch.Tensor): (n_dec, bs, nq, d_model)
            references (torch.Tensor): sigmoid coordinates. (n_dec+1, bs, bq, 4)
            hs_enc (torch.Tensor): (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
            ref_enc (torch.Tensor): sigmoid coordinates. \
                      (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None
        """
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        if self.export:
            spatial_shapes = []
        else:
            spatial_shapes = torch.empty(
                len(srcs), 2, dtype=torch.int32, device=srcs[0].device
            )
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, _, h, w = src.shape
            if self.export:  # Input shape is fixed for export in onnx/tensorRT
                spatial_shapes.append(
                    torch.tensor([[h, w]], dtype=torch.int32, device=srcs[0].device)
                )
            else:  # Used for dynamic input shape
                spatial_shapes[lvl, 0], spatial_shapes[lvl, 1] = h, w

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.level_embed is not None:
                lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)  # bs, \sum{hxw}, c
        if isinstance(spatial_shapes, list):
            spatial_shapes = torch.cat(spatial_shapes, 0)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # two stage
        enc_topk_proposals = enc_refpoint_embed = None

        #########################################################
        # Begin Encoder
        #########################################################
        memory, _, _ = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            ref_token_index=enc_topk_proposals,  # bs, nq
            ref_token_coord=enc_refpoint_embed,  # bs, nq, 4
        )
        #########################################################
        # End Encoder
        # - memory: bs, \sum{hw}, c
        # - mask_flatten: bs, \sum{hw}
        # - lvl_pos_embed_flatten: bs, \sum{hw}, c
        # - enc_intermediate_output: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        # - enc_intermediate_refpoints: None or (nenc+1, bs, nq, c) or (nenc, bs, nq, c)
        #########################################################

        if self.two_stage_type == "standard":
            if self.two_stage_learn_wh:
                input_hw = self.two_stage_wh_embedding.weight[0]
            else:
                input_hw = None
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes, input_hw, export=self.export
            )
            output_memory = self.enc_output_norm(self.enc_output(output_memory))

            if self.two_stage_pat_embed > 0:
                bs, nhw, _ = output_memory.shape
                # output_memory: bs, n, 256; self.pat_embed_for_2stage: k, 256
                output_memory = output_memory.repeat(1, self.two_stage_pat_embed, 1)
                _pats = self.pat_embed_for_2stage.repeat_interleave(nhw, 0)
                output_memory = output_memory + _pats
                output_proposals = output_proposals.repeat(
                    1, self.two_stage_pat_embed, 1
                )

            if self.two_stage_add_query_num > 0:
                assert refpoint_embed is not None
                output_memory = torch.cat((output_memory, tgt), dim=1)
                output_proposals = torch.cat((output_proposals, refpoint_embed), dim=1)

            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            enc_outputs_coord_unselected = (
                self.enc_out_bbox_embed(output_memory) + output_proposals
            )  # (bs, \sum{hw}, 4) unsigmoid
            topk = self.num_queries
            topk_proposals = torch.topk(
                enc_outputs_class_unselected.max(-1)[0], topk, dim=1
            )[1]  # bs, nq

            # gather boxes
            refpoint_embed_undetach = torch.gather(
                enc_outputs_coord_unselected,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
            )  # unsigmoid
            refpoint_embed_ = refpoint_embed_undetach.detach()
            init_box_proposal = torch.gather(
                output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
            ).sigmoid()  # sigmoid

            # gather tgt
            tgt_undetach = torch.gather(
                output_memory,
                1,
                topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
            )
            if self.embed_init_tgt:
                tgt_ = (
                    self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
                )  # nq, bs, d_model
            else:
                tgt_ = tgt_undetach.detach()

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

        elif self.two_stage_type == "no":
            tgt_ = (
                self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, d_model
            refpoint_embed_ = (
                self.refpoint_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
            )  # nq, bs, 4

            if refpoint_embed is not None:
                refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
                tgt = torch.cat([tgt, tgt_], dim=1)
            else:
                refpoint_embed, tgt = refpoint_embed_, tgt_

            if self.num_patterns > 0:
                tgt_embed = tgt.repeat(1, self.num_patterns, 1)
                refpoint_embed = refpoint_embed.repeat(1, self.num_patterns, 1)
                tgt_pat = self.patterns.weight[None, :, :].repeat_interleave(
                    self.num_queries, 1
                )  # 1, n_q*n_pat, d_model
                tgt = tgt_embed + tgt_pat

            init_box_proposal = refpoint_embed_.sigmoid()

        else:
            raise NotImplementedError(
                "unknown two_stage_type {}".format(self.two_stage_type)
            )
        #########################################################
        # End preparing tgt
        # - tgt: bs, NQ, d_model
        # - refpoint_embed(unsigmoid): bs, NQ, d_model
        #########################################################

        #########################################################
        # Begin Decoder
        #########################################################
        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
        )
        #########################################################
        # End Decoder
        # hs: n_dec, bs, nq, d_model
        # references: n_dec+1, bs, nq, query_dim
        #########################################################

        #########################################################
        # Begin postprocess
        #########################################################
        if self.two_stage_type == "standard":
            if self.two_stage_keep_all_tokens:
                hs_enc = output_memory.unsqueeze(0)
                ref_enc = enc_outputs_coord_unselected.unsqueeze(0)
                init_box_proposal = output_proposals

            else:
                hs_enc = tgt_undetach.unsqueeze(0)
                ref_enc = refpoint_embed_undetach.sigmoid().unsqueeze(0)
        else:
            hs_enc = ref_enc = None
        #########################################################
        # End postprocess
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or (n_enc, bs, nq, d_model) or None
        # ref_enc: (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or (n_enc, bs, nq, d_model) or None
        #########################################################

        return hs, references, hs_enc, ref_enc, init_box_proposal
        # hs: (n_dec, bs, nq, d_model)
        # references: sigmoid coordinates. (n_dec+1, bs, bq, 4)
        # hs_enc: (n_enc+1, bs, nq, d_model) or (1, bs, nq, d_model) or None
        # ref_enc: sigmoid coordinates. \
        #           (n_enc+1, bs, nq, query_dim) or (1, bs, nq, query_dim) or None


class TransformerEncoder(nn.Module):
    """Deformable Transfromer Encoder module"""

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        d_model=256,
        num_queries=300,
        enc_layer_share=False,
        enc_layer_dropout_prob=None,
        two_stage_type="no",  # ['no', 'standard']
        export=False,
        activation_checkpoint=True,
    ):
        """Initializes the Transformer Encoder Module"""
        super().__init__()
        # prepare layers
        if num_layers > 0:
            self.layers = _get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )
        else:
            self.layers = []
            del encoder_layer
        self.activation_checkpoint = activation_checkpoint
        self.export = export
        self.query_scale = None
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

        self.enc_layer_dropout_prob = enc_layer_dropout_prob
        if enc_layer_dropout_prob is not None:
            assert isinstance(enc_layer_dropout_prob, list)
            assert len(enc_layer_dropout_prob) == num_layers
            for i in enc_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.two_stage_type = two_stage_type

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device, export=False):
        """get reference points"""
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            if export:  # Fixed dimensions for export in onnx
                H_, W_ = int(H_), int(W_)
            else:
                H_, W_ = spatial_shapes[lvl, 0], spatial_shapes[lvl, 1]
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        key_padding_mask: Tensor,
        ref_token_index: Optional[Tensor] = None,
        ref_token_coord: Optional[Tensor] = None,
    ):
        """Deformable Encoder forward functions.

        Args:
            src (torch.Tensor): [bs, sum(hi*wi), 256].
            pos (torch.Tensor): pos embed for src. [bs, sum(hi*wi), 256].
            spatial_shapes (torch.Tensor): h,w of each level [num_level, 2].
            level_start_index (torch.Tensor): [num_level] start point of level in sum(hi*wi)..
            valid_ratios (torch.Tensor): [bs, num_level, 2].
            key_padding_mask (torch.Tensor): [bs, sum(hi*wi)].
            ref_token_index (torch.Tensor): bs, nq.
            ref_token_coord (torch.Tensor): bs, nq, 4.

        Returns:
            output (torch.Tensor): [bs, sum(hi*wi), 256].
            reference_points (torch.Tensor): [bs, sum(hi*wi), num_level, 2].
        """
        if self.two_stage_type in ["no", "standard"]:
            assert ref_token_index is None

        output = src
        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device, export=self.export
            )

        intermediate_output = []
        intermediate_ref = []
        if ref_token_index is not None:
            out_i = torch.gather(
                output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model)
            )
            intermediate_output.append(out_i)
            intermediate_ref.append(ref_token_coord)

        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            dropflag = False
            if self.enc_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.enc_layer_dropout_prob[layer_id]:
                    dropflag = True

            if not dropflag:
                if self.export or not self.activation_checkpoint:
                    output = layer(
                        src=output,
                        pos=pos,
                        reference_points=reference_points,
                        spatial_shapes=spatial_shapes,
                        level_start_index=level_start_index,
                        key_padding_mask=key_padding_mask,
                    )
                else:
                    output = checkpoint.checkpoint(
                        layer,
                        output,
                        pos,
                        reference_points,
                        spatial_shapes,
                        level_start_index,
                        key_padding_mask,
                        use_reentrant=True,
                    )

            # aux loss
            if (layer_id != self.num_layers - 1) and ref_token_index is not None:
                out_i = torch.gather(
                    output, 1, ref_token_index.unsqueeze(-1).repeat(1, 1, self.d_model)
                )
                intermediate_output.append(out_i)
                intermediate_ref.append(ref_token_coord)

        if self.norm is not None:
            output = self.norm(output)

        if ref_token_index is not None:
            intermediate_output = torch.stack(
                intermediate_output
            )  # n_enc/n_enc-1, bs, \sum{hw}, d_model
            intermediate_ref = torch.stack(intermediate_ref)
        else:
            intermediate_output = intermediate_ref = None

        return output, intermediate_output, intermediate_ref


class TransformerDecoder(nn.Module):
    """Deformable Transfromer Decoder module"""

    def __init__(
        self,
        decoder_layer,
        num_layers,
        norm=None,
        export=False,
        activation_checkpoint=True,
        return_intermediate=False,
        d_model=256,
        query_dim=4,
        modulate_hw_attn=False,
        num_feature_levels=1,
        deformable_decoder=False,
        decoder_query_perturber=None,
        dec_layer_number=None,  # number of queries each layer in decoder
        dec_layer_share=False,
        dec_layer_dropout_prob=None,
        use_detached_boxes_dec_out=False,
    ):
        """Initializes the Transformer Decoder Module"""
        super().__init__()
        self.export = export
        self.activation_checkpoint = activation_checkpoint
        if num_layers > 0:
            self.layers = _get_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share
            )
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        self.query_scale = None

        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
        # for memory
        level_start_index: Optional[Tensor] = None,  # num_levels
        spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        valid_ratios: Optional[Tensor] = None,
    ):
        """Deformable Decoder forward function.

        Args:
            tgt (torch.Tensor): nq, bs, d_model.
            memory (torch.Tensor): hw, bs, d_model.
            pos (torch.Tensor): hw, bs, d_model.
            refpoints_unsigmoid (torch.Tensor): nq, bs, 2/4.
            valid_ratios/spatial_shapes (torch.Tensor): bs, nlevel, 2.
        """
        output = tgt

        intermediate = []
        # Keep dtype consistent with activations/weights for mixed precision
        reference_points = refpoints_unsigmoid.sigmoid().to(output.dtype)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # preprocess ref points
            if (
                self.training
                and self.decoder_query_perturber is not None
                and layer_id != 0
            ):
                reference_points = self.decoder_query_perturber(reference_points)

            if self.deformable_decoder:
                valid_ratios = valid_ratios.to(output.dtype)
                if reference_points.shape[-1] == 4:
                    # nq, bs, nlevel, 4
                    reference_points_input = (
                        reference_points[:, :, None]
                        * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                    )
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[None, :]
                    )
                query_sine_embed = gen_sineembed_for_position(
                    reference_points_input[:, :, 0, :]
                )  # nq, bs, 256*2
            else:
                query_sine_embed = gen_sineembed_for_position(
                    reference_points
                )  # nq, bs, 256*2
                reference_points_input = None

            # conditional query
            raw_query_pos = self.ref_point_head(
                query_sine_embed.to(output.dtype)
            )  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos
            if not self.deformable_decoder:
                query_sine_embed = query_sine_embed[
                    ..., : self.d_model
                ] * self.query_pos_sine_scale(output)

            # modulated HW attentions
            if not self.deformable_decoder and self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 2
                query_sine_embed[..., self.d_model // 2 :] *= (
                    refHW_cond[..., 0] / reference_points[..., 2]
                ).unsqueeze(-1)
                query_sine_embed[..., : self.d_model // 2] *= (
                    refHW_cond[..., 1] / reference_points[..., 3]
                ).unsqueeze(-1)

            # random drop some layers if needed
            dropflag = False
            if self.dec_layer_dropout_prob is not None:
                prob = random.random()
                if prob < self.dec_layer_dropout_prob[layer_id]:
                    dropflag = True
            if not dropflag:
                if self.export or not self.activation_checkpoint:
                    output = layer(
                        tgt=output,
                        tgt_query_pos=query_pos,
                        tgt_query_sine_embed=query_sine_embed,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        tgt_reference_points=reference_points_input,
                        memory=memory,
                        memory_key_padding_mask=memory_key_padding_mask,
                        memory_level_start_index=level_start_index,
                        memory_spatial_shapes=spatial_shapes,
                        memory_pos=pos,
                        self_attn_mask=tgt_mask,
                        cross_attn_mask=memory_mask,
                    )
                else:
                    output = checkpoint.checkpoint(
                        layer,
                        output,
                        query_pos,
                        query_sine_embed,
                        tgt_key_padding_mask,
                        reference_points_input,
                        memory,
                        memory_key_padding_mask,
                        level_start_index,
                        spatial_shapes,
                        pos,
                        tgt_mask,
                        memory_mask,
                        use_reentrant=True,
                    )

            # iter update
            if self.bbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)  # pylint: disable=E1136
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                # select # ref points
                if (
                    self.dec_layer_number is not None
                    and layer_id != self.num_layers - 1
                ):
                    nq_now = new_reference_points.shape[0]
                    select_number = self.dec_layer_number[layer_id + 1]
                    if nq_now != select_number:
                        # pylint: disable=E1136
                        class_unselected = self.class_embed[layer_id](
                            output
                        )  # nq, bs, 91
                        topk_proposals = torch.topk(
                            class_unselected.max(-1)[0], select_number, dim=0
                        )[1]  # new_nq, bs
                        new_reference_points = torch.gather(
                            new_reference_points,
                            0,
                            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
                        )  # unsigmoid

                if self.rm_detach and "dec" in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()
                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))
            if self.dec_layer_number is not None and layer_id != self.num_layers - 1:
                if nq_now != select_number:
                    output = torch.gather(
                        output,
                        0,
                        topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model),
                    )  # unsigmoid

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
        ]


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable Transfromer Encoder Layer module"""

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        add_channel_attention=False,
        export=False,
    ):
        """Initializes the Transformer Encoder Layer"""
        super().__init__()
        self.export = export

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = _get_activation_fn("dyrelu")
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional Embedding to the tensor"""
        return tensor if pos is None else tensor + pos.to(tensor.dtype)

    def forward_ffn(self, src):
        """Feed-forward network forward function"""
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        key_padding_mask=None,
    ):
        """forward function for Encoder Layer"""
        # self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            key_padding_mask,
            export=self.export,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transfromer Decoder Layer module"""

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        key_aware_type=None,
        decoder_sa_type="sa",
        module_seq=["sa", "ca", "ffn"],
        export=False,
    ):
        """Initializes the Transformer Decoder Layer"""
        super().__init__()
        self.module_seq = module_seq
        assert sorted(module_seq) == ["ca", "ffn", "sa"]
        self.export = export

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        if self.export:
            # Starting from PyT 1.14, _scaled_dot_product_attention has been switched to C++ backend
            # which is not exportable as ONNX operator
            # However, the training / eval time can be greatly optimized by Torch selecting the optimal
            # attention mechanism under the hood
            # self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
            raise ValueError(
                "ONNX-compatible MultiheadAttention module implmenetaiton missing."
            )
        else:
            self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)

        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None
        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]

        if decoder_sa_type == "ca_content":
            self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)

    def rm_self_attn_modules(self):
        """Remove self attention module"""
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        """Add positional Embedding to the tensor"""
        return tensor if pos is None else tensor + pos.to(tensor.dtype)

    def forward_ffn(self, tgt):
        """Feed-forward network forward function"""
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_sa(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Self-Attention forward function"""
        # self attention
        if self.self_attn is not None:
            if self.decoder_sa_type == "sa":
                q = k = self.with_pos_embed(tgt, tgt_query_pos)
                tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == "ca_label":
                bs = tgt.shape[1]
                k = v = self.label_embedding.weight[:, None, :].repeat(1, bs, 1)
                tgt2 = self.self_attn(tgt, k, v, attn_mask=self_attn_mask)[0]
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            elif self.decoder_sa_type == "ca_content":
                tgt2 = self.self_attn(
                    self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                    tgt_reference_points.transpose(0, 1).contiguous(),
                    memory.transpose(0, 1),
                    memory_spatial_shapes,
                    memory_level_start_index,
                    memory_key_padding_mask,
                ).transpose(0, 1)
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)
            else:
                raise NotImplementedError(
                    "Unknown decoder_sa_type {}".format(self.decoder_sa_type)
                )

        return tgt

    def forward_ca(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Cross-Attention forward function"""
        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == "mean":
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == "proj_mean":
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError(
                    "Unknown key_aware_type: {}".format(self.key_aware_type)
                )
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
            export=self.export,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        return tgt

    def forward(
        self,
        # for tgt
        tgt: Optional[Tensor],  # nq, bs, d_model
        tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
        tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4
        # for memory
        memory: Optional[Tensor] = None,  # hw, bs, d_model
        memory_key_padding_mask: Optional[Tensor] = None,
        memory_level_start_index: Optional[Tensor] = None,  # num_levels
        memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
        memory_pos: Optional[Tensor] = None,  # pos for memory
        # sa
        self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
        cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
    ):
        """Forward function"""
        for funcname in self.module_seq:
            if funcname == "ffn":
                tgt = self.forward_ffn(tgt)
            elif funcname == "ca":
                tgt = self.forward_ca(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            elif funcname == "sa":
                tgt = self.forward_sa(
                    tgt,
                    tgt_query_pos,
                    tgt_query_sine_embed,
                    tgt_key_padding_mask,
                    tgt_reference_points,
                    memory,
                    memory_key_padding_mask,
                    memory_level_start_index,
                    memory_spatial_shapes,
                    memory_pos,
                    self_attn_mask,
                    cross_attn_mask,
                )
            else:
                raise ValueError(f"Unknown funcname {funcname}")

        return tgt


def _get_clones(module, N, layer_share=False):
    """get clones"""
    if layer_share:
        return nn.ModuleList([module for i in range(N)])

    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
