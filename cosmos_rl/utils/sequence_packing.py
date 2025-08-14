# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional
import torch
from cosmos_rl.utils.logging import logger

def compute_valid_lengths(input_ids, pad_token_id):
    # Find the last non-pad position in each row
    # Reverse along sequence dim, find first True, convert to length
    mask = (input_ids != pad_token_id)  # (B, T)
    flipped_mask = mask.flip(dims=[-1])  # reverse each row
    last_non_pad = flipped_mask.long().argmax(dim=-1)  # index from end
    valid_lengths = mask.size(1) - last_non_pad  # length = total - index from end
    is_all_pad = ~mask.any(dim=-1)
    valid_lengths = valid_lengths.masked_fill(is_all_pad, 0)
    return valid_lengths


def pack_sequences_for_inputs(
    input_ids: torch.Tensor,
    position_ids: torch.Tensor,
    pad_token_id: int,
    pos_seq_dim: int = 1,
    seq_len_multiple: int = 1,
    interested_tokens: Optional[torch.BoolTensor] = None,
    batch_sep_among_seq_len_multiple: bool = False,
):
    args_dict = {}
    with torch.no_grad():
        device = input_ids.device
        # Create mask: True where not padding
        valid_input_len = compute_valid_lengths(input_ids, pad_token_id)
        valid_input_mask = torch.arange(input_ids.size(1), device=device).expand(
            input_ids.size(0), -1
        ) < valid_input_len.unsqueeze(1)
        valid_input_len = valid_input_len.tolist()
        # logger.info(f"valid_input_len: {valid_input_len}")

        packed_input_ids = input_ids[valid_input_mask].unsqueeze(0)
        if batch_sep_among_seq_len_multiple:
            lensum_of_input_ids = [
                sum(
                    valid_input_len[
                        len(valid_input_len) // seq_len_multiple * idx : len(
                            valid_input_len
                        )
                        // seq_len_multiple
                        * (idx + 1)
                    ]
                )
                for idx in range(seq_len_multiple)
            ]
            packed_input_ids_parts = torch.split(packed_input_ids, lensum_of_input_ids, dim=-1)
            for idx, part in enumerate(packed_input_ids_parts):
                valid_input_len[
                    len(
                            valid_input_len
                        )
                        // seq_len_multiple
                        * (idx + 1) - 1
                ] += max(lensum_of_input_ids) - part.size(-1)
            packed_input_ids_parts = [
                torch.nn.functional.pad(
                    part, (0, max(lensum_of_input_ids) - part.size(-1)), value=pad_token_id
                ) for part in packed_input_ids_parts]
            packed_input_ids = torch.cat(packed_input_ids_parts, dim=-1)
        packed_input_ids = torch.nn.functional.pad(
            packed_input_ids, (0, (packed_input_ids.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - packed_input_ids.size(-1)), value=pad_token_id
        )
        args_dict["input_ids"] = packed_input_ids


        if interested_tokens is not None:
            interested_tokens = interested_tokens[valid_input_mask].unsqueeze(0)
            if batch_sep_among_seq_len_multiple:
                interested_tokens_parts = torch.split(interested_tokens, lensum_of_input_ids, dim=-1)
                interested_tokens_parts = [
                    torch.nn.functional.pad(
                        part, (0, max(lensum_of_input_ids) - part.size(-1)), value=0
                    ) for part in interested_tokens_parts]
                interested_tokens = torch.cat(interested_tokens_parts, dim=-1)
            interested_tokens = torch.nn.functional.pad(
                interested_tokens, (0, (interested_tokens.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - interested_tokens.size(-1)), value=0
            )
        args_dict["interested_tokens"] = interested_tokens


        assert valid_input_mask.ndim == 2, "valid_input_mask should be 2D"
        assert pos_seq_dim == position_ids.ndim - 1, "pos_seq_dim must be the last dimension of position_ids"
        position_ids_shape = list(position_ids.shape)
        position_ids_shape[pos_seq_dim] = -1
        if position_ids.ndim > valid_input_mask.ndim:
            for dim in range(position_ids.ndim):
                if dim not in [0, pos_seq_dim]:
                    valid_input_mask = valid_input_mask.unsqueeze(dim)
        else:
            assert position_ids.ndim == valid_input_mask.ndim, (
                "position_ids.ndim must be not less than valid_input_mask.ndim"
            )
       
        position_ids = position_ids[valid_input_mask.expand(position_ids.shape)].view(position_ids_shape[1:]).unsqueeze(0)
        if batch_sep_among_seq_len_multiple:
            position_ids_parts = torch.split(position_ids, lensum_of_input_ids, dim=-1)
            position_ids_parts = [
                torch.nn.functional.pad(
                    part, (0, max(lensum_of_input_ids) - part.size(-1)), value=1
                ) for part in position_ids_parts]
            position_ids = torch.cat(position_ids_parts, dim=-1)
        position_ids = torch.nn.functional.pad(position_ids, (0, (position_ids.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - position_ids.size(-1)), value=1)
        args_dict["position_ids"] = position_ids

        input_packing_mask = []
        for valid_len in valid_input_len:
            input_packing_mask.extend([1] * (valid_len - 1) + [0])
        input_packing_mask = torch.tensor([input_packing_mask], dtype=torch.bool).to(device)
        input_packing_mask = torch.nn.functional.pad(input_packing_mask, (0, (input_packing_mask.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - input_packing_mask.size(-1)), value=False)
        args_dict["input_packing_mask"] = input_packing_mask
        label_packing_mask = []
        for valid_len in valid_input_len:
            label_packing_mask.extend([0] + [1] * (valid_len - 1))
        label_packing_mask = torch.tensor([label_packing_mask], dtype=torch.bool).to(device)
        label_packing_mask = torch.nn.functional.pad(label_packing_mask, (0, (label_packing_mask.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - label_packing_mask.size(-1)), value=False)
        args_dict["label_packing_mask"] = label_packing_mask

        valid_input_len[-1] += (sum(valid_input_len) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - sum(valid_input_len)
        seq_len = torch.tensor(valid_input_len, dtype=torch.int32, device=device)
        prefill_start_pos = (
            torch.cumsum(seq_len, dim=0, dtype=torch.int32) - seq_len
        )
        cu_seqlens = torch.cat(
            [
                prefill_start_pos,
                torch.tensor([torch.sum(seq_len)], dtype=torch.int32).to(device),
            ],
            dim=0,
        )
        args_dict["cu_seqlens"] = cu_seqlens
        args_dict["max_seqlen"] = max(valid_input_len)

    return args_dict

def pack_sequences_for_labels(
    label_ids: torch.Tensor,
    ignore_label_id: int = -100,
    seq_len_multiple: int = 1,
    input_ids: Optional[torch.Tensor] = None,
    pad_token_id: Optional[int] = None,
    batch_sep_among_seq_len_multiple: bool = False,
):
    args_dict = {}
    with torch.no_grad():
        device = label_ids.device
        valid_label_len = compute_valid_lengths(label_ids, ignore_label_id)
        if input_ids is not None and pad_token_id is not None:
            valid_input_len = compute_valid_lengths(input_ids, pad_token_id)
            # Ensure label length does not exceed input length
            valid_label_len = torch.max(valid_label_len, valid_input_len)

        # logger.info(f"valid_label_len: {valid_label_len}")

        valid_label_mask = torch.arange(label_ids.size(1), device=device).expand(
            label_ids.size(0), -1
        ) < valid_label_len.unsqueeze(1)
        valid_label_len = valid_label_len.tolist()
        packed_label_ids = label_ids[valid_label_mask].unsqueeze(0)
        if batch_sep_among_seq_len_multiple:
            lensum_of_label_ids = [
                sum(
                    valid_label_len[
                        len(valid_label_len) // seq_len_multiple * idx : len(
                            valid_label_len
                        )
                        // seq_len_multiple
                        * (idx + 1)
                    ]
                )
                for idx in range(seq_len_multiple)
            ]
            packed_label_ids_parts = torch.split(packed_label_ids, lensum_of_label_ids, dim=-1)
            for idx, part in enumerate(packed_label_ids_parts):
                valid_label_len[
                    len(
                            valid_label_len
                        )
                        // seq_len_multiple
                        * (idx + 1) - 1
                ] += max(lensum_of_label_ids) - part.size(-1)
            packed_label_ids_parts = [
                torch.nn.functional.pad(
                    part, (0, max(lensum_of_label_ids) - part.size(-1)), value=ignore_label_id
                ) for part in packed_label_ids_parts]
            packed_label_ids = torch.cat(packed_label_ids_parts, dim=-1)
        packed_label_ids = torch.nn.functional.pad(packed_label_ids, (0, (packed_label_ids.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - packed_label_ids.size(-1)), value=ignore_label_id)
        args_dict["label_ids"] = packed_label_ids
        label_packing_mask = []
        for valid_len in valid_label_len:
            label_packing_mask.extend([0] + [1] * (valid_len - 1))
        label_packing_mask = torch.tensor([label_packing_mask], dtype=torch.bool).to(device)
        label_packing_mask = torch.nn.functional.pad(label_packing_mask, (0, (label_packing_mask.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - label_packing_mask.size(-1)), value=False)
        args_dict["label_packing_mask"] = label_packing_mask
    return args_dict

def pack_sequences_for_logprobs(
    input_ids: torch.Tensor,
    logprob_masks: torch.Tensor,
    pad_token_id: int = 0,
    advantages: Optional[torch.Tensor] = None,
    seq_len_multiple: int = 1,
    batch_sep_among_seq_len_multiple: bool = False,
):
    args_dict = {}
    with torch.no_grad():
        device = logprob_masks.device
        # Create mask: True where not padding
        valid_input_len = compute_valid_lengths(input_ids, pad_token_id)
        valid_input_mask = torch.arange(input_ids.size(1), device=device).expand(
            input_ids.size(0), -1
        ) < valid_input_len.unsqueeze(1)
        valid_input_len = valid_input_len.tolist()
        packed_logprob_masks = logprob_masks[valid_input_mask].unsqueeze(0)
        if batch_sep_among_seq_len_multiple:
            lensum_of_logprob_masks = [
                sum(
                    valid_input_len[
                        len(valid_input_len) // seq_len_multiple * idx : len(
                            valid_input_len
                        )
                        // seq_len_multiple
                        * (idx + 1)
                    ]
                )
                for idx in range(seq_len_multiple)
            ]
            packed_logprob_masks_parts = torch.split(packed_logprob_masks, lensum_of_logprob_masks, dim=-1)
            for idx, part in enumerate(packed_logprob_masks_parts):
                valid_input_len[
                    len(
                            valid_input_len
                        )
                        // seq_len_multiple
                        * (idx + 1) - 1
                ] += max(lensum_of_logprob_masks) - part.size(-1)
            packed_logprob_masks_parts = [
                torch.nn.functional.pad(
                    part, (0, max(lensum_of_logprob_masks) - part.size(-1)), value=0
                ) for part in packed_logprob_masks_parts]
            packed_logprob_masks = torch.cat(packed_logprob_masks_parts, dim=-1)
        packed_logprob_masks = torch.nn.functional.pad(packed_logprob_masks, (0, (packed_logprob_masks.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - packed_logprob_masks.size(-1)), value=0)
        args_dict["logprob_masks"] = packed_logprob_masks
        label_packing_mask = []
        for valid_len in valid_input_len:
            label_packing_mask.extend([0] + [1] * (valid_len - 1))
        label_packing_mask = torch.tensor([label_packing_mask], dtype=torch.bool).to(device)
        label_packing_mask = torch.nn.functional.pad(label_packing_mask, (0, (label_packing_mask.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - label_packing_mask.size(-1)), value=False)
        args_dict["label_packing_mask"] = label_packing_mask

        concatenated_advantages = [
            adv
            for idx, adv in enumerate(advantages)
            for _ in range(valid_input_len[idx])
        ]
        advantages = torch.tensor(
            [concatenated_advantages],
            dtype=torch.float32,
        ).to(device)
        advantages = torch.nn.functional.pad(advantages, (0, (advantages.size(-1) + seq_len_multiple - 1) // seq_len_multiple * seq_len_multiple - advantages.size(-1)), value=0)
        args_dict["advantages"] = advantages
    return args_dict