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

from typing import Optional, List
import torch
from cosmos_rl.utils.ulysses import slice_input_tensor
from torch.distributed.device_mesh import DeviceMesh


def compute_valid_lengths(input_ids, pad_token_id):
    """
    Compute the valid length at each batch.
    Used for packing sequence.
    """
    # Find the last non-pad position in each row
    # Reverse along sequence dim, find first True, convert to length
    mask = input_ids != pad_token_id  # (B, T)
    flipped_mask = mask.flip(dims=[-1])  # reverse each row
    last_non_pad = flipped_mask.long().argmax(dim=-1)  # index from end
    valid_lengths = mask.size(1) - last_non_pad  # length = total - index from end
    is_all_pad = ~mask.any(dim=-1)
    valid_lengths = valid_lengths.masked_fill(is_all_pad, 0)
    return valid_lengths


def expand_mask(mask, input, pos_seq_dim=1, batch_dim=0):
    """
    Expand the mask to match the shape of the target.
    Used for matching the mask with the target tensor at shape level.
    """

    if input.ndim > mask.ndim:
        for dim in range(input.ndim):
            if dim not in [batch_dim, pos_seq_dim]:
                mask = mask.unsqueeze(dim)
    assert input.ndim == mask.ndim, "input.ndim must be not less than mask.ndim"
    mask = mask.expand(input.shape)
    return mask


def recover_dims(updated, original, seq_dim=1, batch_dim=0):
    """
    Recover the dims layout to match the original tensor.
    Used for maching the mask with the target tensor at dimension level.
    """
    view_shape = list(original.shape)
    view_shape[seq_dim] = -1
    view_shape[batch_dim] = 1
    return updated.view(*view_shape)


def generate_mask(valid_len, input, seq_dim, batch_dim):
    """
    Generate the mask according to the valid length at each batch.
    Used for extracting the valid values and packing them.
    """
    device = input.device
    valid_mask = torch.arange(input.size(seq_dim), device=device).expand(
        input.size(batch_dim), -1
    ) < valid_len.unsqueeze(1)
    return valid_mask


def pack_sequences_info_collect(
    input_ids: torch.Tensor,
    pad_token_id: int,
    label_ids: Optional[torch.Tensor] = None,
    ignore_label_id: int = -100,
    seq_len_multiple: int = 1,
):
    """
    Collect valid length at each batch information used for sequence packing.

    input_ids: the padded input ids in [batch, seq_len] to be input to the model.
    pad_token_id: the token id used for padding the input ids.
    label_ids: the padded label ids in [batch, seq_len] used for supervised fine-tuning loss calculation.
    ignore_label_id: the token id used for ignoring certain labels during loss calculation.
    seq_len_multiple: the value which the processed seq_len must be multiple of.

    Return:
    A Dictionary including tensor "valid_input_len" which is a list of actual sequence length in the batch.
    This information will be used to apply the sequence packing.
    """
    args_dict = {}
    with torch.no_grad():
        device = input_ids.device
        # Create mask: True where not padding
        valid_input_len = compute_valid_lengths(input_ids, pad_token_id)
        if label_ids is not None:
            assert (
                input_ids.shape == label_ids.shape
            ), "Input IDs and label IDs must have the same shape"
            valid_label_len = compute_valid_lengths(label_ids, ignore_label_id)
            # Ensure label length does not exceed input length
            valid_input_len = torch.max(valid_label_len, valid_input_len)
        valid_input_len = valid_input_len.tolist()
        total_valid_input_len = sum(valid_input_len)
        origin_seq_len = input_ids.size(1)
        pad_needed = (
            total_valid_input_len + seq_len_multiple - 1
        ) // seq_len_multiple * seq_len_multiple - total_valid_input_len
        padded_valid_input_len = []
        for seq_len in valid_input_len:
            padded = min(pad_needed, origin_seq_len - seq_len)
            pad_needed -= padded
            seq_len += padded
            padded_valid_input_len.append(seq_len)
        valid_input_len = torch.tensor(
            padded_valid_input_len, dtype=torch.int32, device=device
        )
        args_dict["valid_input_len"] = valid_input_len
    return args_dict


def pack_sequences_for_inputs(
    inputs_embeds: torch.Tensor,
    valid_input_len: torch.Tensor,
    position_ids_list: Optional[List[torch.Tensor]] = None,
    interested_tokens: Optional[torch.BoolTensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    cp_mesh: DeviceMesh = None,
    inputs_seq_dim: int = 1,
    inputs_batch_dim: int = 0,
    position_ids_seq_dim: int = 1,
    position_ids_batch_dim: int = 0,
    interested_tokens_seq_dim: int = 1,
    interested_tokens_batch_dim: int = 0,
    padding_mask_seq_dim: int = 1,
    padding_mask_batch_dim: int = 0,
):
    """
    Processing input tensors inside model forward function after embedding calculation.
    Packing the tensors from different batches into one.

    inputs_embeds: the input embeddings in [batch, seq_len, hidden_size].
    valid_input_len: the valid input lengths in [batch] used for packing.
    position_ids_list: a list of position ids in [batch, seq_len] for each input sequence.
    interested_tokens: a tensor of interested tokens in [batch, seq_len] for each input sequence.
    padding_mask: a tensor of padding masks in [batch, seq_len] for each input sequence.
    cp_mesh: the device mesh for model parallelism.
    inputs_seq_dim: the sequence dimension of the input tensors.
    inputs_batch_dim: the batch dimension of the input tensors.
    position_ids_seq_dim: the sequence dimension of the position ids.
    position_ids_batch_dim: the batch dimension of the position ids.
    interested_tokens_seq_dim: the sequence dimension of the interested tokens.
    interested_tokens_batch_dim: the batch dimension of the interested tokens.
    padding_mask_seq_dim: the sequence dimension of the padding mask.
    padding_mask_batch_dim: the batch dimension of the padding mask.

    Return:
    A Dictionary including all related tensors generated for model inputs after sequence packing.
    Including sequence packed "inputs", "interested_tokens", "padding_mask", and "position_ids".
    After packing, the batch dimension will be 1 and the seq_len will be the sum of all inputs lengths with some paddings in a packing way.
    Also including "cu_seqlens" for accumulated sequence length for the batchs.
    And "max_seqlen" for the max sequence length in the batch.
    These will be used by the model to do the forward and backward in a sequence packing way.
    """

    args_dict = {}
    with torch.no_grad():
        device = inputs_embeds.device
        valid_input_mask = generate_mask(
            valid_input_len, inputs_embeds, inputs_seq_dim, inputs_batch_dim
        )
        packed_inputs_embeds = recover_dims(
            inputs_embeds[valid_input_mask],
            inputs_embeds,
            inputs_seq_dim,
            inputs_batch_dim,
        )
        if cp_mesh is not None:
            packed_inputs_embeds = slice_input_tensor(
                packed_inputs_embeds, inputs_seq_dim, cp_mesh
            )
        args_dict["inputs"] = packed_inputs_embeds

        if interested_tokens is not None:
            assert interested_tokens.ndim == 2, "ndim of interested_tokens must be 2"
            interested_tokens = recover_dims(
                interested_tokens[valid_input_mask],
                interested_tokens,
                interested_tokens_seq_dim,
                interested_tokens_batch_dim,
            )
            if cp_mesh is not None:
                interested_tokens = slice_input_tensor(
                    interested_tokens, interested_tokens_seq_dim, cp_mesh
                )

        args_dict["interested_tokens"] = interested_tokens

        if padding_mask is not None:
            assert padding_mask.ndim == 2, "ndim of padding_mask must be 2"
            padding_mask = recover_dims(
                padding_mask[valid_input_mask],
                padding_mask,
                padding_mask_seq_dim,
                padding_mask_batch_dim,
            )
            if cp_mesh is not None:
                padding_mask = slice_input_tensor(
                    padding_mask, padding_mask_seq_dim, cp_mesh
                )

        args_dict["padding_mask"] = padding_mask

        position_ids_updated = []
        if position_ids_list is not None:
            for position_ids in position_ids_list:
                position_ids_mask = expand_mask(
                    valid_input_mask,
                    position_ids,
                    position_ids_seq_dim,
                    position_ids_batch_dim,
                )
                position_ids = recover_dims(
                    position_ids[position_ids_mask],
                    position_ids,
                    position_ids_seq_dim,
                    position_ids_batch_dim,
                )
                if cp_mesh is not None:
                    position_ids = slice_input_tensor(
                        position_ids, position_ids_seq_dim, cp_mesh
                    )
                position_ids_updated.append(position_ids)
        args_dict["position_ids"] = position_ids_updated

        seq_len = valid_input_len
        prefill_start_pos = torch.cumsum(seq_len, dim=0, dtype=torch.int32) - seq_len
        cu_seqlens = torch.cat(
            [
                prefill_start_pos,
                torch.tensor([torch.sum(seq_len)], dtype=torch.int32).to(device),
            ],
            dim=0,
        )
        args_dict["cu_seqlens"] = cu_seqlens
        args_dict["max_seqlen"] = max(valid_input_len.tolist())
    return args_dict


def pack_sequences_for_masks(
    valid_input_len: torch.Tensor,
    valid_label_len: torch.Tensor,
):
    """
    Generate the related masks for packed input sequences used for loss calculation.

    valid_input_len: the valid input lengths for each sequence in the batch of shape [batch].
    valid_label_len: the valid label lengths for each sequence in the batch of shape [batch].

    Return:
    A Dictionary including all related masks generated for model inputs and labels after sequence packing.
    "input_packing_mask" is the mask for extracting the useful positions from the packed inputs.
    "label_packing_mask" is the mask for extracting the useful positions from the packed labels.
    The extracted useful positions from inputs and labels will be compared to calculate the loss.
    """

    args_dict = {}
    with torch.no_grad():
        device = valid_input_len.device
        valid_input_len = valid_input_len.tolist()
        input_packing_mask = []
        for valid_len in valid_input_len:
            input_packing_mask.extend([1] * (valid_len - 1) + [0])
        input_packing_mask = torch.tensor([input_packing_mask], dtype=torch.bool).to(
            device
        )
        args_dict["input_packing_mask"] = input_packing_mask
        label_packing_mask = []
        for valid_len in valid_label_len.tolist():
            label_packing_mask.extend([0] + [1] * (valid_len - 1))
        label_packing_mask = torch.tensor([label_packing_mask], dtype=torch.bool).to(
            device
        )
        args_dict["label_packing_mask"] = label_packing_mask
    return args_dict


def pack_sequences_for_labels(
    label_ids: torch.Tensor,
    valid_label_len: torch.Tensor,
    label_ids_seq_dim: int = 1,
    label_ids_batch_dim: int = 0,
):
    """
    Generate packed label ids to pack from different batches into one used for SFT loss calculation.

    label_ids: the label ids of shape [batch, seq_len] for each sequence in the batch.
    valid_label_len: the valid label lengths of shape [batch, seq_len] for each sequence in the batch.
    label_ids_seq_dim: the sequence dimension of the label ids tensor.
    label_ids_batch_dim: the batch dimension of the label ids tensor.

    Return:
    The packed label ids after sequence packing.
    It will be used with the packed model outputs to get the loss in SFT.
    """
    with torch.no_grad():
        valid_label_mask = generate_mask(
            valid_label_len, label_ids, label_ids_seq_dim, label_ids_batch_dim
        )
        packed_label_ids = recover_dims(
            label_ids[valid_label_mask],
            label_ids,
            label_ids_seq_dim,
            label_ids_batch_dim,
        )
    return packed_label_ids


def pack_sequences_for_logprobs(
    logprob_masks: torch.Tensor,
    valid_input_len: torch.Tensor,
    advantages: Optional[torch.Tensor] = None,
    input_ids_seq_dim: int = 1,
    input_ids_batch_dim: int = 0,
):
    """
    Generate packed log probabilities and advantages to pack from different batches into one used for GRPO loss calculation.

    logprob_masks: the log probability masks of shape [batch, seq_len] for each sequence in the batch.
    valid_input_len: the valid input lengths of shape [batch] for each sequence in the batch.
    advantages: the advantages of shape [batch] for each sequence in the batch.
    input_ids_seq_dim: the sequence dimension of the input ids tensor.
    input_ids_batch_dim: the batch dimension of the input ids tensor.

    Return:
    A Dictionary including related logprob_masks and advantages after sequence packing.
    "logprob_masks" and "advantages" are the logprob_masks and advantages after sequence packing.
    They will be used to calculate the loss together with the packed model inputs and outputs in RL.
    """
    args_dict = {}
    with torch.no_grad():
        device = logprob_masks.device
        valid_input_mask = generate_mask(
            valid_input_len, logprob_masks, input_ids_seq_dim, input_ids_batch_dim
        )
        packed_logprob_masks = recover_dims(
            logprob_masks[valid_input_mask],
            logprob_masks,
            input_ids_seq_dim,
            input_ids_batch_dim,
        )
        args_dict["logprob_masks"] = packed_logprob_masks
        valid_input_len = valid_input_len.tolist()
        if advantages is not None:
            concatenated_advantages = [
                adv
                for idx, adv in enumerate(advantages)
                for _ in range(valid_input_len[idx])
            ]
            advantages = torch.tensor(
                [concatenated_advantages],
                dtype=torch.float32,
            ).to(device)
            args_dict["advantages"] = advantages
    return args_dict


def pack_sequences_for_teacher_logprobs(
    teacher_logprobs: torch.Tensor,
    valid_input_len: torch.Tensor,
    logprobs_seq_dim: int = 1,
    logprobs_batch_dim: int = 0,
):
    """
    Generate packed teacher log probabilities to pack from different batches into one used for distillation loss calculation.
    teacher_logprobs: the teacher log probabilities of shape [batch, seq_len] for each sequence in the batch.
    valid_input_len: the valid input lengths of shape [batch] for each sequence in the batch.
    logprobs_seq_dim: the sequence dimension of the log probabilities tensor.
    logprobs_batch_dim: the batch dimension of the log probabilities tensor.
    Return:
    The packed teacher log probabilities after sequence packing.
    It will be used with the packed model outputs to get the loss in distillation.
    """
    with torch.no_grad():
        valid_input_mask = generate_mask(
            valid_input_len, teacher_logprobs, logprobs_seq_dim, logprobs_batch_dim
        )
        packed_teacher_logprobs = recover_dims(
            teacher_logprobs[valid_input_mask],
            teacher_logprobs,
            logprobs_seq_dim,
            logprobs_batch_dim,
        )
        return packed_teacher_logprobs
