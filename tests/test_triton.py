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


import random
import torch
import unittest
from typing import List, Dict, Any, Tuple

from cosmos_rl.utils.util import compute_logprobs

PROMPT_LEN_MAX = 1024
PROMPT_LEN_MIN = 64
COMPLETION_LEN_MAX = 4096
COMPLETION_LEN_MIN = 512
VOCAB_SIZE = 152064

# Below are only for kernel debugging
# PROMPT_LEN_MAX = 1
# COMPLETION_LEN_MAX = 4
# VOCAB_SIZE = 8


def simulate_samples(
    bsz, n_ignore_prefix_tokens: int = 0
) -> List[Tuple[List[int], List[int]]]:
    """ "
    Simulate samples for testing.

    Returns:
        A list of simulated samples. Each sample is a tuple of (input_ids, logprob_mask).
    """
    prompt_lens = [random.randint(PROMPT_LEN_MIN, PROMPT_LEN_MAX) for _ in range(bsz)]
    completion_lens = [
        random.randint(COMPLETION_LEN_MIN, COMPLETION_LEN_MAX) for _ in range(bsz)
    ]

    prompt_ids = []
    completion_ids = []

    # generate prompt_ids and completion_ids
    for prompt_len in prompt_lens:
        prompt_ids.append(
            [random.randint(1, VOCAB_SIZE - 2) for _ in range(prompt_len)]
        )

    for completion_len in completion_lens:
        completion_ids.append(
            [random.randint(1, VOCAB_SIZE - 2) for _ in range(completion_len)]
        )

    # generate logprob_masks and input_ids
    logprob_masks = []
    input_ids = []

    for i in range(bsz):
        prompt_ids_i = prompt_ids[i]
        completion_ids_i = completion_ids[i]

        logprob_masks_i = (
            [0] * (len(prompt_ids_i) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids_i) - n_ignore_prefix_tokens)
            + [0]
        )
        input_ids_i = prompt_ids_i + completion_ids_i

        logprob_masks.append(logprob_masks_i)
        input_ids.append(input_ids_i)

    samples = list(zip(input_ids, logprob_masks))
    return samples


def simulate_pad_samples(
    samples: List[Tuple[List[int], List[int]]],
    max_len: int,
    pad_token_id: int = VOCAB_SIZE - 1,
) -> torch.Tensor:
    """
    Pad samples to the same length. simulate `policy_collate_fn` in DataPacker.
    """
    input_ids = torch.tensor(
        [sample[0] + [pad_token_id] * (max_len - len(sample[0])) for sample in samples],
        dtype=torch.long,
    )
    logprob_masks = torch.tensor(
        [sample[1] + [0] * (max_len - len(sample[1])) for sample in samples],
        dtype=torch.bool,
    )

    return input_ids, logprob_masks


def simulate_generate_mini_batch(
    samples: List[Tuple[List[int], List[int]]],
) -> Dict[str, Any]:
    """
    Generate a mini-batch from samples.
    """
    bsz = len(samples)
    input_ids = [sample[0] for sample in samples]
    logprob_masks = [sample[1] for sample in samples]

    max_len_from_samples = max([len(input_id) for input_id in input_ids])
    advantages = [
        random.uniform(-2, 2) for _ in range(bsz)
    ]  # Each sample has a random advantage, in type float32

    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    # expand advantages_t to the same shape as input_ids that padded to max_len_from_samples
    advantages_t = advantages_t.unsqueeze(1).expand(
        -1, max_len_from_samples
    )  # [bsz, max_len_from_samples]

    mini_batch = {}

    input_ids, logprob_masks = simulate_pad_samples(
        samples, max_len=max_len_from_samples
    )

    mini_batch["input_ids"] = input_ids.cuda()  # long
    mini_batch["logprob_masks"] = logprob_masks.cuda()  # bool
    mini_batch["advantages"] = advantages_t.cuda()  # float32

    return mini_batch, max_len_from_samples


def normal_logprobs_computing(mini_batch, full_logits):
    normal_logps, logprob_masks = compute_logprobs(
        mini_batch, full_logits, use_triton=False
    )
    selected_normal_logprobs = torch.masked_select(normal_logps, logprob_masks)

    normal_mean = selected_normal_logprobs.mean()
    normal_mean.backward()

    return selected_normal_logprobs


def triton_logprobs_computing(mini_batch, full_logits):
    triton_logps, cu_seqlens = compute_logprobs(
        mini_batch, full_logits, use_triton=True
    )
    triton_mean = triton_logps.mean()
    triton_mean.backward()
    return triton_logps, cu_seqlens


def test_computing_logprobs_and_loss(bsz: int = 6):
    samples = simulate_samples(bsz=bsz, n_ignore_prefix_tokens=0)
    mini_batch, max_len_from_samples = simulate_generate_mini_batch(samples)

    # now generate the full logits
    full_logits = torch.randn(bsz, max_len_from_samples, VOCAB_SIZE).cuda().bfloat16()

    triton_full_logits = full_logits.clone()
    triton_full_logits.requires_grad = True
    triton_mini_batch = {}
    for k, v in mini_batch.items():
        triton_mini_batch[k] = v.clone()
    full_logits.requires_grad = True

    # Normal Way
    # compute logprobs
    normal_logps = normal_logprobs_computing(mini_batch, full_logits)

    # Triton forward
    triton_logprobs, cu_seqlens = triton_logprobs_computing(
        triton_mini_batch,
        triton_full_logits,
    )

    # Forward compare
    assert normal_logps.data_ptr() != triton_logprobs.data_ptr()
    assert torch.allclose(normal_logps, triton_logprobs)

    # backward compare
    # Gradient values are small, so we lossen the tolerance
    assert torch.allclose(
        full_logits.grad, triton_full_logits.grad, atol=1e-3, rtol=1e-2
    )


class TestTriton(unittest.TestCase):
    def test_computing_logprobs_and_loss(self):
        for bsz in [1, 2, 4, 5]:
            with self.subTest(bsz=bsz):
                test_computing_logprobs_and_loss(bsz=bsz)


if __name__ == "__main__":
    unittest.main()
