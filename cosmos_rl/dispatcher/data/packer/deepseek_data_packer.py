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

from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
    DecoderOnlyLLMDataPacker,
)
from cosmos_rl.policy.config import Config


class DeepSeek_DataPacker(DecoderOnlyLLMDataPacker):
    """
    Data protocol & processing logic for the decoder only LLM for SFT and RL training.
    """

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        self.seq_len = config.policy.model_max_length

    def policy_collate_fn(
        self,
        processed_samples: List[DecoderOnlyLLMDataPacker.RLPolicyInput],
        computed_max_len: int,
    ) -> Dict[str, Any]:
        computed_max_len = max(computed_max_len, self.seq_len)
        input_ids = [x.input_ids for x in processed_samples]
        logprob_masks = [x.logprob_masks for x in processed_samples]
        assert len(input_ids) == len(
            logprob_masks
        ), "The length of input_ids, and logprob_masks should be the same"
        device = torch.cuda.current_device()

        collated_dict = {}
        collated_dict["input_ids"] = torch.tensor(
            [
                x[:computed_max_len]
                + [self.tokenizer.pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in input_ids
            ],
            dtype=torch.long,
        ).to(device)
        collated_dict["logprob_masks"] = torch.tensor(
            [
                x[:computed_max_len] + [0] * (max(0, computed_max_len - len(x)))
                for x in logprob_masks
            ],
            dtype=torch.bool,
        ).to(device)

        return collated_dict

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into a minibatch dictionary passed to the SFT model.
        """
        computed_max_len = max(computed_max_len, self.seq_len)
        # First truncate the samples to the computed_max_len
        list_of_input_ids = [
            x["token_ids"][:computed_max_len] for x in processed_samples
        ]
        list_of_label_ids = [
            x["label_ids"][:computed_max_len] for x in processed_samples
        ]

        # Then pad the samples to the computed_max_len
        input_ids = torch.tensor(
            [
                x[:computed_max_len]
                + [pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_input_ids
            ],
            dtype=torch.long,
        )
        # Model accept unshifted label_ids for loss computation
        label_ids = torch.tensor(
            [
                x[:computed_max_len]
                + [ignore_label_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_label_ids
            ],
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
