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

original_update_prompt_logprobs = None
original_update_sample_logprobs = None


def apply_vllm_gather_logprobs_patch():
    """
    Patch vLLM's LogprobsProcessor to gather logprobs without
    decoding tokens to save memory and reduce overhead.
    """

    import itertools

    from vllm.v1.outputs import LogprobsTensors, LogprobsLists
    import vllm

    NONES = itertools.repeat(None)

    def _update_prompt_logprobs(
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
        prompt_logprobs_tensors: tuple containing the prompt logprobs
                                tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks = prompt_logprobs_tensors

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        decoded_tokens = None
        # We patch this to discard decoded tokens to save memory and reduce overhead.
        # if self.tokenizer is None else (
        #     convert_ids_list_to_tokens(self.tokenizer,
        #                             token_ids.flatten().tolist()))

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening.
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs
            decoded_tokens_for_pos = (
                NONES if decoded_tokens is None else decoded_tokens[offset:offset_end]
            )

            # Update with the Logprob dictionary for this pos.
            self.prompt_logprobs.append(
                self._make_logprob_dict(
                    prompt_logprobs[pos],
                    token_ids[pos],
                    decoded_tokens_for_pos,
                    prompt_token_ranks[pos],
                    self.num_prompt_logprobs,
                )
            )

    global original_update_prompt_logprobs
    if original_update_prompt_logprobs is None:
        original_update_prompt_logprobs = (
            vllm.v1.engine.logprobs.LogprobsProcessor._update_prompt_logprobs
        )
    vllm.v1.engine.logprobs.LogprobsProcessor._update_prompt_logprobs = (
        _update_prompt_logprobs
    )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst = logprobs_lists

        for rank, logprobs, token_ids in zip(ranks_lst, logprobs_lst, token_ids_lst):
            # Detokenize (non-incrementally).
            decoded_tokens = NONES
            # if self.tokenizer is None else (
            #     convert_ids_list_to_tokens(self.tokenizer, token_ids))

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob dictionary for this pos.
            self.logprobs.append(
                self._make_logprob_dict(
                    logprobs,
                    token_ids,
                    decoded_tokens,
                    rank,
                    self.num_logprobs,
                )
            )

    global original_update_sample_logprobs
    if original_update_sample_logprobs is None:
        original_update_sample_logprobs = (
            vllm.v1.engine.logprobs.LogprobsProcessor._update_sample_logprobs
        )
    vllm.v1.engine.logprobs.LogprobsProcessor._update_sample_logprobs = (
        _update_sample_logprobs
    )


def remove_vllm_gather_logprobs_patch():
    """Remove the vLLM patch for gathering prompt logprobs."""
    import vllm

    global original_update_prompt_logprobs
    assert original_update_prompt_logprobs is not None
    vllm.v1.engine.logprobs.LogprobsProcessor._update_prompt_logprobs = (
        original_update_prompt_logprobs
    )
    global original_update_sample_logprobs
    assert original_update_sample_logprobs is not None
    vllm.v1.engine.logprobs.LogprobsProcessor._update_sample_logprobs = (
        original_update_sample_logprobs
    )
