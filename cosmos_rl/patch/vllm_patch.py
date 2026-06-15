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


_flashinfer_rng_patched = False


def apply_flashinfer_isolated_rng_patch():
    """Restore flashinfer's pre-#2295 sampling RNG behavior.

    flashinfer PR #2295 (commit ede764f7, shipped in v0.6.0) changed
    ``get_seed_and_offset`` from using a freshly created, isolated
    ``torch.Generator`` to using torch's *global* cached default CUDA
    generator (``torch.cuda.default_generators[idx]``).

    The global generator's state is advanced by every CUDA RNG op in the
    training loop, so after #2295 the rollout sampling RNG becomes coupled
    to the training-loop RNG state. For on-policy distillation (OPD) this
    couples the sampled rollout trajectory to incidental numerical jitter
    in training, destabilizing the JSD signal and causing catastrophic
    divergence (val_reward regression, entropy blow-up).

    This patch reverts ``get_seed_and_offset`` to the pre-#2295 body: when no
    generator is supplied it uses a fresh ``torch.Generator(device=...)``,
    whose initial state is the fixed torch default seed (67280421310721, 0)
    and is therefore independent of the global RNG. This keeps the flashinfer
    sampling kernel path (same sampling algorithm as the known-good baseline)
    while removing the #2295 coupling.

    NOTE: setting ``SamplingParams.seed`` does NOT fix this — that routes vLLM
    to its PyTorch-native sampler (a different sampling algorithm), which
    produces a valid-but-different trajectory rather than reproducing the
    baseline. Patching flashinfer in place is the only way to keep the kernel
    path AND isolate the RNG.

    Idempotent; safe to call multiple times.
    """
    global _flashinfer_rng_patched
    if _flashinfer_rng_patched:
        return

    import torch
    import flashinfer.sampling as fi_sampling

    def _isolated_get_seed_and_offset(increment, generator=None, device=None):
        # pre-#2295 body, with an optional `device` kwarg for signature
        # compatibility with post-#2295 callers (the value is used only to
        # place the fresh generator; behavior matches the original otherwise).
        if generator is None:
            gdev = (
                device
                if device is not None
                else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
            generator = torch.Generator(device=gdev)
        state = generator.get_state()
        seed, offset = state.view(torch.int64)
        offset += (increment + 3) // 4 * 4
        generator.set_state(
            torch.tensor([seed, offset], dtype=torch.int64).view(torch.uint8)
        )
        return int(seed), int(offset)

    # The sampling closures in flashinfer resolve ``get_seed_and_offset`` as a
    # module global at call time, so overriding the module attribute is enough
    # even though get_sampling_module() may already be cached.
    fi_sampling.get_seed_and_offset = _isolated_get_seed_and_offset
    _flashinfer_rng_patched = True

    from cosmos_rl.utils.logging import logger

    logger.info(
        "[Rollout] Applied flashinfer isolated-RNG patch "
        "(reverts PR #2295 global-generator coupling for sampling)."
    )
