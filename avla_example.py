"""Custom entry point for using Alpamayo dataset with Cosmos-RL (Reasoning VLA).

This script provides a custom dataset wrapper and data packer that adapts the Alpamayo
autonomous driving dataset for use with the Cosmos-RL reinforcement learning framework
when training Reasoning VLA models.
"""

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

from cosmos_rl.policy.trainer import GRPOTrainer
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import compute_loss
from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
import torch
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.distributed as dist_util
import numpy as np
from cosmos_rl.utils.util import (
    compute_mfu,
)

from typing import List, Callable, Dict, Any, Tuple, Optional
from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses
from cosmos_rl.utils.util import is_master_rank, str2torch_dtype
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.dispatcher.replica import Rollout


@TrainerRegistry.register(trainer_type="reasoning_vla_grpo")
class ReasoningVLAGRPOTrainer(GRPOTrainer):
    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        logger.debug("[Policy] Prepare training data.")
        self.metrics = {
            "entropy": 0.0,
            "effective_entropy": 0.0,
        }

        payloads_list = [rollout.prompt for rollout in rollouts]
        completions_list = [rollout.completion for rollout in rollouts]
        advantages_list = [rollout.advantage for rollout in rollouts]
        print(f"Total rollouts in this replica: {len(rollouts)}")
        
        # # Log the mean of component rewards
        # L2_rewards_list = [rollout.report_metrics["traj_reward"] for rollout in rollouts]
        # reasoning_rewards_list = [rollout.report_metrics["reasoning_reward"] for rollout in rollouts]
        # reasoning_consistency_rewards_list = [
        #     (rollout.report_metrics["reasoning_consistency_reward"] if "reasoning_consistency_reward" in rollout.report_metrics else 0.0)
        #     for rollout in rollouts
        # ]
        # collision_rewards_list = [
        #     (rollout.report_metrics["collision_reward"] if "collision_reward" in rollout.report_metrics else 0.0)
        #     for rollout in rollouts
        # ]
        # Optional Positive-NLL support: only compute flags when coefficient > 0
        pos_coef_global = self.config.train.train_policy.positive_nll_coef
        if pos_coef_global is not None and pos_coef_global > 0.0:
            rewards_list = [rollout.reward for rollout in rollouts]
            self._positive_flags_t = torch.tensor(
                [1 if r > 0 else 0 for r in rewards_list],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            self._positive_flags_t = None
        n_ignore_prefix_tokens_list = [
            rollout.n_ignore_prefix_tokens for rollout in rollouts
        ]
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                payloads_list[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(payloads_list))
        ]

        # user_info_keys = list(kwargs.keys())
        advantages_t = torch.tensor(advantages_list).to(self.device)
        batch_size = len(rollouts)
        mini_batch_size = (
            min(self.mini_batch, batch_size) if self.mini_batch > 0 else batch_size
        )
        assert (
            batch_size % mini_batch_size == 0
        ), "Batch size should be divided evenly by mini_batch"
        num_mini_batch = batch_size // mini_batch_size

        # Initialize placeholder for old per-token logprobs
        self.old_per_token_logps = [None for _ in range(num_mini_batch)]
        self.ref_per_token_logps = [None for _ in range(num_mini_batch)]

        acc_n_tokens = 0

        need_compute_ref, kl_beta = self._swap_model_state_dict()

        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        is_computing_refs = [True, False] if need_compute_ref else [False]
        for is_computing_ref in is_computing_refs:
            # Set model to eval mode if reference model is being used
            if is_computing_ref:
                self.model.eval()
            else:
                if need_compute_ref:
                    # Swap model state dict back to the original model
                    need_compute_ref = False
                    self._swap_model_state_dict()
                self.model.train()

            with torch.set_grad_enabled(not is_computing_ref):
                for i_mu in range(1 if is_computing_ref else self.mu_iterations):
                    local_mini_step = 0
                    with torch.cuda.stream(self.train_stream):
                        for i in range(0, batch_size, mini_batch_size):
                            end = min(i + mini_batch_size, batch_size)
                            # Convert advantages from [batch_size] -> [batch_size, max_len] via expanding

                            minibatched_processed_samples = processed_samples[i:end]

                            # TODO(jiaxin): support variable length in PP
                            computed_max_len = (
                                self.config.policy.model_max_length
                                if self.parallel_dims.pp_enabled
                                else self.data_packer.policy_compute_max_len(
                                    minibatched_processed_samples
                                )
                            )

                            computed_max_len = (
                                (computed_max_len + self.seq_len_multiple - 1)
                                // self.seq_len_multiple
                                * self.seq_len_multiple
                            )
                            minibatched_advantages = (
                                advantages_t[i:end]
                                .unsqueeze(1)
                                .expand(-1, computed_max_len)
                                .to(self.device)
                            )

                            user_mini_batch: Dict[str, Any] = (
                                self.data_packer.policy_collate_fn(
                                    minibatched_processed_samples,
                                    computed_max_len=computed_max_len,
                                )
                            )

                            # TP/CP will shard the sequence dimension into n-ranks.
                            # The interested_tokens will be unevenly distributed across ranks.
                            # So do not enable interested_tokens in TP.
                            if (
                                self.parallel_dims.dp_shard_coord[1]
                                == self.parallel_dims.world_size
                            ):
                                user_mini_batch["interested_tokens"] = user_mini_batch[
                                    "logprob_masks"
                                ]

                            # Move all tensor to device
                            for k in user_mini_batch.keys():
                                v = user_mini_batch[k]
                                if (
                                    isinstance(v, torch.Tensor)
                                    and v.device != self.device
                                ):
                                    user_mini_batch[k] = v.to(self.device)

                            # input_ids are different across ranks in dp_shard_cp
                            position_ids, input_ids, pos_seq_dim = (
                                self.model.get_position_ids(**user_mini_batch)
                            )
                            acc_n_tokens += np.prod(input_ids.shape)
                            user_mini_batch["position_ids"] = position_ids
                            padding_mask = user_mini_batch.get("padding_mask", None)

                            input_ids_before_cp = user_mini_batch["input_ids"]
                            position_ids_before_cp = user_mini_batch["position_ids"]
                            padding_mask_before_cp = padding_mask

                            if self.parallel_dims.cp_enabled:
                                [input_ids, position_ids, padding_mask] = (
                                    slice_inputs_for_ulysses(
                                        [input_ids, position_ids, padding_mask],
                                        self.parallel_dims.mesh["cp"],
                                    )
                                )
                                user_mini_batch["position_ids"] = position_ids
                                user_mini_batch["input_ids"] = input_ids
                                if padding_mask is not None:
                                    user_mini_batch["padding_mask"] = padding_mask

                            if self.parallel_dims.pp_enabled:
                                raise NotImplementedError("Pipeline Parallel is not supported for Reasoning VLA")
                            else:
                                model_out = self.model(**user_mini_batch)

                                if self.parallel_dims.cp_enabled:
                                    # reset the position ids and input ids
                                    user_mini_batch["position_ids"] = (
                                        position_ids_before_cp
                                    )
                                    user_mini_batch["input_ids"] = input_ids_before_cp
                                    if padding_mask_before_cp is not None:
                                        user_mini_batch["padding_mask"] = (
                                            padding_mask_before_cp
                                        )

                                # Prefer model-provided loss if available (e.g., AVLA forward)
                                if os.environ.get("COSMOS_USE_SFT_LOSS", "0") == "1" and hasattr(model_out, "loss") and isinstance(model_out.loss, torch.Tensor):
                                    print('[GRPO Trainer] Directly use model-provided loss')
                                    loss = (model_out.loss / num_mini_batch)
                                    per_token_loss = loss
                                    kl_loss = torch.tensor(0.0, device=self.device)

                                    loss.backward()
                                    loss_sum += per_token_loss.item()
                                    kl_loss_sum += kl_loss.item()
                                    loss_count += 1

                                    # maintain step/allreduce cadence
                                    self.mini_step += 1
                                    local_mini_step += 1
                                    if (
                                        local_mini_step
                                        % int(os.environ.get("COSMOS_GRPO_STEP_INTERVAL", 10))
                                        == 0
                                    ) and local_mini_step > 1:
                                        all_reduced = True
                                        grad_norm_sum += self.all_reduce_states(
                                            inter_policy_nccl
                                        )
                                    else:
                                        all_reduced = False
                                    # Done for this micro-batch
                                    continue
                                # Fall back to GRPO/SFT-logprob path. Support HF ModelOutput or raw Tensor
                                raw_logits = model_out.logits if hasattr(model_out, "logits") else model_out
                                if self.config.train.train_policy.temperature > 1e-6:
                                    raw_logits = (
                                        raw_logits
                                        / self.config.train.train_policy.temperature
                                    )
                                # returned shape:
                                # current_per_token_logprobs: [n_tokens_of_logprobs]
                                # cu_seqlens: [batch_size + 1]
                                current_per_token_logprobs, cu_seqlens, metrics = (
                                    self.compute_logprobs(
                                        user_mini_batch,
                                        logits=raw_logits,
                                        is_full_logits=True
                                        if getattr(raw_logits, "ndim", 0) == 3
                                        else False,
                                    )
                                )
                                logprob_masks = user_mini_batch["logprob_masks"]
                                current_advantages = (
                                    logprob_masks * minibatched_advantages
                                )

                                # Compute ref per-token logprobs if needed
                                if is_computing_ref:
                                    assert (
                                        i_mu == 0
                                    ), "Only first iteration should compute ref"
                                    self.ref_per_token_logps[local_mini_step] = (
                                        current_per_token_logprobs.detach()
                                    )
                                    # Skip the rest of the loop
                                    local_mini_step += 1
                                    continue
                                else:
                                    if (
                                        self.old_per_token_logps[local_mini_step]
                                        is None
                                    ):
                                        assert (
                                            i_mu == 0
                                        ), "Only first iteration should append `old_per_token_logps`"
                                        self.old_per_token_logps[local_mini_step] = (
                                            current_per_token_logprobs.detach()
                                        )
                                    else:
                                        assert (
                                            i_mu > 0
                                        ), "Only inner iteration should reuse `old_per_token_logps`"
                                    # Teacher-forcing SFT mode: optimize masked NLL directly
                                    if os.environ.get("COSMOS_USE_SFT_LOSS", "0") == "1":
                                        print('in sft loss, current_per_token_logprobs shape:', current_per_token_logprobs.shape)
                                        sft_loss = -current_per_token_logprobs.mean()
                                        loss = sft_loss / num_mini_batch
                                        per_token_loss = sft_loss / num_mini_batch
                                        kl_loss = torch.tensor(0.0, device=self.device)

                                        loss.backward()
                                        loss_sum += per_token_loss.item()
                                        kl_loss_sum += kl_loss.item()
                                        loss_count += 1
                                    else:
                                        loss, per_token_loss, kl_loss = compute_loss(
                                            current_per_token_logprobs,
                                            self.old_per_token_logps[local_mini_step],
                                            self.ref_per_token_logps[local_mini_step],
                                            current_advantages,
                                            cu_seqlens,
                                            self.config,
                                            logprob_masks,
                                            dp_group=self.parallel_dims.mesh[
                                                "dp"
                                            ].get_group()
                                            if self.parallel_dims.dp_enabled
                                            else None,
                                            ddp_comm=inter_policy_nccl,
                                        )

                                        # Positive Example LM Loss
                                        if (
                                            pos_coef_global is not None
                                            and pos_coef_global > 0.0
                                        ):
                                            pos_flag_batch = self._positive_flags_t[i:end]
                                            pos_mask = pos_flag_batch.unsqueeze(
                                                1
                                            ).expand_as(logprob_masks)
                                            pos_token_mask = pos_mask & logprob_masks
                                            if pos_token_mask.any():
                                                flat_mask = pos_token_mask[logprob_masks]
                                                l_nll = -current_per_token_logprobs[
                                                    flat_mask
                                                ].mean()
                                                loss = loss + pos_coef_global * l_nll

                                        loss = loss / num_mini_batch
                                        per_token_loss = per_token_loss / num_mini_batch
                                        kl_loss = kl_loss / num_mini_batch

                                        loss.backward()
                                        loss_sum += per_token_loss.item()
                                        kl_loss_sum += kl_loss.item()
                                        loss_count += 1
                                    for key in metrics:
                                        self.metrics[key] += metrics[key]
                                        
                            self.mini_step += 1
                            local_mini_step += 1

                            if (
                                local_mini_step
                                % int(os.environ.get("COSMOS_GRPO_STEP_INTERVAL", 10))
                                == 0
                            ) and local_mini_step > 1:
                                all_reduced = True
                                grad_norm_sum += self.all_reduce_states(
                                    inter_policy_nccl
                                )
                            else:
                                all_reduced = False
                        if not is_computing_ref and not all_reduced:
                            grad_norm_sum += self.all_reduce_states(
                                inter_policy_nccl
                            )
        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        # Only step lr scheduler when all the mini-batches are processed
        self.lr_schedulers.step()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                # Calculate the iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = grad_norm_sum.item()

                # report_data["train/traj_reward_avg"] = np.mean(L2_rewards_list)
                # report_data["train/Reasoning_reward_avg"] = np.mean(reasoning_rewards_list)
                # if len(reasoning_consistency_rewards_list) > 0:
                #     report_data["train/Reasoning_consistency_reward_avg"] = np.mean(reasoning_consistency_rewards_list)
                # if len(collision_rewards_list) > 0:
                #     report_data["train/Collision_reward_avg"] = np.mean(collision_rewards_list)

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        n_tokens=acc_n_tokens,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
                if len(self.metrics) > 0:
                    for k, v in self.metrics.items():
                        report_data[f"train/{k}"] = (
                            v.item() if isinstance(v, torch.Tensor) else v
                        ) / loss_count
        # checkpointing
        if is_master_replica and (
            (
                self.config.train.ckpt.enable_checkpoint
                and current_step % self.config.train.ckpt.save_freq == 0
                and current_step > 0
            )
            or (
                self.config.train.ckpt.enable_checkpoint and current_step == total_steps
            )
        ):
            if self.config.train.ckpt.export_safetensors:
                logger.info(
                    f"[Policy] Saving huggingface checkpoint at step {current_step} to {self.config.train.output_dir}..."
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{current_step}",
                    ),
                    trainable_only=False,
                    is_final=current_step == total_steps,
                    dtype=str2torch_dtype(self.config.train.param_dtype),
                )
            logger.info(f"[Policy] Saving cosmos checkpoint at step {current_step}...")
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=total_steps,
                **{
                    "remain_samples_num": remain_samples_num,
                    "is_final": current_step == total_steps,
                },
            )
            self.ckpt_manager.save_check(step=current_step)
        return report_data

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            loss: the loss of shape `[n_tokens]`
            """
            return loss.mean()

        return fake_compute_loss







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
import os
import time
from typing import Any, List, Optional, Tuple

from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult
import torch
import vllm
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.policy.config import Config, RolloutConfig
from cosmos_rl.policy.model import WeightMapper
from cosmos_rl.rollout.rollout_base import RolloutBase
from cosmos_rl.rollout.vllm_rollout.monkey_patch_for_fp8 import apply_fp8_linear_patch
from cosmos_rl.utils import util
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig, AutoTokenizer, GenerationConfig
from vllm import SamplingParams
from vllm.entrypoints.llm import LLM
from cosmos_rl.rollout.rollout_base import RolloutRegistry
from cosmos_rl.utils.parallelism import ParallelDims


def vllm_version_check(rollout_config: RolloutConfig):
    vllm_version = vllm.__version__
    if vllm_version < "0.9.0" and rollout_config.parallelism.pp_size > 1:
        raise NotImplementedError(
            "Pipeline parallelism is not supported for vLLM < 0.9.0, current version is %s"
            % vllm_version
        )



@RolloutRegistry.register("reasoning_vla_vllm_rollout")
class ReasoningVlaVllmRollout(RolloutBase):
    def __init__(
        self,
        config: Config,
        parallel_dims: ParallelDims,
        device: torch.device,
        model_hf_config: Optional[AutoConfig] = None,
        **kwargs,
    ):
        """Rollout with vLLM as the backend.

        Args:
            config: Cosmos Config.
            parallel_dims: Parallel dimensions for the rollout engine.
            device: The device on which the rollout engine will run.
            model_hf_config: the huggingface config to initiallize the generating model in vllm
        """
        super().__init__(config, parallel_dims, device, **kwargs)

        policy_config = self.config.policy
        self.rollout_config = self.config.rollout
        self.validation_config = self.config.validation

        vllm_version_check(self.rollout_config)

        model_path = policy_config.model_name_or_path

        self.model_config = util.retry(AutoConfig.from_pretrained)(
            model_path, trust_remote_code=True
        )
        
        self.tokenizer = util.retry(AutoTokenizer.from_pretrained)(
            self.config.policy.model_name_or_path
        )

        self.pad_token_id = self.tokenizer.pad_token_id

        hf_config_path = self.config.policy.model_name_or_path
        try:
            generation_config = util.retry(GenerationConfig.from_pretrained)(hf_config_path)
            self.eos_token_ids = generation_config.eos_token_id
            if isinstance(self.eos_token_ids, int):
                self.eos_token_ids = [self.eos_token_ids]
        except Exception as e:
            logger.warning(
                f"[Rollout] Failed to load generation config from {hf_config_path}: {str(e)}, use default eos_token_id."
            )
            # self.eos_token_ids = [tokenizer.eos_token_id]
            # TODO(lms): remove this
            self.eos_token_ids = [151645, 151643]
        self._engine_initialized = False
        self.rollout_engine = None
        self._model_param_map = None  # key: compatible name, value: param
        self.global_rank = int(os.environ.get("RANK", 0))

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        logger.info(f"[Rollout] Initializing engine for rank: {self.global_rank}")
        self.val_sampling_params = SamplingParams(
            n=self.config.validation.n_generation,
            logprobs=0,
            top_p=self.config.validation.top_p
            if self.config.validation.top_p is not None
            else self.config.rollout.sampling_config.top_p,
            top_k=self.config.validation.top_k
            if self.config.validation.top_k is not None
            else self.config.rollout.sampling_config.top_k,
            temperature=self.config.validation.temperature
            if self.config.validation.temperature is not None
            else self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.validation.repetition_penalty
            if self.config.validation.repetition_penalty is not None
            else self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.validation.max_response_length
            if self.config.validation.max_response_length is not None
            else self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )
        self.sampling_params = SamplingParams(
            n=self.config.rollout.n_generation,
            logprobs=0,
            top_p=self.config.rollout.sampling_config.top_p,
            top_k=self.config.rollout.sampling_config.top_k,
            temperature=self.config.rollout.sampling_config.temperature,
            repetition_penalty=self.config.rollout.sampling_config.repetition_penalty,
            max_tokens=self.config.rollout.max_response_length,
            stop_token_ids=self.eos_token_ids,
            include_stop_str_in_output=self.config.rollout.include_stop_str_in_output,
            detokenize=True,
        )
        
        # Provide vLLM a text_config and architecture without modifying HF config classes.
        def _avla_vllm_hf_overrides(cfg):
            # Prefer a user-customized base LLM config if provided by the HF config.
            base_cfg = cfg.get_llm_config()
            setattr(cfg, "text_config", base_cfg)
            # Ensure the custom vLLM architecture is discoverable.
            arches = list(getattr(cfg, "architectures", []) or [])
            if "AVLA" not in arches:
                arches.append("AVLA")
            setattr(cfg, "architectures", arches)
            return cfg

        def _reasoning_vla_vllm_hf_overrides(cfg):
            # Prefer the underlying LLM config for ReasoningVLA as text_config
            base_cfg = cfg.get_llm_config()
            setattr(cfg, "text_config", base_cfg)
            # Make vLLM aware of the custom architecture wrapper. We add both
            # the mixed-case and upper-case variants so they match the entries
            # registered in `ModelRegistry`.
            arches = list(getattr(cfg, "architectures", []) or [])
            for name in ("ReasoningVLA", "REASONING_VLA"):
                if name not in arches:
                    arches.append(name)
            setattr(cfg, "architectures", arches)
            return cfg

        if not self._engine_initialized:
            trust_remote_code = True  # set trust remote code default to True.

            model_path = self.config.policy.model_name_or_path


            rollout_parallelism = self.rollout_config.parallelism

            # disable VLLM_DISABLE_COMPILE_CACHE
            os.environ["VLLM_DISABLE_COMPILE_CACHE"] = "1"

            tp_size = rollout_parallelism.tp_size
            pp_size = rollout_parallelism.pp_size

            enable_ep_parallelism = False
            disable_mm_preprocessor_cache = False

            # Check if the model has MoE
            moe_model_type = {"qwen3_moe"}
            multimodal_type = {"qwen2_5_vl"}

            model_type = self.model_config.model_type
            if model_type in moe_model_type:
                enable_ep_parallelism = True
            if model_type in multimodal_type:
                # for vllm nightly, this is only True for multimodal models, check here
                disable_mm_preprocessor_cache = True
            assert tp_size * pp_size == rollout_parallelism.world_size, (
                "[Rollout] For tensor parallel, the tp_size * pp_size must be equal to world size, but got tp_size: %d, pp_size: %d, world_size: %d"
                % (tp_size, pp_size, rollout_parallelism.world_size)
            )

            self.quantization = quantization

            policy_config = self.config.policy

            # Ensure Transformers allows executing custom code in HF configs/models for workers
            # to avoid interactive trust prompts when loading local checkpoints with custom code.
            os.environ.setdefault("HF_ALLOW_CODE_EXECUTION", "1")

            # Allow env overrides to quickly test throughput-sensitive knobs without changing code.
            env_backend = os.getenv("COSMOS_VLLM_EXECUTOR_BACKEND")
            # Prefer in-process multiprocessing backend for single-GPU to reduce control overhead
            if env_backend:
                resolved_backend = env_backend
            else:
                resolved_backend = (
                    "uni" if rollout_parallelism.world_size == 1 else "external_launcher"
                )

            env_enforce_eager = os.getenv("COSMOS_VLLM_ENFORCE_EAGER")
            resolved_enforce_eager = (
                (env_enforce_eager.lower() in ["1", "true", "yes"])
                if env_enforce_eager
                else self.rollout_config.enforce_eager
            )

            env_chunked = os.getenv("COSMOS_VLLM_ENABLE_CHUNKED_PREFILL")
            resolved_chunked = (
                (env_chunked.lower() in ["1", "true", "yes"])
                if env_chunked
                else self.rollout_config.enable_chunked_prefill
            )

            env_gpu_util = os.getenv("COSMOS_VLLM_GPU_MEMORY_UTILIZATION")
            resolved_gpu_util = (
                float(env_gpu_util) if env_gpu_util else self.rollout_config.gpu_memory_utilization
            )

            # max_num_batched_tokens: keep prior behavior but allow override, e.g., 32768 to match test_engine.py
            if 2048 >= policy_config.model_max_length:
                default_max_batched = 2048
            else:
                default_max_batched = policy_config.model_max_length
            env_max_batched = os.getenv("COSMOS_VLLM_MAX_NUM_BATCHED_TOKENS")
            resolved_max_batched = int(env_max_batched) if env_max_batched else default_max_batched

            # Select proper hf_overrides based on model type
            model_type = getattr(self.model_config, "model_type", None)
            if model_type == "avla":
                hf_overrides_fn = _avla_vllm_hf_overrides
            elif model_type == "alpamayo_reasoning_vla":
                hf_overrides_fn = _reasoning_vla_vllm_hf_overrides
            else:
                raise NotImplementedError(f"Model type {model_type} not supported for vLLM rollout.")

            self.rollout_engine = LLM(
                model=model_path,
                hf_overrides=hf_overrides_fn,
                enable_sleep_mode=False,  # enable sleep could corrupt the cuda allocator.
                tensor_parallel_size=tp_size,
                pipeline_parallel_size=pp_size,
                enable_expert_parallel=enable_ep_parallelism,
                distributed_executor_backend="external_launcher",
                dtype="auto",
                enforce_eager=resolved_enforce_eager,  # enable cuda graph
                gpu_memory_utilization=resolved_gpu_util,
                disable_custom_all_reduce=True,
                disable_mm_preprocessor_cache=disable_mm_preprocessor_cache,
                enable_prompt_embeds=False,
                skip_tokenizer_init=False,
                max_model_len=policy_config.model_max_length,
                disable_log_stats=True,
                max_num_batched_tokens=resolved_max_batched,
                enable_chunked_prefill=resolved_chunked,
                enable_prefix_caching=False,
                trust_remote_code=trust_remote_code,
                quantization=self.quantization,
                seed=seed or 42,
                load_format=load_format,
            )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")
            # initialization done.

            # Log effective vLLM engine config for debugging throughput differences
            try:
                mc = self.rollout_engine.llm_engine.get_model_config()
                logger.info(
                    {
                        "vllm_model_config": {
                            "runner_type": getattr(mc, "runner_type", None),
                            "max_model_len": getattr(mc, "max_model_len", None),
                            "dtype": str(getattr(mc, "dtype", None)),
                            "model": getattr(mc, "model", None),
                            "hf_text_config_type": type(
                                getattr(mc, "hf_text_config", None)
                            ).__name__,
                        },
                        "cosmos_vllm_effective": {
                            "distributed_executor_backend": resolved_backend,
                            "enforce_eager": resolved_enforce_eager,
                            "enable_chunked_prefill": resolved_chunked,
                            "gpu_memory_utilization": resolved_gpu_util,
                            "max_num_batched_tokens": resolved_max_batched,
                            "tp_size": tp_size,
                            "pp_size": pp_size,
                        },
                    }
                )
            except Exception:
                pass

            # patch the vllm model to use rowwise fp8
            if self.quantization == "fp8":
                from vllm.config import set_current_vllm_config

                vllm_config = self.rollout_engine.llm_engine.vllm_config
                with set_current_vllm_config(vllm_config):
                    apply_fp8_linear_patch(self.get_underlying_model())
        logger.info(f"[Rollout] Engine initialized for rank: {self.global_rank}")

    def post_init_hook(self, **kwargs):
        pass

    @torch.no_grad()
    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        *args,
        **kwargs,
    ) -> List[RolloutResult]:
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )

        # List of payloads.
        # [
        #   payload,
        #   payload,
        #   ...
        # ]
        payloads = [payload.prompt for payload in payloads]

        # Pack the payloads into prompts for vllm.
        t_prep_start = time.perf_counter()
        prompts = [data_packer.get_rollout_input(payload) for payload in payloads]
        prompts = data_packer.rollout_collate_fn(prompts)
        t_prep_end = time.perf_counter()

        # These lines are needed for avla model to get the prompt embeds
        # model = getattr(self.get_underlying_model(), "model")
        # t_embed_start = time.perf_counter()
        # prompts = model.get_vllm_prompts(**prompts)  # list[{"prompt_embeds": ...}]
        # t_embed_end = time.perf_counter()
        # End of avla model prompt embeds

        # List of completions per prompt.
        # [
        #   [completion_str, completion_str, ...],
        #   [completion_str, completion_str, ...],
        #   ...
        # ]

        response: List[List[str]] = []

        completion_logprobs: List[List[float]] = []

        stream = torch.cuda.current_stream() if stream is None else stream
        try:
            # Use vLLM's own scheduling/streams; avoid wrapping in a custom CUDA stream context
            t_gen_start = time.perf_counter()
            # TODO: Remove this in the future!!!!!
            if is_validation:
                sampling_params = self.val_sampling_params
            else:
                sampling_params = self.sampling_params
            sampling_params.stop_token_ids = [169711] # Traj future end token
            sampling_params.logprobs = 1
            results = self.rollout_engine.generate(
                prompts=prompts,
                sampling_params=sampling_params,
                use_tqdm=False,
            )
            t_gen_end = time.perf_counter()

            total_generated_tokens = 0
            rollout_results : List[RolloutResult] = []
            for idx, output in enumerate(results):
                response.append([output.outputs[i].text for i in range(len(output.outputs))])
                completion_logprobs.append([output.outputs[i].cumulative_logprob for i in range(len(output.outputs))])
                for out in output.outputs:
                    token_ids = getattr(out, "token_ids", None)
                    if token_ids is not None:
                        total_generated_tokens += len(token_ids)

            valid_completions: List[List[str]] = []
            valid_logprobs: List[List[float]] = []
            prompt_indices_to_remove: List[int] = []
            if is_validation:
                for i in range(len(response)):
                    rollout_results.append(
                        RolloutResult(
                            prompt=payloads[i],
                            completions=response[i],
                            cumulative_logprob=completion_logprobs[i]
                        )
                    )
            # Remove empty completions for training
            elif len(response):
                batch_size = len(prompts)
                assert (
                    len(response) == batch_size
                ), f"Error: VLLM returned {len(response)} for {batch_size}"
                for i in range(batch_size):
                    completion = response[i]
                    probs_i = completion_logprobs[i] if completion_logprobs is not None else None

                    skip_output = False
                    total_generation_count = len(completion)
                    empty_generation_count = 0
                    output_texts = []
                    output_probs = [] if probs_i is not None else None

                    for j in range(total_generation_count):
                        output_text = completion[j]
                        output_texts.append(
                            output_text
                            if output_text != ""
                            else self.tokenizer.eos_token
                        )
                        if output_probs is not None:
                            output_probs.append(probs_i[j])
                    # Skip the output if there is one or zero non-empty completions
                    skip_output = (
                        total_generation_count - empty_generation_count
                    ) <= 1
                    if not skip_output:
                        valid_completions.append(output_texts)
                        if output_probs is not None:
                            valid_logprobs.append(output_probs)
                        rollout_results.append(
                            RolloutResult(
                                prompt=payloads[i],
                                completions=response[i],
                                cumulative_logprob=completion_logprobs[i]
                            )
                        )
                    else:
                        prompt_indices_to_remove.append(i)
                        rollout_results.append(
                            RolloutResult(
                                prompt=payloads[i],
                                completions=[]
                            )
                        )

            # Lightweight profiling log
            try:
                prep_s = t_prep_end - t_prep_start
                gen_s = t_gen_end - t_gen_start
                toks_per_s = (total_generated_tokens / gen_s) if gen_s > 0 else float("nan")
                logger.info(
                    {
                        "rollout_profile_s": {
                            "data_prepare_s": round(prep_s, 4),
                            "generate_s": round(gen_s, 4),
                            "generated_tokens": total_generated_tokens,
                            "gen_toks_per_s": round(toks_per_s, 2)
                            if isinstance(toks_per_s, float)
                            else toks_per_s,
                        }
                    }
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"[Rollout] Failed in rollout generation: {str(e)}")
            import traceback

            traceback.print_exc()
            return []

        return rollout_results

    def get_underlying_model(self):
        """Get the underlying parallelized model in vLLM internal."""
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model

    def get_engine(self):
        if not self._engine_initialized:
            raise RuntimeError(
                "[Rollout] Engine is not initialized, please call init_engine first."
            )
        return self.rollout_engine

    def is_engine_initialized(self):
        return self._engine_initialized

    def fp8_quantization(self, weight: torch.Tensor):
        # convert to fp8
        from vllm import _custom_ops as ops

        # quantization of rowwise torch scaled_mm.
        # weight has shape [out_dim, in_dim]
        qweight, weight_scale = ops.scaled_fp8_quant(
            weight, scale=None, use_per_token_if_dynamic=True
        )

        return qweight.t(), weight_scale

    def model_param_map(self, weight_mapper: WeightMapper):
        if self._model_param_map:
            return self._model_param_map
        model = self.get_underlying_model()
        param_map = {}
        for name, param in model.named_parameters():
            compatible_name = weight_mapper._rollout_vllm_name_to_hf(name)
            param_map[compatible_name] = param
        self._model_param_map = param_map
        return self._model_param_map





import os
import sys
from typing import Any, Dict, List, Optional, Union

# Add cosmos-rl to sys.path
from pathlib import Path

_here = Path(__file__).resolve()
_projects_dir = next((p for p in _here.parents if p.name == "projects"), None)
if _projects_dir is not None:
    _cosmos_rl = _projects_dir / "alpamayo_cosmos" / "cosmos-rl"
    print(f"Adding cosmos-rl to sys.path: {_cosmos_rl}")
    if _cosmos_rl.is_dir():
        sys.path.insert(0, str(_cosmos_rl))

_current_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(os.path.dirname(_current_dir))
_src_dir = os.path.join(_repo_root, "src")
_projects_dir = os.path.join(_repo_root, "projects")
if _src_dir not in sys.path or _projects_dir not in sys.path:
    sys.path[:0] = [_src_dir, _projects_dir]

# Ensure locale and heartbeat are set BEFORE importing cosmos_rl modules
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("LANG", "C.UTF-8")
os.environ.setdefault("COSMOS_HEARTBEAT_TIMEOUT", "600")

import hydra
import torch
from alpamayo.data.wds_helpers import basic_collation_fn
from alpamayo.common.vla_constant import IGNORE_INDEX, SPECIAL_TOKENS
from cosmos_rl.dispatcher.data.packer import DataPacker
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model.base import ModelRegistry
from hydra import compose, initialize
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer
from vllm import ModelRegistry as vllm_model_registry
from vllm.inputs import TokensPrompt

from alpamayo_cosmos.helpers import (
    debug_break,
    _log_gpu_info,
    _maybe_pin_gpu_per_role,
    calculate_ade,
    reasoning_trace_reward_common,
    compute_traj_reward_common,
    build_reward_dict,
    custom_reward_fn_common,
)
from reasoning_vla.models.trajectory_fusion import (
    TrajectoryFusionMixin,
)

# Register vLLM wrapper for ReasoningVLA (force re-register to avoid stale mapping).
# IMPORTANT: The architecture name must match the HF config's
# `architectures=["ReasoningVLA"]` exactly so vLLM can resolve it.
try:
    from alpamayo_cosmos.model_reasoning_vla_vllm import ReasoningVLAModelForVLLM
    # Debug: confirm we imported the intended class and that it has required methods.
    print("[INFO] vLLM arch 'ReasoningVLA' ->", ReasoningVLAModelForVLLM,
          "has load_weights:", hasattr(ReasoningVLAModelForVLLM, "load_weights"))
    vllm_model_registry.register_model("ReasoningVLA", ReasoningVLAModelForVLLM)
except Exception as e:
    print("[WARN] Failed to register ReasoningVLA model with vLLM:", e)

# Make the debug function available globally
import builtins

builtins.debug_break = debug_break

# Emit GPU info as early as possible on import for both policy and rollout workers
try:
    _maybe_pin_gpu_per_role()
    _log_gpu_info("module import")
except Exception:
    pass


# Global caches
_alpamayo_dataloaders = None
_alpamayo_tokenizer = None
_alpamayo_traj_tokenizer = None
_alpamayo_tokens_per_future_traj = None
_alpamayo_ckpt_cfg = None
_alpamayo_traj_fuser = None

from reasoning_vla.models.base_model import ReasoningVLA

path = (
    "/lustre/fsw/portfolios/nvr/users/rant/reasoning_vla_pre_trained_ckpts/rvla_stage2_prompt_cot_prod_v025_fix_nccl_fix_tokenizer_eval_bs1_20251023_182629/train/ckpts/checkpoint-50000-expanded_token/"  # noqa: E501
)
global_model = ReasoningVLA.from_pretrained(path)

# Default checkpoint path used by both the main launcher and tests.
DEFAULT_ALPAMAYO_CKPT_PATH = (
    "/lustre/fsw/portfolios/nvr/users/rant/reasoning_vla_pre_trained_ckpts/"
    "rvla_stage2_prompt_cot_prod_v025_fix_nccl_fix_tokenizer_eval_bs1_20251023_182629/"
    "train/ckpts/checkpoint-50000-expanded_token/"
)


# Local constant mirroring vla_hf.utils_data_process.IGNORE_INDEX
IGNORE_INDEX = -100


class _RolloutTrajectoryFusion(torch.nn.Module, TrajectoryFusionMixin):
    """Lightweight helper to reuse ReasoningVLA's fuse_traj_tokens logic in rollout.

    This module only holds trajectory tokenizers and config; it does not construct
    or depend on the full VLM backbone.
    """

    def __init__(self, cfg: AutoConfig, traj_tokenizer, hist_traj_tokenizer=None) -> None:
        super().__init__()
        # ReasoningVLAConfig with traj_token_ids / traj_token_start_idx, etc.
        self.config = cfg

        # Main (future) trajectory tokenizer
        self.traj_tokenizer = traj_tokenizer

        # History trajectory tokenizer (may share weights with future tokenizer)
        if hist_traj_tokenizer is not None:
            self.hist_traj_tokenizer = hist_traj_tokenizer
        else:
            self.hist_traj_tokenizer = traj_tokenizer

        # Match ReasoningVLA._initialize_trajectory_tokenizers
        self.future_token_start_idx = 151665
        self.hist_token_start_idx = 152890


def _pad_1d_sequences(
    sequences: List[torch.Tensor],
    max_len: int,
    padding_side: str,
    padding_value: int | bool,
) -> torch.Tensor:
    """
    Minimal reimplementation of left/right padding for 1D sequences.

    This avoids relying on any custom extensions in vla_hf while keeping
    the behavior equivalent for our use case.
    """
    if padding_side not in ("left", "right"):
        raise NotImplementedError(f"Padding side {padding_side} is not implemented")

    batch_out: List[torch.Tensor] = []
    for seq in sequences:
        # Ensure 1D [T]
        if seq.dim() != 1:
            seq = seq.reshape(-1)
        length = seq.size(0)
        out = seq.new_full((max_len,), padding_value)
        if padding_side == "right":
            out[:length] = seq
        else:  # left
            out[-length:] = seq
        batch_out.append(out)
    return torch.stack(batch_out, dim=0)


def collate_fn(data: dict[str, Any], padding_side: str = "left") -> dict[str, Any]:
    """Collate function for Qwen data."""
    data = basic_collation_fn(data)
    # TODO: get pad_token_id from processor
    pad_token_id = 151643

    if "tokenized_data" in data:
        tokenized_data = {}
        for k in data["tokenized_data"][0].keys():
            if k not in [
                "input_ids",
                "labels_mask",
                "labels",
                "position_ids",
                "attention_mask",
            ]:
                tokenized_data[k] = torch.cat([row[k] for row in data["tokenized_data"]])

        # pad input_ids
        tokenized_data["input_ids"] = torch.nn.utils.rnn.pad_sequence(
            [instance["input_ids"][0] for instance in data["tokenized_data"]],
            batch_first=True,
            padding_value=pad_token_id,
            padding_side=padding_side,
        )

        # pad attention_mask
        tokenized_data["attention_mask"] = tokenized_data["input_ids"].ne(pad_token_id)

        # pad labels, labels_mask if they exist
        if "labels" in data["tokenized_data"][0]:
            tokenized_data["labels"] = torch.nn.utils.rnn.pad_sequence(
                [instance["labels"][0] for instance in data["tokenized_data"]],
                batch_first=True,
                padding_value=IGNORE_INDEX,
                padding_side=padding_side,
            )
        if "labels_mask" in data["tokenized_data"][0]:
            tokenized_data["labels_mask"] = torch.nn.utils.rnn.pad_sequence(
                [instance["labels_mask"][0] for instance in data["tokenized_data"]],
                batch_first=True,
                padding_value=False,
                padding_side=padding_side,
            )
        if "position_ids" in data["tokenized_data"][0]:
            tokenized_data["position_ids"] = torch.nn.utils.rnn.pad_sequence(
                [instance["position_ids"][0] for instance in data["tokenized_data"]],
                batch_first=True,
                padding_value=0,
                padding_side=padding_side,
            )

        # move labels_mask from tokenized_data to data
        if "labels_mask" in tokenized_data:
            data["labels_mask"] = tokenized_data.pop("labels_mask")
        data["tokenized_data"] = tokenized_data

    return data

def build_alpamayo_datastructures(ckpt_path: str):
    """Initialize Alpamayo dataloader and tokenizer once and cache globally."""
    global _alpamayo_dataloaders, _alpamayo_tokenizer, _alpamayo_traj_tokenizer, _alpamayo_tokens_per_future_traj, _alpamayo_ckpt_cfg, _alpamayo_traj_fuser

    if _alpamayo_dataloaders is not None:
        return False

    # Add the src and projects directories to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(current_dir))
    sys.path[:0] = [os.path.join(root, "src"), os.path.join(root, "projects")]

    # Initialize hydra configuration for Alpamayo dataset (reuse AVLA data pipeline)
    with initialize(
        version_base=None,
        config_path="pkg://reasoning_vla",
        job_name="jupyter",
    ):
        cfg = compose(
            config_name="configs/experiments/qwenvl_3b_mlp_v025_eval.yaml",
            overrides=[
                "trainer.per_device_eval_batch_size=1",
                "data.train.num_workers=1",
                "data.val.num_workers=2",
                "trainer.data_seed=1234",
            ],
        )

    ckpt_cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)
    _alpamayo_ckpt_cfg = ckpt_cfg
    _alpamayo_dataloaders = hydra.utils.instantiate(cfg.data, _convert_="partial")

    # Build tokenizer from checkpoint config if available
    try:
        from reasoning_vla.processors.qwen_processor import build_processor

        _alpamayo_tokenizer = build_processor(
            vlm_name_or_path=ckpt_cfg.vlm_name_or_path,
            traj_vocab_size=ckpt_cfg.traj_vocab_size,
            image_height=ckpt_cfg.image_height,
            image_width=ckpt_cfg.image_width,
            min_pixels=ckpt_cfg.min_pixels,
            max_pixels=ckpt_cfg.max_pixels,
            include_camera_ids=ckpt_cfg.include_camera_ids,
            add_special_tokens=ckpt_cfg.add_special_tokens,
        ).tokenizer
        _alpamayo_traj_tokenizer = hydra.utils.instantiate(ckpt_cfg.traj_tokenizer_cfg)

        # Optional separate history trajectory tokenizer, mirroring ReasoningVLA.
        hist_traj_tokenizer = None
        if getattr(ckpt_cfg, "hist_traj_tokenizer_cfg", None) is not None:
            try:
                hist_traj_tokenizer = hydra.utils.instantiate(ckpt_cfg.hist_traj_tokenizer_cfg)
            except Exception:
                print("Failed to instantiate history trajectory tokenizer")
                hist_traj_tokenizer = None

        # Lightweight fusion helper that reuses the same logic as pretraining
        # (TrajectoryFusionMixin.fuse_traj_tokens).
        if _alpamayo_traj_tokenizer is not None:
            _alpamayo_traj_fuser = _RolloutTrajectoryFusion(
                cfg=ckpt_cfg,
                traj_tokenizer=_alpamayo_traj_tokenizer,
                hist_traj_tokenizer=hist_traj_tokenizer,
            )
    except Exception:
        # Fallback: rely on ReasoningVLA processor inside the model path
        print("Failed to build tokenizer from checkpoint config")
        _alpamayo_tokenizer = None
        _alpamayo_traj_tokenizer = None
        _alpamayo_traj_fuser = None

    # Persist tokenizer to checkpoint directory if missing
    # if _alpamayo_tokenizer is not None:
    #     tokenizer_config_path = os.path.join(ckpt_path, "tokenizer_config.json")
    #     if not os.path.exists(tokenizer_config_path):
    #         _alpamayo_tokenizer.save_pretrained(ckpt_path)

    return True


class AlpamayoDataset(Dataset):
    def __init__(self, split: str = "train"):
        self.split = split

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        self.config = config
        _log_gpu_info("AlpamayoDataset.setup")
        split = getattr(self, "split", "train")
        self.dataset = _alpamayo_dataloaders[split].dataset
        self.tokenizer = _alpamayo_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        split = getattr(self, "split", "train")
        sample_idx = str(idx)
        if split == "val" and idx == 16:
            raise StopIteration
        return {"idx": sample_idx, "split": split}

    def get_reference_answer(self, idx: int) -> Any:
        try:
            sample = self.dataset[idx]
        except Exception as e:
            logger.error(f"[AlpamayoDataset] Error getting reference answer: {e}")
            return ""
        if isinstance(sample, dict) and "ego_future_xyz" in sample and "ego_future_rot" in sample:
            return {
                "ego_future_xyz": sample["ego_future_xyz"],
                "ego_future_rot": sample["ego_future_rot"],
                "ego_history_xyz": sample["ego_history_xyz"],
                "ego_history_rot": sample["ego_history_rot"],
                "cot": sample.get("cot", ""),
                "meta_action_strings": sample.get("meta_action_strings", []),
            }
        return ""


class AlpamayoDataPacker(DataPacker):
    Payload = List[Dict[str, Any]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Will be populated in setup; used to de-duplicate image placeholder tokens
        self.image_token_id: Optional[int] = None

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        super().setup(config, tokenizer, *args, **kwargs)
        _log_gpu_info("AlpamayoDataPacker.setup")
	        # Derive the Qwen2.5-VL image_token_id from the HF config so we can
	        # collapse expanded image placeholders back to a single token for vLLM.
        hf_cfg = AutoConfig.from_pretrained(
            config.policy.model_name_or_path,
            trust_remote_code=True,
        )
        self.image_token_id = getattr(hf_cfg, "image_token_id", None)

    def get_rollout_input(self, item: Payload) -> Any:
        idx = int(item["idx"])  # payload idx is str; cast for dataset access
        split = item["split"]
        print('In get_rollout_input, split:', split, 'idx:', idx)
        return _alpamayo_dataloaders[split].dataset[idx]

    def get_cot_input(self, item: Payload) -> Any:
        idx = int(item["idx"])  # payload idx is str; cast for dataset access
        split = item["split"]
        sample = _alpamayo_dataloaders[split].dataset[idx]
        return {k: sample[k] for k in sample.keys() if k == "cot"}

    def rollout_collate_fn(self, items: List[Any]) -> Any:
        """
        Convert Alpamayo world-model items into vLLM-compatible multimodal
        prompts.
 
        Each dataset sample contains:
          - pre-tokenized text under ``sample['tokenized_data']['input_ids']``
          - raw normalized image tensors under ``sample['image_frames']``
 
        We keep the pre-tokenized text as-is (so it stays aligned with the
        ReasoningVLA training pipeline) and attach the per-frame images via
        vLLM's ``multi_modal_data['image']`` interface. vLLM's
        Qwen2.5-VL multimodal processor will then run the official HF
        image processor to produce ``pixel_values`` / ``image_grid_thw``
        that match the placeholder vision tokens in the prompt.
        """
        prompts: List[TokensPrompt] = []
 
        for sample in items:
            if not isinstance(sample, dict):
                raise TypeError(
                    f"Expected Alpamayo sample to be a dict, but got {type(sample)}"
                )
 
            td = sample.get("tokenized_data")
            if td is None:
                raise KeyError(
                    "Expected key 'tokenized_data' in Alpamayo sample for rollout."
                )

            input_ids = td.get("input_ids")
            if input_ids is None:
                raise KeyError(
                    "Expected key 'input_ids' inside sample['tokenized_data'] for rollout."
                )

            # Optionally fuse trajectory tokens into input_ids, mirroring pretraining.
            fused_input_ids = input_ids
         
            if _alpamayo_traj_fuser is not None:
                ego_history_xyz = sample.get("ego_history_xyz", None)
                ego_history_rot = sample.get("ego_history_rot", None)
                ego_future_xyz = sample.get("ego_future_xyz", None)
                ego_future_rot = sample.get("ego_future_rot", None)

                if isinstance(ego_history_xyz, torch.Tensor) and isinstance(
                    ego_history_rot, torch.Tensor
                ):
                    if ego_history_xyz.dim() == 3:
                        ego_history_xyz = ego_history_xyz.unsqueeze(1)
                    if ego_history_rot.dim() == 4:
                        ego_history_rot = ego_history_rot.unsqueeze(1)
                    if isinstance(ego_future_xyz, torch.Tensor) and ego_future_xyz.dim() == 3:
                        ego_future_xyz = ego_future_xyz.unsqueeze(1)
                    if isinstance(ego_future_rot, torch.Tensor) and ego_future_rot.dim() == 4:
                        ego_future_rot = ego_future_rot.unsqueeze(1)

                    if not hasattr(fused_input_ids, "dim") or fused_input_ids.dim() == 1:
                        fused_input_ids = fused_input_ids.unsqueeze(0)

                    traj_data = {
                        "ego_history_xyz": ego_history_xyz,
                        "ego_history_rot": ego_history_rot,
                        "ego_future_xyz": ego_future_xyz,
                        "ego_future_rot": ego_future_rot,
                    }
                    fused_input_ids = _alpamayo_traj_fuser.fuse_traj_tokens(
                        fused_input_ids, traj_data
                    )


            # fused_input_ids is expected to be [1, T]
            if hasattr(fused_input_ids, "dim") and fused_input_ids.dim() == 2:
                token_ids = fused_input_ids[0].tolist()
            else:  # fallback for unexpected shapes / types
                token_ids = (
                    fused_input_ids.tolist()
                    if hasattr(fused_input_ids, "tolist")
                    else list(fused_input_ids)
                )
            # # add debug break
            # import builtins as _b
            # if hasattr(_b, "debug_break"):
            #     _b.debug_break("rollout_collate_fn")
            self.image_token_id = 151655
            if self.image_token_id is not None:
                # Compress runs of image placeholders back to a single token per run;
                # vLLM's Qwen2(VL) multimodal processor will then expand them once.
                img_id = self.image_token_id
                token_ids = [
                    tid for i, tid in enumerate(token_ids)
                    if not (tid == img_id and i > 0 and token_ids[i - 1] == img_id)
                ]
 
            mm_data: Dict[str, Any] = {}
            # Attach image frames if present so vLLM can reconstruct the
            # correct 3D RoPE grids. Shape is (num_groups, frames_per_group, C, H, W).
            image_frames = sample.get("image_frames", None)
            if image_frames is not None:
                # Flatten temporal groups into a single list of frames and
                # move them to CPU for HF processors. Each element is a
                # 3D tensor (C, H, W), which matches `ImageItem`.
                frames = image_frames.flatten(0, 1)  # (N, C, H, W)
                images = [f.cpu() for f in frames]
                if len(images) > 0:
                    mm_data["image"] = images

 
            prompt_dict: TokensPrompt = {"prompt_token_ids": token_ids}
            if mm_data:
                prompt_dict["multi_modal_data"] = mm_data
                mm_kwargs = {
                    "do_rescale": False,
                }
                mm_kwargs = {k: v for k, v in mm_kwargs.items() if v is not None}
                prompt_dict["hf_processor_mm_kwargs"] = mm_kwargs
                prompt_dict["mm_processor_kwargs"] = mm_kwargs
 
            prompts.append(prompt_dict)        
        # Save the prompts[0]['multi_modal_data']['image'] as a npz file 
        # import numpy as np
        # np.savez("reasoning_vla_debug_temp/vllm_input_image.npz", image=prompts[0]['multi_modal_data']['image'])
 
        return prompts


    def decode_rollout_trajectory(
        self,
        to_be_evaluated: str,
        ego_history_xyz: torch.Tensor,
        ego_history_rot: torch.Tensor,
        token_id_offset: int = 151665,
        expected_len: int = 6,
    ) -> float:
        """
        Decode the rollout trajectory and return the predicted future trajectory.
        """
        # Extract inner text between traj start/end markers
        traj_tokens = to_be_evaluated.split("<|traj_future_start|>")[-1].split("<|traj_future_end|>")[0]
        texts = list(traj_tokens) if isinstance(traj_tokens, (list, tuple)) else [traj_tokens]

        # Tokenize to ids
        enc_ids = []
        for t in texts:
            if not isinstance(t, str):
                t = ""
            ids = self.tokenizer(t, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            enc_ids.append(ids)

        # Pad to rectangular [B, L]
        import torch

        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        sequences = torch.nn.utils.rnn.pad_sequence(enc_ids, batch_first=True, padding_value=pad_id)
        traj_token_ids = sequences - token_id_offset

        traj_tokenizer_vocab_size = getattr(_alpamayo_traj_tokenizer, "vocab_size", None)
        if traj_tokenizer_vocab_size is None:
            raise ValueError("traj_tokenizer_vocab_size is not set")

        invalid = (
            torch.any(traj_token_ids < 0)
            or torch.any(traj_token_ids > traj_tokenizer_vocab_size)
            or traj_token_ids.shape[1] != expected_len
        )
        if invalid:
            # We need to clamp the traj_token_ids to the traj_tokenizer_vocab_size
            traj_token_ids = torch.clamp(traj_token_ids, max=traj_tokenizer_vocab_size - 1)
            # We need to pad the traj_token_ids to the expected_len
            traj_token_ids = torch.nn.functional.pad(traj_token_ids, (0, expected_len - traj_token_ids.shape[1]))
            print('Rollout trajectory is invalid, clamped and padded to expected length')
            
        # Align batch dimensions
        ego_hist_xyz = ego_history_xyz
        ego_hist_rot = ego_history_rot
        B = min(ego_hist_xyz.shape[0], traj_token_ids.shape[0])
        if B != traj_token_ids.shape[0]:
            traj_token_ids = traj_token_ids[:B]
        if B != ego_hist_xyz.shape[0]:
            ego_hist_xyz = ego_hist_xyz[:B]
            ego_hist_rot = ego_hist_rot[:B]

        assert ego_hist_xyz.shape[0] == 1
        assert traj_token_ids.shape[0] == 1

        predicted_fut_xyz, predicted_fut_rot, _ = _alpamayo_traj_tokenizer.decode(
            hist_xyz=ego_hist_xyz,
            hist_rot=ego_hist_rot,
            tokens=traj_token_ids,
        )
        return predicted_fut_xyz, predicted_fut_rot


    def get_policy_input(
        self,
        data: dict,
        completions: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        idx = int(data["idx"])  # payload idx is str; cast for dataset access
        split = data["split"]
        data_dict = _alpamayo_dataloaders[split].dataset[idx]
        assert isinstance(data_dict, dict), "data_dict must be a dict"
        
        # add debug break
        import builtins as _b
        if hasattr(_b, "debug_break"):
            _b.debug_break("get_policy_input")
        
        # Stitch generated completion tokens into the traj_future span of input_ids [TODO: make this generic, now it's for traj_future only]
        tokenized = data_dict["tokenized_data"]
        alp_tok = _alpamayo_tokenizer
        # Sanity checks: these should always be available; fail fast if not
        if tokenized is None:
            raise KeyError(
                f"Missing tokenized_data for sample idx={idx}, split={split}"
            )
        if alp_tok is None:
            raise RuntimeError("Global Alpamayo tokenizer is not initialized")
        
        input_ids: torch.Tensor = tokenized.get("input_ids")  # shape: [1, L]
        assert input_ids is not None and input_ids.ndim == 2 and input_ids.shape[0] == 1, "input_ids must be a 2D tensor with shape [1, L]"

        t2id = alp_tok.convert_tokens_to_ids
        end_id = t2id(SPECIAL_TOKENS["traj_future_end"])      # <|traj_future_end|>

        ids_row = input_ids[0]              
        inner_text = completions

        # Default to using the string completions; optionally override with GT tokens
        gen_ids = alp_tok(inner_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]

        use_gt = os.environ.get("COSMOS_USE_SFT_LOSS", "0") == "1"
        #debug_break("get_policy_input")
        if use_gt:
            hist_xyz = data_dict.get("ego_history_xyz")
            hist_rot = data_dict.get("ego_history_rot")
            fut_xyz = data_dict.get("ego_future_xyz")
            fut_rot = data_dict.get("ego_future_rot")
            if (
                hist_xyz is not None
                and hist_rot is not None
                and fut_xyz is not None
                and fut_rot is not None
            ):
                # Use t0 group only to match AVLA decode path
                hx = (
                    hist_xyz[:, -1]
                    if hasattr(hist_xyz, "ndim") and hist_xyz.ndim >= 4
                    else hist_xyz
                )  # [B, Th, 3]
                hr = (
                    hist_rot[:, -1]
                    if hasattr(hist_rot, "ndim") and hist_rot.ndim >= 5
                    else hist_rot
                )  # [B, Th, 3, 3]
                fx = (
                    fut_xyz[:, -1]
                    if hasattr(fut_xyz, "ndim") and fut_xyz.ndim >= 4
                    else fut_xyz
                )  # [B, Tf, 3]
                fr = (
                    fut_rot[:, -1]
                    if hasattr(fut_rot, "ndim") and fut_rot.ndim >= 5
                    else fut_rot
                )  # [B, Tf, 3, 3]
                gt_local = _alpamayo_traj_tokenizer.encode(
                    hist_xyz=hx, hist_rot=hr, fut_xyz=fx, fut_rot=fr
                )
                if hasattr(gt_local, "ndim") and gt_local.ndim == 2:
                    gt_local = gt_local[0]
                gen_ids = (
                    gt_local.to(device=ids_row.device, dtype=ids_row.dtype)
                    + getattr(alp_tok, "traj_token_start_idx", 0)
                )
                # Clamp length to configured tokens_per_future_traj if available
                if _alpamayo_tokens_per_future_traj is not None:
                    gen_ids = gen_ids[: int(_alpamayo_tokens_per_future_traj)]

        if gen_ids.numel() > 0:
            # append tokens and explicit end token
            end_token_tensor = torch.tensor([end_id], dtype=ids_row.dtype, device=ids_row.device)
            new_ids_row = torch.cat([ids_row, gen_ids.to(ids_row.device), end_token_tensor], dim=0)

            # rebuild labels_mask for the appended region only
            new_mask = torch.zeros_like(new_ids_row, dtype=torch.bool)
            new_mask[-(gen_ids.numel()+1):] = True # include the end token

            # extend position_ids to match new length by continuing the sequence
            pos_ids_row = tokenized.get("position_ids")
            if pos_ids_row is not None and pos_ids_row.ndim == 2 and pos_ids_row.shape[0] == 1:
                pos_ids_row0 = pos_ids_row[0]
                last_pos = int(pos_ids_row0[-1].item()) if pos_ids_row0.numel() > 0 else -1
                append_len = new_ids_row.shape[0] - ids_row.shape[0]
                append_positions = torch.arange(
                    last_pos + 1,
                    last_pos + 1 + append_len,
                    dtype=pos_ids_row0.dtype,
                    device=pos_ids_row0.device,
                )
                new_pos_ids_row = torch.cat([pos_ids_row0, append_positions], dim=0)
                data_dict["tokenized_data"]["position_ids"] = new_pos_ids_row.unsqueeze(0)

            # extend attention_mask_4d with a causal mask for the new length (safe default)
            att4d = tokenized.get("attention_mask_4d")
            if att4d is not None and att4d.ndim == 4:
                new_L = new_ids_row.shape[0]
                dtype = att4d.dtype
                device = att4d.device
                # build lower-triangular causal mask
                tri = torch.ones(new_L, new_L, dtype=torch.bool, device=device).tril()
                score_mask = torch.zeros_like(tri, dtype=torch.float32)
                score_mask.masked_fill_(~tri, torch.finfo(score_mask.dtype).min)
                tokenized["attention_mask_4d"] = score_mask.to(dtype)[None, None, ...]

            data_dict["tokenized_data"]["labels_mask"] = new_mask.unsqueeze(0)
            data_dict["tokenized_data"]["input_ids"] = new_ids_row.unsqueeze(0)

            # If we used GT tokens, also override the completions string to keep downstream consistent
            #debug_break("get_policy_input_after_stitching")
            if use_gt:
                gt_inner_text = alp_tok.decode(
                    gen_ids.tolist() if hasattr(gen_ids, "tolist") else gen_ids,
                    skip_special_tokens=False,
                )
                # Wrap with start/end tags for consistency downstream
                start_tok = SPECIAL_TOKENS["traj_future_start"]
                end_tok = SPECIAL_TOKENS["traj_future_end"]
                data_dict["ego_future_completions"] = start_tok + gt_inner_text + end_tok



        predicted_fut_xyz, predicted_fut_rot = self.decode_rollout_trajectory(completions, data_dict["ego_history_xyz"], data_dict["ego_history_rot"])
        data_dict["ego_rollout_xyz"] = predicted_fut_xyz
        data_dict["ego_rollout_rot"] = predicted_fut_rot

        # We need to update the input_ids with the predicted future trajectory
        return data_dict

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        return max([x["tokenized_data"]["input_ids"].shape[1] for x in processed_samples])

    def policy_collate_fn(self, processed_samples: List[Any], computed_max_len: int) -> Dict[str, Any]:
        # Add debug break
        import builtins as _b
        if hasattr(_b, "debug_break"):
            _b.debug_break("policy_collate_fn")
        
        batch = collate_fn(processed_samples)

        # Lift inputs needed by GRPO trainer to top-level
        tokenized = batch.get("tokenized_data", {})
        if "input_ids" in tokenized:
            batch["input_ids"] = tokenized["input_ids"]
        if "position_ids" in tokenized:
            batch["position_ids"] = tokenized["position_ids"]
        if "attention_mask" in tokenized:
            batch["attention_mask"] = tokenized["attention_mask"]
        if "labels_mask" in tokenized:
            lmask = tokenized["labels_mask"]
            batch["labels_mask"] = lmask if lmask.dtype == torch.bool else lmask.bool()
        if "attention_mask_4d" in tokenized:
            batch["attention_mask_4d"] = tokenized["attention_mask_4d"]

        # Build logprob_masks expected by trainer.compute_logprobs
        if "labels_mask" in batch:
            lmask = batch["labels_mask"]
            batch["logprob_masks"] = lmask if lmask.dtype == torch.bool else lmask.bool()
        else:
            if "input_ids" in tokenized:
                B, T = tokenized["input_ids"].shape
                mask = torch.ones(B, T, dtype=torch.bool, device=tokenized["input_ids"].device)
                mask[:, 0] = False
                batch["logprob_masks"] = mask
            else:
                raise KeyError("tokenized_data.input_ids is required to construct logprob_masks")

        return batch


def reasoning_trace_reward(to_be_evaluated: str, gt_reasoning_length: str) -> int:
    return reasoning_trace_reward_common(to_be_evaluated, gt_reasoning_length)


def custom_reward_fn(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> Tuple[float, Dict[str, float]]:
    """
    Return the total reward and the reward dictionary.
    """
    alpamayo_tokenizer = kwargs.get("tokenizer")
    ref_cot = reference.get("cot", "")

    reward_dict = custom_reward_fn_common(
        to_be_evaluated,
        reference,
        tokenizer=alpamayo_tokenizer,
        traj_tokenizer=_alpamayo_traj_tokenizer,
        expected_len=6,
        invalid_penalty=-100.0,
        print_invalid=False,
        traj_weight=1.0,
        reasoning_weight=0.0,
    )
    reward_dict["reward"] = 0.0 * reward_dict["reasoning_reward"] + \
                1.0 * reward_dict["traj_reward"] + \
                0.0 * reward_dict["collision_reward"] + \
                0.0 * reward_dict["reasoning_consistency_reward"]
    
    return reward_dict["reward"], reward_dict


if __name__ == "__main__":
    # Ensure project modules are importable
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(os.path.dirname(current_dir))
    sys.path[:0] = [os.path.join(root, "src"), os.path.join(root, "projects")]

    from alpamayo_cosmos.model_reasoning_vla_cosmos import (
        ReasoningVLAModel,
        ReasoningVLAWeightMapper,
    )

    _maybe_pin_gpu_per_role()
    _log_gpu_info("before launch_worker")

    # Initialize data/tokenizer cache once.
    build_alpamayo_datastructures(DEFAULT_ALPAMAYO_CKPT_PATH)

    # Register ReasoningVLA model + weight mapper so cosmos-rl recognizes model_type "reasoning_vla"
    ModelRegistry.register_model(
        ReasoningVLAModel,
        ReasoningVLAWeightMapper,
        data_packer_cls=AlpamayoDataPacker,
    )

    def get_dataset_factory(config: Config) -> Dataset:
        return AlpamayoDataset(split="train")

    def get_val_dataset_factory(config: Config) -> Dataset:
        return AlpamayoDataset(split="val")

    launch_worker(
        dataset=get_dataset_factory,
        data_packer=AlpamayoDataPacker(),
        reward_fns=[custom_reward_fn],
        val_dataset=get_val_dataset_factory,
        val_data_packer=AlpamayoDataPacker(),
        val_reward_fns=[custom_reward_fn],
    )


