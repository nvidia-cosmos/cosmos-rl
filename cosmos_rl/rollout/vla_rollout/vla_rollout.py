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
from types import SimpleNamespace
import torch
import numpy as np
from typing import Optional, List, Dict, Any
from transformers import AutoConfig

from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import DataFetcherBase
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.rollout.rollout_base import RolloutBase, RolloutRegistry
from cosmos_rl.rollout.schema import RolloutResult
from cosmos_rl.utils import util

from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.utils.replay_buffer import save_trajectory_to_buffer
from cosmos_rl.rollout.vla_rollout.trace_utils import create_tracing_manager


def get_physical_gpu_id():
    """Get the physical GPU ID, accounting for CUDA_VISIBLE_DEVICES.

    When torchrun is used with CUDA_VISIBLE_DEVICES=[4,5,6,7], PyTorch sees
    GPUs 0-3, but we want to return the actual physical IDs 4-7 for logging.

    Returns:
        int: Physical GPU ID
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # Parse CUDA_VISIBLE_DEVICES if set
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        # Handle formats: "4,5,6,7" or "[4,5,6,7]" or "4 5 6 7"
        cuda_visible = cuda_visible.strip("[]").replace(" ", ",")
        gpu_ids = [int(x.strip()) for x in cuda_visible.split(",") if x.strip()]

        # Map local rank to physical GPU ID
        if local_rank < len(gpu_ids):
            return gpu_ids[local_rank]

    # Fallback to local rank if CUDA_VISIBLE_DEVICES not set
    return local_rank


def get_simulator_type(config: Config) -> str:
    """Determine simulator type based on dataset subset.

    Args:
        config: Training configuration

    Returns:
        "b1k" or "libero"
    """
    subset = config.validation.dataset.subset.lower()
    if "b1k" in subset:
        return "b1k"
    elif "libero" in subset:
        return "libero"
    else:
        # Default to libero for backward compatibility
        logger.warning(
            f"Unknown dataset subset '{subset}', defaulting to libero simulator"
        )
        return "libero"


def extract_simulator_config(config: Config):
    """Extract simulator configuration based on simulator type.

    Args:
        config: Training configuration

    Returns:
        SimpleNamespace with simulator-specific configuration
    """
    cfg = SimpleNamespace()
    cfg.num_envs = config.vla.num_envs

    sim_type = get_simulator_type(config)

    if sim_type == "b1k":
        # B1K-specific config
        cfg.height = getattr(config.vla, "height", 256)
        cfg.width = getattr(config.vla, "width", 256)
    elif sim_type == "libero":
        # Libero-specific config
        from cosmos_rl.simulators.libero.utils import (
            LIBERO_MAX_STEPS_MAP,
        )

        cfg.task_suite_name = config.validation.dataset.subset
        cfg.max_steps = LIBERO_MAX_STEPS_MAP.get(cfg.task_suite_name, 512)

    return cfg


@RolloutRegistry.register(rollout_type="vla")
class OpenVLARollout(RolloutBase):
    def __init__(
        self,
        config: Config,
        parallel_dims: ParallelDims,
        device: torch.device,
        **kwargs,
    ):
        super().__init__(config, parallel_dims, device, **kwargs)
        self.num_envs = config.vla.num_envs

        self.obs_keys = ["full_images", "wrist_images", "states"]

    def post_init_hook(self, **kwargs):
        self._model_param_map = None  # Required by RolloutBase.model_param_map()
        self.model_type = self.config.vla.vla_type

        model_cls = ModelRegistry._MODEL_REGISTRY[self.model_type]
        if hasattr(model_cls, "preprocess_hf_config"):
            self.hf_config = model_cls.preprocess_hf_config(self.config)
        else:
            self.hf_config = util.retry(AutoConfig.from_pretrained)(
                self.config.policy.model_name_or_path
            )

        # Determine which environment wrapper to use based on simulator type
        sim_type = get_simulator_type(self.config)
        if sim_type == "b1k":
            from cosmos_rl.simulators.b1k.env_wrapper import (
                B1KEnvWrapper as EnvWrapperCls,
            )

            # B1K/OmniGibson cannot run in subprocess due to Isaac Sim's complex GPU/thread management
            # - "spawn": causes pidfd_getfd permission errors (needs CAP_SYS_PTRACE)
            # - "fork": causes segmentation fault (GPU contexts copied in bad state)
            # Solution: Run in-process
            use_subprocess = False
            logger.info("Initializing B1K simulator in-process (no subprocess)")
        elif sim_type == "libero":
            from cosmos_rl.simulators.libero.env_wrapper import (
                LiberoEnvWrapper as EnvWrapperCls,
            )

            # Libero works fine in subprocess with spawn method
            use_subprocess = True
            logger.info("Initializing LIBERO simulator in subprocess (spawn)")
        else:
            raise ValueError(f"Invalid simulator type: {sim_type}")

        self.env_manager = EnvManager(
            cfg=extract_simulator_config(self.config),
            rank=get_physical_gpu_id(),
            env_cls=EnvWrapperCls,
            use_subprocess=use_subprocess,
        )
        self.env_manager.start_simulator()

        # Initialize tracing system (if enabled via config)
        trace_verbosity = getattr(self.config.vla, "trace_verbosity", 0)
        self.tracing_manager = create_tracing_manager(
            rank=get_physical_gpu_id(),
            output_dir=self.config.train.output_dir,
            trace_verbosity=trace_verbosity,
        )

    def get_underlying_model(self) -> torch.nn.Module:
        return self.model

    def set_underlying_model(self, model: torch.nn.Module):
        self.model = model

    def init_engine(
        self,
        quantization: Optional[str] = None,
        seed: int = 42,
        load_format: str = "dummy",
        **kwargs,
    ):
        if self._engine_initialized:
            return

        self.model = ModelRegistry.build_model(self.config)
        self.processor = self.model.processor
        self.tokenizer = self.model.tokenizer
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        pfn, _ = self.model.parallelize_fn
        pfn(self.model, self.parallel_dims, self.config)

        if self.config.mode != "colocated":
            self.model.load_hf_weights(
                self.config.policy.model_name_or_path,
                self.parallel_dims,
                torch.device("cuda"),
            )
        self.model.eval()
        self.vla_input_keys = self.model.model_input_keys
        self.vla_output_keys = self.model.model_output_keys
        self.vla_train_keys = self.model.model_train_keys
        self._engine_initialized = True
        logger.info("[Rollout] Engine initialized.")

    def _prepare_payload_list(
        self, payloads: List[RLPayload], is_validation: bool
    ) -> List[RLPayload]:
        self.n_generation = (
            max(1, self.config.rollout.n_generation) if not is_validation else 1
        )
        return np.array(
            [[idx for _ in range(self.n_generation)] for idx in range(len(payloads))]
        ).flatten()

    def _setup_parallel_envs(
        self, payloads: List[RLPayload], env_ids: List[int], is_validation: bool
    ):
        task_ids = []
        trial_ids = []
        for payload in payloads:
            task_ids.append(payload.prompt.get("task_id", 0))
            trial_ids.append(payload.prompt.get("trial_id", 0))

        images_and_states, task_descriptions = self.env_manager.reset(
            env_ids, task_ids, trial_ids, [is_validation] * len(env_ids)
        )

        return {
            **images_and_states,
            "task_descriptions": task_descriptions,
        }

    def _setup_parallel_envs_async(
        self, payloads: List[RLPayload], env_ids: List[int], is_validation: bool
    ):
        task_ids = []
        trial_ids = []
        for payload in payloads:
            task_ids.append(payload.prompt.get("task_id", 0))
            trial_ids.append(payload.prompt.get("trial_id", 0))
        self.env_manager.reset_async(
            env_ids, task_ids, trial_ids, [is_validation] * len(env_ids)
        )

    def _get_init_results(
        self,
        sim_results: Dict[str, Any],
        available_env_ids: List[int],
        enqueue_payload_list: List[RLPayload],
        is_validation: bool,
    ):
        with self.tracing_manager.trace("env_reset", env_ids=available_env_ids):
            init_results = self._setup_parallel_envs(
                enqueue_payload_list, available_env_ids, is_validation
            )
        for key in self.obs_keys:
            if sim_results[key] is None:
                data_shape = (
                    self.num_envs,
                    *init_results[key].shape[1:],
                )
                sim_results[key] = np.zeros(data_shape, dtype=init_results[key].dtype)
            sim_results[key][available_env_ids] = init_results[key].copy()
        for i, env_id in enumerate(available_env_ids):
            sim_results["task_descriptions"][env_id] = init_results[
                "task_descriptions"
            ][i]

    def _wait_init_results(
        self, sim_results: Dict[str, Any], async_wait_envs: List[int]
    ):
        wait_env_ids = []
        for env_id in range(len(async_wait_envs)):
            if async_wait_envs[env_id] == 0:
                wait_env_ids.append(env_id)
            elif async_wait_envs[env_id] > 0:
                async_wait_envs[env_id] -= 1

        if wait_env_ids:
            images_and_states, task_descriptions = self.env_manager.reset_wait(
                wait_env_ids
            )
            for key in self.obs_keys:
                if sim_results[key] is None:
                    data_shape = (
                        self.num_envs,
                        *images_and_states[key].shape[1:],
                    )
                    sim_results[key] = np.zeros(
                        data_shape, dtype=images_and_states[key].dtype
                    )
                sim_results[key][wait_env_ids] = images_and_states[key].copy()
            for i, env_id in enumerate(wait_env_ids):
                sim_results["task_descriptions"][env_id] = task_descriptions[i]
            for env_id in wait_env_ids:
                async_wait_envs[env_id] = -1

    @torch.no_grad()
    def _do_rollout(
        self,
        payloads: List[RLPayload],
        payload_indices: np.ndarray,
        is_validation: bool,
        continuous: bool = False,
    ):
        actions = None
        enqueued_payloads = 0
        finished_payloads = 0

        sim_results = {k: None for k in self.obs_keys}
        sim_results["task_descriptions"] = [""] * self.num_envs

        payload_env_mapping = np.full(self.num_envs, -1)

        task_records = [{} for _ in range(len(payload_indices))]
        for i in range(len(payload_indices)):
            payload = payloads[payload_indices[i]]
            task_records[i] = {
                "task_id": payload.prompt.get("task_id", 0),
                "trial_id": payload.prompt.get("trial_id", 0),
                "task_suite_name": payload.prompt.get("task_suite_name", ""),
                "complete": False,
                "finish_step": -1,
            }
            for key in self.vla_train_keys:
                task_records[i][key] = []

        rollout_step = 0
        async_wait_envs = [-1 for _ in range(self.num_envs)]

        active_env_ids = []
        while finished_payloads < len(payload_indices):
            # Step 1: Advance active environments
            if active_env_ids:
                with self.tracing_manager.trace("sim_step", env_ids=active_env_ids):
                    step_results = self.env_manager.chunk_step(active_env_ids, actions)
                active_indices, finished_env_ids = [], []
                for i, env_id in enumerate(active_env_ids):
                    if step_results["active"][i]:
                        active_indices.append(i)
                    else:
                        finished_env_ids.append(env_id)
                        task_idx = payload_env_mapping[env_id]
                        task_records[task_idx]["complete"] = step_results["complete"][i]
                        task_records[task_idx]["active"] = step_results["active"][i]
                        task_records[task_idx]["finish_step"] = step_results[
                            "finish_step"
                        ][i]
                        task_records[task_idx]["end_time"] = time.time()
                        payload_env_mapping[env_id] = -1
                active_env_ids = [active_env_ids[i] for i in active_indices]
                finished_payloads += len(finished_env_ids)
                for key in self.obs_keys:
                    data_shape = (
                        self.num_envs,
                        *step_results[key][active_indices].shape[1:],
                    )
                    sim_results[key] = np.zeros(
                        data_shape, dtype=step_results[key][active_indices].dtype
                    )
                    sim_results[key][active_env_ids] = step_results[key][
                        active_indices
                    ].copy()

                if is_validation and self.config.vla.save_video:
                    rollout_dir = os.path.join(
                        self.config.train.output_dir, "vla_rollouts"
                    )
                    self.env_manager.save_validation_videos(
                        rollout_dir, finished_env_ids
                    )

            # Step 2: Enqueue new payloads if envs become available
            enqueue_payload_list = []
            left_payloads = len(payload_indices) - enqueued_payloads
            if continuous and np.any(payload_env_mapping == -1):
                # continuous rollout, enqueue new payloads if any env becomes available
                available_env_ids = [
                    i for i, pidx in enumerate(payload_env_mapping) if pidx == -1
                ][:left_payloads]
            elif np.all(payload_env_mapping == -1):
                # all envs are idle, enqueue new payloads to all envs
                available_env_ids = list(range(self.num_envs))[:left_payloads]
            else:
                available_env_ids = []

            for env_id in available_env_ids:
                payload_idx = payload_indices[enqueued_payloads]
                payload = payloads[payload_idx]
                payload_env_mapping[env_id] = enqueued_payloads
                enqueue_payload_list.append(payload)
                enqueued_payloads += 1

            # self._wait_init_results(sim_results, async_wait_envs)
            if available_env_ids:
                self._get_init_results(
                    sim_results, available_env_ids, enqueue_payload_list, is_validation
                )
                # self._setup_parallel_envs_async(
                #     enqueue_payload_list, available_env_ids, is_validation
                # )
                # for env_id in available_env_ids:
                #     async_wait_envs[env_id] = 10

            active_env_ids = [
                i
                for i, pidx in enumerate(payload_env_mapping)
                if pidx != -1 and async_wait_envs[i] == -1
            ]
            logger.info(
                f"payload_env_mapping: {payload_env_mapping}, "
                f"finished_payloads: {finished_payloads}/{len(payload_indices)}, "
                f"enqueued_payloads: {enqueued_payloads}/{len(payload_indices)}, "
                f"waiting_envs_left_steps: {async_wait_envs}, active_env_ids: {active_env_ids}"
            )
            if not active_env_ids:
                continue

            active_sim_results = {"task_descriptions": []}
            for k in self.obs_keys:
                active_sim_results[k] = sim_results[k][active_env_ids]
            for env_id in active_env_ids:
                active_sim_results["task_descriptions"].append(
                    sim_results["task_descriptions"][env_id]
                )

            # Step 3: Generate VLA output
            with self.tracing_manager.trace(
                "inference", env_ids=active_env_ids, batch_size=len(active_env_ids)
            ):
                vla_input = self.model.process_input(active_sim_results)
                vla_output = self.model.generate_action(
                    vla_input,
                    is_valid=is_validation,
                    temperature=self.config.rollout.sampling_config.temperature,
                    unnorm_key="libero_10_no_noops",
                )
            for i, env_id in enumerate(active_env_ids):
                task_idx = payload_env_mapping[env_id]
                for key in self.vla_input_keys:
                    task_records[task_idx][key].append(vla_input[key][i])
                for key in self.vla_output_keys:
                    task_records[task_idx][key].append(vla_output[key][i])

            actions = vla_output["action"]
            rollout_step += 1
        return task_records

    def rollout_generation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        is_validation: bool = False,
        **kwargs,
    ):
        self.model._set_fsdp_reshard_after_forward("never")

        # Start a new rollout trace (sets validation state)
        self.tracing_manager.start_rollout(is_validation)

        payload_indices = self._prepare_payload_list(payloads, is_validation)

        # Track time for rollout
        rollout_start = time.time()
        task_records = self._do_rollout(
            payloads,
            payload_indices,
            is_validation,
            self.config.vla.continuous,
        )
        rollout_end = time.time()

        # Calculate simulation FPS
        total_sim_frames = sum(task.get("finish_step", 0) for task in task_records)
        rollout_duration = rollout_end - rollout_start
        sim_fps = total_sim_frames / rollout_duration if rollout_duration > 0 else 0.0

        logger.info(
            f"Rollout generation complete: "
            f"{len(task_records)} tasks, {total_sim_frames} sim frames, "
            f"{rollout_duration:.2f}s, {sim_fps:.2f} sim FPS"
        )

        # Finalize rollout (adds task events, rollout-level trace event, and dumps trace file)
        self.tracing_manager.finalize_rollout(
            task_records=task_records,
            rollout_start_time=rollout_start,
            rollout_end_time=rollout_end,
            continuous=self.config.vla.continuous,
        )

        results = self._pack_grpo_results(
            self.n_generation, payload_indices, task_records, is_validation
        )

        self.model._set_fsdp_reshard_after_forward(
            self.config.train.fsdp_reshard_after_forward
        )
        return results

    def _pack_grpo_results(
        self,
        n_generation: int,
        payload_indices: List[int],
        task_records: List[Dict],
        is_validation: bool,
    ):
        """
        Pack GRPO results and create RolloutResults

        Args:
            n_generation: Number of generations per payload
            payload_indices: List of payload indices
            task_records: List of task metadata dicts
            is_validation: Whether to save validation videos

        Returns:
            List of RolloutResult objects if valid, None if filtered out
        """

        n_payloads = len(payload_indices) // n_generation
        successes = [0] * n_payloads
        for i in range(n_payloads):
            for j in range(n_generation):
                payload_idx = i * n_generation + j
                if task_records[payload_idx]["complete"]:
                    successes[i] += 1
        success_rates = [successes[i] / n_generation for i in range(n_payloads)]
        avg_success_rate = sum(success_rates) / n_payloads * 100

        if is_validation:
            logger.info(
                f"Validation {n_payloads} avg success rate: {avg_success_rate:.2f}%"
            )
        else:
            formatted_rates = ", ".join(
                [f"{rate * 100:.2f}%" for rate in success_rates]
            )
            logger.info(
                f"Rollout {n_payloads}x{n_generation} success rates: [{formatted_rates}], avg {avg_success_rate:.2f}%"
            )

        def _trim_input_ids(
            input_ids: List[torch.Tensor], attention_mask: List[torch.Tensor]
        ):
            """Remove padding tokens from input_ids using attention_mask."""
            trimmed_input_ids, trimmed_attention_mask = [], []
            for step_input_ids, step_attention_mask in zip(input_ids, attention_mask):
                # Convert to CPU for indexing if needed, then create boolean mask
                valid_mask = step_attention_mask.bool()
                trimmed_step_input_ids = step_input_ids[valid_mask]
                trimmed_step_attention_mask = torch.ones_like(trimmed_step_input_ids)
                trimmed_input_ids.append(trimmed_step_input_ids)
                trimmed_attention_mask.append(trimmed_step_attention_mask)
            return trimmed_input_ids, trimmed_attention_mask

        def pack_trajectory(payload_idx: int):
            start_idx = payload_idx * n_generation
            completions = []
            sr = success_rates[payload_idx]
            filter = sr == 0 or sr == 1

            for i in range(n_generation):
                record = task_records[start_idx + i]
                traj = {}
                record["input_ids"], record["attention_mask"] = _trim_input_ids(
                    record["input_ids"], record["attention_mask"]
                )
                for key in self.vla_train_keys:
                    traj[key] = torch.stack(record[key], dim=0)

                trajectory_id = (
                    save_trajectory_to_buffer(
                        traj,
                        buffer_dir=os.path.join(
                            self.config.train.output_dir, "replay_buffer"
                        ),
                    )
                    if not filter
                    else ""
                )
                completions.append(
                    {
                        "complete": bool(task_records[start_idx + i]["complete"]),
                        "finish_step": int(task_records[start_idx + i]["finish_step"]),
                        "trajectory_id": trajectory_id,
                    }
                )
            return RolloutResult(completions=completions)

        results = [pack_trajectory(i) for i in range(n_payloads)]
        return results
