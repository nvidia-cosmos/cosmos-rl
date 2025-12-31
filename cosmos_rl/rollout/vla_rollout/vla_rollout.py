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
import json
import time
from datetime import datetime
from types import SimpleNamespace
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
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
from cosmos_rl.simulators.libero.utils import (
    normalize_gripper_action,
    invert_gripper_action,
    obs_to_vla_input,
    LIBERO_MAX_STEPS_MAP,
)
from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.simulators.libero.env_wrapper import LiberoEnvWrapper
from cosmos_rl.utils.replay_buffer import save_trajectory_to_buffer


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict) -> np.ndarray:
    """Normalize proprioception data using norm stats"""
    mean = norm_stats.get("mean", 0.0)
    std = norm_stats.get("std", 1.0)
    return (proprio - mean) / std


def center_crop_image(image: Image.Image, crop_size: int = 256) -> Image.Image:
    """
    Center crop image with 0.9 scale then resize (matching SimpleVLA-RL)

    This function mimics SimpleVLA-RL's TensorFlow-based center crop:
    - Crops to 90% of the center (zoom in effect)
    - Resizes back to 224x224

    Replaced TensorFlow with torchvision for better compatibility.
    """
    import torchvision.transforms.functional as TF

    crop_scale = 0.9  # Match SimpleVLA-RL

    # Get original image dimensions
    width, height = image.size

    # Calculate crop dimensions (sqrt of scale to match TF implementation)
    crop_ratio = np.sqrt(crop_scale)  # ~0.9487
    crop_height = int(height * crop_ratio)
    crop_width = int(width * crop_ratio)

    # Calculate offsets for center crop
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2

    # Perform center crop
    cropped_image = TF.crop(image, top, left, crop_height, crop_width)

    # Resize to 224x224 (matching SimpleVLA-RL)
    result_image = TF.resize(
        cropped_image, [224, 224], interpolation=TF.InterpolationMode.BILINEAR
    )

    # Ensure RGB format
    result_image = result_image.convert("RGB")

    return result_image


def extract_simulator_config(config: Config):
    cfg = SimpleNamespace()
    cfg.task_suite_name = config.validation.dataset.subset
    cfg.max_steps = LIBERO_MAX_STEPS_MAP.get(cfg.task_suite_name, 512)
    cfg.num_envs = config.vla.num_envs
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

    def post_init_hook(self, **kwargs):
        self.model_type = self.config.vla.vla_type

        model_cls = ModelRegistry._MODEL_REGISTRY[self.model_type]
        if hasattr(model_cls, "preprocess_hf_config"):
            self.hf_config = model_cls.preprocess_hf_config(self.config)
        else:
            self.hf_config = util.retry(AutoConfig.from_pretrained)(
                self.config.policy.model_name_or_path
            )

        self.env_manager = EnvManager(
            cfg=extract_simulator_config(self.config),
            rank=torch.distributed.get_rank(),
            env_cls=LiberoEnvWrapper,
        )
        self.env_manager.start_simulator()

        # Initialize task timing accumulator
        self.accumulated_task_records = []
        self.rollout_call_count = 0
        self.last_dump_time = time.time()
        self.dump_dir = os.path.join(self.config.train.output_dir, "task_timing_logs")
        os.makedirs(self.dump_dir, exist_ok=True)

        # Configuration for periodic dumping
        self.dump_every_n_seconds = 120

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

        obs_keys = ["full_images", "wrist_images", "states"]
        sim_results = {k: None for k in obs_keys}
        sim_results["task_descriptions"] = [""] * self.num_envs

        vla_input_keys = ["input_ids", "attention_mask", "pixel_values"]
        vla_output_keys = ["responses", "action"]
        if not is_validation:
            vla_output_keys.append("old_log_probs")
        vla_history_keys = vla_input_keys + vla_output_keys

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
                "start_time": None,
                "end_time": None,
                "env_id": None,
            }
            for key in vla_history_keys:
                task_records[i][key] = []

        active_env_ids = []
        while finished_payloads < len(payload_indices):
            # Step 1: Advance active environments
            if active_env_ids:
                step_results = self.env_manager.chunk_step(active_env_ids, actions)
                active_indices, finished_env_ids = [], []
                for i, env_id in enumerate(active_env_ids):
                    if step_results["active"][i]:
                        active_indices.append(i)
                    else:
                        finished_env_ids.append(env_id)
                        payload_idx = payload_env_mapping[env_id]
                        task_records[payload_idx]["complete"] = step_results[
                            "complete"
                        ][i]
                        task_records[payload_idx]["active"] = step_results["active"][i]
                        task_records[payload_idx]["finish_step"] = step_results[
                            "finish_step"
                        ][i]
                        task_records[payload_idx]["end_time"] = time.time()
                        payload_env_mapping[env_id] = -1
                active_env_ids = [active_env_ids[i] for i in active_indices]
                finished_payloads += len(finished_env_ids)
                for key in obs_keys:
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
                payload_env_mapping[env_id] = payload_idx
                enqueue_payload_list.append(payload)
                # Record start time and env_id for this task
                task_records[payload_idx]["start_time"] = time.time()
                task_records[payload_idx]["env_id"] = env_id
                enqueued_payloads += 1

            logger.debug(
                f"payload_env_mapping: {payload_env_mapping}, "
                f"finished_payloads: {finished_payloads}/{len(payload_indices)}, "
                f"enqueued_payloads: {enqueued_payloads}/{len(payload_indices)}."
            )
            if available_env_ids:
                init_results = self._setup_parallel_envs(
                    enqueue_payload_list, available_env_ids, is_validation
                )
                for key in obs_keys:
                    if sim_results[key] is None:
                        sim_results[key] = init_results[key].copy()
                    else:
                        sim_results[key][available_env_ids] = init_results[key].copy()
                for i, env_id in enumerate(available_env_ids):
                    sim_results["task_descriptions"][env_id] = init_results[
                        "task_descriptions"
                    ][i]

            # Step 3: Generate VLA output
            vla_input = self._process_input(sim_results)
            vla_output = self._generate_one_step_oft(vla_input)
            active_env_ids = [
                i for i, pidx in enumerate(payload_env_mapping) if pidx != -1
            ]
            for env_id in active_env_ids:
                payload_idx = payload_env_mapping[env_id]
                for key in vla_input_keys:
                    task_records[payload_idx][key].append(vla_input[key][env_id])
                for key in vla_output_keys:
                    task_records[payload_idx][key].append(vla_output[key][env_id])
            actions = vla_output["action"][active_env_ids]

        return task_records

    def _extract_timing_info(self, task_records: List[Dict], rank: int) -> List[Dict]:
        """
        Extract timing and environment information from task records.
        Converts to Chrome tracing format for visualization in chrome://tracing.

        Args:
            task_records: List of task record dictionaries
            rank: Rank ID to use as process ID in tracing

        Returns:
            List of Chrome tracing event dictionaries (JSON serializable)
        """
        tracing_events = []

        for idx, task in enumerate(task_records):
            start_time = task.get("start_time")
            end_time = task.get("end_time")

            if start_time is None or end_time is None:
                continue

            # Convert to microseconds (Chrome tracing format requirement)
            ts_us = int(start_time * 1_000_000)
            dur_us = int((end_time - start_time) * 1_000_000)

            # Convert numpy types to native Python types for JSON serialization
            task_id = int(task.get("task_id", -1))
            trial_id = int(task.get("trial_id", -1))
            env_id = int(task.get("env_id", -1))
            complete = bool(task.get("complete", False))
            finish_step = int(task.get("finish_step", -1))

            task_name = f"Task_{task_id}.{trial_id}"

            # Chrome tracing duration event (ph: "X")
            # Use rank as process ID (pid) and env_id as thread ID (tid)
            # Color: green for complete, red for incomplete
            event = {
                "name": task_name,
                "cat": "task_execution",
                "ph": "X",  # Complete event (duration)
                "ts": ts_us,
                "dur": dur_us,
                "pid": rank,  # Rank as process ID
                "tid": env_id,  # Environment as thread ID within the rank
                "cname": "good"
                if complete
                else "bad",  # Green for complete, red for incomplete
                "args": {
                    "rank": rank,
                    "task_id": task_id,
                    "trial_id": trial_id,
                    "task_suite_name": str(task.get("task_suite_name", "")),
                    "env_id": env_id,
                    "complete": complete,
                    "finish_step": finish_step,
                    "status": "success" if complete else "failed",
                },
            }
            tracing_events.append(event)

        return tracing_events

    def _should_dump_timing_data(self) -> bool:
        """
        Check if timing data should be dumped based on count or time threshold.

        Returns:
            True if data should be dumped, False otherwise
        """
        # Check time-based trigger
        time_since_last_dump = time.time() - self.last_dump_time
        if (
            self.dump_every_n_seconds > 0
            and time_since_last_dump >= self.dump_every_n_seconds
        ):
            return True

        return False

    def _dump_timing_data(self):
        """
        Dump accumulated task timing data in Chrome tracing format.
        Gathers data from all ranks and merges into a single trace file.
        Can be visualized in chrome://tracing or https://ui.perfetto.dev/
        """
        if not self.accumulated_task_records:
            logger.debug("No timing data to dump")
            return

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

        # Prepare metadata for this rank
        local_metadata = {
            "rank": int(rank),
            "num_tasks": len(self.accumulated_task_records),
            "rollout_calls": int(self.rollout_call_count),
        }

        # Gather data from all ranks to rank 0
        if world_size > 1:
            # Gather all events to rank 0
            all_events = [None] * world_size
            all_metadata = [None] * world_size

            torch.distributed.gather_object(
                self.accumulated_task_records, all_events if rank == 0 else None, dst=0
            )

            torch.distributed.gather_object(
                local_metadata, all_metadata if rank == 0 else None, dst=0
            )

            if rank == 0:
                # Merge all events from all ranks
                merged_events = []
                for rank_events in all_events:
                    if rank_events:
                        merged_events.extend(rank_events)

                # Create combined metadata
                total_tasks = sum(m["num_tasks"] for m in all_metadata if m)
                combined_metadata = {
                    "dump_timestamp": datetime.now().isoformat(),
                    "world_size": world_size,
                    "total_tasks": total_tasks,
                    "per_rank_info": all_metadata,
                }
            else:
                # Non-rank-0 processes just clear and return
                self.accumulated_task_records = []
                self.last_dump_time = time.time()
                return
        else:
            # Single rank case
            merged_events = self.accumulated_task_records
            combined_metadata = {
                "dump_timestamp": datetime.now().isoformat(),
                "world_size": 1,
                "total_tasks": len(self.accumulated_task_records),
                "per_rank_info": [local_metadata],
            }

        # Only rank 0 writes the combined trace file
        if rank == 0:
            tracing_path = os.path.join(
                self.dump_dir, f"trace_all_ranks_{timestamp_str}.json"
            )

            # Add process and thread name metadata events for better visualization
            metadata_events = []

            # Add process names for each rank
            for r in range(world_size):
                metadata_events.append(
                    {
                        "name": "process_name",
                        "ph": "M",  # Metadata event
                        "pid": r,
                        "args": {"name": f"Rank {r}"},
                    }
                )

            # Add thread names for each environment ID
            # Extract unique (pid, tid) pairs from events
            seen_threads = set()
            for event in merged_events:
                thread_key = (event["pid"], event["tid"])
                if thread_key not in seen_threads:
                    seen_threads.add(thread_key)
                    metadata_events.append(
                        {
                            "name": "thread_name",
                            "ph": "M",  # Metadata event
                            "pid": event["pid"],
                            "tid": event["tid"],
                            "args": {"name": f"Env {event['tid']}"},
                        }
                    )

            # Chrome tracing format requires a specific structure
            tracing_data = {
                "traceEvents": metadata_events + merged_events,
                "displayTimeUnit": "ms",
                "metadata": combined_metadata,
            }

            with open(tracing_path, "w") as f:
                json.dump(tracing_data, f, indent=2)

            logger.info(
                f"Dumped {len(merged_events)} task traces from {world_size} rank(s) to: {tracing_path}"
            )
            logger.info("  View in Chrome: chrome://tracing (Load button)")
            logger.info("  Or online: https://ui.perfetto.dev/ (Open trace file)")
            logger.info("  Visualization: Each rank is a process, each env is a thread")

        # Clear accumulated data after dumping
        self.accumulated_task_records = []
        self.last_dump_time = time.time()

    def flush_timing_data(self):
        """
        Force dump of any remaining timing data.
        Should be called at the end of training or on shutdown.
        """
        if self.accumulated_task_records:
            logger.info(
                f"Flushing remaining {len(self.accumulated_task_records)} task records"
            )
            self._dump_timing_data()
        else:
            logger.debug("No remaining timing data to flush")

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

        payload_indices = self._prepare_payload_list(payloads, is_validation)
        task_records = self._do_rollout(
            payloads,
            payload_indices,
            is_validation,
            self.config.vla.continuous,
        )

        # Accumulate timing information
        self.rollout_call_count += 1
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        timing_info = self._extract_timing_info(task_records, rank)
        self.accumulated_task_records.extend(timing_info)

        # Periodically dump timing data
        if self._should_dump_timing_data():
            self._dump_timing_data()

        results = self._pack_grpo_results(
            self.n_generation, payload_indices, task_records, is_validation
        )

        self.model._set_fsdp_reshard_after_forward(
            self.config.train.fsdp_reshard_after_forward
        )
        return results

    def _process_input(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Process inputs for VLA model (matching SimpleVLA-RL's process_input)

        Args:
            inputs: List of observation dictionaries
            task_descriptions: List of task description strings

        Returns:
            Processed batch data for VLA model
        """
        full_images = inputs["full_images"]
        wrist_images = inputs["wrist_images"]
        task_descriptions = inputs["task_descriptions"]

        vla_type = self.config.vla.vla_type

        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}

        batch_size = full_images.shape[0]
        for i in range(batch_size):
            full_image = obs_to_vla_input(full_images[i])
            full_image = Image.fromarray(full_image).convert("RGB")
            full_image = center_crop_image(full_image)
            desp = task_descriptions[i]

            prompt = f"In: What action should the robot take to {desp.lower()}?\nOut:"
            batch_feature = self.processor(prompt, full_image)
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature.get(
                "attention_mask", torch.ones_like(input_ids)
            )
            pixel_values = batch_feature["pixel_values"]

            if (
                hasattr(self.config, "use_wrist_camera")
                and self.config.use_wrist_camera
            ):
                wrist_image = obs_to_vla_input(wrist_images[i])
                wrist_image = Image.fromarray(wrist_images[i]).convert("RGB")
                wrist_image = center_crop_image(wrist_image)
                wrist_feature = self.processor(prompt, wrist_image)
                pixel_values = torch.cat(
                    [pixel_values, wrist_feature["pixel_values"]], dim=1
                )

            # Handle OpenVLA-OFT specific formatting
            if vla_type == "openvla-oft":
                # Add space token if needed (matching SimpleVLA-RL)
                space_token_id = 29871  # Space token for LLaMA-based models
                if not torch.all(input_ids[:, -1] == space_token_id):
                    input_ids = torch.cat(
                        (
                            input_ids,
                            torch.tensor(
                                [[space_token_id]],
                                dtype=input_ids.dtype,
                                device=input_ids.device,
                            ),
                        ),
                        dim=1,
                    )
                    attention_mask = torch.cat(
                        (
                            attention_mask,
                            torch.tensor(
                                [[True]],
                                dtype=attention_mask.dtype,
                                device=attention_mask.device,
                            ),
                        ),
                        dim=1,
                    )

            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)

        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vla_type == "openvla-oft":
            # OpenVLA-OFT specific batch processing
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [
                x.transpose(0, 1) for x in batchdata["attention_mask"]
            ]

            batchdata["input_ids"] = (
                pad_sequence(
                    batchdata["input_ids"],
                    batch_first=True,
                    padding_value=self.pad_token_id,
                )
                .squeeze(-1)
                .to(device)
            )
            batchdata["attention_mask"] = (
                pad_sequence(
                    batchdata["attention_mask"], batch_first=True, padding_value=0
                )
                .squeeze(-1)
                .to(device)
            )

            # Handle padding and sorting (matching SimpleVLA-RL)
            padding_mask = batchdata["input_ids"].ne(self.pad_token_id)
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int()
            sorted_indices = torch.argsort(
                padding_mask, dim=1, descending=True, stable=True
            )
            batchdata["input_ids"] = torch.gather(
                batchdata["input_ids"], 1, sorted_indices
            )
            batchdata["attention_mask"] = torch.gather(
                batchdata["attention_mask"], 1, sorted_indices
            )

            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"], dim=0).to(
                device
            )
        else:
            # Standard batch processing
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)
        return batchdata

    def _generate_one_step_oft(
        self, prompts: Dict[str, torch.Tensor], is_valid: bool = False
    ) -> Dict[str, Any]:
        """Generate one step for OpenVLA-OFT (matching SimpleVLA-RL)"""
        input_ids = prompts["input_ids"]
        attention_mask = prompts["attention_mask"]
        pixel_values = prompts["pixel_values"]
        proprio = prompts.get("proprio", None)

        # Generation parameters
        temperature = self.config.rollout.sampling_config.temperature

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Try to call the VLA model's generation method
            actions, responses, logprobs = self.model.model.generate_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
                proprio=proprio,
                attention_mask=attention_mask,
                padding_idx=self.pad_token_id,
                do_sample=not is_valid,
                unnorm_key=getattr(self.config, "unnorm_key", "libero_10_no_noops"),
                temperature=temperature,
            )

            actions = normalize_gripper_action(actions)
            actions = invert_gripper_action(actions)

            return {
                "action": actions,
                "responses": responses,
                "old_log_probs": logprobs,
            }

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
                f"Rollout {n_payloads}x{n_generation} avg success rate: {avg_success_rate:.2f}%, success rates: [{formatted_rates}]"
            )

        pack_keys = ["pixel_values", "responses"]
        if not is_validation:
            pack_keys.append("old_log_probs")

        def pack_trajectory(payload_idx: int):
            start_idx = payload_idx * n_generation
            completions = []
            for i in range(n_generation):
                traj = {
                    "input_ids": pad_sequence(
                        task_records[start_idx + i]["input_ids"],
                        batch_first=True,
                        padding_value=self.pad_token_id,
                    ),
                    "attention_mask": pad_sequence(
                        task_records[start_idx + i]["attention_mask"],
                        batch_first=True,
                        padding_value=0,
                    ),
                }
                for key in pack_keys:
                    try:
                        traj[key] = torch.stack(task_records[start_idx + i][key], dim=0)
                    except Exception as e:
                        logger.error(
                            f"Error packing trajectory {start_idx + i} {key}: {e}"
                        )
                        exit(0)

                trajectory_id = save_trajectory_to_buffer(
                    traj,
                    buffer_dir=os.path.join(
                        self.config.train.output_dir, "replay_buffer"
                    ),
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
