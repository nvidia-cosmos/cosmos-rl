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
)
from cosmos_rl.simulators.env_manager import EnvManager
from cosmos_rl.simulators.libero.env_wrapper import LiberoEnvWrapper
from cosmos_rl.utils.replay_buffer import save_trajectory_to_buffer

MAX_STEPS_MAP = {
    # LIBERO tasks
    "libero_spatial": 512,
    "libero_object": 512,
    "libero_goal": 512,
    "libero_10": 512,
    "libero_90": 512,
    # RoboTwin 2.0 tasks
    "robotwin2_click_bell": 200,
    "robotwin2_move_can_pot": 200,
    "robotwin2_place_phone_stand": 200,
    "robotwin2_place_a2b_left": 200,
    "robotwin2_place_a2b_right": 200,
    "robotwin2_handover_mic": 200,
    "robotwin2_pick_dual_bottles": 100,
    "robotwin2_lift_pot": 200,
    "robotwin2_put_bottles_dustbin": 800,
    "robotwin2_stack_blocks_two": 400,
    "robotwin2_stack_bowls_two": 400,
    "robotwin2_handover_block": 400,
    "robotwin2_place_empty_cup": 200,
    "robotwin2_shake_bottle": 75,
    "robotwin2_move_stapler_pad": 200,
    "robotwin2_place_container_plate": 150,
    "robotwin2_blocks_ranking_rgb": 600,
    "robotwin2_beat_block_hammer": 200,
    "robotwin2_place_mouse_pad": 200,
    "robotwin2_place_shoe": 250,
    "robotwin2_move_pillbottle_pad": 200,
}


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
    cfg.max_steps = MAX_STEPS_MAP.get(cfg.task_suite_name, 512)
    cfg.num_envs = config.rollout.n_generation
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
        if not self._engine_initialized:
            model_path = self.config.policy.model_name_or_path

            if self.model_type == "openvla-oft":
                from cosmos_rl.policy.model.vla.openvla_oft.processing_prismatic import (
                    PrismaticProcessor,
                )
            elif self.model_type == "openvla":
                from cosmos_rl.policy.model.vla.openvla.processing_prismatic import (
                    PrismaticProcessor,
                )
            else:
                raise ValueError(f"Unsupported vla model type: {self.model_type}")

            self.processor = PrismaticProcessor.from_pretrained(
                model_path, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            self.pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

            self.model = ModelRegistry.build_model(self.config)

            pfn, _ = self.model.parallelize_fn
            pfn(self.model, self.parallel_dims, self.config)

            self.model.eval()

            if self.config.mode != "colocated":
                self.model.load_hf_weights(
                    self.config.policy.model_name_or_path,
                    self.parallel_dims,
                    torch.device("cuda"),
                )
            self._engine_initialized = True
            logger.info("[Rollout] Engine initialized.")

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

        if is_validation:
            results = self._rollout_validation(
                payloads, stream, data_packer, data_fetcher, **kwargs
            )
        else:
            results = self._rollout_collection(
                payloads, stream, data_packer, data_fetcher, **kwargs
            )

        self.model._set_fsdp_reshard_after_forward(
            self.config.train.fsdp_reshard_after_forward
        )

        return results

    def _rollout_collection(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        **kwargs,
    ):
        n_generation = max(1, self.config.rollout.n_generation)
        num_batches = len(payloads)
        collected_results = []
        for i in range(num_batches):
            try:
                group_results = self._generate_minibatch(
                    payloads=[payloads[i]] * n_generation,
                    is_validation=False,
                )
                collected_results.extend(group_results)
            except Exception as e:
                logger.warning(
                    f"[Rollout Collection] Prompt failed likely due to sim timeout, prompt dropped...: {e}"
                )
                self._destroy_parallel_envs()
                collected_results.append(RolloutResult(completions=[]))
        return collected_results

    def _rollout_validation(
        self,
        payloads: List[RLPayload],
        stream: torch.cuda.Stream,
        data_packer: BaseDataPacker,
        data_fetcher: DataFetcherBase,
        **kwargs,
    ):
        batch_size = max(1, self.config.validation.batch_size)
        num_batches = (len(payloads) + batch_size - 1) // batch_size
        valid_results = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(payloads))
            payload_batch = payloads[start_idx:end_idx]
            while True:
                try:
                    group_results = self._generate_minibatch(
                        payloads=payload_batch,
                        is_validation=True,
                    )
                    break
                except Exception as e:
                    logger.warning(
                        f"[Rollout Validation] Batch failed likely due to sim timeout, retrying...: {e}"
                    )
                    self._destroy_parallel_envs()
                    continue
            valid_results.extend(group_results)
        return valid_results

    def _setup_parallel_envs(self, payloads: List[RLPayload], is_validation: bool):
        env_ids = list(range(len(payloads)))
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

    def _destroy_parallel_envs(self):
        """
        Cleanly destroy parallel environment workers and reset simulation pool

        Side Effects:
            - Terminates processes in self.sim_processes
            - Resets all self.sim_* member variables to empty
        """
        if not self.sim_processes:
            return  # Nothing to clean up

        logger.debug(f"Destroying {len(self.sim_processes)} simulation processes...")

        # Send termination signal to all workers
        for q in self.sim_input_queues:
            try:
                q.put(None, timeout=1)
            except Exception:
                pass  # Ignore errors, process termination will handle stuck workers

        # Wait for processes to finish, terminate if hung
        for p in self.sim_processes:
            p.join(timeout=20)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)  # Wait again after terminate
                if p.is_alive():
                    p.kill()
                    p.join(timeout=2)  # Final wait after kill

        # Reset simulation pool state
        num_destroyed = len(self.sim_processes)
        self.sim_processes = []
        self.sim_input_queues = []
        self.sim_output_queues = []

        logger.debug(f"Destroyed {num_destroyed} simulation processes, pool reset")

    def _process_input(
        self, inputs: Dict[str, Any], task_descriptions: List[str]
    ) -> Dict[str, torch.Tensor]:
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
        vla_history: List[Dict],
        task_records: List[Dict],
        group_size: int,
        is_validation: bool,
    ):
        """
        Pack GRPO results: apply filtering, compute old_log_probs, create RolloutResults

        This function:
        1. Checks GRPO filtering criteria (if enabled)
        2. If group is invalid, returns None (skips expensive log prob computation)
        3. If group is valid, extracts trajectories and computes old_log_probs
        4. Creates and returns RolloutResult objects

        Args:
            vla_history: List of step_data dicts from _parallel_inference_and_sim
            task_records: List of task metadata dicts
            group_size: Number of episodes in the group (n_generation)
            enable_filtering: Whether to apply GRPO filtering

        Returns:
            List of RolloutResult objects if valid, None if filtered out
        """
        n_success = sum([task_records[i]["complete"] for i in range(group_size)])
        if not is_validation:
            task_id = task_records[0]["task_id"]
            trial_id = task_records[0]["trial_id"]
            logger.info(
                f"[Rollout] task_id: {task_id}, trial_id: {trial_id}, success rate: {n_success}/{group_size}"
            )
        else:
            logger.info(f"[Validation] success rate: {n_success}/{group_size}")

        trajectories = [
            {
                "input_ids": pad_sequence(
                    vla_history[i]["input_ids"],
                    batch_first=True,
                    padding_value=self.pad_token_id,
                ),
                "attention_mask": pad_sequence(
                    vla_history[i]["attention_mask"],
                    batch_first=True,
                    padding_value=0,
                ),
            }
            for i in range(group_size)
        ]
        pack_keys = ["pixel_values", "responses"] + (
            ["old_log_probs"] if not is_validation else []
        )

        for episode_idx in range(group_size):
            for key in pack_keys:
                trajectories[episode_idx][key] = torch.stack(
                    vla_history[episode_idx][key], dim=0
                )

        # Compute old_log_probs for each episode by replaying trajectory
        completions = []
        for episode_idx in range(group_size):
            traj = trajectories[episode_idx]
            trajectory_id = save_trajectory_to_buffer(
                traj,
                buffer_dir=os.path.join(self.config.train.output_dir, "replay_buffer"),
            )
            completions.append(
                {
                    "complete": bool(task_records[episode_idx]["complete"]),
                    "finish_step": int(task_records[episode_idx]["finish_step"]),
                    "trajectory_id": trajectory_id,
                }
            )
        if is_validation:
            return [RolloutResult(completions=[c]) for c in completions]
        else:
            return [RolloutResult(completions=completions)]

    @torch.no_grad()
    def _generate_minibatch(
        self, payloads: List[RLPayload], is_validation: bool = False
    ):
        """
        Run parallel VLA inference and simulation for a minibatch

        Uses separate processes for each environment to avoid shared OpenGL/MuJoCo state.

        Args:
            payloads: List of RLPayload objects containing task information
            is_validation: Whether to save validation videos

        Returns:
            Tuple of (vla_history, task_records)
        """
        env_ids = list(range(len(payloads)))
        gen_indices = (
            [0] * len(payloads) if is_validation else [i for i in range(len(payloads))]
        )

        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()  # Free up cached GPU memory
        logger.debug(
            f"Released GPU resources before spawning {len(payloads)} sim workers"
        )

        # Setup parallel environments (populates self.sim_processes, self.sim_input_queues, self.sim_output_queues)
        initial_data = self._setup_parallel_envs(payloads, is_validation)

        # Unpack initial data
        task_descriptions = initial_data["task_descriptions"]

        # Episode execution loop
        step = 0
        vla_history = []
        env_states = self.env_manager.get_env_states(env_ids)
        task_records = [{} for _ in range(len(payloads))]
        for i in range(len(payloads)):
            task_records[i] = {
                "task_id": payloads[i].prompt.get("task_id", 0),
                "trial_id": payloads[i].prompt.get("trial_id", 0),
                "gen_idx": gen_indices[i],
                "task_suite_name": payloads[i].prompt.get("task_suite_name", ""),
                "active": env_states[i].active,
                "complete": env_states[i].complete,
                "finish_step": env_states[i].step,
            }

        from cosmos_rl.policy.model.vla.openvla_oft.constants import NUM_ACTIONS_CHUNK

        vla_input_keys = ["input_ids", "attention_mask", "pixel_values"]
        vla_output_keys = ["responses", "action"]
        if not is_validation:
            vla_output_keys.append("old_log_probs")
        vla_history_keys = vla_input_keys + vla_output_keys
        vla_history = [
            {key: [] for key in vla_history_keys} for _ in range(len(payloads))
        ]

        active_indices = active_env_ids = env_ids
        current_inputs = {
            "full_images": initial_data["full_images"],
            "wrist_images": initial_data["wrist_images"],
            "states": initial_data["states"],
        }
        while True:
            if not active_indices:
                break
            vla_input = self._process_input(current_inputs, task_descriptions)
            vla_output = self._generate_one_step_oft(vla_input, is_validation)
            step_results = self.env_manager.chunk_step(
                active_env_ids, vla_output["action"]
            )
            for i, env_id in enumerate(active_env_ids):
                for key in vla_input_keys:
                    vla_history[env_id][key].append(vla_input[key][i])
                for key in vla_output_keys:
                    vla_history[env_id][key].append(vla_output[key][i])
                for key in ["active", "complete", "finish_step"]:
                    task_records[env_id][key] = step_results[key][i]

            # update active indices, prepare data for next chunk
            active_indices = [
                i
                for i, env_id in enumerate(active_env_ids)
                if task_records[env_id]["active"]
            ]
            active_env_ids = [active_env_ids[i] for i in active_indices]
            for key in ["full_images", "wrist_images", "states"]:
                current_inputs[key] = step_results[key][active_indices].copy()
            step += NUM_ACTIONS_CHUNK

        if is_validation and self.config.vla.save_video:
            rollout_dir = os.path.join(self.config.train.output_dir, "vla_rollouts")
            self.env_manager.save_validation_videos(rollout_dir, env_ids)

        return self._pack_grpo_results(
            vla_history, task_records, len(payloads), is_validation
        )
