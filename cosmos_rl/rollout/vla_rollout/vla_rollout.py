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

from collections import defaultdict
import os
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Any
from multiprocessing import Process, Queue
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
from cosmos_rl.rollout.vla_rollout.libero_utils import (
    obs_to_vla_input,
    save_rollout_video,
)
from cosmos_rl.rollout.vla_rollout.env_worker import (
    libero_env_worker,
    robotwin_env_worker,
)
from cosmos_rl.policy.model.vla.openvla_oft.constants import (
    NUM_ACTIONS_CHUNK,
)
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

        # rollout will not reshard after forward pass to avoid repeated all-gathers
        self.config.train.fsdp_reshard_after_forward = "never"

        self.sim_processes = []
        self.sim_input_queues = []
        self.sim_output_queues = []

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

            self.model = ModelRegistry.build_model(self.config)

            pfn, _ = self.model.parallelize_fn
            pfn(self.model, self.parallel_dims, self.config)

            self.model.eval()

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
        if is_validation:
            return self._rollout_validation(
                payloads, stream, data_packer, data_fetcher, **kwargs
            )

        return self._rollout_collection(
            payloads, stream, data_packer, data_fetcher, **kwargs
        )

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

    def _setup_parallel_envs(
        self, payloads: List[RLPayload], gen_indices: List[int], is_validation: bool
    ):
        """
        Setup parallel environment workers

        Args:
            payloads: List of RLPayload objects containing task information
            gen_indices: List of generation indices (for varying random seeds)
            is_validation: Whether to save validation videos

        Returns:
            Dict with initial_data containing: task_descriptions, inputs, task_records, valid_video

        Side Effects:
            Populates self.sim_processes, self.sim_input_queues, self.sim_output_queues
        """
        batch_size = len(payloads)

        # Extract task information from payloads and construct env configs
        task_suite_names = []
        task_ids = []
        trial_ids = []
        max_steps_list = []

        for payload in payloads:
            task_suite_names.append(payload.prompt.get("task_suite_name"))
            task_ids.append(payload.prompt.get("task_id", 0))
            trial_ids.append(payload.prompt.get("trial_id", 0))
            max_steps_list.append(
                MAX_STEPS_MAP.get(payload.prompt.get("task_suite_name"), 512)
            )

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if len(cuda_visible_devices.split(",")) >= 8:
            os.environ["CUDA_VISIBLE_DEVICES"] = f"{torch.distributed.get_rank()}"
        # logger.info(f"Setting CUDA_VISIBLE_DEVICES to {cuda_visible_devices}")
        # Spawn worker processes for each environment (stored in member variables/[0].content
        for idx in range(batch_size):
            task_name = task_suite_names[idx]
            t_id = task_ids[idx]
            tr_id = trial_ids[idx]
            max_steps = max_steps_list[idx]
            input_q = Queue()
            output_q = Queue()

            # Determine worker function based on task type
            if "libero" in task_name.lower():
                worker_fn = libero_env_worker
            elif "robotwin" in task_name.lower():
                worker_fn = robotwin_env_worker
            else:
                logger.warning(f"Unknown task type {task_name}, defaulting to LIBERO")
                worker_fn = libero_env_worker

            args = (task_name, t_id, tr_id, input_q, output_q, is_validation, max_steps)
            p = Process(target=worker_fn, args=args)
            p.start()
            self.sim_processes.append(p)
            self.sim_input_queues.append(input_q)
            self.sim_output_queues.append(output_q)

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        logger.debug(
            f"Spawned {len(self.sim_processes)} worker processes (total sim pool: {len(self.sim_processes)})"
        )

        # Collect initial observations from workers
        task_descriptions = []
        inputs = []
        task_records = []
        valid_video = defaultdict(list)

        for idx in range(batch_size):
            init_data = self.sim_output_queues[idx].get(timeout=120)
            assert (
                init_data["type"] == "init"
            ), f"Expected 'init', got '{init_data['type']}'"

            task_descriptions.append(init_data["task_description"])
            inputs.append(
                obs_to_vla_input(
                    init_data["obs"],
                    is_robotwin="robotwin" in task_suite_names[idx].lower(),
                )
            )
            task_records.append(
                {
                    "active": init_data["active"],
                    "complete": init_data["complete"],
                    "finish_step": init_data["finish_step"],
                    "task_file_name": init_data["task_file_name"],
                    "task_id": task_ids[idx],
                    "trial_id": trial_ids[idx],
                    "gen_idx": gen_indices[idx],
                    "task_suite_name": task_suite_names[idx],
                }
            )

            # Collect initial video frames
            if is_validation:
                valid_video[init_data["task_file_name"]].extend(
                    init_data["valid_images"]
                )

        initial_data = {
            "task_descriptions": task_descriptions,
            "inputs": inputs,
            "task_records": task_records,
            "valid_video": valid_video,
        }

        return initial_data

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
        self, inputs: List[Dict], task_descriptions: List[str]
    ) -> Dict[str, torch.Tensor]:
        """
        Process inputs for VLA model (matching SimpleVLA-RL's process_input)

        Args:
            inputs: List of observation dictionaries
            task_descriptions: List of task description strings

        Returns:
            Processed batch data for VLA model
        """
        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}

        vla_type = self.config.vla.vla_type
        use_proprio = self.config.vla.use_proprio

        if use_proprio:
            batchdata["proprio"] = []

        for i in range(len(inputs)):
            input_data = inputs[i]
            task_description = task_descriptions[i]

            # Process main image
            if "full_image" in input_data:
                image_array = input_data["full_image"]
            elif "agentview_image" in input_data:
                image_array = input_data["agentview_image"]
            else:
                raise RuntimeError(
                    f"No image found in input_data, expected full_image or agentview_image, got {input_data.keys()}"
                )

            image = Image.fromarray(image_array).convert("RGB")

            # Center crop if configured
            image = center_crop_image(image)

            # Create prompt (matching SimpleVLA-RL format)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature = self.processor(prompt, image)
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature.get(
                "attention_mask", torch.ones_like(input_ids)
            )
            pixel_values = batch_feature["pixel_values"]

            # Handle multi-view images (wrist cameras, etc.)
            pixel_values_list = [pixel_values]

            # Process additional camera views
            if (
                hasattr(self.config, "use_wrist_camera")
                and self.config.use_wrist_camera
            ):
                if "wrist_image" in input_data:
                    wrist_image = Image.fromarray(input_data["wrist_image"]).convert(
                        "RGB"
                    )
                    if hasattr(self.config, "center_crop") and self.config.center_crop:
                        wrist_image = center_crop_image(wrist_image)

                    try:
                        wrist_feature = self.processor(
                            "", wrist_image
                        )  # Empty prompt for additional views
                        pixel_values_list.append(wrist_feature["pixel_values"])
                    except Exception:
                        pass  # Skip if processing fails

            # Concatenate pixel values
            if len(pixel_values_list) > 1:
                pixel_values = torch.cat(pixel_values_list, dim=1)
            else:
                pixel_values = pixel_values_list[0]

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

            # Process proprioception for robotwin
            if use_proprio and "state" in input_data:
                norm_stats = self.hf_config.norm_stats.get("proprio", {})
                proprio = input_data["state"]
                proprio = normalize_proprio(proprio, norm_stats)
                batchdata["proprio"].append(torch.from_numpy(proprio))

        # Device placement
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if vla_type == "openvla-oft":
            # OpenVLA-OFT specific batch processing
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [
                x.transpose(0, 1) for x in batchdata["attention_mask"]
            ]

            pad_token_id = getattr(self.processor.tokenizer, "pad_token_id", 0)
            batchdata["input_ids"] = (
                pad_sequence(
                    batchdata["input_ids"], batch_first=True, padding_value=pad_token_id
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
            padding_mask = batchdata["input_ids"].ne(pad_token_id)
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

            if use_proprio:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(
                    device
                )
        else:
            # Standard batch processing
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

            if use_proprio:
                batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(
                    device
                )
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
            actions, responses = self.model.model.generate_action(
                input_ids=input_ids,
                pixel_values=pixel_values,
                proprio=proprio,
                attention_mask=attention_mask,
                padding_idx=getattr(self.processor.tokenizer, "pad_token_id", 0),
                do_sample=not is_valid,
                unnorm_key=getattr(self.config, "unnorm_key", "libero_10_no_noops"),
                temperature=temperature,
            )

            # Convert actions to numpy if needed (might already be numpy from _unnormalize_actions)
            if isinstance(actions, torch.Tensor):
                actions_np = actions.cpu().numpy()
            else:
                actions_np = actions

            return {
                "action": actions_np,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
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
        # Check GRPO filtering criteria first (before expensive log prob computation)
        num_chunks = len(vla_history)
        trajectories = [{} for _ in range(group_size)]
        for episode_idx in range(group_size):
            for key in ["input_ids", "attention_mask", "pixel_values", "responses"]:
                trajectories[episode_idx][key] = torch.stack(
                    [
                        vla_history[step_idx][key][episode_idx]
                        for step_idx in range(num_chunks)
                    ],
                    dim=0,
                )
            # for k, v in trajectories[episode_idx].items():
            #     logger.info(f"episode {episode_idx} steps {task_records[episode_idx]['finish_step']} trajectories: {k} {v.shape}")

        # #load saved batch from /root/SimpleVLA-RL/saved_training_batches
        # batch_dir = '/root/SimpleVLA-RL/saved_training_batches'
        # from cosmos_rl.utils.saved_batch_loader import SavedBatchLoader, SavedBatchIterator
        # loader = SavedBatchLoader(batch_dir=batch_dir, episodes_per_step=128, device='cpu')
        # iterator = SavedBatchIterator(loader)
        # policy_inputs_all, advantages_all, meta_info = next(iterator)
        # saved_episode = policy_inputs_all[0]
        #
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # stacked_input_ids = torch.stack([saved_episode.per_step_data[i]['input_ids'] for i in range(len(saved_episode.per_step_data))]).to(device)
        # stacked_attention_mask = torch.stack([saved_episode.per_step_data[i]['attention_mask'] for i in range(len(saved_episode.per_step_data))]).to(device)
        # stacked_pixel_values = torch.stack([saved_episode.pixel_values[i] for i in range(len(saved_episode.pixel_values))]).to(device)
        # stacked_responses = torch.stack([saved_episode.per_step_data[i]['responses'] for i in range(len(saved_episode.per_step_data))]).to(device)
        # stacked_old_log_prob = torch.stack([saved_episode.per_step_data[i]['old_log_prob'] for i in range(len(saved_episode.per_step_data))]).to(device)
        #
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        # with torch.no_grad():
        # outputs = self.vla_model.forward_with_trajectory_structure(
        # input_ids=stacked_input_ids,
        # pixel_values=stacked_pixel_values,
        # attention_mask=stacked_attention_mask,
        # labels=stacked_responses,
        # temperature=1.6,
        # proprio=None
        # )
        # rollout_logits = outputs.logits
        # rollout_log_probs = outputs.logprobs.squeeze(0)
        # diff = (rollout_log_probs - stacked_old_log_prob).abs()
        # if torch.distributed.get_rank() == 0:
        # logger.info(f"saved_log_probs {stacked_old_log_prob.shape}, saved_log_probs {stacked_old_log_prob}")
        # logger.info(f"  Absolute differences:")
        # logger.info(f"    Mean: {diff.mean().item():.6f}")
        # logger.info(f"    Max: {diff.max().item():.6f}")
        # logger.info(f"    Min: {diff.min().item():.6f}")
        # logger.info(f"    Std: {diff.std().item():.6f}")
        # logger.info(f"    diff[0:10]: {diff[0:10]}")

        # Compute old_log_probs for each episode by replaying trajectory
        completions = []
        with (
            torch.no_grad(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            for episode_idx in range(group_size):
                traj = trajectories[episode_idx]
                traj["old_log_probs"] = self.model.forward_with_trajectory_structure(
                    input_ids=traj["input_ids"],
                    pixel_values=traj["pixel_values"],
                    attention_mask=traj["attention_mask"],
                    labels=traj["responses"],
                    temperature=self.config.rollout.sampling_config.temperature,
                    proprio=None,
                ).logprobs

                trajectory_id = save_trajectory_to_buffer(
                    traj,
                    buffer_dir=os.path.join(
                        self.config.train.output_dir, "replay_buffer"
                    ),
                )
                # {'active': False, 'complete': True, 'finish_step': 164, 'task_file_name': 'libero_10_task_5_trial_13', 'task_id': 5, 'trial_id': 13, 'gen_idx': 0, 'task_suite_name': 'libero_10'}
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
        # Extract task info from payloads (already available, no need to pass through initial_data)
        task_suite_names = []
        task_ids = []
        trial_ids = []
        max_steps = -1
        for payload in payloads:
            task_suite_names.append(payload.prompt.get("task_suite_name"))
            task_ids.append(payload.prompt.get("task_id", 0))
            trial_ids.append(payload.prompt.get("trial_id", 0))
            max_steps = max(
                max_steps, MAX_STEPS_MAP.get(payload.prompt.get("task_suite_name"), 512)
            )
        gen_indices = (
            [0] * len(payloads) if is_validation else [i for i in range(len(payloads))]
        )

        # CRITICAL: Release GPU resources before spawning sim workers
        # After NCCL broadcast, GPU SM resources may still be occupied by:
        # - Lingering CUDA kernels from weight sync
        # - VLA model occupying GPU memory
        # - CUDA context fragmentation
        # Without this, sim workers timeout trying to initialize rendering contexts
        torch.cuda.synchronize()  # Wait for all CUDA operations to complete
        torch.cuda.empty_cache()  # Free up cached GPU memory
        logger.debug(
            f"Released GPU resources before spawning {len(payloads)} sim workers"
        )

        # Setup parallel environments (populates self.sim_processes, self.sim_input_queues, self.sim_output_queues)
        initial_data = self._setup_parallel_envs(payloads, gen_indices, is_validation)

        # Unpack initial data
        task_descriptions = initial_data["task_descriptions"]
        inputs = initial_data["inputs"]
        task_records = initial_data["task_records"]
        valid_video = initial_data["valid_video"]

        # Use member variables for process/queue management
        batch_size = len(self.sim_processes)

        # Episode execution loop
        step = 0
        vla_history = []

        while step < max_steps:
            # Find active environments
            active_indices = [i for i, r in enumerate(task_records) if r["active"]]
            if not active_indices:
                break

            # VLA model inference on all inputs
            current_inputs = inputs
            current_task_descriptions = task_descriptions

            vla_input = self._process_input(current_inputs, current_task_descriptions)
            vla_output = self._generate_one_step_oft(vla_input, is_validation)

            # if task_ids[0] == 0 and trial_ids[0] == 0 and gen_indices[0] == 0:
            #     # logger.info(f"task_suite_name: {task_suite_names[0]}, task_id: {task_ids[0]}, trial_id: {trial_ids[0]}, gen_idx: {gen_indices[0]}")
            #     # logger.info(f"current_inputs.full_image {current_inputs[0]['full_image'].shape}, {current_inputs[0]['full_image']}")
            #     # logger.info(f"current_task_descriptions {current_task_descriptions}")
            #     for k, v in vla_input.items():
            #         logger.info(f"vla_input {k} {v.shape}")
            #     for k, v in vla_output.items():
            #         logger.info(f"vla_output {k} {v.shape}")

            step_data = {
                "input_ids": vla_input["input_ids"],
                "attention_mask": vla_input["attention_mask"],
                "pixel_values": vla_input["pixel_values"],
                "responses": vla_output["responses"],
                "action": vla_output["action"],
            }

            vla_history.append(step_data)

            # Send actions to active workers
            for idx in active_indices:
                self.sim_input_queues[idx].put(step_data["action"][idx])

            # Collect results from active workers
            new_inputs = inputs.copy()
            for idx in active_indices:
                result = self.sim_output_queues[idx].get(timeout=30)
                assert (
                    result["type"] == "step"
                ), f"Expected 'step', got '{result['type']}'"

                new_inputs[idx] = obs_to_vla_input(
                    result["obs"],
                    is_robotwin="robotwin" in task_suite_names[idx].lower(),
                )
                task_records[idx]["active"] = result["active"]
                task_records[idx]["complete"] = result["complete"]
                task_records[idx]["finish_step"] = result["finish_step"]

                # Collect video frames
                if is_validation and len(result["valid_images"]) > 0:
                    valid_video[task_records[idx]["task_file_name"]].extend(
                        result["valid_images"]
                    )

                if not result["active"]:
                    status = "✅ SUCCESS" if result["complete"] else "❌ FAILED"
                    logger.debug(
                        f"Task {idx} [task_id={task_ids[idx]}, trial_id={trial_ids[idx]}, gen={gen_indices[idx]}]: {status} (steps={result['finish_step']})"
                    )

            inputs = new_inputs
            step += NUM_ACTIONS_CHUNK

        # Save rollout videos
        if valid_video:
            rollout_dir = os.path.join(self.config.train.output_dir, "vla_rollouts")

            for task_file, images in valid_video.items():
                if len(images) > 0:
                    complete = any(
                        r["complete"]
                        for r in task_records
                        if r["task_file_name"] == task_file
                    )

                    try:
                        save_rollout_video(images, rollout_dir, task_file, 0, complete)
                    except Exception as e:
                        logger.warning(f"  ⚠️  Failed to save {task_file}: {e}")
                        import traceback

                        traceback.print_exc()

        self._destroy_parallel_envs()

        return self._pack_grpo_results(
            vla_history, task_records, batch_size, is_validation
        )
