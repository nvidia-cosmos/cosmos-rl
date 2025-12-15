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
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import Optional, List, Dict, Any
from PIL import Image
from transformers import AutoConfig, AutoTokenizer

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
from cosmos_rl.rollout.vla_rollout.trace_utils import create_tracing_manager

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img

def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])

def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


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

        self.obs_keys = ["full_images", "wrist_images", "states"]
        self.vla_input_keys = ["input_ids", "attention_mask", "pixel_values"]
        self.vla_output_keys = ["responses", "old_log_probs"]
        self.vla_train_keys = self.vla_input_keys + self.vla_output_keys

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

        self.env_manager = EnvManager(
            cfg=extract_simulator_config(self.config),
            rank=torch.distributed.get_rank(),
            env_cls=LiberoEnvWrapper,
        )
        self.env_manager.start_simulator()

        # Initialize tracing system (if enabled via config)
        trace_verbosity = getattr(self.config.vla, "trace_verbosity", 0)
        self.tracing_manager = create_tracing_manager(
            rank=torch.distributed.get_rank(),
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

            self._wait_init_results(sim_results, async_wait_envs)
            if available_env_ids:
                # self._get_init_results(sim_results, available_env_ids, enqueue_payload_list, is_validation)
                self._setup_parallel_envs_async(
                    enqueue_payload_list, available_env_ids, is_validation
                )
                for env_id in available_env_ids:
                    async_wait_envs[env_id] = 10

            active_env_ids = [
                i
                for i, pidx in enumerate(payload_env_mapping)
                if pidx != -1 and async_wait_envs[i] == -1
            ]
            logger.debug(
                f"payload_env_mapping: {payload_env_mapping}, "
                f"finished_payloads: {finished_payloads}/{len(payload_indices)}, "
                f"enqueued_payloads: {enqueued_payloads}/{len(payload_indices)}, "
                f"waiting_envs_left_steps: {async_wait_envs}, active_env_ids: {active_env_ids}"
            )
            if not active_env_ids:
                continue

            # if torch.distributed.get_rank() == 0:
            #     logger.info(sim_results)

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
                vla_input = self._process_input(active_sim_results)
                vla_output = self._generate_one_step_oft(vla_input)
            for i, env_id in enumerate(active_env_ids):
                task_idx = payload_env_mapping[env_id]
                for key in self.vla_input_keys:
                    task_records[task_idx][key].append(vla_input[key][i])
                for key in self.vla_output_keys:
                    task_records[task_idx][key].append(vla_output[key][i])

            actions = vla_output["action"]
            rollout_step += 1
        return task_records

    def _load_pi05_norm_stats(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Load PI05 norm_stats from canonical OpenPI path."""
        import json

        resolved = util.resolve_model_path(model_path)
        p = os.path.join(resolved, "assets", "physical-intelligence", "libero", "norm_stats.json")
        if not os.path.isfile(p):
            return None
        with open(p, "r") as f:
            raw = json.load(f)
        return raw['norm_stats']

    def _normalize_pi05_state(self, state: np.ndarray) -> np.ndarray:
        """official_openpi Normalize(use_quantiles=True): map state into [-1, 1] using q01/q99."""
        s_stats = self.norm_stats.get("state") or self.norm_stats.get("proprio")
        if not isinstance(s_stats, dict) or ("q01" not in s_stats) or ("q99" not in s_stats):
            raise ValueError("PI05 norm_stats must contain state.q01 and state.q99 (quantiles).")
        x = np.asarray(state, dtype=np.float32).reshape(-1)
        q01 = np.asarray(s_stats["q01"], dtype=np.float32).reshape(-1)[: x.shape[0]]
        q99 = np.asarray(s_stats["q99"], dtype=np.float32).reshape(-1)[: x.shape[0]]
        return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0

    def _unnormalize_pi05_actions(self, actions: np.ndarray) -> np.ndarray:
        """official_openpi Unnormalize(use_quantiles=True): map actions from [-1, 1] back using q01/q99."""
        a_stats = self.norm_stats.get("actions") or  self.norm_stats.get("action")
        x = np.asarray(actions, dtype=np.float32)
        dim = x.shape[-1]
        q01 = np.asarray(a_stats["q01"], dtype=np.float32).reshape(-1)
        q99 = np.asarray(a_stats["q99"], dtype=np.float32).reshape(-1)
        if q01.size < dim:
            # official_openpi: apply to first dim(q01), keep the rest unchanged
            y0 = (x[..., : q01.size] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            return np.concatenate([y0, x[..., q01.size :]], axis=-1)
        q01 = q01[:dim]
        q99 = q99[:dim]
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


    def _process_input_pi05(
        self, inputs: List[Dict], task_descriptions: List[str]
    ) -> Any:

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = len(inputs)

        def _to_pi05_img(arr: np.ndarray) -> torch.Tensor:
            # 1. Convert to PIL Image
            pil_img = np.ascontiguousarray(arr)
            # 2. Resize with padding (matching official_openpi)
            pil_img = convert_to_uint8(resize_with_pad(pil_img, 224, 224))
            # 3. Convert to torch.Tensor
            return torch.from_numpy(pil_img)  # [H, W, C], values in [0, 255]

        base_imgs = []
        wrist_imgs = []
        for inp in inputs:
            if "full_image" in inp:
                base_imgs.append(_to_pi05_img(inp["full_image"]))
            elif "agentview_image" in inp:
                base_imgs.append(_to_pi05_img(inp["agentview_image"]))
            else:
                raise RuntimeError(
                    f"No image found in input_data, expected full_image or agentview_image, got {inp.keys()}"
                )

            wrist_imgs.append(_to_pi05_img(inp["wrist_image"]))

        base_imgs_t = torch.stack(base_imgs, 0).to(device)  # [B, H, W, C]
        wrist_imgs_t = torch.stack(wrist_imgs, 0).to(device)

        images = {
            "base_0_rgb": base_imgs_t,
            "left_wrist_0_rgb": wrist_imgs_t,
            "right_wrist_0_rgb": torch.zeros_like(base_imgs_t),
        }
        image_masks = {
            "base_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "left_wrist_0_rgb": torch.ones(batch_size, dtype=torch.bool, device=device),
            "right_wrist_0_rgb": torch.zeros(batch_size, dtype=torch.bool, device=device),
        }


        state = torch.stack([
            torch.as_tensor(self._normalize_pi05_state(inp["state"]), dtype=torch.float32, device=device)
            for inp in inputs
        ], dim=0)
        
        tokenized_prompt = []
        tokenized_prompt_mask = []
        for i in range(batch_size):
            prompt = task_descriptions[i]
            # If discrete_state_input=False, pass None to tokenizer (state is used as continuous input)
            # If discrete_state_input=True, pass state to tokenizer (state is discretized into tokens)
            if self.hf_config.discrete_state_input:
                state_np = state[i].detach().float().cpu().numpy().astype(np.float32)
                tokens, mask = self.tokenizer.tokenize_openpi(prompt, state=state_np)
            else:
                tokens, mask = self.tokenizer.tokenize_openpi(prompt, state=None)
            
            tokenized_prompt.append(torch.from_numpy(tokens).to(torch.long))
            tokenized_prompt_mask.append(torch.from_numpy(mask).to(torch.bool))
        tokenized_prompt = torch.stack(tokenized_prompt, dim=0).to(device)
        tokenized_prompt_mask = torch.stack(tokenized_prompt_mask, dim=0).to(device)
        return SimpleNamespace(
            images=images,
            image_masks=image_masks,
            state=state,
            tokenized_prompt=tokenized_prompt,
            tokenized_prompt_mask=tokenized_prompt_mask,
        )

    def _generate_one_step_pi05(self, observation: Any, mode: str = "train"):
        """Generate one PI05 action chunk and adapt it to the LIBERO env_worker format.
        
        Args:
            observation: PI05 observation object
            mode: "train" or "eval"
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.model.sample_actions(self.device, observation, mode=mode)
            actions = outputs["actions"]

        actions = actions[:, :self.hf_config.action_horizon, :self.hf_config.action_env_dim].to(torch.float32)
        actions_np = actions.detach().cpu().numpy()
        outputs["action"] = self._unnormalize_pi05_actions(actions_np)
        # Clip actions to valid range (important for gripper which can slightly exceed [-1, 1])
        # outputs["action"] = np.clip(outputs["action"], -1.0, 1.0)

        return outputs

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
                for key in [
                    "input_ids",
                    "attention_mask",
                    "pixel_values",
                    "responses",
                    "old_log_probs",
                ]:
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
        max_steps = -1
        for payload in payloads:
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

        from cosmos_rl.policy.model.vla.openvla_oft.constants import NUM_ACTIONS_CHUNK

        while step < max_steps:
            # Find active environments
            active_indices = [i for i, r in enumerate(task_records) if r["active"]]
            if not active_indices:
                break

            # VLA model inference on all inputs
            current_inputs = inputs
            current_task_descriptions = task_descriptions

            if self.model_type in ("pi05", "pi0"):
                pi05_obs = self._process_input_pi05(current_inputs, current_task_descriptions)
                vla_output = self._generate_one_step_pi05(
                    pi05_obs, mode="eval" if is_validation else "train"
                )

                image_keys = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                step_data = {
                    "action": vla_output["action"],
                    "chains": vla_output["chains"],
                    "denoise_inds": vla_output["denoise_inds"],
                    "old_log_probs": vla_output["old_log_probs"],
                    "images": torch.stack([pi05_obs.images[k] for k in image_keys], dim=1),
                    "image_masks": torch.stack([pi05_obs.image_masks[k] for k in image_keys], dim=1),
                    "state": pi05_obs.state,
                    "tokenized_prompt": pi05_obs.tokenized_prompt,
                    "tokenized_prompt_mask": pi05_obs.tokenized_prompt_mask,
                }
            else:
                vla_input = self._process_input(current_inputs, current_task_descriptions)
                vla_output = self._generate_one_step_oft(vla_input, is_validation)
                step_data = {
                    "input_ids": vla_input["input_ids"],
                    "attention_mask": vla_input["attention_mask"],
                    "pixel_values": vla_input["pixel_values"],
                    "responses": vla_output["responses"],
                    "action": vla_output["action"],
                }

            # if task_ids[0] == 0 and trial_ids[0] == 0 and gen_indices[0] == 0:
            #     # logger.info(f"task_suite_name: {task_suite_names[0]}, task_id: {task_ids[0]}, trial_id: {trial_ids[0]}, gen_idx: {gen_indices[0]}")
            #     # logger.info(f"current_inputs.full_image {current_inputs[0]['full_image'].shape}, {current_inputs[0]['full_image']}")
            #     # logger.info(f"current_task_descriptions {current_task_descriptions}")
            #     for k, v in vla_input.items():
            #         logger.info(f"vla_input {k} {v.shape}")
            #     for k, v in vla_output.items():
            #         logger.info(f"vla_output {k} {v.shape}")

            vla_history.append(step_data)

            if self.model_type in ("pi05", "pi0"):
                replan_steps = self.config.custom["pi05"]["replan_steps"]
                assert replan_steps > 0 and replan_steps <= self.hf_config.action_horizon

            # Send actions to active workers
            for idx in active_indices:
                if self.model_type in ("pi05", "pi0"):
                    self.sim_input_queues[idx].put(step_data["action"][idx][:replan_steps])
                else:
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
            if self.model_type in ("pi05", "pi0"):
                step += replan_steps
            else:
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

        # PI05/PI0 path in grpo packing
        if self.model_type in ("pi05", "pi0"):
            if is_validation:
                # Eval mode: no training data needed
                completions = [
                    {
                        "complete": bool(task_records[i]["complete"]),
                        "finish_step": int(task_records[i]["finish_step"]),
                    }
                    for i in range(group_size)
                ]
                return [RolloutResult(completions=[c]) for c in completions]
            
            # Training mode: build trajectories for GRPO (RLinf-style chain replay)
            keys = (
                "chains",
                "denoise_inds",
                "old_log_probs",
                "images",
                "image_masks",
                "state",
                "tokenized_prompt",
                "tokenized_prompt_mask",
            )
            if not vla_history:
                raise RuntimeError("PI05 rollout produced empty vla_history in training mode.")
            missing = [k for k in keys if k not in vla_history[0]]
            if missing:
                raise KeyError(
                    f"PI05 rollout missing keys={missing} in step_data. Keys={list(vla_history[0].keys())}"
                )

            stacked = {k: torch.stack([s[k] for s in vla_history], 0) for k in keys}  # [T, B, ...]
            trajectories = [
                {k: stacked[k][:, i].detach().cpu() for k in keys}
                for i in range(group_size)
            ]

            buffer_dir = os.path.join(self.config.train.output_dir, "replay_buffer")
            completions = [
                {
                    "complete": bool(task_records[i]["complete"]),
                    "finish_step": int(task_records[i]["finish_step"]),
                    "trajectory_id": save_trajectory_to_buffer(
                        trajectories[i], buffer_dir=buffer_dir
                    ),
                }
                for i in range(group_size)
            ]
            return [RolloutResult(completions=completions)]

        return self._pack_grpo_results(
            vla_history, task_records, batch_size, is_validation
        )
