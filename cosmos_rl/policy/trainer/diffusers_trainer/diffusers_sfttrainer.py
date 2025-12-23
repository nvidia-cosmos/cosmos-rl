import os
import torch
import numpy as np

import torch.distributed as dist

from functools import partial

from typing import Optional


from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
from cosmos_rl.policy.config import (
    Config as CosmosConfig,
)
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.utils.logging import logger

import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.dispatcher.data.packer import BaseDataPacker

from cosmos_rl.policy.trainer import DiffusersTrainer
from cosmos_rl.policy.trainer.base import TrainerRegistry

from diffusers.utils import export_to_video
from PIL import Image

@TrainerRegistry.register(trainer_type="diffusers_sft")
class Diffusers_SFTTrainer(DiffusersTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        **kwargs,
    ):
        super(Diffusers_SFTTrainer, self).__init__(
            config,
            parallel_dims,
            train_stream,
            data_packer,
            val_data_packer,
            **kwargs,
        )

        self.is_video = config.policy.diffusers_config.is_video

        if self.parallel_dims.dp_shard_enabled:
            dp_group = self.parallel_dims.mesh["dp_shard"].get_group()
        else:
            dp_group = None

        
        if self.parallel_dims.cp_enabled:
            cp_group = self.parallel_dims.mesh["cp"].get_group()
        else:
            cp_group = None
        
        self.is_lora = config.policy.lora is not None

    def load_model(self):
        ckpt_total_steps = 0
        train_step = 0
        if (
            not self.parallel_dims.dp_replicate_enabled
        ) or self.parallel_dims.dp_replicate_coord[0] == 0:
            if self.config.train.resume:
                try:
                    # early init the lr_schedulers to avoid it is not initialized when loading the checkpoint
                    ckpt_extra_vars = self.model_resume_from_checkpoint()
                    ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                    train_step = ckpt_extra_vars.get("step", 0)
                except Exception as e:
                    logger.error(
                        f"Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                    )
                    self.lr_schedulers = None
                    self.build_optimizers()
                    self.model.load_hf_weights(
                        self.config.policy.model_name_or_path,
                        self.parallel_dims,
                        self.device,
                        revision=self.config.policy.model_revision,
                    )
            else:
                self.model_load_from_hf()

            # TODO (yy) support multi-replica
        return ckpt_total_steps, train_step


    def model_resume_from_checkpoint(self):
        ckpt_extra_vars, self.lr_schedulers = self.ckpt_manager.load_checkpoint(
            model=self.model.trained_model[0],
            optimizer=self.optimizers,
            scheduler=partial(build_lr_schedulers, self.optimizers, self.config),
            model_name_or_path=self.config.policy.model_name_or_path,
            revision=self.config.policy.model_revision,
            strict=not self.is_lora, # For LoRA training, ckpt only save lora adapter's parameters, should load with restrict=False
        )
        return ckpt_extra_vars

    def checkpointing(
        self,
        total_steps: int,
        train_step: int,
        save_freq: int,
        is_last_step: bool = False,
        pp_last_stage: bool = False,
        val_score: Optional[float] = None,
    ):
        # Support save safetensor
        if (
            is_last_step or (train_step % save_freq == 0 and train_step > 0)
            ) and self.parallel_dims.dp_replicate_coord[0] == 0:
            if self.config.train.ckpt.enable_checkpoint:
                logger.info(f"Saving cosmos checkpoint at step {train_step}...")
                model_state_dict = self.model.get_trained_model_state_dict()
                self.ckpt_manager.save_checkpoint(
                    model=model_state_dict,
                    optimizer=self.optimizers,
                    scheduler=self.lr_schedulers,
                    step=train_step,
                    total_steps=total_steps,
                )
                # TODO(yy): support save safetensor
                # self.ckpt_manager.save_check(
                #     step=train_step,
                #     val_score=val_score,
                #     pp_enabled=self.parallel_dims.pp_enabled,
                #     pp_last_stage=pp_last_stage,
                #     pp_master_rank=self.parallel_dims.world_size
                #     - self.parallel_dims.world_size / self.parallel_dims.pp,
                # )

    def step_training(
        self,
        global_batch,
        total_steps,
        train_step,
        save_freq
    ):
        if self.lr_schedulers is None:
            assert (
                train_step == 0
            ), "`SFTTrainer.lr_schedulers` should be None if training is from scratch"
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )
        acc_loss = torch.zeros(1, device=self.device)
        self.optimizers.zero_grad()
        global_batch_size = len(global_batch)
        # split global_batch into mini_batches
        mini_batch_begin_idxs = list(
            range(
                0,
                global_batch_size,
                self.config.train.train_policy.mini_batch,
            )
        )
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        for i in mini_batch_begin_idxs:
            # gradient accumulation
            raw_batch = global_batch[i : i + self.config.train.train_policy.mini_batch]
            batch = self.data_packer.sft_collate_fn(
                raw_batch
            )
            loss_term = self.model.training_step(batch['visual'], batch['prompt'])
            loss_term['loss'].mean().backward()

        acc_loss += loss_term['loss'].detach()
        all_params = [
            p
            for m in [model for model in self.model.trained_model if model is not None]
            for p in m.parameters()
        ]
        grad_norm = dist_util.gradient_norm_clipping(
            all_params,
            self.config.train.optm_grad_norm_clip,
            foreach=True,
            pp_mesh=self.parallel_dims.mesh["pp"]
            if self.parallel_dims.pp_enabled
            else None,
            return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        end_event.record()

        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(acc_loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(acc_loss, self.parallel_dims.mesh["dp_cp"]),
            )
        else:
            global_avg_loss = global_max_loss = acc_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if util.is_master_rank(self.parallel_dims, self.global_rank):
                # Calculate last iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds

                report_data = {
                    "train/iteration_time": iter_time,
                    "train/loss_avg": global_avg_loss,
                    "train/loss_max": global_max_loss,
                    "train/learning_rate": self.lr_schedulers.get_last_lr()[0],
                    "train/grad_norm": grad_norm if grad_norm is not None else -1,
                }

                # TODO (yy): support MFU calculation for diffusers
        return report_data

    def save_visual_output(self, visual_output, save_dir, filename):
        suffix = 'mp4' if self.is_video else 'png'
        filename = f"{save_dir}/{filename}.{suffix}"
        if self.is_video:
            export_to_video(visual_output, filename, fps=16)
        else:
            visual_output.save(filename)

    def step_validation(self, val_global_batch, train_step: int, total_steps: int):
        if not self.config.validation.enable:
            return
        if self.parallel_dims.dp_replicate_coord[0] != 0:
            return
        
        val_batch = self.val_data_packer.sft_collate_fn(
            val_global_batch,
            is_validation=True,
        )
        save_dir = os.path.join(self.config.train.output_dir, "visual_output", str(train_step))
        os.makedirs(save_dir, exist_ok=True)
        for batch in val_batch:
            if self.is_video:
                frame = batch['frames']
            else:
                frame = None
            visual_output = self.model.inference(
                inference_step = batch['inference_step'],
                height = batch['height'],
                width = batch['width'],
                guidance_scale=batch['guidance_scale'],
                prompt_list=[batch['prompt']],
                frames=frame,
            )[0]
            self.save_visual_output(visual_output, save_dir, batch['prompt'][:10])
        return 0.0
