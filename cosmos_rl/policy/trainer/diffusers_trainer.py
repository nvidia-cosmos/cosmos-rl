import torch
import os
import time
import random
import numpy as np
import json
import threading
from typing import Optional, Dict

from cosmos_rl.policy.trainer.base import Trainer
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model import ModelRegistry
from cosmos_rl.policy.trainer.optm import build_optimizers, build_lr_schedulers

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.utils.checkpoint import CheckpointMananger

class DiffusersTrainer(Trainer):
    def __init__(
        self, 
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: Optional[torch.cuda.Stream] = None,
        data_packer: BaseDataPacker = None,
        val_data_packer: BaseDataPacker = None,
        **kwargs,
        ):
        super(DiffusersTrainer, self).__init__(
            config = config,
            parallel_dims = parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs
        )

        if config.train.seed:
            torch.manual_seed(config.train.seed)
            torch.cuda.manual_seed(config.train.seed)
            torch.cuda.manual_seed_all(config.train.seed)
            random.seed(config.train.seed)
            np.random.seed(config.train.seed)

        if config.train.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(mode=True, warn_only=True)
        
        # This model contains all part for a diffusers pipeline (transformers, vae, text_encoder)
        model = ModelRegistry.build_model(config)

        # Add low precision support

        try:
            # Apply parallelism to the model
            parallelize_fn, _ = model.parallelize_fn
            # `pp_scheduler` is used for both `sft` and `RLHF`
            # `pp_scheduler_val` is used only for `sft`, since `RLHF` does not require policy model via validation
            self.pp_scheduler, self.pp_scheduler_val = parallelize_fn(
                model, parallel_dims, config, pp_loss_fn=None
            )
            # Enable gradient checkpointing for the model
            model.set_gradient_checkpointing_enabled(
                config.policy.model_gradient_checkpointing
            )

            torch.cuda.empty_cache()
            self.model_parts = model.separate_model_parts()
            self.model = model
            # util.add_nan_checks(model)
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e

        self.ckpt_manager = CheckpointMananger(
            self.config, self.parallel_dims, self.global_rank
        )

        self.build_optimizers()
        self.lr_schedulers = None        


    def build_optimizers(self):
        # TODO (yy): Add low precision support
        self.optimizers = build_optimizers(self.model.trained_model, self.config)

    def build_lr_schedulers(self):
        pass

    def step_training(self):
        pass

    def step_validation(self):
        pass
    
    def export_safetensors(
        self,
        output_dir: str,
        rel_path: str,
        trainable_only: bool = False,
        is_final=False,
        dtype: Optional[torch.dtype] = None,
    ):
        # TODO (yy): support safetensor exporting
        pass

    def model_load_from_hf(self):
        # TODO (yy): meta init not support now
        pass