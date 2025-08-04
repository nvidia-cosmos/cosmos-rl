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

from cosmos_rl.policy.model.deepseek_v3.configs.model_config import (
    TrainingConfig,
    ExperimentalConfig,
    ActivationCheckpointConfig,
    Float8Config,
    CheckpointConfig,
    OptimizerConfig,
    CommConfig,
    DeepseekConfig,
)


def create_default_deepseek_config():
    training_dict = {
        "compile": False,
        "data_parallel_shard_degree": -1,
        "data_parallel_replicate_degree": 1,
        "tensor_parallel_degree": 1,
        "context_parallel_degree": 1,
        "expert_parallel_degree": 64,
        "pipeline_parallel_degree": 1,
        "disable_loss_parallel": False,
        "mixed_precision_param": "bfloat16",
        "mixed_precision_reduce": "float32",
        "enable_cpu_offload": False,
        "warmup_steps": 1000,
        "steps": 500000,
        "use_linear_decay": False,
        "use_cosine_decay": True,
        "fsdp_reshard_after_forward": "default",
    }
    training = TrainingConfig(**training_dict)
    experimental_dict = {
        "enable_async_tensor_parallel": False,
        "enable_compiled_autograd": False,
    }
    experimental = ExperimentalConfig(**experimental_dict)
    activation_checkpoint_dict = {
        "mode": "selective",
        "models": "llm",
        "selective_ac_option": "op",
    }
    activation_checkpoint = ActivationCheckpointConfig(**activation_checkpoint_dict)
    float8_dict = {
        "enable_float8_linear": False,
    }
    float8 = Float8Config(**float8_dict)
    checkpoint_dict = {
        "enable_checkpoint": False,
        "folder": "checkpoint",
        "interval_type": "steps",
        "interval": 500,
        "model_weights_only": False,
        "export_dtype": "float32",
        "async_mode": "disabled",
        "create_seed_checkpoint": False,
    }
    checkpoint = CheckpointConfig(**checkpoint_dict)
    optimizer_dict = {
        "name": "AdamW",
        "lr": 1e-06,
        "init_lr": 1e-07,
        "end_lr": 1e-06,
        "fused": True,
        "early_step_in_backward": False,
        "lr_multiplier_llm": 1.0,
    }
    optimizer = OptimizerConfig(**optimizer_dict)
    comm_dict = {
        "init_timeout_seconds": 300,
        "train_timeout_seconds": 100,
        "trace_buf_size": 20000,
    }
    comm = CommConfig(**comm_dict)
    flat_dict = {
        "max_batch_size": 1,
        "max_seq_len": 16384,
        "training_seq_len": 10240,
        "use_fsdp2": True,
        "use_rope_from_torchtitan": True,
        "precision": "bfloat16",
        "fsdp_enabled": False,
        "z_loss_coeff": 0.0,
        "freeze_llm": False,
        "seed": 0,
        "deterministic": False,
        "use_cache": False,
        "loss_per_token": True,
        "dtype": "bfloat16",
        "vocab_size": 129280,
        "dim": 7168,
        "inter_dim": 18432,
        "moe_inter_dim": 2048,
        "n_layers": 61,
        "n_dense_layers": 3,
        "n_heads": 128,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "n_activated_experts": 8,
        "n_expert_groups": 8,
        "n_limited_groups": 4,
        "train_gate": True,
        "gate_bias_update_factor": 0.01,
        "score_func": "sigmoid",
        "route_scale": 2.5,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "original_seq_len": 4096,
        "rope_theta": 10000.0,
        "rope_factor": 40,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "tokenizer_type": "deepseek_r1",
        "enable_deepep": False,
        "fake_balanced_gate": False,
    }
    return DeepseekConfig(
        training=training,
        experimental=experimental,
        activation_checkpoint=activation_checkpoint,
        float8=float8,
        checkpoint=checkpoint,
        optimizer=optimizer,
        comm=comm,
        **flat_dict,
    )
