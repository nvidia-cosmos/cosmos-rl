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

import torch
import os
from datetime import timedelta
import torch.distributed as dist

from transformers import AutoConfig

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils import util
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import ParallelismConfig
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.model.gpt import GPT
import toml
import random

from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses

"""
To run this test, execute like: `CP_SIZE=2 TP_SIZE=1 DP_SIZE=2 torchrun --nproc_per_node=4 test_context_parallel.py`
"""


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG_TOML = """
redis = "12808"

[train]
resume = false
epoch = 1
output_dir = "./outputs/qwen2-5-7b-p-fsdp2-cp2-r-tp2-pp1-grpo"
epsilon = 1e-6
optm_name = "AdamW"
optm_lr = 1e-6
optm_impl = "fused"
optm_weight_decay = 0.01
optm_betas = [0.9, 0.999]
optm_warmup_steps = 20
optm_grad_norm_clip = 1.0
async_tp_enabled = false
compile = true
param_dtype = "bfloat16"
fsdp_reduce_dtype = "float32"
fsdp_offload = false
fsdp_reshard_after_forward = "default"
train_batch_per_replica = 8
sync_weight_interval = 1

[rollout]
gpu_memory_utilization = 0.7
enable_chunked_prefill = false
max_response_length = 2048
n_generation = 16
batch_size = 1
quantization = "none"


[policy]
model_name_or_path = "Qwen/Qwen2.5-7B-Instruct"
model_max_length = 4096
model_gradient_checkpointing = true

[logging]
logger = ['console', 'wandb']
project_name = "cosmos_rl"
experiment_name = "None"

[train.train_policy]
type = "grpo"
dataset.name = "JiaxinTsao/math_examples"
prompt_column_name = "prompt"
response_column_name = "result"
reward_function = "boxed_math"
dataset.split = "train"
temperature = 0.9
epsilon_low = 0.2
epsilon_high = 0.2
kl_beta = 0.0
mu_iterations = 1
min_filter_prefix_tokens = 1

[train.ckpt]
enable_checkpoint = false
save_freq = 20
save_mode = "async"

[rollout.parallelism]
n_init_replicas = 1
tp_size = 2
pp_size = 1

[rollout.sampling_config]
temperature = 0.9
top_p = 1.0
top_k = 10

[policy.parallelism]
n_init_replicas = 1
tp_size = 1
cp_size = 2
dp_shard_size = 2
pp_size = 1
dp_replicate_size = 1

"""


def test_cp_forward_and_backward(CP_SIZE, TP_SIZE, DP_SIZE):
    # read the model config from raw string
    config_dict = toml.loads(CONFIG_TOML)
    loaded_config = CosmosConfig.from_dict(config_dict)

    # modify the config
    loaded_config.policy.parallelism.cp_size = CP_SIZE
    loaded_config.policy.parallelism.tp_size = TP_SIZE
    loaded_config.policy.parallelism.dp_shard_size = DP_SIZE

    # int dist
    torch.distributed.init_process_group(
        backend="cuda:nccl,cpu:gloo",
        timeout=timedelta(seconds=600),
    )

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    assert (
        world_size == CP_SIZE * DP_SIZE * TP_SIZE
    ), f"world_size: {world_size} != CP_SIZE * DP_SIZE * TP_SIZE: {CP_SIZE * DP_SIZE * TP_SIZE}"

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    hf_config = AutoConfig.from_pretrained(loaded_config.policy.model_name_or_path)

    batch_size = 1
    seqlen_multiple = CP_SIZE * TP_SIZE
    if CP_SIZE > 1:
        seqlen_multiple = (seqlen_multiple + 16 - 1) // 16 * 16

    # 1. Run the CP model.
    # init cp mesh
    parallel_dims = ParallelDims.from_config(
        ParallelismConfig(cp_size=CP_SIZE, tp_size=TP_SIZE, dp_shard_size=DP_SIZE)
    )
    parallel_dims.build_mesh(device_type="cuda")

    cp_mesh = parallel_dims.mesh["cp"]

    # simulate the disptach_rollouts in grpo_trainer.py
    vocab_size = hf_config.vocab_size
    seqlen_max = 4096
    taotal_num_of_sample = DP_SIZE  # this is the same logic like dispatch_rollouts.

    scatter_input_ids = [[] for _ in range(world_size)]

    if global_rank == 0:
        total_input_ids = []
        for _ in range(taotal_num_of_sample):
            seqlen_cur = (
                (random.randint(256, seqlen_max) + seqlen_multiple - 1)
                // seqlen_multiple
                * seqlen_multiple
            )
            input_ids_cur = torch.randint(
                low=0, high=vocab_size, size=(batch_size, seqlen_cur)
            )
            total_input_ids.append(input_ids_cur)

        dp_id = 0
        for sample in total_input_ids:
            for i in range(world_size):
                fsdp_rank_for_i = parallel_dims.get_rank_in_dim("dp_shard", i)
                if fsdp_rank_for_i == dp_id:
                    scatter_input_ids[i].append(sample)
            dp_id += 1
            if dp_id >= DP_SIZE:
                dp_id = 0

    inputs_id_list_each_rank = [[]]
    input_ids = None
    if world_size == 1:
        input_ids = scatter_input_ids[0]
    else:
        dist.scatter_object_list(inputs_id_list_each_rank, scatter_input_ids, src=0)
        input_ids = inputs_id_list_each_rank[0]
    input_ids = input_ids[0].to(device)  # change cpu tensor to cuda tensor

    assert (
        input_ids.shape[1] % (CP_SIZE * TP_SIZE) == 0
    ), "Input sequence length must be multiple of CP_SIZE * TP_SIZE"

    # 1. CP part.
    # Only DP and CP to simpify test. So each rank will load all the model weights.
    with torch.device("meta"):
        with util.cosmos_default_dtype(torch.bfloat16):
            model = GPT.from_pretrained(
                hf_config,
                loaded_config.policy.model_name_or_path,
                max_position_embeddings=4096,
            )

    model.to_empty(device=device)

    # parallel model
    try:
        # Apply parallelism to the model
        parallelize_fn, _ = model.parallelize_fn
        _, _ = parallelize_fn(model, parallel_dims, loaded_config, None)
        model.post_to_empty_hook(loaded_config)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    # load hf weight
    model.load_hf_weights(
        loaded_config.policy.model_name_or_path,
        parallel_dims=parallel_dims,
        device=device,
    )

    # Now we get the input ids for each rank.
    user_mini_batch = {
        "input_ids": input_ids,
    }

    position_ids, input_ids, pos_seq_dim = model.get_position_ids(**user_mini_batch)
    user_mini_batch["position_ids"] = position_ids

    input_ids_before_cp = user_mini_batch["input_ids"]
    position_ids_before_cp = user_mini_batch["position_ids"]

    input_ids, position_ids = slice_inputs_for_ulysses(
        [input_ids, position_ids],
        cp_mesh,
    )

    user_mini_batch["position_ids"] = position_ids
    user_mini_batch["input_ids"] = input_ids

    ulysses_logits = model(**user_mini_batch)
    # for foward, each rank will have the same output.
    mean_ulysses_logits = ulysses_logits.mean()

    # backward
    mean_ulysses_logits.backward()

    ulysses_grad_for_o_proj_weight = model.layers["0"].self_attn.o_proj.weight.grad
    global_ulysses_grad_shape = ulysses_grad_for_o_proj_weight.shape
    local_ulysses_grad_for_o_proj_weight = (
        ulysses_grad_for_o_proj_weight.to_local()
        if isinstance(ulysses_grad_for_o_proj_weight, dist.tensor.DTensor)
        else ulysses_grad_for_o_proj_weight.to_local()
    )
    for_comp_mean_ulysses_logits = mean_ulysses_logits.detach()

    # 2. Non-cp part. Run the non-CP model, normal inference.
    del model
    torch.cuda.empty_cache()

    logger.info("Test: Run normal inference.")
    # reset the input ids and position ids, fsdp is not changed, so input_ids are not changed.
    user_mini_batch["input_ids"] = input_ids_before_cp
    user_mini_batch["position_ids"] = position_ids_before_cp

    # change the loaded_config, fsdp is not changed, but change merge cp into tp.
    loaded_config.policy.parallelism.cp_size = 1
    loaded_config.policy.parallelism.tp_size = TP_SIZE * CP_SIZE
    loaded_config.policy.parallelism.dp_shard_size = DP_SIZE

    parallel_dims = ParallelDims.from_config(
        ParallelismConfig(cp_size=1, tp_size=TP_SIZE * CP_SIZE, dp_shard_size=DP_SIZE)
    )
    parallel_dims.build_mesh(device_type="cuda")

    with torch.device("meta"):
        with util.cosmos_default_dtype(torch.bfloat16):
            model = GPT.from_pretrained(
                hf_config,
                loaded_config.policy.model_name_or_path,
                max_position_embeddings=4096,
            )

    model.to_empty(device=device)

    # parallel model
    try:
        # Apply parallelism to the model
        parallelize_fn, _ = model.parallelize_fn
        _, _ = parallelize_fn(model, parallel_dims, loaded_config, None)
        model.post_to_empty_hook(loaded_config)
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e

    model.load_hf_weights(
        loaded_config.policy.model_name_or_path,
        parallel_dims=parallel_dims,
        device=device,
    )

    # run the normal inference
    normal_logits = model(**user_mini_batch)

    mean_normal_logits = normal_logits.mean()

    # backward
    mean_normal_logits.backward()

    # copy the grad of the embedding weight
    normal_grad_for_o_proj_weight = model.layers["0"].self_attn.o_proj.weight.grad
    global_normal_grad_shape = normal_grad_for_o_proj_weight.shape
    local_normal_grad_for_o_proj_weight = (
        normal_grad_for_o_proj_weight.to_local()
        if isinstance(normal_grad_for_o_proj_weight, dist.tensor.DTensor)
        else normal_grad_for_o_proj_weight.to_local()
    )
    for_comp_mean_normal_logits = mean_normal_logits.detach()

    if global_rank == 0:
        with torch.no_grad():
            logger.info(
                f"[Global Rank {global_rank}] ulysses_logits shape: {ulysses_logits.shape}"
            )
            logger.info(
                f"[Global Rank {global_rank}] normal_logits shape: {normal_logits.shape}"
            )
            assert ulysses_logits.data_ptr() != normal_logits.data_ptr()

            logger.info(
                f"[Global Rank {global_rank}] ulysses_logits: {ulysses_logits[0, 1]}"
            )
            logger.info(
                f"[Global Rank {global_rank}] normal_logits: {normal_logits[0, 1]}"
            )

            logger.info(
                f"[Global Rank {global_rank}] for_comp_mean_ulysses_logits: {for_comp_mean_ulysses_logits}"
            )
            logger.info(
                f"[Global Rank {global_rank}] for_comp_mean_normal_logits: {for_comp_mean_normal_logits}"
            )
            torch.testing.assert_close(
                for_comp_mean_ulysses_logits,
                for_comp_mean_normal_logits,
                rtol=1e-1,
                atol=1e-1,
            )

        assert (
            local_ulysses_grad_for_o_proj_weight.data_ptr()
            != local_normal_grad_for_o_proj_weight.data_ptr()
        )
        logger.info(
            f"local_ulysses_grad_for_o_proj_weight.shape: {local_ulysses_grad_for_o_proj_weight.shape}"
        )
        logger.info(
            f"local_normal_grad_for_o_proj_weight.shape: {local_normal_grad_for_o_proj_weight.shape}"
        )
        # TP+FSDP weight sharding for o_proj is both in rowwise
        unit_part_of_row_dim = global_ulysses_grad_shape[0] // (DP_SIZE * CP_SIZE)
        unit_part_of_col_dim = global_normal_grad_shape[1] // (TP_SIZE * CP_SIZE)
        logger.info(f"unit_part_of_row_dim: {unit_part_of_row_dim}")
        logger.info(f"unit_part_of_col_dim: {unit_part_of_col_dim}")
        rank_0_ulysses_weight = local_ulysses_grad_for_o_proj_weight[
            :unit_part_of_row_dim, :unit_part_of_col_dim
        ]
        rank_0_normal_weight = local_normal_grad_for_o_proj_weight[
            :unit_part_of_row_dim, :unit_part_of_col_dim
        ]

        logger.info(f"rank_0_ulysses_weight.shape: {rank_0_ulysses_weight.shape}")
        logger.info(f"rank_0_normal_weight.shape: {rank_0_normal_weight.shape}")
        logger.info(f"rank_0_ulysses_weight: {rank_0_ulysses_weight.flatten()[-10:]}")
        logger.info(f"rank_0_normal_weight: {rank_0_normal_weight.flatten()[-10:]}")

        torch.testing.assert_close(
            rank_0_ulysses_weight, rank_0_normal_weight, atol=1e-1, rtol=1e-4
        )


if __name__ == "__main__":
    if "CP_SIZE" not in os.environ:
        raise ValueError("CP_SIZE must be set")
    if "TP_SIZE" not in os.environ:
        raise ValueError("TP_SIZE must be set")
    if "DP_SIZE" not in os.environ:
        raise ValueError("DP_SIZE must be set")

    CP_SIZE = int(os.environ.get("CP_SIZE"))
    TP_SIZE = int(os.environ.get("TP_SIZE"))
    DP_SIZE = int(os.environ.get("DP_SIZE"))

    test_cp_forward_and_backward(CP_SIZE=CP_SIZE, TP_SIZE=TP_SIZE, DP_SIZE=DP_SIZE)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
