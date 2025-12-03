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

from cosmos_rl.dispatcher.command import RolloutToRolloutBroadcastCommand
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.utils.distributed import init_distributed, destroy_distributed
from cosmos_rl.policy.trainer import GRPOTrainer
from cosmos_rl.colocated.rl_worker import ColocatedRLControlWorker
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.policy.config import Config as CosmosConfig
import torch
from cosmos_rl.dispatcher.api.client import APIClient
from typing import List
from cosmos_rl.dispatcher.data.schema import Rollout
import math
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)


def mock_for_decoupled_loss():
    orig_dispatch_rollouts = RLPolicyWorker.dispatch_rollouts

    def dispatch_rollouts(self):
        ret: List[Rollout] = orig_dispatch_rollouts(self)
        for rollout in ret:
            assert not rollout.completion
            assert len(rollout.completion_token_ids) == len(rollout.completion_logprobs)
            assert len(rollout.completion_token_ids) > 0
        return ret

    RLPolicyWorker.dispatch_rollouts = dispatch_rollouts

    orig_compute_logprobs = GRPOTrainer.compute_logprobs

    def compute_logprobs(
        self,
        minibatch,
        logits,
        is_full_logits,
    ):
        assert "rollout_logprobs" in minibatch
        ret = orig_compute_logprobs(
            self,
            minibatch,
            logits,
            is_full_logits,
        )
        logprobs = ret[0].tolist()
        rollout_logprobs = [
            i for sublist in minibatch["rollout_logprobs"] for i in sublist
        ]
        assert len(logprobs) == len(
            rollout_logprobs
        ), f"{len(logprobs)} vs {len(rollout_logprobs)}"
        all_same = True
        for lp, rlp in zip(logprobs, rollout_logprobs):
            assert (
                math.exp(lp - rlp) < 2.0
            ), f"Logprob: {lp}, Rollout logprob: {rlp} : {lp - rlp} {math.exp(lp - rlp)}"
            if lp != rlp:
                all_same = False
        assert not all_same, "All logprobs are the same, something is wrong."

        return ret

    GRPOTrainer.compute_logprobs = compute_logprobs


def mock_for_custom_rollout():
    orig_dispatch_rollouts = RLPolicyWorker.dispatch_rollouts

    def dispatch_rollouts(self):
        ret: List[Rollout] = orig_dispatch_rollouts(self)
        for rollout in ret:
            assert rollout.completion
            assert isinstance(rollout.completion, str)
            assert len(rollout.completion) > 0
        return ret

    RLPolicyWorker.dispatch_rollouts = dispatch_rollouts

    orig_compute_logprobs = GRPOTrainer.compute_logprobs

    def compute_logprobs(
        self,
        minibatch,
        logits,
        is_full_logits,
    ):
        ret = orig_compute_logprobs(
            self,
            minibatch,
            logits,
            is_full_logits,
        )
        if not hasattr(self, "computed_cnt"):
            self.computed_cnt = 0
        self.computed_cnt += 1
        assert torch.any(ret[0] != 0)
        return ret

    GRPOTrainer.compute_logprobs = compute_logprobs


def mock_for_colocated():
    origin_broadcast_to_all_rollout_replica = (
        ColocatedRolloutControlWorker.broadcast_to_all_rollout_replica
    )

    def broadcast_to_all_rollout_replica_fake(
        self,
        broadcast_command,
    ) -> None:
        origin_broadcast_to_all_rollout_replica(
            self,
            broadcast_command,
        )
        if broadcast_command.replica_should_stop():
            assert (
                self.current_weight_version
                == broadcast_command.total_steps
                == broadcast_command.weight_step
            )
            assert self.current_weight_version == 2

    ColocatedRolloutControlWorker.register_rollout_command_handler(
        RolloutToRolloutBroadcastCommand
    )(broadcast_to_all_rollout_replica_fake)


def main(*args, **kwargs):
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

    api_client = APIClient(role="POLICY")
    metadata = api_client.get_controller_metadata()

    if metadata["config"] is None:
        raise RuntimeError(
            f"[Policy] Please first go to http://{api_client.remote_ips}:{api_client.remote_port} to configure training parameters."
        )

    cosmos_config = CosmosConfig.from_dict(metadata["config"])

    logger.info(f"[Policy] Loaded configuration: {cosmos_config.model_dump()}")

    parallel_dims = ParallelDims.from_config(
        parallesim_config=cosmos_config.policy.parallelism
    )
    init_distributed()
    parallel_dims.build_mesh(device_type="cuda")

    policy_type = cosmos_config.train.train_policy.type

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str)
    args = parser.parse_args()

    if args.test == "decoupled_loss":
        # Apply the mock for decoupled loss testing
        mock_for_decoupled_loss()
    elif args.test == "custom_rollout":
        # Apply the mock for custom rollout testing
        mock_for_custom_rollout()
    elif args.test == "colocated":
        mock_for_colocated()

    try:
        if cosmos_config.mode == "colocated":
            logger.info("Starting colocated RL worker...")
            policy_worker = ColocatedRLControlWorker(
                config=cosmos_config,
                parallel_dims=parallel_dims,
                **kwargs,
            )
            policy_worker.main_loop()
        elif policy_type == "grpo":
            logger.info("Starting GRPO training...")
            worker = RLPolicyWorker(
                config=cosmos_config,
                parallel_dims=parallel_dims,
                dataset=kwargs.get("dataset", None),
                data_packer=kwargs.get("data_packer", None),
                val_dataset=kwargs.get("val_dataset", None),
                val_data_packer=kwargs.get("val_data_packer", None),
            )
            worker.main_loop()
            if args.test == "custom_rollout":
                assert worker.trainer.computed_cnt == 4
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise e
    finally:
        destroy_distributed()
        logger.info("Process group destroyed.")


if __name__ == "__main__":
    main()
