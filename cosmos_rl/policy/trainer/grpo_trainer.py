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

from cosmos_rl.policy.trainer import Trainer
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.parallelism import (
    ParallelDims,
)
import torch
import inspect
import os
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import compute_mfu
import cosmos_rl.utils.distributed as dist_util
import time
import torch.distributed as dist
import numpy as np
import requests
import threading
import asyncio
from queue import Queue, Empty
from cosmos_rl.dispatcher.command import (
    Command,
    BuildMeshCommand,
    PolicyToPolicyBroadcastCommand,
    PolicyToRolloutUnicastCommand,
    WeightResumeCommand,
    PolicyToPolicyUnicastCommand,
    DataFetchCommand,
)
import atexit
from cosmos_rl.utils.util import (
    list_to_b64,
    msgpack_c_long,
    msgunpack_c_long,
    fix_data_type_size,
)
from cosmos_rl.utils.parallelism_map import (
    ParallelTopoMapperGroup,
)
from functools import cached_property
from typing import List, Callable, Dict, Any, Tuple, Optional
import types
from functools import partial
import msgpack
from cosmos_rl.utils.network_util import make_request_with_retry
from cosmos_rl.utils.ulysses import slice_input_for_ulysses
from cosmos_rl.utils.util import is_master_rank
from cosmos_rl.utils import constant
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.dispatcher.replica import Rollout
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_POLICY_TRAIN_ACK_SUFFIX,
)
from cosmos_rl.utils.pynccl import (
    create_nccl_uid,
    create_nccl_comm,
    nccl_send,
)
from cosmos_rl.utils.util import compute_logprobs as logprobs_computing


def compute_loss(
    current_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    old_per_token_logps: torch.Tensor,  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    ref_per_token_logps: Optional[
        torch.Tensor
    ],  # per-token logprobs of shape `[n_tokens_of_logprobs]`
    current_advantages: torch.Tensor,  # of shape `[batch_size, max_len]`
    cu_seqlens: torch.Tensor,  # of shape `[batch_size + 1]`
    config: CosmosConfig,
    logprob_masks: torch.Tensor,  # of shape `[batch_size, max_len]`
) -> Tuple[torch.Tensor, torch.Tensor]:
    # Turn current_advantages from [batch_size, max_len] to [n_logprob_tokens]
    current_advantages = torch.masked_select(current_advantages, logprob_masks)

    assert (
        current_token_logps.shape == current_advantages.shape
    ), "current_token_logps and current_advantages should have the same shape"
    assert (
        old_per_token_logps.shape == current_token_logps.shape
    ), "old_per_token_logps and ref_per_token_logps should have the same shape"
    if ref_per_token_logps is not None:
        assert (
            ref_per_token_logps.shape == current_token_logps.shape
        ), "ref_per_token_logps and current_token_logps should have the same shape, but got {} and {}".format(
            ref_per_token_logps.shape, current_token_logps.shape
        )

    # Compute the KL divergence between the model and the reference model
    if config.train.train_policy.kl_beta != 0.0:
        assert (
            not ref_per_token_logps.requires_grad
        ), "ref_per_token_logps should not require gradient"
        """
            With reference model used for KL. The logic should be further reviewed to verify.
        """
        per_token_kl = (
            torch.exp(ref_per_token_logps - current_token_logps)
            - (ref_per_token_logps - current_token_logps)
            - 1
        )

    # Same processing as `verl`
    # Clamp coef_1 for stability
    coef_1 = torch.clamp(current_token_logps - old_per_token_logps, min=-20.0, max=20.0)
    coef_1 = torch.exp(coef_1)

    if config.train.train_policy.aipo_rho is not None:
        # Due to the asynchronous update of the reference model, the rollout is not necessarily
        # the exact previous iterate of latest policy. So a more natural motivation is correct
        # for the off-policyness of samples generated under previous policy, to construct
        # approximate on-policy update to latest policy.
        # A difference from double-sided clipping of PPO, we use one-sided clipping.
        rho = config.train.train_policy.aipo_rho
        per_token_loss = -torch.clamp(coef_1, max=rho) * current_advantages
    else:
        # the standard grpo loss with dual-clip PPO: https://arxiv.org/pdf/1912.09729
        coef_2 = torch.clamp(
            coef_1,
            1 - config.train.train_policy.epsilon_low,
            1 + config.train.train_policy.epsilon_high,
        )
        per_token_loss1 = coef_1 * current_advantages
        per_token_loss2 = coef_2 * current_advantages
        per_token_loss3 = (
            -config.train.train_policy.lower_bound_ratio * current_advantages
        )
        clip_losses1 = -torch.min(per_token_loss1, per_token_loss2)
        clip_losses2 = torch.min(per_token_loss3, clip_losses1)
        per_token_loss = torch.where(current_advantages < 0, clip_losses2, clip_losses1)

    if config.train.train_policy.kl_beta != 0.0:
        """
            With reference model used for KL. The logic should be further reviewed to verify.
        """
        kl_loss = config.train.train_policy.kl_beta * per_token_kl
        per_token_loss += kl_loss
    else:
        kl_loss = torch.zeros_like(per_token_loss)

    bsz, max_len = logprob_masks.shape
    per_token_loss_seq_sum = torch.zeros(
        bsz, device=per_token_loss.device, dtype=per_token_loss.dtype
    )  # [bsz,]
    kl_loss_seq_sum = torch.zeros(
        bsz, device=kl_loss.device, dtype=kl_loss.dtype
    )  # [bsz,]
    for i in range(bsz):
        per_token_loss_seq_sum[i] = per_token_loss[
            cu_seqlens[i] : cu_seqlens[i + 1]
        ].sum()
        kl_loss_seq_sum[i] = kl_loss[cu_seqlens[i] : cu_seqlens[i + 1]].sum()
    shifted_length = cu_seqlens[1:] - cu_seqlens[:-1]

    if config.train.train_policy.loss_type == "seq-mean-token-mean":
        # seq-mean-token-sum
        # If Dr.GRPO is used, we need to normalize the loss by the max tokens for unbiased loss
        if (
            config.train.train_policy.unbiased_loss_max_tokens is not None
            and config.train.train_policy.unbiased_loss_max_tokens > 0
        ):
            norm_factor = config.train.train_policy.unbiased_loss_max_tokens
        else:
            norm_factor = max_len

        per_token_loss = (per_token_loss_seq_sum / norm_factor).mean()
        kl_loss = (kl_loss_seq_sum / norm_factor).mean()
        return per_token_loss, kl_loss
    elif config.train.train_policy.loss_type == "seq-mean-token-sum":
        # seq-mean-token-sum
        per_token_loss = per_token_loss_seq_sum / max_len
        kl_loss = kl_loss_seq_sum / max_len
        return per_token_loss.mean(), kl_loss.mean()
    elif config.train.train_policy.loss_type == "token-mean":
        # token-mean
        per_token_loss = per_token_loss_seq_sum / shifted_length
        kl_loss = kl_loss_seq_sum / shifted_length
        return per_token_loss.mean(), kl_loss.mean()
    else:
        raise ValueError(f"Invalid loss type: {config.train.train_policy.loss_type}")


class GRPOTrainer(Trainer):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims):
        super().__init__(config, parallel_dims)
        self.reference_state_dict = {}

        if parallel_dims.dp_replicate > 1:
            raise ValueError(
                f"DP replicate size {parallel_dims.dp_replicate} is not supported for GRPO"
                "Please use elastic scaling feature instead."
            )

        self.grpo_config = self.config.train.train_policy
        # For model load
        self.model_ready = False

        # For mesh build
        self.inter_policy_nccl = HighAvailabilitylNccl(
            replica_name=self.replica_name,
            global_rank=self.global_rank,
            controller_hosts=self.remote_hosts,
        )
        self.rollouts_comm = {}
        self.kv_store = dist_util.DistKVStore(
            group=dist.distributed_c10d._get_default_group(),
            master_rank=0,
            shutdown_event=self.shutdown_signal,
        )

        # For command fetch
        self.fetch_command_buffer = Queue()
        self.command_buffer = Queue()

        # For rollouts fetch
        self.data_queue = Queue()

        # Parallel parameters
        self.dp_rank, self.dp_world_size = 0, 1
        if parallel_dims.dp_enabled:
            self.dp_rank = parallel_dims.mesh["dp"].get_local_rank()
            self.dp_world_size = parallel_dims.mesh["dp"].size()

        # Init redis controller
        self.init_redis()

        # For iteration control
        self.mini_step = 0
        self.replica_batch_for_this_step = 0
        self.mini_batch = self.grpo_config.mini_batch

        # For Polocy to Rollout weight mapping
        self.parallel_mapper = None
        self.policy_to_rollout_insts = None

        # For GRPO
        self.max_length = config.policy.model_max_length
        self.mu_iterations = self.config.train.train_policy.mu_iterations
        self.optimizers.zero_grad()
        self.fetch_command_thread = None
        self.fetch_rollouts_thread = None
        atexit.register(self.handle_shutdown)
        self.p2r_related_ranks = None
        self.p2r_nccl_uuids = {}

        # Flag for determining if the current replica is the master replica,
        # The master replica needs to:
        # - Save the checkpoint/safetensors
        self.is_master_replica = True

    def handle_shutdown(self):
        if not hasattr(self, "_handle_shutdown_called"):
            self._handle_shutdown_called = True

            self.shutdown_signal.set()
            self.inter_policy_nccl.shutdown()
            if self.fetch_rollouts_thread is not None:
                self.fetch_rollouts_thread.join()
                self.fetch_rollouts_thread = None

            if self.fetch_command_thread is not None:
                self.fetch_command_thread.join()
                self.fetch_command_thread = None

            if hasattr(self, "heartbeat_thread") and self.heartbeat_thread is not None:
                self.heartbeat_thread.join()
                self.heartbeat_thread = None

            # Manually unregister from controller
            self.unregister_from_controller()

            if hasattr(self, "upload_thread") and self.upload_thread is not None:
                logger.info("[Policy] Waiting for upload thread to finish...")
                self.upload_thread.join()
                logger.info("[Policy] Upload thread finished.")
                self.upload_thread = None

            # TODO(jiaxin)
            # The background threads are daemon threads, so that they will exit when the main thread exits
            # However, the previous `.join()` may not really wait for them to stop.
            # So we need to wait for a while to ensure they have a chance to exit to prevent `exitcode:-6`

            # Another notice is that make sure the background threads detect the shutdown event in less than 15 seconds
            # Otherwise, the main thread may exit before the background threads detect the shutdown event
            time.sleep(15)

    def model_load_from_hf(self):
        self.model.load_hf_weights(
            self.config.policy.model_name_or_path,
            self.parallel_dims,
            self.device,
        )
        self.model.train()
        self.model_ready = True

    def model_resume_from_checkpoint(self):
        self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizers,
            scheduler=self.lr_schedulers,
        )
        self.model.train()
        self.model_ready = True

    async def fetch_rollouts(self):
        assert self.global_rank == 0, "Only rank 0 can fetch rollouts"
        while not self.shutdown_signal.is_set():
            rollouts = []
            try:
                rollouts = [
                    Rollout.from_dict(msgpack.unpackb(x))
                    for x in self.redis_controller.subscribe_rollout(self.replica_name)
                ]
            except Exception as e:
                logger.debug(f"Failed to get rollouts: {e}, wait for next round")
            for rollout in rollouts:
                self.data_queue.put_nowait(rollout)

    def wrap_to_cuda_tensor(self, key, obj, in_place=False):
        """
        wrap the object to cuda tensor for sync parameters using nccl.
        """
        if isinstance(obj, torch.Tensor):
            if isinstance(obj, torch.distributed.tensor.DTensor):
                obj = obj.to_local()

            if obj.device != self.device:
                if in_place:
                    raise ValueError(
                        f"Object {key} is not on the same device as the model. Please set in_place to False."
                    )
                obj = obj.to(self.device)
            return obj
        elif isinstance(obj, np.ndarray):
            if in_place:
                raise ValueError(
                    f"Object {key} is not a tensor. Please set in_place to False."
                )
            obj = torch.from_numpy(obj).to(self.device)
            return obj
        else:
            if in_place:
                raise ValueError(
                    f"Object {key} is not a tensor. Please set in_place to False."
                )
            if isinstance(obj, tuple):
                obj = tuple(
                    [x.tolist() if isinstance(x, np.ndarray) else x for x in obj]
                )
                obj = fix_data_type_size(obj)
            bytes = msgpack.packb(obj, default=msgpack_c_long)
            obj = torch.frombuffer(bytes, dtype=torch.uint8).to(self.device)
            return obj

    def extract_from_cuda_tensor(self, key, obj, tensor):
        """
        Extract the object from cuda tensor for sync parameters using nccl.
        """
        if isinstance(obj, torch.distributed.tensor.DTensor):
            assert (
                obj.device == self.device
            ), "DTensor is not on the same device as the model."
        elif isinstance(obj, torch.Tensor):
            if obj.device != self.device:
                obj.copy_(tensor)
        elif isinstance(obj, np.ndarray):
            if obj.shape != tensor.shape:
                raise ValueError(
                    f"Object {key} is not the same shape as the tensor. Please check the data consistency."
                )
            x = tensor.cpu()
            obj.copy_(x.numpy())
        else:
            np_arr = tensor.cpu()
            obj_new = msgpack.unpackb(bytes(np_arr.numpy()), ext_hook=msgunpack_c_long)
            if isinstance(obj, tuple):
                assert len(obj) == len(obj_new)
                obj = tuple(
                    [
                        np.array(obj_new[idx])
                        if isinstance(x, np.ndarray)
                        else tuple(obj_new[idx])
                        if isinstance(x, tuple)
                        else obj_new[idx]
                        for idx, x in enumerate(obj)
                    ]
                )
        return obj

    def sync_all_states(self, is_send: bool, send_hook: callable, recv_hook: callable):
        """
        Sync all states of the model and optimizer.
        """
        len_params = 0
        model_state_dict = [self.model.state_dict()]

        # If KL-divergence is enabled, we need to also sync the reference model state dict
        if self.config.train.train_policy.kl_beta != 0.0:
            if len(self.reference_state_dict) == 0:
                assert (
                    not is_send
                ), "Reference model state dict should be populated before sending"
                for key, value in model_state_dict[0].items():
                    self.reference_state_dict[key] = torch.empty_like(
                        value, device="cpu"
                    )
            model_state_dict.append(self.reference_state_dict)

        # 1. Sync all model states
        for state_to_sync in model_state_dict:
            for dest_name in sorted(state_to_sync.keys()):
                obj = state_to_sync[dest_name]
                assert isinstance(obj, torch.Tensor)
                local_view = self.wrap_to_cuda_tensor(
                    dest_name, obj, in_place=obj.is_cuda
                )
                if is_send:
                    send_hook(local_view)
                else:
                    recv_hook(local_view)
                    if isinstance(obj, torch.distributed.tensor.DTensor):
                        to_write = obj.to_local()
                    else:
                        to_write = obj

                    # Copy again for offloaded tensor since it is not inplace received
                    if not to_write.is_cuda:
                        to_write.copy_(local_view)
                len_params += 1

        # 2. Sync optimizer states
        optimizer_state = self.optimizers.state_dict()
        for dest_name in sorted(optimizer_state.keys()):
            obj = optimizer_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if local_view.data_ptr() is None:
                # skip the optimizer state if the data pointer is None
                continue
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                optimizer_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.optimizers.load_state_dict(optimizer_state)

        # 3. Sync lr_scheduler states
        lr_sheduler_state = self.lr_schedulers.state_dict()
        for dest_name in sorted(lr_sheduler_state.keys()):
            obj = lr_sheduler_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                lr_sheduler_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.lr_schedulers.load_state_dict(lr_sheduler_state)

        # 4. Sync rng_state
        rng_state = self.ckpt_manager.get_rng_state()
        for dest_name in sorted(rng_state.keys()):
            obj = rng_state[dest_name]
            local_view = self.wrap_to_cuda_tensor(dest_name, obj)
            if is_send:
                # nccl send
                send_hook(local_view)
            else:
                # nccl recv
                recv_hook(local_view)
                rng_state[dest_name] = self.extract_from_cuda_tensor(
                    dest_name, obj, local_view
                )
            len_params += 1
        if not is_send:
            self.ckpt_manager.set_rng_state(rng_state)
        return len_params

    @Trainer.register_policy_command_handler(PolicyToPolicyBroadcastCommand)
    def execute_policy_to_policy_broadcast(
        self, command: PolicyToPolicyBroadcastCommand
    ):
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name in command.dst_replica_names and not send
        if not send and not recv:
            return True
        st = time.time()
        # TODO(zjx): there need failure tolerance for nccl send and recv, so get nccl param from command
        send_recv_hook = partial(
            self.inter_policy_nccl.broadcast, src_replica=command.src_replica_name
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_recv_hook,
            recv_hook=send_recv_hook,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Broadcast {len_params} parameters from {command.src_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.src_replica_name)}) to {len(command.dst_replica_names)} replicas took {time_eclapsed:.3f} seconds."
        )
        return False

    @Trainer.register_policy_command_handler(PolicyToPolicyUnicastCommand)
    def execute_policy_to_policy_unicast(self, command: PolicyToPolicyUnicastCommand):
        send = self.replica_name == command.src_replica_name
        recv = self.replica_name == command.dst_replica_name
        if not send and not recv:
            return False
        st = time.time()
        # TODO(zjx): there need failure tolerance for nccl send and recv, so get nccl param from command
        send_hook = partial(
            self.inter_policy_nccl.send, dst_replica=command.dst_replica_name
        )
        recv_hook = partial(
            self.inter_policy_nccl.recv, src_replica=command.src_replica_name
        )
        len_params = self.sync_all_states(
            is_send=send,
            send_hook=send_hook,
            recv_hook=recv_hook,
        )
        if recv:
            self.model_ready = True
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] Policy2Policy Unicast {len_params} parameters from {command.src_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.src_replica_name)}) to {command.dst_replica_name} (rank {self.inter_policy_nccl.get_replica_rank(command.dst_replica_name)}) as sender {send} took {time_eclapsed:.3f} seconds."
        )
        return False

    @cached_property
    def map_w_from_policy_to_rollout(self):
        """
        Generate a mapping from local parameters into shape/layout that rollout requires.
        The mapping is created by iterating through the named parameters of both models
        and replacing certain substrings in the parameter names.
        """
        name_to_transform = {}
        assert len(self.model.sorted_hf_key_n_rank) > 0, "No sorted parameters found."
        for name, transform_block in self.model.weight_sync_transforms:
            assert isinstance(transform_block, Callable) or isinstance(
                transform_block, torch.Tensor
            )
            name_to_transform[name] = transform_block
        return name_to_transform

    def pre_P2R_collect_parameters(self):
        needed_tensors = []
        for inst in self.policy_to_rollout_insts:
            dest_name = inst[3]
            needed_tensors.append(dest_name)
        prepared_tensor_to_rollout = {}
        for dest_name, local_view in self.map_w_from_policy_to_rollout.items():
            if isinstance(
                local_view, Callable
            ) and self.model.weight_mapper.policy_pre_P2R_gather_required_for_sync(
                dest_name
            ):
                view = local_view()
                if dest_name in needed_tensors:
                    prepared_tensor_to_rollout[dest_name] = view
        return prepared_tensor_to_rollout

    @Trainer.register_policy_command_handler(PolicyToRolloutUnicastCommand)
    def execute_policy_to_rollout_unicast(self, command: PolicyToRolloutUnicastCommand):
        assert command.src_replica_size == self.world_size
        if self.parallel_mapper is None:
            self.parallel_mapper = ParallelTopoMapperGroup(
                self.config.policy.parallelism,
                self.config.rollout.parallelism,
                self.world_size,
                command.dst_replica_size,
                hf_config=self.hf_config,
                weight_mapper=self.model.weight_mapper,
            )
        if not command.src_replica_name == self.replica_name:
            logger.error(
                f"Policy {self.replica_name} received P2R command from {command.src_replica_name}, but it is not the source replica."
            )
            return False

        if self.policy_to_rollout_insts is None:
            # Ordered list of (hf_key, tensor_dim)
            hf_key_n_rank: List[Tuple[str, int]] = self.model.sorted_hf_key_n_rank
            self.policy_to_rollout_insts = (
                self.parallel_mapper.prepare_policy_to_rollout_manifest(
                    hf_key_n_rank, self.global_rank
                )
            )
            self.p2r_related_ranks = [set() for _ in range(command.src_replica_size)]
            for rank in range(command.src_replica_size):
                insts_at_rank = self.parallel_mapper.prepare_policy_to_rollout_manifest(
                    hf_key_n_rank, rank
                )
                for i in insts_at_rank:
                    p_rank, r_rank, _, _, _ = i
                    self.p2r_related_ranks[rank].add(r_rank)

        comm_id = {}
        # Create nccl id for one policy replica to another rollout replica
        for p_rank in range(command.src_replica_size):
            for r_rank in sorted(self.p2r_related_ranks[p_rank]):
                mesh_key = command.src_replica_name + "_" + command.dst_replica_name

                if mesh_key not in self.p2r_nccl_uuids:
                    nccl_uuid = None
                    if self.global_rank == 0:
                        # Only create nccl group id in rank 0.
                        nccl_uuid = create_nccl_uid()
                        base64_nccl_group_id = list_to_b64(nccl_uuid)
                        logger.debug(f"[Policy] mesh_key: {mesh_key}")
                        try:
                            make_request_with_retry(
                                partial(
                                    requests.post,
                                    json={
                                        "unique_pair_name": mesh_key,
                                        "handle_base64": base64_nccl_group_id,
                                    },
                                ),
                                self.get_alternative_urls(
                                    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX
                                ),
                                max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                            )
                        except Exception as e:
                            raise RuntimeError(
                                f"[Policy] Failed in post nccl group_id to controller after retries {e}."
                            )
                    # broadcast the nccl group id to all ranks
                    nccl_uuid = dist_util.broadcast_object_cpu(nccl_uuid)
                    self.p2r_nccl_uuids[mesh_key] = nccl_uuid

        for r_rank in sorted(self.p2r_related_ranks[self.global_rank]):
            mesh_key = command.src_replica_name + "_" + command.dst_replica_name
            if mesh_key not in self.rollouts_comm:
                assert mesh_key in self.p2r_nccl_uuids
                nccl_uuid = self.p2r_nccl_uuids[mesh_key]
                logger.debug(
                    f"[Policy] Creating nccl communicator for `P2R` with mesh_key: {mesh_key}"
                )
                comm_id[r_rank] = create_nccl_comm(
                    nccl_uuid,
                    self.global_rank,
                    self.world_size + command.dst_replica_size,
                )
                logger.debug(
                    f"[Policy] `P2R` nccl comm: {comm_id[r_rank]} for `P2R` with mesh_key: {mesh_key} is created."
                )
                self.rollouts_comm[mesh_key] = comm_id[r_rank]
            else:
                comm_id[r_rank] = self.rollouts_comm[mesh_key]
        assert (
            self.map_w_from_policy_to_rollout is not None
        ), "No parameters to sync found."
        st = time.time()
        # sort the param list by the dest_name, same as rollout
        total_bytes_sent = 0
        # There is a local-replica comm in training step
        # Here we use another comm to send weight to rollout
        # NCCL announces that multi-comm could lead to deadlocks if not synchronized
        with torch.cuda.stream(self.train_stream):
            pre_P2R_collected_tensors: Dict[str, torch.Tensor] = (
                self.pre_P2R_collect_parameters()
            )
            for inst in self.policy_to_rollout_insts:
                p_rank, r_rank, tensor_split_strategys, dest_name, _ = inst
                if dest_name not in self.map_w_from_policy_to_rollout:
                    raise RuntimeError(
                        f"dest_name {dest_name} not in map_w_from_policy_to_rollout"
                    )
                local_view = self.map_w_from_policy_to_rollout[dest_name]
                if dest_name in pre_P2R_collected_tensors:
                    local_view = pre_P2R_collected_tensors[dest_name]
                elif isinstance(local_view, Callable):
                    local_view = local_view()
                else:
                    pass

                view = (
                    local_view.cosmos_slice(tensor_split_strategys).contiguous().cuda()
                )
                assert self.global_rank == p_rank
                nccl_send(
                    view,
                    self.world_size + r_rank,
                    comm_id[r_rank],
                )
                total_bytes_sent += view.numel() * view.element_size()
        # make sure all the send operations of all ranks are finished
        time_eclapsed = time.time() - st
        logger.debug(
            f"[Policy] All {len(self.policy_to_rollout_insts)} at step {command.weight_step} send operations of finished in {time_eclapsed:.3f} seconds with {total_bytes_sent / (1024 * 1024)} MB sent."
        )
        return False

    @Trainer.register_policy_command_handler(WeightResumeCommand)
    def execute_weight_resume(self, command: WeightResumeCommand = None):
        # If KL-divergence is enabled, hf model should always be loaded from checkpoint
        model_loaded = False
        if self.config.train.train_policy.kl_beta != 0.0:
            self.model_load_from_hf()
            model_loaded = True
            # Clone the state dict of hf model so that it can be used for KL-divergence calculation
            self.reference_state_dict = {}
            state_dict = self.model.state_dict()
            for key, value in state_dict.items():
                self.reference_state_dict[key] = value.detach().cpu()

        if self.config.train.resume:
            try:
                # Need to reload again from checkpoint to make sure the model is in the correct state
                self.model_resume_from_checkpoint()
                model_loaded = True
            except Exception as e:
                if isinstance(e, FileNotFoundError):
                    logger.info(
                        f"Fail to resume from {self.config.train.resume} because the checkpoint file does not exist, trying to load from HuggingFace..."
                    )
                else:
                    logger.error(
                        f"Cannot resume from {self.config.train.resume} {e}. Trying to load from HuggingFace..."
                    )
                if not model_loaded:
                    self.model_load_from_hf()
                    model_loaded = True
        elif not model_loaded:
            logger.info("Resume not set. Trying to load from HuggingFace...")
            self.model_load_from_hf()
            model_loaded = True

        assert model_loaded, "Model weight must be populated before training starts."
        logger.info("[Policy] Model loaded from checkpoint.")
        assert (
            self.map_w_from_policy_to_rollout is not None
        ), "No parameters to sync found."
        return False

    @Trainer.register_policy_command_handler(DataFetchCommand)
    def execute_data_fetch(self, command: DataFetchCommand):
        if command.do_profile:
            self.profiler.start_dynamic(
                active_steps=command.active_steps,
                rank_filter=command.rank_filter,
                record_shape=command.record_shape,
                profile_memory=command.profile_memory,
                with_stack=command.with_stack,
                with_modules=command.with_modules,
            )

        assert self.replica_name == command.replica_name
        self.replica_batch_for_this_step = command.items_count

        is_fake_step = self.replica_batch_for_this_step == 0
        if not is_fake_step:
            report_data = self.train(
                current_step=command.global_step,
                total_steps=command.total_steps,
                remain_samples_num=command.remain_samples_num,
            )
        else:
            report_data = {}
            logger.info(
                f"[Policy] No data to fetch for global step {command.global_step}, skip this step."
            )

        # Train ACK
        if is_master_rank(self.parallel_dims, self.global_rank):
            try:
                make_request_with_retry(
                    partial(
                        requests.post,
                        json={
                            "replica_name": self.replica_name,
                            "weight_step": command.global_step,
                            "total_steps": command.total_steps,
                            "profile_finished": self.profiler.check_finished(),
                            "report_data": report_data,
                        },
                    ),
                    self.get_alternative_urls(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX),
                    max_retries=constant.COSMOS_HTTP_RETRY_CONFIG.max_retries,
                )
            except Exception as e:
                raise RuntimeError(
                    f"[Policy] Failed in in send train ack to controller after retries {e}."
                )

        logger.debug(f"[Policy] Train ack sent for global step {command.global_step}.")
        return command.replica_should_stop()

    def execute_all_reduce(self):
        """
        # Add nccl allreduce operations for all parameters and necessary states.
        """
        for model_part in self.model_parts:
            # Model part may use same physical mesh for different logical mesh,
            # which is not supported by DTensor operands like `torch.nn.utils.get_total_norm`
            # So we need to do allreduce for each model part
            if model_part is not None:
                dist_util.gradient_reduce_across_dp_replicas_(
                    [p for p in model_part.parameters()], self.inter_policy_nccl
                )

            if self.config.train.optm_grad_norm_clip > 0:
                # Then clipping gradient norm
                dist_util.gradient_norm_clipping(
                    # Must pass empty list even if model_part is None,
                    # GradNorm across pp stages will fail if some rank does not join the barrier
                    [p for p in model_part.parameters()]
                    if model_part is not None
                    else [],
                    self.config.train.optm_grad_norm_clip,
                    foreach=True,
                    pp_mesh=self.parallel_dims.mesh["pp"]
                    if self.parallel_dims.pp_enabled
                    else None,
                )
        self.optimizers.step()
        self.lr_schedulers.step()
        self.optimizers.zero_grad()
        return True

    async def fetch_command(self):
        # assert self.global_rank == 0, "Only rank 0 can fetch command"
        while not self.shutdown_signal.is_set():
            # TODO(zjx): will remove separate BuildMeshCommand, and here only fetch other commands
            if self.global_rank == 0:
                # rank 0 will get command from redis
                # and broadcast the buildmesh command to all ranks
                commands = []
                try:
                    commands = self.redis_controller.subscribe_command(
                        self.replica_name
                    )
                except Exception as e:
                    logger.debug(
                        f"Failed to get commands : {e} at replica {self.replica_name}, wait for next round"
                    )
                try:
                    for x in commands:
                        command = Command.depack(x)
                        if isinstance(command, BuildMeshCommand):
                            """ directly push the buildmesh command to the nccl comm, will not block main thread """
                            # broadcast the buildmesh command to all ranks
                            cmd = self.kv_store.broadcast_command(command, src=0)
                            self.is_master_replica = (
                                cmd.replica_name_to_rank[self.replica_name] == 0
                            )
                            self.inter_policy_nccl.push_cmd(cmd)
                            continue
                        self.fetch_command_buffer.put_nowait(command)
                except Exception as e:
                    logger.error(e)
                    raise e

            else:
                try:
                    bmcmd = self.kv_store.broadcast_command(None, src=0)
                    if bmcmd:
                        assert isinstance(
                            bmcmd, BuildMeshCommand
                        ), "Only buildmesh command is supported"
                        self.is_master_replica = (
                            bmcmd.replica_name_to_rank[self.replica_name] == 0
                        )
                        self.inter_policy_nccl.push_cmd(bmcmd)
                except Exception as e:
                    logger.error(f"Failed to broadcast on slave workers: {e}")
                    raise e

    def execute_command(self, command: Command):
        logger.debug(f"[Policy] Process command {command._serialize()}")

        handler = self.get_policy_command_handler(type(command))
        if handler is None:
            raise Exception(f"No such command supoorted in policy {command}")
        should_abort = handler(self, command)
        logger.debug(
            f"[Policy] Command {command._serialize()} executed with abort: {should_abort}"
        )
        return should_abort

    def broadcast_command(self):
        command = []
        if self.global_rank == 0:
            while len(self.fetch_command_buffer.queue) > 0:
                command.append(self.fetch_command_buffer.get_nowait())
        command = dist_util.broadcast_object_cpu(
            command, src=0, device=torch.device("cpu")
        )
        if len(command) > 0:
            for c in command:
                self.command_buffer.put_nowait(c)

    def main_loop(self):
        def fetch_command_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_command())
            new_loop.stop()
            new_loop.close()
            return

        def fetch_rollouts_helper(trainer: GRPOTrainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_rollouts())
            new_loop.stop()
            new_loop.close()
            return

        # Start the thread with daemon=True, so it will exit when the main program exits.
        # we need all ranks have fetch_command_thread, so that buildmesh command can be broadcasted to all ranks
        # TODO(zjx): we will only let rank 0 fetch and broadcast command
        self.fetch_command_thread = threading.Thread(
            target=fetch_command_helper,
            args=(self,),
            daemon=True,
            name="fetch_command_thread",
        ).start()

        if self.global_rank == 0:
            self.fetch_rollouts_thread = threading.Thread(
                target=fetch_rollouts_helper,
                args=(self,),
                daemon=True,
                name="fetch_rollouts_thread",
            ).start()

        abort = False
        while True:
            abort_at_this_round = abort
            if abort_at_this_round:
                # Wait 30s to make sure the final potential P->R is received to finalize the Rollouts
                time.sleep(30)

            self.broadcast_command()
            while len(self.command_buffer.queue) > 0:
                cmd = self.command_buffer.get_nowait()
                abort = self.execute_command(cmd) or abort

            if abort_at_this_round:
                break
        logger.info("[Policy] Main loop finished. Shutdown background task event set.")
        self.train_stream.synchronize()
        self.handle_shutdown()

    def dispatch_rollouts(self):
        rollouts = [[]]
        scattered_rollouts = [[] for _ in range(self.world_size)]
        if self.global_rank == 0:
            batch_for_this_step = (
                self.replica_batch_for_this_step
                // self.dp_world_size
                * self.dp_world_size
            )
            assert batch_for_this_step % self.dp_world_size == 0

            dp_id = 0
            for _ in range(batch_for_this_step):
                try:
                    rollout = self.data_queue.get(block=True, timeout=None)
                except Empty:
                    logger.error(
                        "[Policy] Rollouts queue is empty, please check the dispatcher."
                    )
                    raise Empty
                for i in range(self.world_size):
                    if self.parallel_dims.get_rank_in_dim("dp", i) == dp_id:
                        scattered_rollouts[i].append(rollout)
                        # logger.info(f"[Policy] Rollout {dp_id} dispatched to rank {i}, dp world_size {self.dp_world_size}")
                dp_id += 1
                if dp_id >= self.dp_world_size:
                    dp_id = 0
        if self.world_size == 1:
            return scattered_rollouts[0]
        dist.scatter_object_list(
            rollouts,
            scattered_rollouts,
            src=0,
        )
        return rollouts[0]

    def compute_logprobs(
        self,
        minibatch: Dict[str, Any],
        full_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the per-token log probabilities and advantages

        Args:
            minibatch: a dictionary containing the input_ids and logprob_masks
            full_logits: the logits of the model

        Returns:
            logps: the per-token log probabilities
            logprob_masks: the logprob_masks
        """
        assert "input_ids" in minibatch, "input_ids is required for computing logprobs"
        assert (
            "logprob_masks" in minibatch
        ), "logprob_masks is required for computing logprobs"
        return logprobs_computing(
            minibatch["input_ids"], minibatch["logprob_masks"], full_logits
        )

    def _swap_model_state_dict(self):
        kl_beta = self.config.train.train_policy.kl_beta
        if kl_beta != 0.0:
            with torch.cuda.stream(self.train_stream):
                model_state_dict = self.model.state_dict()
                reference_state_dict = self.reference_state_dict
                for key, value in model_state_dict.items():
                    # clone the reference state dict to avoid inplace operation
                    ref_clone = reference_state_dict[key].clone()
                    # copy the current model state dict to the reference state dict
                    reference_state_dict[key].copy_(value)
                    # copy the reference state dict to the current model state dict
                    value.copy_(ref_clone)
            return True, kl_beta
        else:
            return False, 0.0

    def forward_backward_step(
        self,
        batched_datas: List[Any],
        batched_advantages: Optional[List[Any]],
        old_per_token_logps: Optional[List[Any]],
        ref_per_token_logps: Optional[List[Any]],
        num_pp_micro_batches: int,
        only_forward: bool,
        pp_last_stage: bool = False,
    ):
        """
        Do forward and backward for each batched_data. Support both pipeline parallel and data parallel.
        Attention: this function will not do allreduce and update optimizer, you need to do it by yourself.

        Here we have two mode:
        1. only_forward = True, we will only compute the per-token logprobs, and skip the rest of the loop
        2. only_forward = False, we will compute the loss and backward for each batched_data

        Args:
            batched_datas: List[Any], the datas to be processed, when pp_enabled, it's a list of length 1.
            batched_advantages: Optional[List[Any]], the advantages to be processed, when pp_enabled, it's a list of length 1.
            old_per_token_logps: Optional[List[Any]], the old per-token logprobs to be processed, when pp_enabled, it's a list of length 1.
            ref_per_token_logps: Optional[List[Any]], the ref per-token logprobs to be processed, when pp_enabled, it's a list of length 1.
            num_pp_micro_batches: int, pp_schedule will split the input batch_data into num_pp_micro_batches micro_batches.
            only_forward: bool, whether to only forward, if True, we will skip backward.
            pp_last_stage: bool, whether the current stage is the last stage of the pipeline.

        Return:
            losses: List[float]
            batched_logps: List[torch.Tensor] or List[List[torch.Tensor]] in pp_enabled
            acc_n_tokens: int, the number of tokens processed so far
        """

        n_batches = len(batched_datas)

        if self.parallel_dims.pp_enabled:
            # TODO(zjx): this is a hack to disable backward in pp_scheduler, may cause some unexpected behavior
            # Originally, pp_scheduler set has_backward by detecting if loss_fn is passed in or not.
            if not only_forward and not hasattr(
                self, "pp_scheduler_fwd_bwd_initialized"
            ):
                # because we use pp_scheduler with ony_forward multi times at beginning, when forward and backward one batch_data,
                # some backward states is not initialized, so need to initialize the pp_scheduler with fwd_bwd once.
                setattr(self, "pp_scheduler_fwd_bwd_initialized", True)

                # let pp_scheduler re-initialize the stage after toggle has_backward status.
                self.pp_scheduler._has_backward = True
                self.pp_scheduler._stage_initialized = False

            self.pp_scheduler._stage.has_backward = not only_forward

            # because pp_scheduler will split input batch_data into micro_batches,
            # so we directly pass the full batch_data.
            assert 1 == len(batched_datas), "pp_enabled mode only support 1 batch_data"

        if batched_advantages is not None:
            assert (
                n_batches == len(batched_advantages)
            ), f"batch num of batched_advantages should be {n_batches}, but got {len(batched_advantages)}"
        if old_per_token_logps is not None:
            # It's a little different from batched_advantages, we pass the whole mini_batch to pp_stage,
            # so here we assume the length of old_per_token_logps is the same as batch_size.
            # user can see more details in `_swizzle_pp_grpo_forward` function.
            if self.parallel_dims.pp_enabled:
                assert (
                    1 == len(old_per_token_logps)
                ), f"batch num of old_per_token_logps should be 1, but got {len(old_per_token_logps)}"
            else:
                assert (
                    n_batches == len(old_per_token_logps)
                ), f"batch num of old_per_token_logps should be {n_batches}, but got {len(old_per_token_logps)}"
        if ref_per_token_logps is not None:
            # the value of ref_per_token_logps can be None
            if self.parallel_dims.pp_enabled:
                # same reason as old_per_token_logps
                assert (
                    1 == len(ref_per_token_logps)
                ), f"batch num of ref_per_token_logps should be 1, but got {len(ref_per_token_logps)}"
            else:
                assert (
                    n_batches == len(ref_per_token_logps)
                ), f"batch num of ref_per_token_logps should be {n_batches}, but got {len(ref_per_token_logps)}"

        if only_forward:
            batched_advantages = [None] * n_batches
            old_per_token_logps = [None] * n_batches
            ref_per_token_logps = [None] * n_batches
        else:
            assert batched_advantages is not None
            assert old_per_token_logps is not None
            assert ref_per_token_logps is not None

        losses = []
        kl_losses = []
        acc_n_tokens = 0

        # we use this container collect logps from _swizzle_pp_grpo_forward function.
        # TODO(zjx) will remove in the future, and get logps driect from model's output.
        if self.parallel_dims.pp_enabled:
            # the reset pp stage hold list of None, too keep same size like pp_last_stage
            self.batched_logps = [None] * num_pp_micro_batches
        else:
            self.batched_logps = [None] * n_batches

        for local_batch_step, (
            batched_data,
            batched_advantage,
            old_per_token_logp,
            ref_per_token_logp,
        ) in enumerate(
            zip(
                batched_datas,
                batched_advantages,
                old_per_token_logps,
                ref_per_token_logps,
            )
        ):
            # TODO(jiaxin): support variable length in PP
            computed_max_len = (
                self.config.policy.model_max_length
                if self.parallel_dims.pp_enabled
                else self.data_packer.policy_compute_max_len(batched_data)
            )

            computed_max_len = (
                (computed_max_len + self.seq_len_multiple - 1)
                // self.seq_len_multiple
                * self.seq_len_multiple
            )
            # Convert advantages from [batch_size] -> [batch_size, max_len] via expanding
            user_batched_advantages = (
                (
                    batched_advantage.unsqueeze(1)
                    .expand(-1, computed_max_len)
                    .to(self.device)
                )
                if batched_advantage is not None
                else None
            )

            user_batch: Dict[str, Any] = self.data_packer.policy_collate_fn(
                batched_data,
                computed_max_len=computed_max_len,
            )

            # Move all tensor to device
            for k in user_batch.keys():
                v = user_batch[k]
                if isinstance(v, torch.Tensor) and v.device != self.device:
                    user_batch[k] = v.to(self.device)

            # input_ids are different across ranks in dp_shard_cp
            position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                **user_batch
            )
            acc_n_tokens += np.prod(input_ids.shape)
            user_batch["position_ids"] = position_ids

            input_ids_before_cp = user_batch["input_ids"]
            position_ids_before_cp = user_batch["position_ids"]

            if self.parallel_dims.cp_enabled:
                input_ids, position_ids = slice_input_for_ulysses(
                    input_ids,
                    position_ids,
                    self.parallel_dims.mesh["cp"],
                )
                user_batch["position_ids"] = position_ids
                user_batch["input_ids"] = input_ids

            with (
                torch.set_grad_enabled(not only_forward),
                torch.cuda.stream(self.train_stream),
            ):
                if self.parallel_dims.pp_enabled:
                    # [micro_batch_size, 1]: indicating the index of mini-batch
                    pp_first_stage = self.parallel_dims.pp_coord[0] == 0
                    # Pipeline Parallel forward / backward inside step() call
                    pp_losses = []
                    if pp_last_stage:
                        # Only the pp_last_stage use swizzle_pp_grpo_forward, so it' can parse those parameters
                        # Inject the `mini-batch` and `micro-batch` ids to the input so that the last stage can know which microbatch it is processing
                        user_batch["pp_batch_ids"] = (
                            torch.arange(
                                num_pp_micro_batches, dtype=torch.int
                            ).unsqueeze(1)
                            // self.config.policy.parallelism.micro_batch_size
                        )
                        user_batch["loss_scaling"] = torch.tensor(
                            [
                                [
                                    1
                                    / n_batches
                                    / self.config.policy.parallelism.micro_batch_size
                                ]
                            ]
                            * num_pp_micro_batches,
                            dtype=torch.float32,
                        )

                        user_batch["only_forward"] = torch.tensor(
                            [only_forward] * num_pp_micro_batches,
                            dtype=torch.bool,
                        )
                        # if the per_token_logp is None, we won't pass them in to user_batch
                        if old_per_token_logp is not None:
                            user_batch["old_per_token_logprob"] = old_per_token_logp
                        if ref_per_token_logp is not None:
                            user_batch["ref_per_token_logprob"] = ref_per_token_logp
                        if user_batched_advantages is not None:
                            user_batch["advantages"] = user_batched_advantages

                    if pp_first_stage or pp_last_stage:
                        # First/Last stage: pass all inputs
                        if self.parallel_dims.cp_enabled:
                            # This is for recover these two tensors after ulysses
                            user_batch["input_ids_before_cp"] = input_ids_before_cp
                            user_batch["position_ids_before_cp"] = (
                                position_ids_before_cp
                            )

                        # TODO(zjx): stage will return a merged stage output, maybe we can use it to get the current_per_token_logprobs
                        self.pp_scheduler.step(
                            **user_batch,
                            losses=pp_losses,
                            target=torch.empty(
                                [num_pp_micro_batches, 1],
                                device=self.device,
                            ),
                        )
                    else:
                        # Middle stages: forward data from previous stage
                        self.pp_scheduler.step(
                            position_ids=user_batch["position_ids"],
                        )

                    if only_forward:
                        # Continue to next mini-batch since loss is not needed for reference model
                        # The current_per_token_logprobs already set in the swizzle_pp_grpo_forward
                        continue

                    if pp_last_stage:
                        loss = torch.mean(torch.stack(pp_losses)).to(self.device)
                        # TODO(zjx): maybe we can record kl_loss here
                        # kl_losses.append(0.0)
                        losses.append(loss.item())
                else:
                    # without pp_enabled
                    raw_logits = self.model(**user_batch)

                    if self.parallel_dims.cp_enabled:
                        # reset the position ids and input ids
                        user_batch["position_ids"] = position_ids_before_cp
                        user_batch["input_ids"] = input_ids_before_cp

                    if self.config.train.train_policy.temperature > 1e-6:
                        raw_logits = (
                            raw_logits / self.config.train.train_policy.temperature
                        )

                    current_per_token_logprobs, cu_seqlens = self.compute_logprobs(
                        user_batch,
                        full_logits=raw_logits,
                    )

                    # Compute ref per-token logprobs if needed
                    if only_forward:
                        self.batched_logps[local_batch_step] = (
                            current_per_token_logprobs.detach()
                        )
                        # Skip the rest of the loop
                        continue

                    # Continue compute loss and backward
                    logprob_masks = user_batch["logprob_masks"]
                    current_advantages = logprob_masks * user_batched_advantages

                    loss, kl_loss = compute_loss(
                        current_per_token_logprobs,
                        old_per_token_logp,
                        ref_per_token_logp,
                        current_advantages,
                        cu_seqlens,
                        self.config,
                        logprob_masks,
                    )
                    if n_batches > 1:
                        loss /= n_batches
                        kl_loss /= n_batches
                    loss.backward()
                    losses.append(loss.item())
                    kl_losses.append(kl_loss.item())

        batched_logps = self.batched_logps
        delattr(self, "batched_logps")

        if self.parallel_dims.pp_enabled:
            assert len(batched_logps) == num_pp_micro_batches
            if only_forward:
                # the pp_schedule will split the input batch_data into num_pp_micro_batches micro_batches,
                # here we keep the same size as batched_datas (which is 1), so the code can be simpler.
                batched_logps = (
                    [torch.cat(batched_logps, dim=0)]
                    if pp_last_stage
                    else [batched_logps]
                )
        else:
            assert len(batched_logps) == n_batches

        return (losses, kl_losses), batched_logps, acc_n_tokens

    def train(self, current_step: int, total_steps: int, remain_samples_num: int):
        pp_last_stage = (
            self.parallel_dims.pp_coord[0] == self.parallel_dims.pp_coord[1] - 1
        )

        # Do it once
        if (
            pp_last_stage
            and self.parallel_dims.pp_enabled
            and not hasattr(self, "swizzled_forward")
        ):
            # Swizzle the forward function to return the current per-token logprobs.
            orig_forward = self.model.forward
            self.model.forward = types.MethodType(
                partial(
                    _swizzle_pp_grpo_forward,
                    self,
                    orig_forward,
                    self.config,
                ),
                self.model,
            )
            self.swizzled_forward = True

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        logger.debug("[Policy] Prepare training data.")
        rollouts: List[Rollout] = self.dispatch_rollouts()

        payloads_list = [rollout.payload for rollout in rollouts]
        completions_list = [rollout.completion for rollout in rollouts]
        advantages_list = [rollout.advantage for rollout in rollouts]
        n_ignore_prefix_tokens_list = [
            rollout.n_ignore_prefix_tokens for rollout in rollouts
        ]
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                payloads_list[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(payloads_list))
        ]

        # user_info_keys = list(kwargs.keys())
        advantages_t = torch.tensor(advantages_list).to(self.device)
        # Currently, we only support no cp parallelism for policy training.
        assert not self.parallel_dims.cp_enabled
        batch_size = len(rollouts)
        mini_batch_size = (
            min(self.mini_batch, batch_size) if self.mini_batch > 0 else batch_size
        )
        assert (
            batch_size % mini_batch_size == 0
        ), "Batch size should be divided evenly by mini_batch"
        num_mini_batches = batch_size // mini_batch_size

        micro_batch_size = (
            min(self.config.policy.parallelism.micro_batch_size, mini_batch_size)
            if self.config.policy.parallelism.micro_batch_size > 0
            else mini_batch_size
        )
        assert (
            mini_batch_size % micro_batch_size == 0
        ), "Mini batch size should be divided evenly by micro_batch"
        num_micro_batches = mini_batch_size // micro_batch_size

        # Initialize placeholder for old per-token logprobs
        self.old_per_token_logps = [[] for _ in range(num_mini_batches)]
        self.ref_per_token_logps = [[] for _ in range(num_mini_batches)]

        # used to compute mfu
        acc_n_tokens = 0

        # Validate the PP parallelism configuration
        if self.parallel_dims.pp_enabled:
            n_microbatches = (
                mini_batch_size // self.config.policy.parallelism.micro_batch_size
            )
            assert (
                n_microbatches % self.parallel_dims.pp == 0
            ), f"n_microbatches {n_microbatches} should be divided evenly by pp size of {self.parallel_dims.pp}"

        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0

        # 0. split processed_samples into num_mini_batches * num_micro_batches mini_batches
        processed_samples_list = [[] for _ in range(num_mini_batches)]
        batched_advantages_list = [[] for _ in range(num_mini_batches)]
        for i_mini_batch in range(num_mini_batches):
            # it's little different for pp_enabled, pp_schedule.step() will auto split the inputs into micro_batches
            # so we don't need to split the inputs into micro_batches here
            if self.parallel_dims.pp_enabled:
                batch_start = i_mini_batch * mini_batch_size
                batch_end = min(batch_start + mini_batch_size, batch_size)
                processed_samples_list[i_mini_batch] = [
                    processed_samples[batch_start:batch_end]
                ]
                batched_advantages_list[i_mini_batch] = [
                    advantages_t[batch_start:batch_end]
                ]
            else:
                for i_micro_batch in range(0, mini_batch_size, micro_batch_size):
                    # avoid out of index for last micro_batch
                    batch_start = i_mini_batch * mini_batch_size + i_micro_batch
                    batch_end = min(batch_start + micro_batch_size, batch_size)
                    processed_samples_list[i_mini_batch].append(
                        processed_samples[batch_start:batch_end]
                    )
                    batched_advantages_list[i_mini_batch].append(
                        advantages_t[batch_start:batch_end]
                    )

        # 1. compute ref per-token logprobs
        # TODO(zjx): maybe we can only do this once, because the ref model state dict is not changed in the following loop
        need_compute_ref, kl_beta = self._swap_model_state_dict()

        for i_mini_batch, mini_batch_data in enumerate(processed_samples_list):
            if need_compute_ref:
                _, ref_logps, _ = self.forward_backward_step(
                    mini_batch_data,
                    None,
                    None,
                    None,
                    num_pp_micro_batches=num_micro_batches
                    if self.parallel_dims.pp_enabled
                    else micro_batch_size,
                    only_forward=True,
                    pp_last_stage=pp_last_stage,
                )
            else:
                # set some fake value
                ref_logps = [None] * len(mini_batch_data)
            self.ref_per_token_logps[i_mini_batch] = ref_logps

        # swap model state dict back to the original model
        self._swap_model_state_dict()
        self.model.train()

        for i_mu in range(self.mu_iterations):
            # 2. compute old per-token logprobs
            if i_mu == 0:
                for i_mini_batch, mini_batch_data in enumerate(processed_samples_list):
                    _, old_logps, _ = self.forward_backward_step(
                        mini_batch_data,
                        None,
                        None,
                        None,
                        num_pp_micro_batches=num_micro_batches
                        if self.parallel_dims.pp_enabled
                        else micro_batch_size,
                        only_forward=True,
                        pp_last_stage=pp_last_stage,
                    )
                    self.old_per_token_logps[i_mini_batch] = old_logps

            # 3. compute loss and backward for each mini_batch
            for i_mini_batch, (mini_batch_data, mini_advantage_batch_data) in enumerate(
                zip(processed_samples_list, batched_advantages_list)
            ):
                # TODO(zjx): maybe we can pass a mini_batch data to speed up do_fwd_bwd()
                (losses, kl_losses), _, acc_n_tokens_ = self.forward_backward_step(
                    mini_batch_data,
                    mini_advantage_batch_data,
                    self.old_per_token_logps[i_mini_batch],
                    self.ref_per_token_logps[i_mini_batch],
                    num_pp_micro_batches=num_micro_batches
                    if self.parallel_dims.pp_enabled
                    else micro_batch_size,
                    only_forward=False,
                    pp_last_stage=pp_last_stage,
                )
                # do allreduce and optimizer.step()
                self.execute_all_reduce()

                loss_sum += sum(losses)
                loss_count += len(losses)
                kl_loss_sum += sum(kl_losses)
                acc_n_tokens += acc_n_tokens_
                self.mini_step += 1

        # clean up per_token_logps cache
        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                # Calculate the iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss

                # FIXME(dinghaoy): only compute MFU of rank 0, if enable tp or pp,
                # it will be inaccurate. Need a reduce for all the metrics.
                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        n_tokens=acc_n_tokens,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
        # checkpointing
        if self.is_master_replica and (
            (
                self.config.train.ckpt.enable_checkpoint
                and current_step % self.config.train.ckpt.save_freq == 0
                and current_step > 0
            )
            or (
                self.config.train.ckpt.enable_checkpoint and current_step == total_steps
            )
        ):
            if self.config.train.ckpt.export_safetensors:
                logger.info(
                    f"[Policy] Saving huggingface checkpoint at step {current_step} to {self.config.train.output_dir}..."
                )
                self.export_safetensors(
                    output_dir=self.config.train.output_dir,
                    rel_path=os.path.join(
                        "safetensors",
                        f"step_{current_step}",
                    ),
                    trainable_only=False,
                    is_final=current_step == total_steps,
                )
            logger.info(f"[Policy] Saving cosmos checkpoint at step {current_step}...")
            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizers,
                scheduler=self.lr_schedulers,
                step=current_step,
                total_steps=total_steps,
                **{
                    "remain_samples_num": remain_samples_num,
                    "is_final": current_step == total_steps,
                },
            )
            self.ckpt_manager.save_check(step=current_step)

        # For profiling
        self.profiler.step()

        return report_data

    @property
    def pp_loss_fn(self):
        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """
            loss: the loss of shape `[n_tokens]`
            """
            return loss.mean()

        return fake_compute_loss


# TODO: (lms) May be it's better to register this func as a hook to the last stage model.
# That way is more clean. I think it's feasible but need to be compatible with torch Pipelie schedule.
def _swizzle_pp_grpo_forward(
    trainer: GRPOTrainer, ori_forward: Callable, config: CosmosConfig, *args, **kwargs
):
    args = args[1:]  # Skip self
    """
    Swizzle the forward function (only to last stage) to return the loss directly.
    """
    # [mini_batch_size]: the mini-batch index of the sample with respect to the whole batch
    # [micro_batch_size]: the micro-batch index of the sample with respect to the mini-batch
    # TODO(zjx), template rename those key.
    micro_batch_ids = kwargs.pop("pp_batch_ids")
    loss_scaling = kwargs.pop("loss_scaling")
    only_forward = kwargs.pop("only_forward")
    advantages = kwargs.pop("advantages") if "advantages" in kwargs else None
    # if per_token_logp not in kwargs, we assume them is None
    old_per_token_logprob = (
        kwargs.pop("old_per_token_logprob")
        if "old_per_token_logprob" in kwargs
        else None
    )
    ref_per_token_logprob = (
        kwargs.pop("ref_per_token_logprob")
        if "ref_per_token_logprob" in kwargs
        else None
    )

    micro_batch_id = micro_batch_ids[0].item()
    loss_scaling = loss_scaling[0].item()
    only_forward = only_forward[0].item()

    # User defined input
    user_input = kwargs.copy()

    assert torch.all(
        micro_batch_ids == micro_batch_id
    ), f"micro_batch_ids are not all the same: {micro_batch_ids}"
    del micro_batch_ids

    n_args = len(args)
    if n_args > 0:
        # remove the first `n_args` arguments from kwargs
        signature = list(inspect.signature(ori_forward).parameters.keys())[:n_args]
        for key in signature:
            if key in kwargs:
                kwargs.pop(key)

    raw_logits = ori_forward(*args, **kwargs)

    # recover the input ids and position ids
    if "input_ids_before_cp" in kwargs:
        user_input["input_ids"] = kwargs["input_ids_before_cp"]
    if "position_ids_before_cp" in kwargs:
        user_input["position_ids"] = kwargs["position_ids_before_cp"]

    if config.train.train_policy.temperature > 1e-6:
        raw_logits = raw_logits / config.train.train_policy.temperature
    # [n_tokens, n_vocab]
    current_per_token_logprobs, cu_seqlens = trainer.compute_logprobs(
        minibatch={
            **user_input,
        },
        full_logits=raw_logits,
    )

    if only_forward:
        assert hasattr(
            trainer, "batched_logps"
        ), "trainer must provide batched_logps to save logps"
        assert isinstance(trainer.batched_logps, list)
        # here we will record n_micro_batches logps in trainer.batched_logps
        # In some case, the micro_batch may re-compute the logps, so we can't use list.append()
        trainer.batched_logps[micro_batch_id] = current_per_token_logprobs.detach()
        # Skip the rest logic since we are computing logp
        # we must return a Tensor, because torch.distributed.pipelining.stage.PipelineStage._shape_inference
        # will cast and concat this value. if not, it will raise an error
        return torch.tensor([0.0], device=trainer.device)

    logprob_masks = user_input["logprob_masks"]
    current_advantages = logprob_masks * advantages

    # check old/ref per_token_logprobs
    # [batch_size, max_len]
    assert old_per_token_logprob is not None
    assert isinstance(old_per_token_logprob, torch.Tensor)
    assert (
        old_per_token_logprob.shape == current_per_token_logprobs.shape
    ), f"old_per_token_logprobs.shape: {old_per_token_logprob.shape}, while it should be {current_per_token_logprobs.shape}"

    if ref_per_token_logprob is not None:
        # [batch_size, max_len]
        assert isinstance(ref_per_token_logprob, torch.Tensor)
        assert (
            ref_per_token_logprob.ndim == 2
        ), f"ref_per_token_logprobs.ndim: {ref_per_token_logprob.ndim}, while it should be 2"
        assert (
            ref_per_token_logprob.shape == current_per_token_logprobs.shape
        ), f"ref_per_token_logprobs.shape: {ref_per_token_logprob.shape}, while it should be {current_per_token_logprobs.shape}"

    # TODO(zjx): maybe we can move compute_loss into Trainer.pp_loss_fn ?
    loss, _ = compute_loss(
        current_per_token_logprobs,
        old_per_token_logprob,
        ref_per_token_logprob,
        current_advantages,
        cu_seqlens,
        config,
        logprob_masks,
    )

    return loss.unsqueeze(0) * loss_scaling
