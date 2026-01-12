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

import time
from cosmos_rl.reference.engine.torch_engine import TorchEngine
import atexit
import asyncio
import threading
from typing import List, Optional, Union, Callable, Dict
from torch.utils.data import Dataset
from queue import Queue
import torch.distributed as dist
from queue import Empty
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker
from cosmos_rl.dispatcher.data.data_fetcher import WorkerDataFetcher
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.utils.distributed import destroy_distributed
import copy
from cosmos_rl.utils.util import (
    setup_tokenizer,
)
from cosmos_rl.policy.worker.base import PolicyWorkerBase
from cosmos_rl.dispatcher.protocol import Role


class TeacherWorker(PolicyWorkerBase):
    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        # parallel_dims is built from distillation parallelism config
        config = self.update_config(config)
        assert isinstance(
            config, CosmosConfig
        ), "config must be a CosmosConfig object for this trainer"

        kwargs["role"] = Role.REFERENCE

        super(TeacherWorker, self).__init__(
            config, parallel_dims=parallel_dims, **kwargs
        )

        self.student_tokenizer = setup_tokenizer(config.policy.model_name_or_path)

        # Initialize the teacher
        dataset = kwargs.get("dataset", None)
        data_packer = kwargs.get("data_packer", None)
        self.build_runner(
            dataset=dataset,
            data_packer=data_packer,
        )

        # For rollouts fetch
        self.data_queue = Queue()
        self.fetch_rollouts_thread = None
        self.end_event = threading.Event()

        atexit.register(self.handle_shutdown)

    def check_config(self):
        assert self.config.distillation.enable, "Distillation must be enabled"
        assert (
            self.config.distillation.batch_size_per_replica > 0
        ), "Batch size per replica must be greater than 0"
        assert (
            self.config.distillation.parallelism.dp_shard_size > 0
        ), "DP shard size must be greater than 0"
        assert (
            self.config.distillation.parallelism.dp_replicate_size == 1
        ), "DP replicate size must be 1"
        dp_shard_size = self.config.distillation.parallelism.dp_shard_size
        assert dp_shard_size == self.parallel_dims.dp_shard
        assert (
            self.config.distillation.batch_size_per_replica % dp_shard_size == 0
        ), "Batch size per replica must be divisible by DP shard size"
        logger.info("[Reference] Config checked successfully")

    def execute(self):
        """
        Execute the training.
        """
        assert self.engine is not None, "[Reference] Engine has not been built."
        try:
            self.main_loop()
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise e
        finally:
            self.destroy_worker()

    def update_config(self, config: CosmosConfig):
        # Update train config to reference config to reuse the logic of policy trainer
        config.train.seed = config.distillation.seed
        config.train.compile = config.distillation.compile
        config.train.master_dtype = config.distillation.master_dtype
        config.train.param_dtype = config.distillation.param_dtype
        config.train.logprob_dtype = config.distillation.logprob_dtype
        config.train.fsdp_reduce_dtype = config.distillation.fsdp_reduce_dtype
        config.train.fsdp_offload = config.distillation.fsdp_offload
        config.train.fsdp_reshard_after_forward = (
            config.distillation.fsdp_reshard_after_forward
        )
        config.policy.model_name_or_path = config.distillation.model_name_or_path
        config.policy.model_revision = config.distillation.model_revision
        config.policy.parallelism = config.distillation.parallelism
        return config

    def setup(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[BaseDataPacker] = None,
    ):
        # setup data packer first
        self.init_data_packer(
            data_packer=data_packer,
        )
        # Set up data fetcher
        self.data_fetcher = WorkerDataFetcher(
            config=self.config,
            dataset=dataset,
            data_packer=self.data_packer,
            val_data_packer=None,
            is_rl=True,
        )

    def handle_shutdown(self):
        if not hasattr(self, "_handle_shutdown_called"):
            self._handle_shutdown_called = True

            self.shutdown_signal.set()
            self.shutdown_mp_signal.set()
            if self.fetch_rollouts_thread is not None:
                self.fetch_rollouts_thread.join()
                self.fetch_rollouts_thread = None

            if hasattr(self, "heartbeat_thread") and self.heartbeat_thread is not None:
                self.heartbeat_thread.join()
                self.heartbeat_thread = None

            # Manually unregister from controller
            self.unregister_from_controller()

    async def fetch_rollouts(self):
        assert self.global_rank == 0, "Only rank 0 can fetch rollouts"
        running = True
        while running:
            teacher_requests = []
            logger.debug("[Reference] Fetching rollouts from redis")
            try:
                teacher_requests = self.redis_controller.subscribe_teacher_request(
                    self.replica_name, count=self.engine.batch_size
                )
                logger.debug(
                    f"[Reference] Fetched {len(teacher_requests)} rollouts from redis"
                )
            except Exception as e:
                logger.debug(
                    f"[Reference] Failed to get rollouts: {e}, wait for next round"
                )
            for rollout in teacher_requests:
                assert (
                    len(rollout["teacher_result_uuid"])
                    == len(rollout["completion_token_ids"])
                ), "Number of teacher result uuids and completion token ids must be the same"
                for tokens, uuid in zip(
                    rollout["completion_token_ids"], rollout["teacher_result_uuid"]
                ):
                    logger.debug(
                        f"[Reference] Putting rollout with uuid {uuid} into data queue"
                    )
                    rollout_item = copy.deepcopy(rollout)
                    rollout_item["completion_token_ids"] = tokens
                    rollout_item["teacher_result_uuid"] = uuid
                    self.data_queue.put_nowait(rollout_item)
                if "is_end" in rollout:
                    logger.info("[Reference] Exiting fetch rollouts")
                    self.data_queue.put_nowait({"is_end": True})
                    running = False

    def dispatch_rollouts(self) -> List[Rollout]:
        def preprocess_rollouts(rollouts: List[Dict]):
            updated_rollouts: List[Rollout] = []
            for i in range(len(rollouts)):
                if "is_end" in rollouts[i]:
                    logger.info("[Reference] Setting end event")
                    self.end_event.set()
                    continue
                updated_rollouts.append(
                    Rollout(
                        prompt=self.data_fetcher.get_payload_by_index(
                            rollouts[i]["prompt_idx"]
                        ),
                        prompt_idx=rollouts[i]["prompt_idx"],
                        teacher_result_uuid=rollouts[i]["teacher_result_uuid"],
                        completion_token_ids=rollouts[i]["completion_token_ids"],
                    )
                )
            return updated_rollouts

        rollouts = [[]]
        scattered_rollouts = [[] for _ in range(self.world_size)]
        if self.global_rank == 0:
            batch_for_this_step = None
            assert (
                self.engine.batch_size
                >= self.dp_world_size * self.config.distillation.mini_batch
            ), (
                f"Batch size {self.engine.batch_size} must be greater than or equal to "
                f"dp_world_size {self.dp_world_size} * mini_batch {self.config.distillation.mini_batch}"
            )
            while batch_for_this_step is None or batch_for_this_step <= 0:
                time.sleep(0.1)
                current_queue_size = self.data_queue.qsize()
                if current_queue_size >= self.engine.batch_size:
                    batch_for_this_step = self.engine.batch_size
                else:
                    batch_for_this_step = current_queue_size - (
                        current_queue_size
                        % (self.dp_world_size * self.config.distillation.mini_batch)
                    )
                    if batch_for_this_step <= 0 and current_queue_size > 0:
                        batch_for_this_step = current_queue_size
            dp_id = 0
            for _ in range(batch_for_this_step):
                try:
                    rollout = self.data_queue.get(block=True, timeout=None)
                    if "is_end" in rollout:
                        for i in range(self.world_size):
                            logger.info(
                                f"[Reference] Appending end rollout to rank {i}"
                            )
                            scattered_rollouts[i].append(rollout)
                        break
                except Empty:
                    raise Empty(
                        "[Policy] Rollouts queue is empty, please check the dispatcher."
                    )
                for i in range(self.world_size):
                    if self.parallel_dims.get_rank_in_dim("dp", i) == dp_id:
                        scattered_rollouts[i].append(rollout)
                dp_id += 1
                if dp_id >= self.dp_world_size:
                    dp_id = 0
            # Pad rollouts to mini_batch size
            for i in range(self.world_size):
                if len(scattered_rollouts[i]) < self.config.distillation.mini_batch:
                    for _ in range(
                        self.config.distillation.mini_batch - len(scattered_rollouts[i])
                    ):
                        scattered_rollouts[i].append(scattered_rollouts[0][0])
            for i in range(self.world_size):
                assert (
                    len(scattered_rollouts[i]) == len(scattered_rollouts[0])
                ), f"Rank {i} has {len(scattered_rollouts[i])} rollouts, but rank 0 has {len(scattered_rollouts[0])} rollouts"
        if self.world_size == 1:
            data = preprocess_rollouts(scattered_rollouts[0])
        logger.debug(
            f"[Reference] Preprocessed rollouts, global rank {self.global_rank} number of rollouts: {[len(scattered_rollouts[i]) for i in range(self.world_size)]}"
        )
        dist.scatter_object_list(
            rollouts,
            scattered_rollouts,
            src=0,
        )
        rollouts = rollouts[0]
        data = preprocess_rollouts(rollouts)
        return data

    def main_loop(self):
        self.engine.model_load_from_hf()

        def fetch_rollouts_helper(trainer):
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            new_loop.run_until_complete(trainer.fetch_rollouts())
            new_loop.stop()
            new_loop.close()
            return

        if self.global_rank == 0:
            self.fetch_rollouts_thread = threading.Thread(
                target=fetch_rollouts_helper,
                args=(self,),
                daemon=True,
                name="fetch_rollouts_thread",
            ).start()

        while True:
            if self.end_event.is_set():
                logger.info("[Reference] End event set, breaking loop")
                break
            rollouts = self.dispatch_rollouts()
            if len(rollouts) == 0:
                continue
            data = self.engine.step_forward(rollouts)
            for item in data:
                id = item.pop("teacher_result_uuid")
                logger.debug(f"[Reference] Setting teacher result for uuid {id}")
                if not self.redis_controller.set_teacher_result(
                    id, item, self.replica_name
                ):
                    logger.error(f"[Reference] Failed to set teacher result {id}")
                logger.debug(f"[Reference] Teacher result set for uuid {id}")
        logger.info(
            "[Reference] Main loop finished. Shutdown background task event set."
        )
        self.train_stream.synchronize()
        self.handle_shutdown()

    def build_runner(
        self,
        dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
        data_packer: Optional[BaseDataPacker] = None,
    ):
        # Initialize data packer and setup data fetcher first.
        self.setup(
            dataset=dataset,
            data_packer=data_packer,
        )
        self.engine = TorchEngine(
            self.config,
            self.parallel_dims,
            device=self.device,
            train_stream=self.train_stream,
            data_packer=self.data_packer,
            student_tokenizer=self.student_tokenizer,
        )

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Reference] Process group destroyed.")
