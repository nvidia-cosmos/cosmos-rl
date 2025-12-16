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

from cosmos_rl.policy.worker.base import PolicyWorkerBase
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


class TeacherWorker(PolicyWorkerBase):
    def update_config(self, config: CosmosConfig):
        # Update train config to reference config to reuse the logic of policy trainer
        config.train.seed = config.reference.seed
        config.train.compile = config.reference.compile
        config.train.master_dtype = config.reference.master_dtype
        config.train.param_dtype = config.reference.param_dtype
        config.train.logprob_dtype = config.reference.logprob_dtype
        config.train.fsdp_reduce_dtype = config.reference.fsdp_reduce_dtype
        config.train.fsdp_offload = config.reference.fsdp_offload
        config.train.fsdp_reshard_after_forward = (
            config.reference.fsdp_reshard_after_forward
        )
        config.policy.model_name_or_path = config.reference.model_name_or_path
        config.policy.model_max_length = config.reference.model_max_length
        config.policy.model_revision = config.reference.model_revision
        config.policy.parallelsim = config.reference.parallelsim
        return config

    def __init__(self, config: CosmosConfig, parallel_dims: ParallelDims, **kwargs):
        config = self.update_config(config)
        assert isinstance(
            config, CosmosConfig
        ), "config must be a CosmosConfig object for this trainer"
        super().__init__(config, parallel_dims=parallel_dims)

        # For hooks and custom logger functions
        self.custom_logger_fns = kwargs.get("custom_logger_fns", [])
        self.hook_fns = kwargs.get("hook_fns", {})

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
        while not self.shutdown_signal.is_set():
            teacher_requests = []
            try:
                teacher_requests = self.redis_controller.subscribe_teacher_request(
                    self.replica_name, count=self.engine.batch_size
                )
            except Exception as e:
                logger.debug(f"Failed to get rollouts: {e}, wait for next round")
            for rollout in teacher_requests:
                self.data_queue.put_nowait(rollout)

    def dispatch_rollouts(self) -> List[Rollout]:
        def preprocess_rollouts(rollouts: List[Dict]):
            updated_rollouts: List[Rollout] = []
            for i in range(len(rollouts)):
                if "is_end" in rollouts[i]:
                    self.end_event.set()
                    continue
                updated_rollouts.append(
                    Rollout(
                        prompt=self.data_fetcher.get_payload_by_index(
                            rollouts[i]["prompt_idx"]
                        ),
                        prompt_idx=rollouts[i]["prompt_idx"],
                        uuid=rollouts[i]["uuid"],
                        completion_token_ids=rollouts[i]["completion_token_ids"],
                    )
                )
            return updated_rollouts

        rollouts = [[]]
        scattered_rollouts = [[] for _ in range(self.world_size)]
        if self.global_rank == 0:
            batch_for_this_step = self.engine.batch_size
            assert batch_for_this_step % self.dp_world_size == 0
            dp_id = 0
            for _ in range(batch_for_this_step):
                try:
                    rollout = self.data_queue.get(block=True, timeout=None)
                    if "is_end" in rollout:
                        for i in range(self.world_size):
                            scattered_rollouts[i].append(rollout)
                        break
                except Empty:
                    raise Empty(
                        "[Policy] Rollouts queue is empty, please check the dispatcher."
                    )
                for i in range(self.world_size):
                    if self.parallel_dims.get_rank_in_dim("dp", i) == dp_id:
                        scattered_rollouts[i].append(rollout)
                        # logger.info(f"[Policy] Rollout {dp_id} dispatched to rank {i}, dp world_size {self.dp_world_size}")
                dp_id += 1
                if dp_id >= self.dp_world_size:
                    dp_id = 0
            for i in range(self.world_size):
                assert (
                    len(scattered_rollouts[i]) == len(scattered_rollouts[0])
                ), f"Rank {i} has {len(scattered_rollouts[i])} rollouts, but rank 0 has {len(scattered_rollouts[0])} rollouts"
        if self.world_size == 1:
            data = preprocess_rollouts(scattered_rollouts[0])

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
            rollouts = self.dispatch_rollouts()
            data = self.engine.step_forward(rollouts)
            for item in data:
                id = item.pop("uuid")
                self.redis_controller.set_teacher_result(id, item, self.replica_name)
            if self.end_event.is_set():
                break
        logger.info("[Policy] Main loop finished. Shutdown background task event set.")
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
            train_stream=self.teacher_stream,
            data_packer=self.data_packer,
        )

    def destroy_worker(self):
        destroy_distributed()
        logger.info("[Reference] Process group destroyed.")
