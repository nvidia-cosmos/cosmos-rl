# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""torchrun --standalone --nproc-per-node=2 tests/state_tensor_canary.py"""

from datetime import timedelta
import os

import numpy as np
import torch
import torch.distributed as dist

from cosmos_rl.policy.trainer.llm_trainer import llm_trainer as module
from test_p2p_batching import _state_sync_trainer


class Broadcast:
    supports_packing = True

    def __call__(self, tensor):
        assert tensor.is_cuda
        # Cosmos P2P transports raw bytes. ProcessGroupNCCL does not directly
        # support every fixture dtype (e.g. int16), so mirror that byte contract.
        dist.broadcast(tensor.view(-1).view(torch.uint8), src=0)

    def batch(self, tensors):
        for tensor in tensors:
            self(tensor)


def main():
    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=timedelta(seconds=60))
    assert dist.get_world_size() == 2
    try:
        for packed in (False, True):
            module._P2P_SYNC_BUCKET_SIZE_BYTES = 64
            module._P2P_SYNC_PACK_TENSORS = packed
            trainer = _state_sync_trainer(7 if rank == 0 else 0)
            trainer.device = torch.device("cuda", local)
            owner = np.full((3, 4), 7 if rank == 0 else 0, dtype=np.float32)
            trainer.optimizers.state["array"] = owner[:, ::2]
            trainer.lr_schedulers.state["array"] = np.full(
                (), 7 if rank == 0 else 0, dtype=np.float64
            )
            rng = np.random.RandomState(173 if rank == 0 else 91)
            rng.normal(size=7)
            trainer.ckpt_manager.state["numpy"] = rng.get_state()
            alias = trainer.optimizers.state["array"]
            hook = Broadcast()
            trainer.sync_all_states(rank == 0, hook, hook)
            assert trainer.optimizers.state["array"] is alias
            np.testing.assert_array_equal(owner[:, ::2], 7)
            np.testing.assert_array_equal(owner[:, 1::2], 7 if rank == 0 else 0)
            np.testing.assert_array_equal(trainer.lr_schedulers.state["array"], 7)
            received = trainer.ckpt_manager.state["numpy"]
            assert received[1].dtype == np.uint32
            rng.set_state(received)
            expected = np.random.RandomState(173)
            expected.normal(size=7)
            np.testing.assert_array_equal(rng.normal(size=16), expected.normal(size=16))
            for tensor in trainer.model.state_dict().values():
                torch.testing.assert_close(tensor, torch.full_like(tensor, 7))
            print(f"STATE_TENSOR_NATIVE_PASS rank={rank} packed={packed}", flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
