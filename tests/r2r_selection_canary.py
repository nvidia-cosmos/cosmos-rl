# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real synchronous R2R selection, plus a local pre-native rejection control.

Run with two torchrun ranks, on one or two nodes. This does not claim distributed
failure propagation: the invalid-state arm forbids native entry on both ranks.
"""

import argparse
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist

from cosmos_rl.rollout.worker import rollout_control as control
from cosmos_rl.utils.pynccl import nccl_abort
from test_r2r_command_selection import command, worker
from weight_adoption_canary import initialize_transport


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", action="store_true")
    args = parser.parse_args()
    rank, device, comm = initialize_transport()
    instance, parameters = worker(synced=rank == 0, frozen_received=rank == 0)
    instance.replica_name = "rollout-a" if rank == 0 else "rollout-b"
    instance.rank_in_rollout_repicas = rank
    instance.global_commnicator_idex = comm
    instance.device = device
    instance.inference_stream = torch.cuda.Stream()
    instance.config.rollout.r2r_sync_pack_tensors = args.packed
    instance.config.rollout.r2r_sync_bucket_size_bytes = 1024
    model = torch.nn.Module()
    model.register_parameter(
        "trainable", torch.nn.Parameter(torch.zeros(2, device=device))
    )
    model.register_buffer("frozen", torch.zeros(3, device=device))
    instance.rollout.get_underlying_model = lambda: model
    parameters.update(model.state_dict())
    try:
        with torch.no_grad():
            for step, full_path, trainable_only in (
                (1, False, False),
                (2, False, True),
                (3, True, True),
                (4, False, False),
            ):
                instance.config.rollout.broadcast_all_params = full_path
                if full_path:
                    # Full-state receipt must be recorded even if the command
                    # hint is trainable-only and the local sticky flag is false.
                    instance.non_trainable_params_received = False
                if rank == 0:
                    parameters["trainable"].fill_(step)
                    if not trainable_only or full_path:
                        parameters["frozen"].fill_(10 * step)
                torch.cuda.synchronize()
                dist.barrier()
                update = command(trainable_only=trainable_only)
                update.weight_step = step
                instance._execute_rollout_broadcast(update)
                instance.inference_stream.synchronize()
                torch.testing.assert_close(
                    parameters["trainable"],
                    torch.full_like(parameters["trainable"], step),
                )
                frozen_step = 1 if step == 2 else step
                torch.testing.assert_close(
                    parameters["frozen"],
                    torch.full_like(parameters["frozen"], 10 * frozen_step),
                )
                assert instance.non_trainable_params_received
                assert instance.current_weight_version == step
                dist.barrier()

        instance.config.rollout.broadcast_all_params = False
        instance.non_trainable_params_received = False
        native = Mock(side_effect=AssertionError("invalid state entered native R2R"))
        with patch.object(control, "do_nccl_broadcast_tensors", native):
            try:
                instance._execute_rollout_broadcast(command(trainable_only=True))
                raise AssertionError("invalid state was accepted")
            except RuntimeError as error:
                assert "previously received full weights" in str(error)
        native.assert_not_called()
        assert instance.current_weight_version == 4
        dist.barrier()
        print(
            f"R2R_SELECTION_PASS rank={rank} packed={args.packed} "
            "updates=4 invalid_state_native_calls=0",
            flush=True,
        )
    finally:
        nccl_abort(comm)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
