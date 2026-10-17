# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Four ranks, two policy replicas: native readiness and optimizer participation.

Run with torchrun --nproc-per-node=4 (or two nodes with two ranks each).
TCPStore is used for bootstrap. --cpu is a Gloo control, never a GPU substitute.
This probes lifecycle coordination, not the NCCL/UCXX payload wire protocol.
"""

import argparse
from datetime import timedelta
import os
import threading
from types import SimpleNamespace
from queue import Queue

import torch
import torch.distributed as dist

from cosmos_rl.policy.trainer import prefetch
from cosmos_rl.utils import distributed as dist_util
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.dispatcher.command import PayloadPrefetchCommand
from cosmos_rl.dispatcher.data.schema import Rollout
from test_trainer_payload_prefetch import Packer, cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    device = torch.device("cpu")
    if not args.cpu:
        assert torch.cuda.is_available(), "Required GPU control cannot skip"
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
    local_ranks = ([0, 1], [2, 3])
    cross_ranks = ([0, 2], [1, 3])
    locals_ = [dist.new_group(ranks, backend="gloo") for ranks in local_ranks]
    crosses = [dist.new_group(ranks, backend="gloo") for ranks in cross_ranks]
    local = locals_[rank // 2]
    cross = crosses[rank % 2]
    native = None

    def local_max(tensor, op):
        dist.all_reduce(tensor, op=op, group=local)
        return tensor

    original = dist_util.all_reduce_tensor_object_cpu
    dist_util.all_reduce_tensor_object_cpu = local_max
    comm = SimpleNamespace(
        allreduce=lambda send, recv, op: dist.all_reduce(recv, op=op, group=cross)
    )
    try:
        if not args.cpu:
            from cosmos_rl.utils.distributed import HighAvailabilitylNccl
            from cosmos_rl.utils.pynccl import create_nccl_comm, create_nccl_uid

            ids = [create_nccl_uid() if rank < 2 else None]
            dist.broadcast_object_list(ids, src=rank % 2, group=cross)
            native = HighAvailabilitylNccl(
                f"replica-{rank // 2}",
                rank % 2,
                SimpleNamespace(post_nccl_comm_error=lambda *args: None),
            )
            native.replica_name_to_rank = {"replica-0": 0, "replica-1": 1}
            native.default_timeout_ms = 30000
            native.max_retry = 1
            native.is_comm_ready.set()
            native.comm_idx = create_nccl_comm(ids[0], rank // 2, 2, timeout_ms=30000)
            comm = native
        # Exercise the real background TCPStore path within each two-rank
        # policy, independently of both local and cross-replica collectives.
        for scenario in ("healthy", "duplicate", "wrong-session", "missing-peer"):
            packer = Packer()
            packer._setup_prefetch(prefetch_timeout=5)
            pipeline = prefetch.TrainerPayloadPrefetch(packer)
            instance = RLPolicyWorker.__new__(RLPolicyWorker)
            instance.payload_prefetch = pipeline
            instance.data_packer = packer
            instance.device, instance.inter_policy_nccl = device, comm
            instance.replica_name = f"replica-{rank // 2}"
            instance.global_rank, instance.world_size, instance.dp_world_size = (
                rank % 2,
                2,
                2,
            )
            instance.api_client = SimpleNamespace(_report_session_id=f"session-{rank}")
            instance.parallel_dims = SimpleNamespace(get_rank_in_dim=lambda _dim, r: r)
            pipeline.take(cmd(1), lambda: ["current"])
            store = dist_util.DistKVStore.__new__(dist_util.DistKVStore)
            store.rank, store.world_size, store.counter = rank % 2, 2, 0
            store.shutdown_event = threading.Event()
            store.local_store = dist.PrefixStore(
                f"prefetch-{scenario}-{rank // 2}",
                dist.distributed_c10d._get_default_store(),
            )
            cohort = [
                (
                    f"replica-{replica}",
                    [
                        (local_rank, f"session-{2 * replica + local_rank}")
                        for local_rank in range(2)
                    ],
                )
                for replica in range(2)
            ]
            if scenario == "wrong-session":
                cohort[1][1][1] = (1, "restarted")
            notice = PayloadPrefetchCommand(
                instance.replica_name,
                f"notification-{scenario}",
                2,
                cohort,
                [
                    Rollout(prompt_idx=i, completion=f"future-{i}").model_dump()
                    for i in range(2)
                ],
            )
            errors = []

            def receive():
                try:
                    delivered = store.broadcast_command_bounded(
                        notice if rank % 2 == 0 else None, timeout_s=1
                    )
                    instance.receive_payload_notification(delivered)
                    if scenario == "duplicate":
                        instance.receive_payload_notification(delivered)
                except Exception as error:
                    errors.append(error)

            thread = None
            if not (scenario == "missing-peer" and rank == 3):
                thread = threading.Thread(target=receive)
                thread.start()
                thread.join(4)
                assert not thread.is_alive()
            try:

                def ready_notification():
                    if errors:
                        raise errors[0]
                    if pipeline.pending is None:
                        raise ValueError("Missing notification recipient")
                    pipeline.complete(1)
                    rows, _ = pipeline.take(cmd(2, current=notice.batch_id), lambda: ())
                    assert len(rows) == 1 and rows[0].prompt_idx == rank % 2

                failed = False
                try:
                    prefetch.cohort_payload_call(instance, ready_notification)
                except RuntimeError:
                    failed = True
                assert failed == (scenario in ("wrong-session", "missing-peer"))
                print(
                    f"PREFETCH_NOTIFICATION_COHORT_PASS rank={rank} scenario={scenario} failed={failed}",
                    flush=True,
                )
            finally:
                pipeline.stop_notifications()
                store.shutdown_event.set()
                packer.shutdown_prefetch()
            dist.barrier()
        for scenario in (
            "healthy",
            "missing",
            "malformed",
            "fetch-error",
            "wrong-identity",
            "drain-pending",
            "healthy-after",
        ):
            packer = Packer()
            packer._setup_prefetch(prefetch_timeout=5)
            pipeline = prefetch.TrainerPayloadPrefetch(packer)
            worker = SimpleNamespace(
                device=device,
                inter_policy_nccl=comm,
                payload_prefetch=pipeline,
            )
            parameter = torch.nn.Parameter(torch.ones((), device=device))
            optimizer = torch.optim.SGD([parameter], lr=0.1, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=1, gamma=0.9
            )
            before = scheduler.last_epoch

            def ready():
                if rank == 3:
                    if scenario in ("missing", "malformed"):
                        metadata = SimpleNamespace(
                            global_rank=0,
                            world_size=1,
                            dp_world_size=1,
                            replica_batch_for_this_step=1,
                            data_queue=Queue(),
                            data_packer=SimpleNamespace(_prefetch_timeout_s=0.01),
                        )
                        if scenario == "malformed":
                            metadata.data_queue.put("not a Rollout")
                        RLPolicyWorker._dispatch_prefetch_rollouts(metadata)
                    if scenario == "fetch-error":

                        def fail_fetch(tasks):
                            raise ValueError("injected decoder error")

                        packer._fetch_batch = fail_fetch
                    if scenario in ("wrong-identity", "drain-pending"):
                        pipeline.pending = prefetch.PendingPayloadBatch(
                            "owned", 2, (), None
                        )
                        if scenario == "drain-pending":
                            pipeline.drain()
                        pipeline.validate(cmd(2, current="wrong"))
                pipeline.take(cmd(1), lambda: ["sample"])
                return True

            failed = False
            try:
                with prefetch.payload_cohort_scope(worker):
                    prefetch.cohort_payload_call(worker, ready)
            except RuntimeError as error:
                assert "no update ACK" in str(error)
                failed = True
            finally:
                packer.shutdown_prefetch()
            if failed:
                assert scenario not in ("healthy", "healthy-after")
                assert parameter.item() == 1 and optimizer.state == {}
                assert scheduler.last_epoch == before and worker.payload_prefetch.failed
            else:
                assert scenario in ("healthy", "healthy-after")
                (parameter * 2).backward()
                # The same real cross-replica collective is reached everywhere.
                comm.allreduce(parameter.grad, parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(2)
                optimizer.step()
                scheduler.step()
                torch.testing.assert_close(parameter, torch.tensor(0.8, device=device))
                assert scheduler.last_epoch == before + 1
            print(
                f"PREFETCH_COHORT_PASS rank={rank} scenario={scenario} failed={failed}",
                flush=True,
            )
        dist.barrier()
    finally:
        dist_util.all_reduce_tensor_object_cpu = original
        if native is not None:
            from cosmos_rl.utils.pynccl import nccl_abort

            native.shutdown()
            nccl_abort(native.comm_idx)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
