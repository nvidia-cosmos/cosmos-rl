"""Real collective ordering with an independent toy backend; no VLA forward claim.

torchrun --standalone --nproc-per-node=2 tests/empty_rollout_batch_canary.py [--cuda]
The same entrypoint supports multi-node torchrun. Gloo carries commands/prompts;
the optional toy forward computes on CUDA but has no cross-rank collectives.
"""

import argparse
from datetime import timedelta
import os
from queue import Queue
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.distributed as dist

from cosmos_rl.rollout import State
from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker as Worker,
)
from cosmos_rl.utils.distributed import broadcast_object_cpu


def run_case(prefetch, supported, device):
    rank, size = dist.get_rank(), dist.get_world_size()
    assert size == 2, "This canary requires exactly two DP slices"
    worker = object.__new__(Worker)
    worker.global_rank = rank
    worker.replica_name = "empty-slot-canary"
    worker.rank_in_rollout_repicas = 0
    worker._prompt_fetch_lock = threading.Lock()
    worker._prompt_queue = Queue(maxsize=2)
    worker.parallel_dims = SimpleNamespace(
        world_size=size,
        mesh={
            "dp": SimpleNamespace(
                size=lambda: size,
                get_local_rank=lambda: rank,
                get_group=lambda: dist.group.WORLD,
            )
        },
    )
    worker.config = SimpleNamespace(
        train=SimpleNamespace(
            local_dataset=False,
            train_policy=SimpleNamespace(
                data_dispatch_as_rank_in_mesh=False, allowed_outdated_steps=0
            ),
        ),
        rollout=SimpleNamespace(
            prefetch_rollout=prefetch,
            prefetch_queue_maxsize=2,
            async_r2r_sync="disabled",
        ),
    )
    fetched, generated, prepared, commands = [], [], [], []

    def fetch(*args, **kwargs):
        assert rank == 0
        fetched.append(True)
        if len(fetched) == 1:
            return [dict(prompt_idx=10, prompt="10", weight_version=3)], False
        assert len(fetched) == 2, "duplicate fetch"
        return [
            dict(prompt_idx=i, prompt=str(i), weight_version=v)
            for i, v in ((20, 4), (21, 7), (22, 4))
        ], True

    def forward(payloads, **kwargs):
        assert payloads, "empty rank entered forward"
        for payload in payloads:
            result = torch.tensor(payload.prompt_idx, device=device) * 2
            assert result.item() == payload.prompt_idx * 2
            generated.append((payload.prompt_idx, worker.current_weight_version))
        return payloads

    worker.rollout = SimpleNamespace(
        supports_empty_dp_batches=supported,
        rollout_generation=forward,
        submit_setup=lambda payloads: prepared.extend(p.prompt_idx for p in payloads),
    )
    worker.api_client = SimpleNamespace(get_next_prompt=fetch)
    worker.state = State()
    worker.state.set_weight_synced()
    worker.current_weight_version = 0
    worker.shutdown_signal = threading.Event()
    worker.validation_flag = threading.Event()
    worker._is_async_rollout = False
    worker.batch_size = 2
    worker.should_report = False
    worker._bind_prefetch_context_once = Mock()
    worker._maybe_emit_mainloop_summary = Mock()
    worker.report_rollouts = Mock(return_value=(None, False, None, True))

    def consume(**kwargs):
        turn = broadcast_object_cpu(len(commands) + 1 if rank == 0 else None)
        commands.append(turn)
        worker.current_weight_version = 0 if turn == 1 else 3 if turn == 2 else 7
        if turn == 6:
            worker.shutdown_signal.set()

    worker.consume_command = consume
    worker.one_step_generation = lambda: worker._call_rollout_generation(
        payloads=worker._prompt_queue.get_nowait()
    )
    rejected = False
    try:
        worker._main_loop_impl()
    except ValueError as error:
        assert "empty DP slices" in str(error)
        rejected = True
    reports = [None] * size
    dist.all_gather_object(reports, (rejected, generated, prepared, commands))
    if not supported:
        assert all(
            rejected and not generated and not prepared
            for rejected, generated, prepared, _ in reports
        ), reports
    else:
        assert not rejected and worker._prompt_queue.empty()
        assert reports[0][3] == reports[1][3] == list(range(1, 7)), reports
        assert sorted(item for _, output, _, _ in reports for item in output) == [
            (10, 3),
            (20, 7),
            (21, 7),
            (22, 7),
        ]
        if prefetch:
            assert sorted(i for _, _, output, _ in reports for i in output) == [
                10,
                20,
                21,
                22,
            ]
    print(
        f"EMPTY_ROLLOUT_PASS rank={rank} prefetch={prefetch} supported={supported} device={device}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()
    device = torch.device("cpu")
    if args.cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo", timeout=timedelta(seconds=90))
    try:
        for prefetch in (False, True):
            for supported in (False, True):
                run_case(prefetch, supported, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
