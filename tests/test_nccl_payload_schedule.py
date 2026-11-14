# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Shared-stream progress across peer orderings, batches and callers."""

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations, product
from pathlib import Path
import subprocess
import sys
import threading

import pytest
import torch

from cosmos_rl.utils.payload_transport.nccl import strategy as nccl
from cosmos_rl.utils.payload_transport.nccl.streams import get_receive_lock
from tests.test_receive_memory import make_receiver, refs_for


def can_complete(queues):
    queues = [list(queue) for queue in queues]
    while any(queues):
        heads = {queue[0] for queue in queues if queue}
        ready = {
            transfer
            for transfer in heads
            if all(transfer not in queue or queue[0] == transfer for queue in queues)
        }
        if not ready:
            return False
        for queue in queues:
            if queue and queue[0] in ready:
                queue.pop(0)
    return True


def test_dependency_cycle_includes_individually_sorted_partial_batches():
    ax, ay, bx, by = ("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y")
    sends = [[ay, ax], [bx, by]]
    receives = [[ax, bx], [by, ay]]
    orders = product(
        permutations(sends[0]),
        permutations(sends[1]),
        permutations(receives[0]),
        permutations(receives[1]),
    )
    assert sum(not can_complete(order) for order in orders) == 2
    assert not can_complete(sends + receives)
    # Each one-element partial fetch is sorted, yet the combined stream still
    # cycles. The invariant must hold across fetch boundaries and callers.
    assert not can_complete(
        sends + [sum([sorted([r]) for r in q], []) for q in receives]
    )


@pytest.mark.parametrize("bounded", [False, True])
def test_complete_and_retire_before_next_rendezvous(monkeypatch, bounded):
    receiver, _, seen = make_receiver(monkeypatch)
    rendezvous = receiver._rendezvous_one
    pending = set()
    deadlines = []

    class Completion:
        def query(self):
            pending.clear()
            return True

    class Deadline:
        def __init__(self, seconds, context):
            assert seconds == 30  # unchanged cold budget
            self.closed = False
            deadlines.append(self)

        def remaining_ms(self):
            return 30_000

        def wait_event(self, event, **kwargs):
            if event is not None:
                assert event.query()

        def close(self):
            self.closed = True

    def accept(ref, pynccl):
        assert not pending, "previous receive still occupies the shared stream"
        assert all(deadline.closed for deadline in deadlines)
        result = rendezvous(ref, pynccl)
        pending.add(ref["transfer_id"])
        return result

    monkeypatch.setattr(receiver, "_rendezvous_one", accept)
    monkeypatch.setattr(nccl, "TransportDeadline", Deadline)
    monkeypatch.setattr(nccl, "record_event", lambda stream=None: Completion())
    refs = refs_for([3, 5, 9])
    # Repeated transfers on one pair must keep their original indices/order.
    refs = [(9, refs[2][1]), (2, refs[0][1]), (7, refs[1][1])]
    fetch = receiver._fetch_all if bounded else receiver._fetch_unbounded
    batch, _, _ = fetch(refs)
    assert list(batch) == [9, 2, 7]
    assert seen == [ref["transfer_id"] for _, ref in refs]
    assert all(deadline.closed for deadline in deadlines)
    batch.release()


@pytest.mark.parametrize("separate_strategy", [False, True])
def test_foreground_cannot_accept_during_prefetch_completion(
    monkeypatch, separate_strategy
):
    receiver, _, _ = make_receiver(monkeypatch)
    second = make_receiver(monkeypatch)[0] if separate_strategy else receiver
    receiver._recv_lock = second._recv_lock = None
    # Unbounded path has no per-strategy bounded-fetch lock to hide the race.
    first_accept = receiver._rendezvous_one
    second_accept = second._rendezvous_one
    waiting, release, attempted, accepted = (threading.Event() for _ in range(4))

    def first(ref, pynccl):
        if threading.current_thread().name.startswith("prefetch"):
            return first_accept(ref, pynccl)
        accepted.set()
        return second_accept(ref, pynccl)

    receiver._rendezvous_one = first
    if separate_strategy:

        def other(ref, pynccl):
            accepted.set()
            return second_accept(ref, pynccl)

        second._rendezvous_one = other

    class Deadline:
        def __init__(self, *args):
            self.waited = False

        def remaining_ms(self):
            return 30_000

        def wait_event(self, event, **kwargs):
            if not self.waited and threading.current_thread().name.startswith(
                "prefetch"
            ):
                self.waited = True
                waiting.set()
                assert release.wait(3)

        def close(self):
            pass

    def foreground():
        attempted.set()
        return second._fetch_unbounded(refs_for([5]))

    monkeypatch.setattr(nccl, "TransportDeadline", Deadline)
    with ThreadPoolExecutor(1, thread_name_prefix="prefetch") as background:
        future = background.submit(receiver._fetch_unbounded, refs_for([3]))
        try:
            assert waiting.wait(2)
            with ThreadPoolExecutor(1) as foreground_pool:
                other = foreground_pool.submit(foreground)
                try:
                    assert attempted.wait(2)
                    assert not accepted.wait(0.1)
                finally:
                    release.set()
                other.result(3)[0].release()
            future.result(3)[0].release()
        finally:
            release.set()


def test_receive_device_aliases_share_schedule(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert get_receive_lock(3) is get_receive_lock("cuda:3")
    assert get_receive_lock(torch.device("cuda", 3)) is get_receive_lock("cuda")
    assert get_receive_lock(3) is not get_receive_lock(2)


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA devices")
def test_four_gpu_adversarial_payload_schedule():
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("nccl_payload_schedule_canary.py")),
        ],
        check=True,
        timeout=300,
    )
