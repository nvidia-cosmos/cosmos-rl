# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Four-GPU shared-stream cycle witness and native receiver progress gate.

FileStore supplies deterministic acceptance/launch scheduling; payloads use
real pair NCCL communicators and the production receive/identity/lease path.
The negative control deliberately queues both receives before completion,
matching the previous scheduling rule. No failed native owner is cleaned up.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist

from cosmos_rl.utils import pynccl
from cosmos_rl.utils.payload_transport.layout import TensorLayout
from cosmos_rl.utils.payload_transport.nccl.header import HEADER_NBYTES, build_header
from cosmos_rl.utils.payload_transport.nccl.strategy import (
    NCCLTransportStrategy,
    _PreparedReceive,
)
from cosmos_rl.utils.payload_transport.nccl.streams import (
    get_transfer_stream_pool,
    record_event,
    wait_event,
)
from cosmos_rl.utils.payload_transport.receive_memory import (
    ReceiveBudget,
    ReceiveBufferPool,
)
from cosmos_rl.utils.trajectory import TensorSpec
from cosmos_rl.utils.transport_failure import TransportDeadline


def wait_keys(store, keys):
    deadline = time.monotonic() + 15
    while not store.check(keys):
        assert time.monotonic() < deadline, f"Control schedule stalled: {keys}"
        time.sleep(0.001)


def identity(sender, receiver, sequence):
    return f"{sender}:{receiver}-{sequence}"


def value(sender, receiver, sequence):
    return 32 * sender + 8 * receiver + sequence


def worker(args):
    rank = int(os.environ["RANK"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    store = dist.FileStore(args.store, 4)
    pairs = {}
    for sender in (0, 1):
        for receiver in (2, 3):
            uid = [pynccl.create_nccl_uid() if rank == sender else None]
            dist.broadcast_object_list(uid, src=sender)
            if rank in (sender, receiver):
                pairs[sender, receiver] = pynccl.create_nccl_comm(
                    uid[0],
                    0 if rank == sender else 1,
                    2,
                    timeout_ms=10_000,
                )
            dist.barrier()
    layout = TensorLayout.compile(
        [TensorSpec(name="sample", shape=(args.nbytes,), dtype=np.uint8)]
    )
    frame_bytes = HEADER_NBYTES + layout.payload_nbytes
    # Communicator creation does not connect every data channel. A first P2P
    # launch can block the host waiting for its peer before it returns an
    # enqueue. Warm the actual message size in matching pair order so the
    # adversarial schedule exercises the steady-state shared-stream cycle.
    warm_buffer = torch.zeros(frame_bytes, dtype=torch.uint8, device=device)
    warm_stream = get_transfer_stream_pool(device=device).acquire()
    wait_event(warm_stream, record_event())
    for sender in (0, 1):
        for receiver in (2, 3):
            if rank in (sender, receiver):
                operation = TransportDeadline(30, f"warm {sender}:{receiver}")
                native = pynccl.nccl_send if rank == sender else pynccl.nccl_recv
                native(
                    warm_buffer,
                    1 if rank == sender else 0,
                    pairs[sender, receiver],
                    stream=warm_stream,
                    timeout_ms=operation.remaining_ms(),
                )
                operation.wait_event(record_event(warm_stream))
                operation.close()
            dist.barrier()
    sequences = (
        16 if args.case == "benchmark" else 4 if args.case == "concurrent" else 2
    )

    def key(sender, receiver, sequence):
        return f"accepted/{sender}/{receiver}/{sequence}"

    # Each receiver advertises only requests it has actually accepted. Wait
    # until both enter their first device-completion wait before send launch.
    # The old schedule has four accepted transfers; the fix has two. Producers
    # choose adversarial order only among requests that already exist.
    class Deadline(TransportDeadline):
        def wait_event(self, event, **kwargs):
            store.set(f"waiting/{rank}", "1")
            return super().wait_event(event, **kwargs)

    if rank < 2:
        buffers = {}
        for receiver in (2, 3):
            for sequence in range(sequences):
                transfer = identity(rank, receiver, sequence)
                buffer = torch.full(
                    (frame_bytes,),
                    value(rank, receiver, sequence),
                    dtype=torch.uint8,
                    device=device,
                )
                header = build_header(
                    transfer_id=transfer,
                    payload_nbytes=layout.payload_nbytes,
                    layout_id=layout.layout_id,
                    sender_incarnation="schedule-canary",
                )
                buffer[:HEADER_NBYTES] = torch.tensor(
                    list(header), dtype=torch.uint8, device=device
                )
                buffers[receiver, sequence] = buffer
        ready = record_event()
        ready.synchronize()
        dist.barrier()
        wait_keys(store, ["waiting/2", "waiting/3"])
        if args.case == "cycle":
            assert all(store.check([key(rank, r, 0)]) for r in (2, 3))
        stream = get_transfer_stream_pool(device=device).acquire()
        sent = {2: 0, 3: 0}
        order = (3, 2) if rank == 0 else (2, 3)
        while sum(sent.values()) < 2 * sequences:
            progressed = False
            for receiver in order:
                request = f"request/{rank}/{receiver}/{sent[receiver] + 1}"
                if not store.check([request]):
                    continue
                sequence = int(store.get(request))
                operation = TransportDeadline(
                    5, f"send {identity(rank, receiver, sequence)}"
                )
                wait_event(stream, ready)
                print(
                    f"SEND_LAUNCH sender={rank} receiver={receiver} sequence={sequence}",
                    flush=True,
                )
                pynccl.nccl_send(
                    buffers[receiver, sequence],
                    1,
                    pairs[rank, receiver],
                    stream=stream,
                    timeout_ms=operation.remaining_ms(),
                )
                operation.wait_event(record_event(stream))
                operation.close()
                sent[receiver] += 1
                progressed = True
            if not progressed:
                time.sleep(0.001)
    else:

        class Cache:
            def __init__(self):
                self.pins = 0

            def get(self, pair):
                return pairs[int(pair[0]), rank]

            def unpin(self, pair):
                self.pins -= 1

        def make_receiver(bounded):
            strategy = NCCLTransportStrategy()
            strategy._device = device
            strategy._receiver_rank = rank
            strategy._rendezvous = object()
            strategy._streams = get_transfer_stream_pool(device=device)
            strategy._warm_pairs = set()
            strategy._comm_cache = Cache()
            strategy._receive_pool = ReceiveBufferPool()
            if bounded:
                strategy._receive_budget = ReceiveBudget(8 * frame_bytes, 2)
            import threading

            strategy._bounded_fetch_lock = threading.Lock()

            def accept(ref, _pynccl):
                sender, sequence = int(ref["sender_replica"]), ref["sequence"]
                lease = strategy._receive_pool.acquire(
                    layout.layout_id, frame_bytes, device
                )
                if strategy._receive_budget is not None:
                    strategy._receive_budget.attribute(raw=frame_bytes)
                ordinal = store.add(f"ordinal/{sender}/{rank}", 1)
                operation = Deadline(5, f"receive {ref['transfer_id']}")
                strategy._comm_cache.pins += 1
                store.set(key(sender, rank, sequence), str(ordinal))
                store.set(f"request/{sender}/{rank}/{ordinal}", str(sequence))
                print(
                    f"ACCEPT receiver={rank} sender={sender} sequence={sequence}",
                    flush=True,
                )
                return _PreparedReceive(
                    pairs[sender, rank], lease.tensor, lease, operation, None
                )

            strategy._rendezvous_one = accept
            return strategy

        def refs(sequence):
            senders = (0, 1) if rank == 2 or args.case == "benchmark" else (1, 0)
            result = []
            for sender in senders:
                ref = dict(
                    transfer_id=identity(sender, rank, sequence),
                    sender_replica=str(sender),
                    sender_rank=0,
                    sender_incarnation="schedule-canary",
                    sequence=sequence,
                    layout=layout,
                    layout_id=layout.layout_id,
                )
                result.append((100 * sequence + sender, ref))
            return result

        def fetch(strategy, references):
            before = [(idx, id(ref)) for idx, ref in references]
            batch, _, _ = strategy._fetch_all(references)
            assert [(idx, id(ref)) for idx, ref in references] == before
            assert list(batch) == [idx for idx, _ in references]
            for idx, ref in references:
                sender = int(ref["sender_replica"])
                assert torch.all(
                    batch[idx]["sample"] == value(sender, rank, ref["sequence"])
                )
            batch.release()
            assert strategy._comm_cache.pins == 0
            if strategy._receive_budget is not None:
                assert strategy._receive_budget.used == 0

        strategy = make_receiver(args.case == "bounded")
        dist.barrier()
        started = time.perf_counter()
        if args.case == "cycle":
            # Negative control: previous enqueue-whole-batch rule, retaining
            # operands until completion or the unchanged fatal deadline.
            owned = []
            stream = strategy._streams.acquire()
            for _, ref in refs(0):
                prepared = strategy._rendezvous_one(ref, pynccl)
                wait_event(stream, record_event())
                pynccl.nccl_recv(
                    prepared.buffer,
                    0,
                    prepared.comm_idx,
                    stream=stream,
                    timeout_ms=prepared.operation.remaining_ms(),
                )
                owned.append((ref, prepared, record_event(stream)))
            print(f"CYCLE_RECVS_POSTED receiver={rank} count={len(owned)}", flush=True)
            for _ref, prepared, event in owned:
                prepared.operation.wait_event(event)
            raise AssertionError(
                "Adversarial shared-stream cycle unexpectedly completed"
            )
        elif args.case == "concurrent":
            other = make_receiver(False)
            with ThreadPoolExecutor(2) as pool:
                a = pool.submit(fetch, strategy, refs(0) + refs(1))
                b = pool.submit(fetch, other, refs(2) + refs(3))
                a.result(20)
                b.result(20)
        elif args.case == "partial":
            for sequence in range(sequences):
                for reference in refs(sequence):
                    fetch(strategy, [reference])
        elif args.case == "benchmark":
            references = [r for sequence in range(sequences) for r in refs(sequence)]
            references.sort(key=lambda item: item[1]["sender_replica"])
            fetch(strategy, references)
        else:
            fetch(strategy, refs(0) + refs(1))
        if args.case == "benchmark":
            elapsed = time.perf_counter() - started
            print(
                "METRIC "
                + json.dumps(
                    dict(
                        rank=rank,
                        payload_bytes=args.nbytes,
                        seconds=elapsed,
                        received_bytes=2 * sequences * args.nbytes,
                    )
                ),
                flush=True,
            )
        print(f"RECEIVER_PASS rank={rank} case={args.case}", flush=True)
    dist.barrier()
    for comm in pairs.values():
        pynccl.nccl_abort(comm)
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--case", default="all")
    parser.add_argument("--expect-cycle", action="store_true")
    parser.add_argument("--store")
    # Small messages may fit eagerly in NCCL's internal buffers and complete
    # despite the unsafe dependency graph. Exercise a large trajectory frame.
    parser.add_argument("--nbytes", type=int, default=256 * 1024**2)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    for case in (
        ("cycle", "opposite", "bounded", "partial", "concurrent")
        if args.case == "all"
        else (args.case,)
    ):
        with tempfile.TemporaryDirectory(prefix="payload-schedule-") as directory:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc-per-node=4",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--case",
                    case,
                    "--store",
                    directory + "/control",
                    "--nbytes",
                    str(args.nbytes),
                ],
                env={**os.environ, "NCCL_LAUNCH_ORDER_IMPLICIT": "0"},
                capture_output=True,
                text=True,
                timeout=50,
            )
            output = result.stdout + result.stderr
            negative_control = case == "cycle" or args.expect_cycle
            if negative_control:
                assert result.returncode != 0 and "[Transport FATAL]" in output, output
                for receiver in (2, 3):
                    if case == "cycle":
                        assert (
                            f"CYCLE_RECVS_POSTED receiver={receiver} count=2" in output
                        ), output
                    for sender in (0, 1):
                        assert (
                            f"ACCEPT receiver={receiver} sender={sender} sequence=0"
                            in output
                        ), output
                for sender, receiver in ((0, 3), (1, 2)):
                    assert (
                        f"SEND_LAUNCH sender={sender} receiver={receiver} sequence=0"
                        in output
                    ), output
            else:
                assert result.returncode == 0, output
                for receiver in (2, 3):
                    assert f"RECEIVER_PASS rank={receiver} case={case}" in output, (
                        output
                    )
            print(
                json.dumps(
                    dict(
                        case=case,
                        negative_control=negative_control,
                        passed=True,
                        returncode=result.returncode,
                    )
                ),
                flush=True,
            )
            for line in output.splitlines():
                if line.startswith("METRIC ") or (
                    negative_control
                    and line.startswith(
                        (
                            "ACCEPT ",
                            "CYCLE_RECVS_POSTED ",
                            "SEND_LAUNCH ",
                            "[Transport FATAL]",
                        )
                    )
                ):
                    print(line, flush=True)
    print("NCCL_PAYLOAD_SCHEDULE_PASS", flush=True)


if __name__ == "__main__":
    main()
