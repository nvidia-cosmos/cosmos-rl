# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Known rejections are not unknown cache misses or native-completion failures."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest

from test_receive_memory import make_receiver, refs_for
from cosmos_rl.utils.payload_transport.nccl import strategy as nccl
from cosmos_rl.utils.payload_transport.nccl.rendezvous import RendezvousResult
from cosmos_rl.utils.payload_transport.prefetch_mixin import PrefetchDataPackerMixin
from cosmos_rl.utils.payload_transport.receive_memory import ReceiveMemoryError
from cosmos_rl.utils.trajectory import serialize_schema
from cosmos_rl.utils.transport_failure import TransportUnusableError


class ConcretePacker:
    def get_policy_input(
        self, sample, rollout_output, n_ignore_prefix_tokens=0, **kwargs
    ):
        return rollout_output


class Packer(PrefetchDataPackerMixin, ConcretePacker):
    pass


def wire_ref(ref):
    return {
        "_nccl": True,
        "_transfer_id": ref["transfer_id"],
        "_sender_replica": ref["sender_replica"],
        "_sender_rank": ref["sender_rank"],
        "_schema": serialize_schema(ref["schema"]),
    }


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("outcome", ["missing", "no_schema"])
def test_known_rejection_skips_without_unleased_refetch(monkeypatch, outcome, prepared):
    receiver, _, _ = make_receiver(monkeypatch)
    ref = wire_ref(refs_for([3])[0][1])
    if outcome == "missing":
        receiver._rendezvous = SimpleNamespace(
            initiate=lambda **kwargs: RendezvousResult(
                status=nccl.TransferStatus.MISSING
            )
        )
        receiver._rendezvous_one = MethodType(
            nccl.NCCLTransportStrategy._rendezvous_one, receiver
        )
    else:
        ref.pop("_schema")
    batch = receiver.fetch_batch([(4, ref)])
    packer = Packer()
    packer._transport_strategy = receiver
    packer._prefetch_cache = {} if prepared else batch
    if prepared:
        packer._preparation_local = SimpleNamespace(cache=batch)
    packer._sync_fetch = Mock(side_effect=receiver.sync_fetch)
    try:
        assert packer.get_policy_input(rollout_output=ref) is None
        packer._sync_fetch.assert_not_called()
        assert not receiver._receive_budget.closed
    finally:
        if hasattr(batch, "release"):
            batch.release()
    assert receiver._receive_budget.used == 0


def test_header_mismatch_remains_terminal_after_native_completion(monkeypatch):
    receiver, _, _ = make_receiver(monkeypatch)
    original = receiver._rendezvous_one

    def wrong_header(ref, pynccl):
        communicator, raw = original(ref, pynccl)
        raw[0] = 0
        return communicator, raw

    receiver._rendezvous_one = wrong_header
    receiver._resync_pair = Mock()
    ref = wire_ref(refs_for([3])[0][1])
    with pytest.raises(
        TransportUnusableError, match="Accepted payload identity mismatch"
    ):
        receiver.fetch_batch([(4, ref)])
    assert receiver._receive_budget.closed
    receiver._resync_pair.assert_not_called()


def test_unknown_absence_still_cannot_refetch_outside_lease(monkeypatch):
    receiver, _, _ = make_receiver(monkeypatch)
    receiver._rendezvous_one = lambda *args: None  # No proof of a safe rejection.
    ref = wire_ref(refs_for([3])[0][1])
    batch = receiver.fetch_batch([(4, ref)])
    packer = Packer()
    packer._transport_strategy = receiver
    packer._prefetch_cache = batch
    try:
        with pytest.raises(ReceiveMemoryError, match="cache-miss refetch is disabled"):
            packer.get_policy_input(rollout_output=ref)
    finally:
        batch.release()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("all_rejected", [False, True])
def test_mixed_received_and_rejected_batch_keeps_lease_and_clears_outcomes(
    monkeypatch, prepared, all_rejected
):
    receiver, _, _ = make_receiver(monkeypatch)
    receive = receiver._rendezvous_one
    receiver._rendezvous = SimpleNamespace(
        initiate=lambda **kwargs: RendezvousResult(status=nccl.TransferStatus.MISSING)
    )
    missing = MethodType(nccl.NCCLTransportStrategy._rendezvous_one, receiver)
    receiver._rendezvous_one = lambda ref, pynccl: (
        missing(ref, pynccl)
        if all_rejected or ref["transfer_id"] == "episode-0"
        else receive(ref, pynccl)
    )
    tasks = [(i, wire_ref(ref)) for i, ref in refs_for([3, 5])]
    packer = Packer()
    packer._transport_strategy = receiver
    packer._filter_prefetch_tasks = lambda tasks: tasks
    packer._setup_prefetch(prefetch_timeout=5)

    def resolve():
        return [packer.get_policy_input(rollout_output=ref) for _, ref in tasks]

    try:
        if prepared:
            future = packer.start_prepared_prefetch(tasks, resolve)
            values = future.result(timeout=3)
            packer.release_prepared_prefetch(future)
        else:
            packer.start_prefetch(tasks)
            packer.wait_prefetch()
            values = resolve()
        assert values[0] is None
        if all_rejected:
            assert values[1] is None
        else:
            assert set(values[1]) == {"odd", "aligned"}
        batch = packer._prefetch_cache
        assert batch.rejected_keys == (
            {"episode-0", "episode-1"} if all_rejected else {"episode-0"}
        )
        assert (batch.nbytes == 0) == all_rejected
        assert receiver._receive_budget.used == batch.nbytes
        del values
        if prepared:
            del future
        packer.release_prefetch()
        assert not batch.rejected_keys and receiver._receive_budget.used == 0
        with pytest.raises(ReceiveMemoryError, match="cache-miss refetch is disabled"):
            packer.get_policy_input(rollout_output=tasks[0][1])
    finally:
        packer.release_prefetch()
        packer.shutdown_prefetch()
