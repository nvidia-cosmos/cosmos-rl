# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from copy import copy

import pytest

from cosmos_rl.dispatcher.receipt import OrderedReceipt, request_digest


def test_lost_ack_replays_an_immutable_response_without_a_new_mutation():
    receipt = OrderedReceipt()
    token, cached = receipt.begin(0, "first")
    assert token is not None and cached is None
    result = {"ids": [3, 7]}
    receipt.commit(token, result)
    result["ids"].append(9)
    duplicate, cached = receipt.begin(0, "first")
    assert duplicate is None and cached == {"ids": [3, 7]}
    cached["ids"].append(12)
    assert receipt.begin(0, "first") == (None, {"ids": [3, 7]})


@pytest.mark.parametrize(
    "sequence,digest", [(0, "changed"), (2, "new"), (-1, "old"), (True, "bool")]
)
def test_rejected_identity_cannot_change_receipt(sequence, digest):
    receipt = OrderedReceipt()
    token, _ = receipt.begin(0, "first")
    receipt.commit(token, {"ok": True})
    with pytest.raises(ValueError):
        receipt.begin(sequence, digest)
    assert receipt.sequence == 0 and receipt.pending is None
    assert receipt.begin(0, "first") == (None, {"ok": True})


def test_receipt_is_bounded_and_old_retries_are_not_reinterpreted():
    receipt = OrderedReceipt()
    for sequence in range(100):
        token, _ = receipt.begin(sequence, str(sequence))
        receipt.commit(token, [sequence])
    assert receipt.response == [99]
    with pytest.raises(ValueError, match="expired"):
        receipt.begin(0, "0")
    assert receipt.begin(99, "99") == (None, [99])


def test_pending_and_failed_operations_cannot_replay_partial_mutations():
    receipt = OrderedReceipt()
    token, _ = receipt.begin(0, "first")
    with pytest.raises(ValueError, match="unacknowledged"):
        receipt.begin(0, "first")
    receipt.fail(token)
    with pytest.raises(RuntimeError, match="unusable"):
        receipt.begin(0, "first")
    with pytest.raises(RuntimeError):
        receipt.commit(token, {"late": "success"})


def test_equal_but_unowned_token_cannot_commit_or_poison_receipt():
    receipt = OrderedReceipt()
    token, _ = receipt.begin(0, "first")
    for operation in (
        lambda: receipt.commit(copy(token), None),
        lambda: receipt.fail(copy(token)),
    ):
        with pytest.raises(RuntimeError, match="own"):
            operation()
    assert not receipt.failed and receipt.pending is token
    receipt.commit(token, None)
    assert receipt.begin(0, "first") == (None, None)


def test_snapshot_failure_poisoned_after_mutation_is_not_retriable():
    class BrokenSnapshot:
        def __deepcopy__(self, memo):
            raise RuntimeError("injected copy failure")

    receipt = OrderedReceipt()
    token, _ = receipt.begin(0, "first")
    with pytest.raises(RuntimeError, match="injected"):
        receipt.commit(token, BrokenSnapshot())
    assert receipt.failed and receipt.pending is token
    with pytest.raises(RuntimeError, match="unusable"):
        receipt.begin(0, "first")


def test_digest_is_canonical_and_rejects_nonfinite_input():
    assert request_digest({"a": 1, "b": 2}) == request_digest({"b": 2, "a": 1})
    assert request_digest({"a": 1}) != request_digest({"a": 2})
    with pytest.raises(ValueError):
        request_digest({"a": float("nan")})
