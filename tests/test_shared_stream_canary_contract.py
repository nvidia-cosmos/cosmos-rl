# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Evidence checks must reject setup crashes and absent second-peer admission."""

import pytest

from transfer_contract_canary import check_shared_stream_output


BASE = "\n".join(
    [
        "SHARED_PAYLOAD_PASS rank=1 step=0 exact=True",
        "SHARED_PAYLOAD_PASS rank=2 step=0 exact=True",
        "SHARED_ACCEPTED peer=consumer1",
        "SHARED_ACCEPTED peer=consumer2",
        "SHARED_NATIVE_POST shared_stream=True",
    ]
)
FAILURE = BASE + "\nSHARED_STREAM_INJECT t=100.0\n[Transport FATAL] deadline\n"
HEALTHY = (
    BASE
    + "\n"
    + "\n".join(f"SHARED_PAYLOAD_PASS rank={rank} step=1 exact=True" for rank in (1, 2))
)


def test_healthy_and_fault_outcomes():
    assert check_shared_stream_output(HEALTHY, 0, "shared-stream-healthy", 108) is None
    assert check_shared_stream_output(FAILURE, 1, "shared-stream-peer-stall", 108) == 8


@pytest.mark.parametrize("missing", BASE.splitlines() + ["[Transport FATAL]"])
def test_missing_evidence_is_not_a_pass(missing):
    with pytest.raises(AssertionError):
        check_shared_stream_output(
            FAILURE.replace(missing, ""), 1, "shared-stream-peer-stall", 108
        )


@pytest.mark.parametrize("now", [99, 120, 140])
def test_failure_must_be_bounded(now):
    with pytest.raises(AssertionError):
        check_shared_stream_output(FAILURE, 1, "shared-stream-peer-stall", now)


def test_failure_cannot_return_cleanly_or_publish_withheld_payload():
    with pytest.raises(AssertionError):
        check_shared_stream_output(FAILURE, 0, "shared-stream-peer-stall", 108)
    with pytest.raises(AssertionError):
        check_shared_stream_output(
            FAILURE + "SHARED_PAYLOAD_PASS rank=1 step=1 exact=True",
            1,
            "shared-stream-peer-stall",
            108,
        )


@pytest.mark.parametrize("rank", [1, 2])
def test_healthy_control_requires_both_final_payloads(rank):
    with pytest.raises(AssertionError):
        check_shared_stream_output(
            HEALTHY.replace(f"SHARED_PAYLOAD_PASS rank={rank} step=1 exact=True", ""),
            0,
            "shared-stream-healthy",
            108,
        )
