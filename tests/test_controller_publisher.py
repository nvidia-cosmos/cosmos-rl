# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The controller outbox capacity follows the replica cohort size."""

import threading

from cosmos_rl.dispatcher.publication import ControllerPublisher, outbox_capacity


class _BlockingTransport:
    def __init__(self):
        self.release = threading.Event()

    def publish_plan(self, plan):
        self.release.wait()


def test_outbox_capacity_scales_with_replicas():
    assert outbox_capacity(24, 32) == 448
    assert outbox_capacity(48, 80) == 1024
    assert outbox_capacity(1, 1) == 128


def test_outbox_rejects_only_beyond_its_capacity():
    transport = _BlockingTransport()
    failures = []
    publisher = ControllerPublisher(
        transport, on_failure=failures.append, max_pending=outbox_capacity(48, 80)
    )
    try:
        for _ in range(1024):
            publisher.publish_command(b"x", "replica")
        assert failures == []
        try:
            publisher.publish_command(b"x", "replica")
        except RuntimeError as error:
            assert "outbox is full" in str(error)
        else:
            raise AssertionError("the 1025th pending publication must be rejected")
        assert len(failures) == 1
    finally:
        transport.release.set()
        publisher._writer.shutdown(wait=False, cancel_futures=True)
