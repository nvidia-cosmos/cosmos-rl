# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A transient command-store error must not poison later successful delivery."""

from unittest.mock import Mock, patch

import pytest

from cosmos_rl.utils.distributed import DistKVStore


@pytest.mark.parametrize("rank", [0, 1])
def test_broadcast_advances_once_after_transient_store_error(rank):
    store = object.__new__(DistKVStore)
    store.world_size = 2
    store.rank = rank
    store.counter = 3
    store.local_store = Mock()
    store.local_store.get.return_value = b"command"
    store.blocking_wait = Mock(side_effect=[RuntimeError("transient"), None, None])
    # Bound the old implementation's sticky-error spin without relying on time.
    store.shutdown_event = Mock()
    store.shutdown_event.is_set.side_effect = [False, False, True]
    command = Mock()
    with patch("cosmos_rl.utils.distributed.Command.depack", return_value=command):
        assert store.broadcast_command(command, src=0) is command
    assert store.counter == 4
    assert store.shutdown_event.is_set.call_count == 2
    if rank == 0:
        assert store.local_store.delete_key.call_count == 3
    else:
        store.local_store.delete_key.assert_not_called()


def test_shutdown_before_delivery_does_not_advance_command_cursor():
    store = object.__new__(DistKVStore)
    store.world_size = 2
    store.rank = 1
    store.counter = 3
    store.local_store = Mock()
    store.shutdown_event = Mock()
    store.shutdown_event.is_set.return_value = True
    assert store.broadcast_command(None) is None
    assert store.counter == 3
    store.local_store.get.assert_not_called()
