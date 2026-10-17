# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real Redis receipts cover the current and reserved batch as one operation."""

import pytest
import redis

from cosmos_rl.dispatcher.command import (
    Command,
    DataFetchCommand,
    PayloadPrefetchCommand,
)
import test_dispatch_publication as publication_fixture
from test_trainer_payload_prefetch import manager, advance

client = publication_fixture.client


def test_lost_reply_keeps_current_and_lookahead_exactly_once(client):
    status, replica = manager(4)
    status.try_trigger_data_fetch_and_training()
    plan = status.redis_handler.publish_plan.call_args.args[0].for_server(
        client.info("server")["run_id"]
    )

    class LostReply:
        def eval(self, *args):
            client.eval(*args)
            raise redis.ConnectionError("injected lost commit reply")

    with pytest.raises(redis.ConnectionError):
        plan.publish(LostReply(), maxlen=100)
    receipt = plan.publish(client, maxlen=100)
    assert plan.publish(client, maxlen=100) == receipt
    assert client.xlen(replica.name + "_rollout") == 2
    assert client.xlen(replica.name + "_command") == 2
    command = Command.depack(client.xrange(replica.name + "_command")[0][1][b"command"])
    assert command.global_step == 1
    notice = Command.depack(client.xrange(replica.name + "_command")[1][1][b"command"])
    assert isinstance(notice, PayloadPrefetchCommand)
    assert notice.batch_id == status._payload_lookahead[0]
    assert notice.global_step == 2
    assert status.current_step == 1 and status.samples_on_the_fly == 4
    advance(status, replica)
    next_plan = status.redis_handler.publish_plan.call_args.args[0].for_server(
        client.info("server")["run_id"]
    )
    next_plan.publish(client, maxlen=100)
    assert (
        client.xlen(replica.name + "_rollout") == 2
    )  # B is carried only by identity-bearing commands.
    commands = [
        Command.depack(data[b"command"])
        for _, data in client.xrange(replica.name + "_command")
    ]
    assert [
        command.global_step
        for command in commands
        if isinstance(command, DataFetchCommand)
    ] == [1, 2]
    assert commands[2].prefetched_batch_id == notice.batch_id
    assert commands[2].payload_notification == notice._serialize()


def test_partial_lookahead_publication_cannot_repeat_or_train(client):
    status, replica = manager(4)
    status.try_trigger_data_fetch_and_training()
    plan = status.redis_handler.publish_plan.call_args.args[0].for_server(
        client.info("server")["run_id"]
    )
    client.set(plan.key, plan.digest + ":pending", ex=60)
    # Current metadata exists, but neither the ordinary command nor notification.
    for stream, kind, payload in plan.entries[:2]:
        client.xadd(stream, {kind: payload})
    with pytest.raises(redis.ResponseError, match="incomplete"):
        plan.publish(client, maxlen=100)
    assert client.xlen(replica.name + "_rollout") == 2
    assert not client.exists(replica.name + "_command")
    assert status.dispatched_rollouts_by_step == {1: 2}


def test_late_notification_has_its_own_idempotent_receipt(client):
    status, replica = manager(4)
    rows = [status.rollout_buffer.get_nowait() for _ in range(4)]
    for row in rows[:2]:
        status.rollout_buffer.put_nowait(row)
    status.try_trigger_data_fetch_and_training()
    first = status.redis_handler.publish_plan.call_args.args[0].for_server(
        client.info("server")["run_id"]
    )
    first.publish(client, maxlen=100)
    assert status._payload_lookahead is None
    for row in rows[2:]:
        status.rollout_buffer.put_nowait(row)
    status.try_trigger_data_fetch_and_training()
    later = status.redis_handler.publish_plan.call_args.args[0].for_server(
        client.info("server")["run_id"]
    )
    assert later.operation_id != first.operation_id
    assert len(later.entries) == 1
    notice = Command.depack(later.entries[0][2])
    assert isinstance(notice, PayloadPrefetchCommand)
    receipt = later.publish(client, maxlen=100)
    assert later.publish(client, maxlen=100) == receipt
    assert client.xlen(replica.name + "_command") == 2
    assert status.current_step == 1 and status.dispatched_rollouts_by_step == {1: 2}
