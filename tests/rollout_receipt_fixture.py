# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Install a real registered reporting atom around focused controller fixtures."""

import asyncio

from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.protocol import Role, RolloutRequest
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity


def install_report_source(controller, request):
    if not isinstance(request, RolloutRequest):
        request = RolloutRequest(**vars(request))
    name = request.src_replica_name
    atom = Atom(
        0,
        "127.0.0.1",
        "fixture",
        None,
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        name,
        report_session_id="fixture-session",
    )
    replica = Replica(name, Role.ROLLOUT, [atom])
    # Focused route tests begin after the prompt-fetch boundary. Allocate the
    # controller slots explicitly instead of treating bare results as issued work.
    delivered = []
    for payload in request.payloads:
        count = len(payload.completions) if payload.completions is not None else 0
        if count:
            work = replica.producer_reservations.issue(payload.weight_version, count)
            payload.training_work_id = work
            payload.training_completion_slots = list(range(count))
            delivered.extend(
                TrainingCompletionIdentity(work_id=work, slot=i) for i in range(count)
            )
    if request.completion_identities is not None:
        assert len(delivered) == len(request.completion_identities)
        request.completion_identities = [
            identity.model_copy(update={"reservation": coordinate})
            for identity, coordinate in zip(request.completion_identities, delivered)
        ]
        failures = []
        for failure in request.completion_failures:
            work = replica.producer_reservations.issue(
                failure.identity.weight_version, 1
            )
            coordinate = TrainingCompletionIdentity(work_id=work, slot=0)
            identity = failure.identity.model_copy(update={"reservation": coordinate})
            failures.append(failure.model_copy(update={"identity": identity}))
            request.training_rejections.append(coordinate)
        request.completion_failures = failures
    else:
        for key in ("discarded_samples", "filtered_positive", "filtered_negative"):
            count = request.metrics.get(key, 0)
            if type(count) is int and count > 0:
                version = request.metrics.get("discarded_weight_version", 0)
                if type(version) is not int or version < 0:
                    version = 0
                work = replica.producer_reservations.issue(version, count)
                request.training_rejections.extend(
                    TrainingCompletionIdentity(work_id=work, slot=i)
                    for i in range(count)
                )
    previous = getattr(controller, "rollout_status_manager", None)

    class Sources(dict):
        def __getattr__(self, attribute):
            return getattr(previous, attribute)

    controller.rollout_status_manager = Sources({name: replica})
    if not isinstance(getattr(controller, "life_cycle_lock", None), asyncio.Lock):
        controller.life_cycle_lock = asyncio.Lock()
    if not hasattr(controller, "_put_application_rollouts_locked"):
        controller._put_application_rollouts_locked = (
            lambda request,
            rollouts,
            **kwargs: Controller._put_application_rollouts_locked(
                controller, request, rollouts, **kwargs
            )
        )
    if hasattr(controller, "policy_status_manager"):
        controller.policy_status_manager.terminal_error = None
        # Mirror Controller initialization: capacity settlement and original
        # version quota refill use this callback, not a second HTTP mutation.
        from cosmos_rl.dispatcher.status import PolicyStatusManager

        if isinstance(
            controller.policy_status_manager, PolicyStatusManager
        ) and hasattr(controller, "register_discarded_samples_for_refill"):
            controller.policy_status_manager.set_discard_refill_hook(
                controller.register_discarded_samples_for_refill
            )
    if not hasattr(controller, "completion_admission"):
        controller.completion_admission = None
    if not hasattr(controller.config, "mode"):
        controller.config.mode = "disaggregated"
    request.src_global_rank = 0
    request.report_session_id = "fixture-session"
    request.report_sequence = 0
    return request
