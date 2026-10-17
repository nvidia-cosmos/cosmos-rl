# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Preserve controller reservation identities through producer-side selection."""

from cosmos_rl.dispatcher.data.schema import TrainingCompletionIdentity


def training_identity(payload, index):
    if payload.training_work_id is None:
        return None
    slots = payload.training_completion_slots
    if slots is None or index >= len(slots):
        raise ValueError("Training source lost its completion slots")
    return TrainingCompletionIdentity(
        work_id=payload.training_work_id, slot=slots[index]
    )


def select_training_slots(payload, indices):
    """Select structural survivors without changing generated training values."""
    if payload.training_work_id is None:
        return payload
    slots = payload.training_completion_slots
    if slots is None or any(
        type(index) is not int or not 0 <= index < len(slots) for index in indices
    ):
        raise ValueError("Generation exceeds its controller-reserved slots")
    if len(set(indices)) != len(indices):
        raise ValueError("Generation repeats a controller-reserved slot")
    return payload.model_copy(
        update={"training_completion_slots": [slots[index] for index in indices]}
    )


def discarded_training_slots(originals, survivors):
    kept = {
        (payload.training_work_id, slot)
        for payload in survivors
        for slot in (payload.training_completion_slots or [])
    }
    return [
        TrainingCompletionIdentity(work_id=payload.training_work_id, slot=slot)
        for payload in originals
        if payload.training_work_id is not None
        for slot in payload.training_completion_slots
        if (payload.training_work_id, slot) not in kept
    ]


def attach_training_rejections(report, source_payloads):
    """Account for quality selection and whole-group removal, not metrics counts.

    ``source_payloads`` precedes metadata-only/DAPO removal. Reward selection
    already carries its excluded slots separately on each source payload.
    Existing reward, advantage and serialization behavior is unchanged.
    """
    sent = {}
    for payload in report.payloads:
        if payload.training_work_id is None:
            continue
        slots = payload.training_completion_slots
        if slots is None or len(slots) != len(payload.completions):
            raise ValueError("Training payload lost its completion slots")
        selected = sent.setdefault(payload.training_work_id, set())
        for slot in slots:
            if slot in selected:
                raise ValueError("Training payload repeats a completion slot")
            selected.add(slot)
    rejected = {(item.work_id, item.slot) for item in report.training_rejections}
    for payload in source_payloads:
        work_id = payload.training_work_id
        if work_id is None:
            continue
        if payload.training_completion_slots is None:
            raise ValueError("Training source lost its completion slots")
        selected = sent.get(work_id, set())
        rejected.update((work_id, slot) for slot in payload.training_rejected_slots)
        rejected.update(
            (work_id, slot)
            for slot in payload.training_completion_slots
            if slot not in selected
        )
    if any(slot in sent.get(work_id, ()) for work_id, slot in rejected):
        raise ValueError("Training slot cannot be both delivered and rejected")
    report.training_rejections = [
        TrainingCompletionIdentity(work_id=work_id, slot=slot)
        for work_id, slot in sorted(rejected)
    ]
    return report
