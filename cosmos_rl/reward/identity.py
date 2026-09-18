# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Producer reservations survive selection; only the controller settles them."""

import itertools
import threading

from cosmos_rl.dispatcher.data.admission import CompletionFailure, CompletionIdentity
from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.dispatcher.protocol import RolloutRequest


class CompletionReporter:
    def __init__(self, replica_name, global_rank):
        self.replica_name = replica_name
        self.global_rank = global_rank
        self._sequences = itertools.count()
        self._lock = threading.Lock()

    def reserve(self, payloads, count, weight_version):
        if count < 1:
            raise ValueError("Completion reservation count must be positive")
        with self._lock:
            for payload in payloads:
                if payload.completion_sequences is not None:
                    raise ValueError("Prompt already has completion reservations")
                payload.weight_version = weight_version
                payload.completion_sequences = [
                    next(self._sequences) for _ in range(count)
                ]

    def generation_failure(self, payloads, reason):
        return RolloutRequest(
            src_replica_name=self.replica_name,
            src_global_rank=self.global_rank,
            payloads=[],
            completion_identities=[],
            completion_failures=[
                CompletionFailure(
                    identity=CompletionIdentity(
                        sequence=sequence, weight_version=payload.weight_version
                    ),
                    reason=reason,
                )
                for payload in payloads
                for sequence in payload.completion_sequences
            ],
        )

    def report(self, payloads, packer):
        identities = []
        failures = []
        accepted = []
        for payload in payloads:
            if payload.completion_sequences is None or len(
                payload.completion_sequences
            ) != len(payload.completions):
                raise ValueError("Reward processing lost completion reservations")
            identities.extend(
                CompletionIdentity(
                    sequence=sequence, weight_version=payload.weight_version
                )
                for sequence in payload.completion_sequences
            )
            for rejection in payload.completion_rejections:
                # Use the same application serializer for rejected references.
                # They go only to cleanup, never to a trainer.
                completion = rejection["completion"]
                rejected = None
                if completion is not None:
                    completions, _, _, _, _ = packer.get_rollout_output(
                        [completion],
                        [rejection["completed_conversations"]],
                        [rejection["completion_logprobs"]],
                        [rejection["completion_token_ids"]],
                    )
                    rejected = Rollout(
                        completion=completions[0],
                        weight_version=rejection["weight_version"],
                    )
                failures.append(
                    CompletionFailure(
                        identity=CompletionIdentity(
                            sequence=rejection["sequence"],
                            weight_version=rejection["weight_version"],
                        ),
                        reason=rejection["reason"],
                        payload=rejected,
                    )
                )
            if payload.completions:
                accepted.append(payload)
        return RolloutRequest(
            src_replica_name=self.replica_name,
            src_global_rank=self.global_rank,
            payloads=accepted,
            completion_identities=identities,
            completion_failures=failures,
        )
