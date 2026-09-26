# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One worker's serial validation fetch/report receipts for a sealed round."""

from cosmos_rl.dispatcher.protocol import ValidationReportRequest


def validation_round_for_command(command):
    """None explicitly suppresses a completed round; missing protocol is unsafe."""
    if getattr(command, "validation_protocol_version", 0) != 1:
        raise ValueError("Validation requires matching controller/worker protocol")
    return command.validation_round_id


class ValidationSession:
    def __init__(self, client, round_id, step, replica, rank):
        if not round_id:
            raise ValueError("Validation command has no round identity")
        self.client = client
        self.round_id = round_id
        self.step = step
        self.replica = replica
        self.rank = rank
        self.fetch_sequence = 0
        self.report_sequence = 0

    def fetch(self, batch_size, rank_in_mesh=None):
        result = self.client.get_next_prompt(
            batch_size,
            validation_step=self.step,
            rank_in_mesh=rank_in_mesh,
            validation_round_id=self.round_id,
            src_replica_name=self.replica,
            fetch_sequence=self.fetch_sequence,
        )
        self.fetch_sequence += 1
        return result

    def report(self, payloads, *, is_end=False):
        self.client.post_validation_report(
            ValidationReportRequest(
                src_replica_name=self.replica,
                src_global_rank=self.rank,
                validation_step=self.step,
                validation_round_id=self.round_id,
                report_sequence=self.report_sequence,
                payloads=payloads,
                is_end=is_end,
            )
        )
        self.report_sequence += 1
