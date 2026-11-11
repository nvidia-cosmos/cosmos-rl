# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Required synchronous generation sink; persistence precedes settlement."""

from cosmos_rl.dispatcher.data.schema import Rollout
from cosmos_rl.execution.corpus import CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord, output_id


class CorpusResultSink:
    def __init__(self, output, *, actor_digest=None, metadata=None, packer=None):
        self.output = output
        self.actor_digest = actor_digest
        self.sampling = {
            key: metadata[key]
            for key in ("seed", "temperature", "max_response_length", "sampling")
            if metadata and key in metadata
        }
        self.packer = packer
        self.writer = CorpusWriter(
            output.path,
            run_id=output.run_id,
            max_record_bytes=output.max_record_bytes,
            metadata=metadata,
        )

    def __call__(self, event):
        if event.occurrence_id is None:
            raise ValueError(
                "Durable generation requires a controller occurrence identity"
            )
        group = output_id(self.output.run_id, event.occurrence_id, 0)
        for outcome in event.outcomes:
            identity = output_id(self.output.run_id, event.occurrence_id, outcome.slot)
            record = SampleRecord(
                sample_id=identity,
                occurrence_id=identity,
                group_id=group,
                group_size=len(event.outcomes),
                group_slot=outcome.slot,
                producer=ModelIdentity(
                    role="actor",
                    name=event.model_name_or_path,
                    revision=event.model_revision,
                    artifact_digest=self.actor_digest,
                    behavior_version=event.weight_version,
                ),
                sampling=self.sampling,
                annotations={
                    "status": outcome.status,
                    "quality_trainable": outcome.quality_trainable,
                    "quality_reason": outcome.quality_reason,
                    "failure_reason": outcome.failure_reason,
                    "scorer_metrics": outcome.scorer_metrics,
                },
                attempt_id=f"{event.execution_id}:{event.work_id}:{outcome.slot}",
                rollout=Rollout(
                    prompt=event.prompt,
                    conversation=event.conversation,
                    extra_info=event.extra_info,
                    completion=outcome.completion,
                    completed_conversation=outcome.completed_conversation,
                    completion_token_ids=outcome.token_ids,
                    completion_logprobs=outcome.logprobs,
                    reward=outcome.score,
                ),
            )
            export = getattr(
                self.packer() if self.packer else None, "export_record", None
            )
            if export is not None:
                exported = export(record)
                if not isinstance(exported, SampleRecord) or (
                    exported.sample_id,
                    exported.occurrence_id,
                    exported.group_id,
                    exported.group_size,
                    exported.group_slot,
                    exported.producer,
                ) != (
                    record.sample_id,
                    record.occurrence_id,
                    record.group_id,
                    record.group_size,
                    record.group_slot,
                    record.producer,
                ):
                    raise ValueError(
                        "Packer export_record must preserve sample identity and provenance"
                    )
                record = exported
            self.writer.commit(record)

    def close(self):
        self.writer.close()


def corpus_result_factory(config, *, packer=None):
    from cosmos_rl.utils.application_callbacks import CallbackFactory

    output = config.execution.output
    actor = config.execution.actor
    return CallbackFactory(
        lambda _: CorpusResultSink(
            output,
            actor_digest=actor.artifact_digest if actor else None,
            metadata=generation_metadata(config),
            packer=packer,
        )
    )


def generation_metadata(config):
    from cosmos_rl.execution.config import validate_generation_identity
    from cosmos_rl.execution.models import actor_config

    validate_generation_identity(config)
    config = actor_config(config)
    actor = config.execution.actor
    policy = config.train.train_policy
    return {
        "model": config.policy.model_name_or_path,
        "revision": config.policy.model_revision,
        "artifact_digest": actor.artifact_digest if actor else None,
        "resumable": config.execution.output.resume,
        "generation_fingerprint": config.execution.output.generation_fingerprint,
        "dataset": policy.dataset.model_dump(mode="json"),
        "epochs": config.train.epoch,
        "n_generation": config.rollout.n_generation,
        "seed": config.rollout.seed,
        "temperature": policy.temperature,
        "sampling": config.rollout.sampling_config.model_dump(mode="json"),
        "source_seed": policy.dataloader_seed,
        "source_shuffle": policy.dataloader_shuffle,
        "prompt_column": policy.prompt_column_name,
        "response_column": policy.response_column_name,
        "max_response_length": config.rollout.max_response_length,
        "bypass_reward": policy.bypass_reward,
        "reward_function": policy.model_dump(mode="json")["reward_function"],
    }


class GenerationLedger:
    """Persist input occurrence identity before dispatch, then skip committed work."""

    def __init__(self, config):
        import json
        from pathlib import Path

        self.output = config.execution.output
        root = Path(self.output.path)
        if not self.output.resume and (
            (root / "complete.json").exists()
            or any((root / "dispatch").glob("*.json"))
            or any((root / "records").glob("*.json"))
        ):
            raise ValueError(
                "One-off generation cannot reopen journaled or committed work; "
                "use a fresh output path and run_id"
            )
        self.writer = CorpusWriter(
            self.output.path,
            run_id=self.output.run_id,
            max_record_bytes=self.output.max_record_bytes,
            metadata=generation_metadata(config),
        )
        self.completed = {}
        from cosmos_rl.execution.corpus import _read_record

        for path in (self.writer.root / "records").glob("*.json"):
            record = _read_record(
                self.writer.root, path, self.output.max_record_bytes, materialize=False
            )
            identity = json.loads(record.sample_id)
            run_id, occurrence, slot = identity
            if run_id != self.output.run_id:
                raise ValueError("Committed record belongs to another generation run")
            self.completed.setdefault(occurrence, set()).add(slot)
        self.slots = set(range(config.rollout.n_generation))

    def reserve(self, payload, occurrence):
        from cosmos_rl.execution.corpus import ArtifactCodec, _json, _publish

        if (self.writer.root / "complete.json").exists() and self.completed.get(
            occurrence, set()
        ) != self.slots:
            raise ValueError("A completed generation run cannot admit new occurrences")
        codec = ArtifactCodec(self.writer.root, self.output.max_record_bytes)
        data = _json(
            codec.encode(
                {
                    "prompt": payload.prompt,
                    "conversation": payload.conversation,
                    "extra_info": payload.extra_info,
                }
            )
        )
        if len(data) + codec.used_bytes > self.output.max_record_bytes:
            raise ValueError("Generation input exceeds its durable byte budget")
        path = self.writer.root / "dispatch" / f"{occurrence}.json"
        if not _publish(path, data) and path.read_bytes() != data:
            raise ValueError(
                "Generation source changed at an existing input occurrence"
            )
        return self.completed.get(occurrence, set()) == self.slots

    def finalize(self, occurrences):
        return self.writer.finalize(expected_records=occurrences * len(self.slots))
