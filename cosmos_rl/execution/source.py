# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native corpus loading, scoring and checkpoint compatibility."""

import copy
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class RecordPreparationContext:
    """Independent randomness for one corpus occurrence in one training epoch.

    Applications construct local generators from ``seed`` instead of reseeding
    process-wide Python, NumPy or Torch randomness. The v1 seed derivation is
    independent of source location, rank, loader workers and prefetch timing.
    """

    seed: int
    epoch: int


class CorpusSupervisedDataset:
    """Expose committed successful results to the native SFT dataloader.

    Failed generation stays in the corpus for diagnostics, without fabricating
    a supervised target. Quality decisions remain application annotations.
    """

    def __init__(self, corpus, *, max_batch_bytes=None, preparation_seed=None):
        self.corpus = corpus
        self.preparation_seed = preparation_seed
        self.epoch = 0
        self.max_batch_bytes = (
            corpus.max_record_bytes if max_batch_bytes is None else max_batch_bytes
        )
        self.scorer = None
        self.indices = []
        self.group_advantages = {}
        for i, fields in enumerate(corpus.record_metadata):
            if fields["status"] != "generation_failed":
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def preparation_context(self, record):
        if self.preparation_seed is None:
            return None
        identity = [
            "cosmos-record-preparation-v1",
            self.preparation_seed,
            self.corpus.revision,
            self.epoch,
            record.sample_id,
            record.occurrence_id,
        ]
        digest = hashlib.sha256(
            json.dumps(identity, ensure_ascii=True, separators=(",", ":")).encode()
        ).digest()
        return RecordPreparationContext(
            seed=int.from_bytes(digest[:8], "big") % 2**63, epoch=self.epoch
        )

    def __getitem__(self, index):
        if index is None:
            record = self.corpus[self.indices[0]]
            record.annotations["empty_contribution"] = True
            return record
        record = self.corpus[self.indices[index]]
        if record.sample_id in self.group_advantages:
            advantage, eligible = self.group_advantages[record.sample_id]
            record.annotations.update(group_advantage=advantage, rl_eligible=eligible)
        if self.scorer is not None:
            self.scorer(record)
        return record

    def check_batch_budget(self, indices):
        total = sum(
            self.corpus.record_bytes(self.indices[0 if i is None else i])
            for i in indices
        )
        if total > self.max_batch_bytes:
            raise ValueError(
                f"Corpus batch requires {total} bytes, exceeding its {self.max_batch_bytes} byte budget"
            )

    def validate_policy_groups(self, *, advantages="recorded"):
        import math

        groups = {}
        for fields in self.corpus.record_metadata:
            if advantages == "recorded":
                self.group_advantages[fields["sample_id"]] = (
                    None,
                    fields["status"] != "generation_failed"
                    and fields["quality_trainable"] is not False,
                )
                continue
            group_id, size, slot = (
                fields["group_id"],
                fields["group_size"],
                fields["group_slot"],
            )
            if group_id is None or size is None or slot is None or not 0 <= slot < size:
                raise ValueError(
                    "Recorded policy groups require an explicit group identity, size and slot"
                )
            groups.setdefault(group_id, []).append(fields)
        for group in groups.values():
            size = group[0]["group_size"]
            if any(row["group_size"] != size for row in group) or sorted(
                row["group_slot"] for row in group
            ) != list(range(size)):
                raise ValueError(
                    "Recorded policy group is incomplete or contains duplicate slots"
                )
            eligible = [
                row
                for row in group
                if row["status"] != "generation_failed"
                and row["quality_trainable"] is not False
            ]
            if advantages == "group_standardized":
                rewards = [row["score"] for row in eligible]
                if any(
                    not isinstance(score, (int, float)) or not math.isfinite(score)
                    for score in rewards
                ):
                    raise ValueError(
                        "Group-standardized advantages require finite committed group scores"
                    )
                mean = sum(rewards) / max(len(rewards), 1)
                deviation = math.sqrt(
                    sum((score - mean) ** 2 for score in rewards) / max(len(rewards), 1)
                )
                values = {
                    row["sample_id"]: (score - mean) / (deviation + 1e-8)
                    for row, score in zip(eligible, rewards)
                }
            else:
                values = {}
            eligible_ids = (
                {row["sample_id"] for row in eligible} if len(eligible) >= 2 else set()
            )
            for row in group:
                self.group_advantages[row["sample_id"]] = (
                    values.get(row["sample_id"]),
                    row["sample_id"] in eligible_ids,
                )


class RecordScorer:
    """Reuse scalar reward functions over owned records without a generator."""

    def __init__(self, config, packer, reward_fns=None):
        from cosmos_rl.dispatcher.algo.reward import Reward

        source = config.execution.source
        if not source.reward_function and not reward_fns:
            raise ValueError(
                "Record scoring requires a configured reward function or application reward_fns"
            )
        self.scorer = Reward(
            config,
            tokenier=getattr(packer, "tokenizer", None),
            reward_function=source.reward_function or {},
            explicit_reward_fn=reward_fns,
            explicit_filter_reward_fn=[],
            data_packer=packer,
        )
        self.revision = source.scorer_revision

    def __call__(self, record):
        import math

        prompt = (
            record.inputs.get("prompt")
            if isinstance(record.inputs, dict)
            else record.inputs
        )
        scores, _, metrics = self.scorer.compute_reward(
            [record.targets],
            record.annotations.get("reference"),
            prompt=prompt,
            record=record,
        )
        score = float(scores[0])
        if not math.isfinite(score):
            raise ValueError("Record scoring returned a nonfinite reward")
        record.annotations.update(
            score=score, scorer_metrics=metrics[0], scorer_revision=self.revision
        )


class ExactCorpusSampler:
    """Pad distributed schedules with masked occurrences, never real samples."""

    def __init__(
        self, dataset, *, num_replicas, rank, shuffle=False, drop_last=False, seed=0
    ):
        self.dataset, self.replicas, self.rank = dataset, num_replicas, rank
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0
        if drop_last:
            raise ValueError("Corpus scheduling does not drop training records")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        import math

        return math.ceil(len(self.dataset) / self.replicas)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            import random

            random.Random(self.seed + self.epoch).shuffle(indices)
        indices += [None] * (len(self) * self.replicas - len(indices))
        yield from indices[self.rank :: self.replicas]


def prepare_supervised_record(record, packer, *, context=None):
    """Keep application packers in charge of model-specific representations."""
    custom = getattr(packer, "prepare_record", None)
    if custom is not None:
        prepared = (
            custom(record, context=context) if context is not None else custom(record)
        )
    elif isinstance(record.targets, dict) and "label_ids" in record.targets:
        prepared = copy.deepcopy(record.targets)
    elif isinstance(record.targets, str):
        inputs = record.inputs
        conversation = inputs.get("conversation") if isinstance(inputs, dict) else None
        prompt = inputs.get("prompt", "") if isinstance(inputs, dict) else inputs
        messages = (
            copy.deepcopy(conversation)
            if conversation
            else [{"role": "user", "content": prompt}]
        )
        messages.append({"role": "assistant", "content": record.targets})
        prepared = packer.sft_process_sample(messages)
    else:
        raise ValueError(
            "This corpus representation requires a packer.prepare_record adapter"
        )
    if not isinstance(prepared, dict):
        raise TypeError("prepare_record must return a trainer input mapping")
    if record.annotations.get("empty_contribution") and "label_ids" in prepared:
        prepared["label_ids"] = [-100] * len(prepared["label_ids"])
    prepared["_execution_record"] = record
    return prepared


def source_compatibility(dataset, execution, training=None, *, legacy=False):
    """Stable input/objective identity persisted with learner checkpoints."""
    settings = execution.model_dump(mode="json")
    if settings.get("source") is not None:
        # Location and admission do not identify training data or its schedule.
        # Prepared-batch limits remain pinned: they can change window grouping.
        for key in (
            ("allow_partial",)
            if legacy
            else ("path", "max_prefetch_bytes", "allow_partial", "revision")
        ):
            settings["source"].pop(key, None)
    if (
        settings.get("source") is not None
        and settings["source"].get("max_prepared_bytes") is None
    ):
        # Preserve checkpoint compatibility when the new optional limit is unused.
        settings["source"].pop("max_prepared_bytes", None)
    if (
        settings.get("source") is not None
        and settings["source"].get("preparation_seed") is None
    ):
        # Keep existing deterministic-preparation checkpoint fingerprints.
        settings["source"].pop("preparation_seed", None)
    value = {
        "source_revision": dataset.revision,
        "execution": settings,
        "training": training,
    }
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
