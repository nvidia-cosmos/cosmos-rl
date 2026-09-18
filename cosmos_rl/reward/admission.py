# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from numbers import Real
import re
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Sequence
import uuid

from cosmos_rl.dispatcher.data.schema import RLPayload

if TYPE_CHECKING:
    from cosmos_rl.rollout.schema import RolloutResult


COMPLETION_ADMISSION_METRIC_PREFIX = "rollout/completion_admission_"
COMPLETION_ADMISSION_REPORT_ID_KEY = "completion_admission_report_id"
COMPLETION_ADMISSION_WEIGHT_VERSION_KEY = "completion_admission_weight_version"
DISCARDED_SAMPLES_KEY = "discarded_samples"
DISCARD_REPORT_ID_KEY = "discard_report_id"
DISCARDED_WEIGHT_VERSION_KEY = "discarded_weight_version"


_COMPLETION_ALIGNED_FIELDS = (
    "completions",
    "completed_conversations",
    "n_ignore_prefix_tokens",
    "rewards",
    "advantages",
    "filter_rewards",
    "completion_token_ids",
    "completion_logprobs",
    "cumulative_logprob",
    "report_metrics",
    "teacher_result_uuids",
    "completion_sequences",
)

_ROLLOUT_RESULT_COMPLETION_ALIGNED_FIELDS = (
    "completions",
    "completed_conversations",
    "completion_trainable",
    "completion_drop_reasons",
    "completion_logprobs",
    "completion_token_ids",
    "cumulative_logprob",
)

_KNOWN_DROP_REASONS = frozenset(
    {
        "empty_completion",
        "generation_error",
        "invalid_completion",
        "missing_log_probs",
        "timeout",
        "unspecified",
    }
)


def _metric_reason(reason: Optional[str]) -> str:
    value = (reason or "unspecified").strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    value = value or "unspecified"
    return value if value in _KNOWN_DROP_REASONS else "other"


def _metric_component(value: Any) -> str:
    component = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return (component or "metric")[:64]


def aggregate_excluded_reward_metrics(
    reward_metrics: Iterable[Optional[Dict[str, Any]]],
) -> Dict[str, int | float]:
    """Aggregate numeric telemetry for completions excluded from training.

    Admission metadata is the only reporting path that survives when an entire
    group is rejected, so emit sum/count pairs there. Non-numeric and non-finite
    values are intentionally skipped; they cannot be safely reduced across
    workers.
    """

    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for metrics in reward_metrics:
        for key, value in (metrics or {}).items():
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except (RuntimeError, TypeError, ValueError):
                    continue
            if isinstance(value, bool) or not isinstance(value, Real):
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                continue
            component = _metric_component(key)
            sums[component] = sums.get(component, 0.0) + numeric
            counts[component] = counts.get(component, 0) + 1

    aggregate: Dict[str, int | float] = {}
    for component, total in sums.items():
        prefix = f"rollout/completion_admission_excluded_reward_{component}"
        aggregate[f"{prefix}_sum"] = total
        aggregate[f"{prefix}_count"] = counts[component]
    return aggregate


@dataclass(frozen=True)
class CompletionAdmission:
    """Resolved training admission for one completion group."""

    explicit: bool
    original_size: int
    eligible_indices: List[int]
    excluded_indices: List[int]
    drop_reason_counts: Dict[str, int]
    minimum_trainable_completions: int
    group_excluded: bool

    @property
    def eligible_size(self) -> int:
        return len(self.eligible_indices)

    @property
    def training_discarded_count(self) -> int:
        if not self.explicit:
            return 0
        return self.original_size if self.group_excluded else len(self.excluded_indices)

    def metrics(self) -> Dict[str, int]:
        if not self.explicit:
            return {}
        metrics = {
            "rollout/completion_admission_group_count": 1,
            "rollout/completion_admission_original_count": self.original_size,
            "rollout/completion_admission_eligible_count": self.eligible_size,
            "rollout/completion_admission_excluded_count": len(self.excluded_indices),
            "rollout/completion_admission_insufficient_group_count": int(
                self.group_excluded
            ),
            "rollout/completion_admission_training_discarded_count": self.training_discarded_count,
        }
        for reason, count in self.drop_reason_counts.items():
            metrics[f"rollout/completion_admission_reason_{reason}_count"] = count
        return metrics


def resolve_completion_admission(
    payload: RLPayload,
    minimum_trainable_completions: int,
    *,
    enabled: bool = True,
) -> CompletionAdmission:
    """Validate and resolve a producer-provided completion admission mask.

    Admission is explicit only when ``enabled`` is true and a mask is present.
    This distinction preserves the historical behavior for producers that do
    not provide a mask, including one-completion groups.
    """

    if payload.completions is None:
        raise ValueError("completion admission requires payload.completions")
    original_size = len(payload.completions)
    if minimum_trainable_completions < 1:
        raise ValueError("minimum_trainable_completions must be at least 1")

    # Validation and other admission-disabled paths ignore producer admission
    # metadata completely. Stale training-only masks must not make validation
    # fail before it can score every completion.
    if not enabled:
        return CompletionAdmission(
            explicit=False,
            original_size=original_size,
            eligible_indices=list(range(original_size)),
            excluded_indices=[],
            drop_reason_counts={},
            minimum_trainable_completions=minimum_trainable_completions,
            group_excluded=False,
        )

    mask = payload.completion_trainable
    reasons = payload.completion_drop_reasons

    if mask is not None and len(mask) != original_size:
        raise ValueError(
            "completion_trainable must have the same length as completions: "
            f"got {len(mask)} and {original_size}"
        )
    if reasons is not None and len(reasons) != original_size:
        raise ValueError(
            "completion_drop_reasons must have the same length as completions: "
            f"got {len(reasons)} and {original_size}"
        )

    explicit = mask is not None
    if not explicit:
        return CompletionAdmission(
            explicit=False,
            original_size=original_size,
            eligible_indices=list(range(original_size)),
            excluded_indices=[],
            drop_reason_counts={},
            minimum_trainable_completions=minimum_trainable_completions,
            group_excluded=False,
        )

    eligible_indices = [i for i, trainable in enumerate(mask) if trainable]
    excluded_indices = [i for i, trainable in enumerate(mask) if not trainable]
    reason_counts = Counter(
        _metric_reason(reasons[i] if reasons is not None else None)
        for i in excluded_indices
    )
    return CompletionAdmission(
        explicit=True,
        original_size=original_size,
        eligible_indices=eligible_indices,
        excluded_indices=excluded_indices,
        drop_reason_counts=dict(reason_counts),
        minimum_trainable_completions=minimum_trainable_completions,
        group_excluded=len(eligible_indices) < minimum_trainable_completions,
    )


def select_completion_aligned_value(
    value: Any,
    indices: Sequence[int],
    original_size: int,
    field_name: str,
) -> Any:
    if value is None:
        return None
    try:
        value_size = len(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be completion-aligned") from exc
    if value_size != original_size:
        raise ValueError(
            f"{field_name} must have the same length as completions: "
            f"got {value_size} and {original_size}"
        )
    if isinstance(value, list):
        return [value[i] for i in indices]
    if isinstance(value, tuple):
        return tuple(value[i] for i in indices)
    try:
        return value[list(indices)]
    except (IndexError, TypeError):
        return [value[i] for i in indices]


def select_completion_aligned_mapping(
    extra_info: Optional[Dict[str, Any]],
    indices: Sequence[int],
    original_size: int,
) -> Optional[Dict[str, Any]]:
    if extra_info is None:
        return None
    selected = {}
    for key, value in extra_info.items():
        is_aligned_sequence = (
            isinstance(value, (list, tuple)) and len(value) == original_size
        )
        is_aligned_array = (
            hasattr(value, "shape")
            and len(value.shape) > 0
            and value.shape[0] == original_size
        )
        selected[key] = (
            select_completion_aligned_value(
                value, indices, original_size, f"extra_info[{key!r}]"
            )
            if is_aligned_sequence or is_aligned_array
            else value
        )
    return selected


def select_rollout_result_completions(
    result: RolloutResult, indices: Sequence[int]
) -> RolloutResult:
    """Select every completion-aligned rollout-result field identically."""

    original_size = len(result.completions)
    selected = result.model_copy(deep=False)
    for field_name in _ROLLOUT_RESULT_COMPLETION_ALIGNED_FIELDS:
        setattr(
            selected,
            field_name,
            select_completion_aligned_value(
                getattr(result, field_name), indices, original_size, field_name
            ),
        )
    selected.extra_info = select_completion_aligned_mapping(
        result.extra_info, indices, original_size
    )
    return selected


def normalize_rollout_results(generated: Sequence[Any]) -> List[RolloutResult]:
    """Accept the standard RolloutResult contract and legacy completion lists."""

    from cosmos_rl.rollout.schema import RolloutResult

    return [
        result
        if isinstance(result, RolloutResult)
        else RolloutResult(completions=result)
        for result in generated
    ]


def apply_rollout_result_to_payload(
    payload: RLPayload,
    result: RolloutResult,
    *,
    include_completed_conversations: bool,
) -> RLPayload:
    """Propagate producer output, including admission, into its payload."""

    payload.completions = result.completions
    payload.completion_trainable = result.completion_trainable
    payload.completion_drop_reasons = result.completion_drop_reasons
    payload.completion_logprobs = result.completion_logprobs
    payload.completion_token_ids = result.completion_token_ids
    payload.prompt_logprobs = result.prompt_logprobs
    payload.prompt_token_ids = result.prompt_token_ids
    payload.cumulative_logprob = result.cumulative_logprob
    payload.extra_info = result.extra_info
    if include_completed_conversations:
        payload.completed_conversations = result.completed_conversations
    return payload


def select_payload_completions(
    payload: RLPayload,
    admission: CompletionAdmission,
) -> RLPayload:
    """Return the training payload selected by an admission decision.

    An insufficient group intentionally selects no completions, including the
    otherwise eligible ones, because the entire group is excluded.
    """

    selected = payload.model_copy(deep=False)
    selected_indices = [] if admission.group_excluded else admission.eligible_indices
    selected.completion_rejections = list(payload.completion_rejections)
    if payload.completion_sequences is not None:
        if len(payload.completion_sequences) != admission.original_size:
            raise ValueError("Completion reservations must be completion-aligned")
        selected_set = set(selected_indices)
        for index, sequence in enumerate(payload.completion_sequences):
            if index not in selected_set:
                selected.completion_rejections.append(
                    {
                        "sequence": sequence,
                        "weight_version": payload.weight_version,
                        "reason": (
                            "insufficient_group"
                            if admission.group_excluded
                            else (
                                payload.completion_drop_reasons
                                or [None] * admission.original_size
                            )[index]
                            or "quality"
                        ),
                        "completion": payload.completions[index],
                        **{
                            name: getattr(payload, name)[index]
                            if getattr(payload, name) is not None
                            else None
                            for name in (
                                "completed_conversations",
                                "completion_logprobs",
                                "completion_token_ids",
                            )
                        },
                    }
                )
    for field_name in _COMPLETION_ALIGNED_FIELDS:
        setattr(
            selected,
            field_name,
            select_completion_aligned_value(
                getattr(payload, field_name),
                selected_indices,
                admission.original_size,
                field_name,
            ),
        )
    selected.extra_info = select_completion_aligned_mapping(
        payload.extra_info, selected_indices, admission.original_size
    )
    # Admission has been consumed. Clearing the producer fields prevents an
    # already-filtered payload from being interpreted as a new group later.
    selected.completion_trainable = None
    selected.completion_drop_reasons = None
    selected.completion_admission_metrics = admission.metrics() or None
    return selected


def consume_completion_admission_metrics(
    payloads: List[RLPayload],
) -> tuple[List[RLPayload], Dict[str, int | float]]:
    """Collect admission counters and remove metadata-only rejected groups."""

    kept_payloads = []
    aggregate: Dict[str, int | float] = {}
    for payload in payloads:
        metrics = payload.completion_admission_metrics
        if metrics is None:
            kept_payloads.append(payload)
            continue
        for key, value in metrics.items():
            aggregate[key] = aggregate.get(key, 0) + value
        payload.completion_admission_metrics = None
        if metrics.get("rollout/completion_admission_insufficient_group_count", 0) == 0:
            kept_payloads.append(payload)
    return kept_payloads, aggregate


def prepare_completion_admission_report(
    metrics: Dict[str, int | float],
    weight_version: int,
    *,
    report_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Add stable routing and deduplication metadata to admission metrics.

    The returned dictionary is safe to reuse for an HTTP retry: its report ID
    is created once here rather than by the controller.  A training discard is
    carried in the same report so admission telemetry, in-flight settlement,
    and same-weight replacement capacity share one idempotency boundary.
    """

    report: Dict[str, Any] = dict(metrics)
    if not any(key.startswith(COMPLETION_ADMISSION_METRIC_PREFIX) for key in report):
        return report

    stable_report_id = report_id or uuid.uuid4().hex
    report[COMPLETION_ADMISSION_REPORT_ID_KEY] = stable_report_id
    report[COMPLETION_ADMISSION_WEIGHT_VERSION_KEY] = weight_version

    discarded_count = report.get(
        f"{COMPLETION_ADMISSION_METRIC_PREFIX}training_discarded_count", 0
    )
    if type(discarded_count) is int and discarded_count > 0:
        report[DISCARDED_SAMPLES_KEY] = discarded_count
        report[DISCARD_REPORT_ID_KEY] = stable_report_id
        report[DISCARDED_WEIGHT_VERSION_KEY] = weight_version
    return report
