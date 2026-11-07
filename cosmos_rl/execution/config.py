# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Small, declarative execution configuration; no model or transport imports."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["dataset", "corpus"] = "dataset"
    path: str | None = None
    revision: str | None = None
    max_prefetch_bytes: int = Field(default=256 * 1024**2, gt=0)
    reward_weighting: Literal["none", "nonnegative"] = "none"
    teacher_targets: Literal["recorded", "online"] = "recorded"
    score_records: bool = False
    reward_function: str | dict[str, float] | None = None
    scorer_revision: str | None = None
    advantages: Literal["recorded", "group_standardized"] = "recorded"

    @model_validator(mode="after")
    def check(self):
        if self.kind == "corpus" and not self.path:
            raise ValueError("A corpus source requires execution.source.path")
        if self.score_records and self.advantages == "group_standardized":
            raise ValueError(
                "Group advantages use committed scores; commit a rescored corpus before group standardization"
            )
        return self


class ObjectiveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supervised: float = Field(default=0, ge=0, allow_inf_nan=False)
    policy_gradient: float = Field(default=0, ge=0, allow_inf_nan=False)
    distillation: float = Field(default=0, ge=0, allow_inf_nan=False)
    policy_clip: float = Field(default=0.2, gt=0, lt=1, allow_inf_nan=False)


class ActorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weight_source: Literal["learner", "checkpoint"] = "learner"
    model_name_or_path: str | None = None
    model_revision: str | None = None
    artifact_digest: str | None = None


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    run_id: str
    max_record_bytes: int = Field(default=1024**3, gt=0)

    @model_validator(mode="after")
    def check(self):
        if not self.path or not self.run_id:
            raise ValueError("An output requires a nonempty path and logical run_id")
        return self


class ExecutionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Literal["train_from_data", "train_from_generation", "generate"] | None = None
    source: SourceConfig | None = None
    objectives: ObjectiveConfig | None = None
    actor: ActorConfig | None = None
    output: OutputConfig | None = None


@dataclass(frozen=True)
class ExecutionPlan:
    path: str
    learners: int
    generators: int
    teachers: int
    actor_weight_source: str | None
    supervised: float
    policy_gradient: float
    distillation: float

    @property
    def sync_actor_weights(self):
        return self.generators > 0 and self.actor_weight_source == "learner"


def resolve_execution(config) -> ExecutionPlan:
    """Project old and new recipes into the same role and objective contract.

    Legacy configuration keeps its historical counts/algorithms. Explicit
    execution paths opt into validation and can never silently allocate a role
    which the selected path does not need.
    """
    execution = config.execution
    if execution.path is None and any(
        value is not None
        for value in (
            execution.source,
            execution.objectives,
            execution.actor,
            execution.output,
        )
    ):
        raise ValueError("New execution options require an explicit execution.path")
    policy = config.train.train_policy
    is_sft = policy.type == "sft"
    path = execution.path or (
        "generate"
        if config.rollout.rollout_only
        else "train_from_data"
        if is_sft
        else "train_from_generation"
    )
    objectives = execution.objectives
    supervised = objectives.supervised if objectives else float(is_sft)
    pg = (
        objectives.policy_gradient
        if objectives
        else float(not is_sft and path != "generate")
    )
    kd = objectives.distillation if objectives else float(config.distillation.enable)
    learners = 0 if path == "generate" else config.policy.parallelism.n_init_replicas
    generators = (
        config.rollout.parallelism.n_init_replicas if path != "train_from_data" else 0
    )
    recorded_teacher = (
        path == "train_from_data"
        and execution.source is not None
        and execution.source.teacher_targets == "recorded"
    )
    teachers = (
        config.distillation.parallelism.n_init_replicas
        if kd > 0 and not recorded_teacher
        else 0
    )
    actor_source = (
        None
        if not generators
        else (
            execution.actor.weight_source
            if execution.actor
            else "checkpoint"
            if path == "generate"
            else "learner"
        )
    )
    if execution.path is not None:
        if path == "generate":
            if config.policy.parallelism.n_init_replicas != 0:
                raise ValueError("generate requires zero learner replicas")
            if supervised or pg or kd:
                raise ValueError("generate cannot enable training objectives")
        elif learners < 1 or not (supervised or pg or kd):
            raise ValueError("Training requires a learner and an enabled objective")
        if path == "train_from_data":
            if config.rollout.parallelism.n_init_replicas != 0:
                raise ValueError("train_from_data requires zero rollout replicas")
            if execution.actor is not None:
                raise ValueError("train_from_data has no actor")
            if (
                execution.source is not None
                and execution.source.kind == "corpus"
                and learners != 1
            ):
                raise ValueError(
                    "Corpus training initially supports one learner replica with data parallelism"
                )
            if (
                execution.source is not None
                and execution.source.kind == "corpus"
                and any(
                    getattr(config.policy.parallelism, d) != 1
                    for d in ("tp_size", "pp_size", "cp_size", "ep_size")
                )
            ):
                raise ValueError(
                    "Corpus training initially requires pure data parallelism"
                )
            if kd and not recorded_teacher:
                raise ValueError(
                    "Data training requires recorded teacher targets; use live OPD for online teacher scoring"
                )
        elif generators < 1:
            raise ValueError("Generation requires at least one rollout replica")
        if kd and not recorded_teacher and not config.distillation.enable:
            raise ValueError("A distillation objective requires a configured teacher")
        if execution.output is not None and path != "generate":
            raise ValueError("execution.output currently belongs to generate")
        if execution.source is not None and path != "train_from_data":
            raise ValueError("execution.source belongs to train_from_data")
        if execution.source is not None and execution.source.kind != "corpus":
            if execution.source.path is not None or execution.source.score_records:
                raise ValueError(
                    "Corpus paths and record scoring require source.kind=corpus; ordinary datasets use train.train_policy.dataset"
                )
        if actor_source == "learner" and execution.actor is not None:
            actor = execution.actor
            if actor.model_name_or_path not in (None, config.policy.model_name_or_path):
                raise ValueError(
                    "Learner weight synchronization requires the learner model"
                )
            if actor.model_revision not in (None, config.policy.model_revision):
                raise ValueError(
                    "A synchronized actor cannot use a different model revision"
                )
        if actor_source == "checkpoint" and path == "train_from_generation":
            if getattr(policy, "on_policy", False):
                raise ValueError("An independent checkpoint actor is not on-policy")
            if (
                config.mode != "disaggregated"
                or config.validation.enable
                or config.rollout.async_r2r_sync != "disabled"
                or config.rollout.backend == "trtllm"
                or getattr(policy, "data_dispatch_as_rank_in_mesh", False)
                or getattr(policy, "variant", "grpo") != "grpo"
            ):
                raise ValueError(
                    "Independent actor training initially supports disaggregated shared-queue GRPO without validation or weight sync"
                )
            if (
                not policy.use_rollout_logprobs_for_loss
                or policy.max_retry_for_on_policy > 0
            ):
                raise ValueError(
                    "Independent actors require original rollout log probabilities for loss and no learner-version retry budget"
                )
        if path == "train_from_data" and not is_sft:
            raise ValueError(
                "Data training uses the SFT worker with execution objectives; live GRPO workers require generation"
            )
        if path == "generate" and actor_source != "checkpoint":
            raise ValueError("Generation without a learner requires checkpoint weights")
        if path == "train_from_generation" and objectives is not None:
            raise ValueError(
                "Live GRPO and OPD use their existing trainer settings; explicit combined objectives currently belong to train_from_data"
            )
        if path == "train_from_data" and objectives is not None:
            if policy.type != "sft" or policy.trainer_type != "sft":
                raise ValueError(
                    "Recorded token objectives currently use the native sft trainer"
                )
            if (
                any(
                    getattr(config.policy.parallelism, d) != 1
                    for d in ("tp_size", "pp_size", "cp_size", "ep_size")
                )
                or learners != 1
            ):
                raise ValueError(
                    "Recorded token objectives currently require one pure-DP learner replica"
                )
            if config.train.sequence_packing or policy.enable_dp_load_balancing:
                raise ValueError(
                    "Recorded token objectives require unpacked fixed-schedule batches"
                )
            if config.policy.enable_liger_fused_cross_entropy:
                raise ValueError(
                    "Recorded token objectives require materialized logits; disable fused cross entropy"
                )
    return ExecutionPlan(
        path, learners, generators, teachers, actor_source, supervised, pg, kd
    )


def normalize_execution(data: dict) -> dict:
    """Fill only absent role counts for explicit paths before config validation."""
    execution = data.get("execution", {})
    if isinstance(execution, ExecutionConfig):
        execution = execution.model_dump()
    if not isinstance(execution, dict):
        return data
    execution = {
        key: value.model_dump() if isinstance(value, BaseModel) else value
        for key, value in execution.items()
    }
    path = execution.get("path")
    if path == "generate":
        data.setdefault("rollout", {})["rollout_only"] = True
        data.setdefault("policy", {}).setdefault("parallelism", {}).setdefault(
            "n_init_replicas", 0
        )
    elif path == "train_from_data":
        data.setdefault("rollout", {}).setdefault("parallelism", {}).setdefault(
            "n_init_replicas", 0
        )
        source = execution.get("source") or {}
        if isinstance(source, dict) and source.get("kind") == "corpus":
            data.setdefault("train", {}).setdefault("train_policy", {}).setdefault(
                "dataloader_drop_last", False
            )
    objectives = execution.get("objectives")
    if objectives is not None and objectives.get("distillation", 0) == 0:
        data.setdefault("distillation", {})["enable"] = False
    source = execution.get("source") or {}
    if (
        path == "train_from_data"
        and isinstance(source, dict)
        and source.get("kind") == "corpus"
        and source.get("teacher_targets", "recorded") == "recorded"
    ):
        data.setdefault("distillation", {})["enable"] = False
    return data
