# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Explicit model relationships; learner steps are not actor identities."""


def checkpoint_actor(config):
    execution = getattr(config, "execution", None)
    actor = getattr(execution, "actor", None)
    return getattr(actor, "weight_source", None) == "checkpoint"


def local_actor_weights(config):
    return getattr(
        getattr(config, "rollout", None), "rollout_only", False
    ) is True or checkpoint_actor(config)


def actor_config(config):
    """Project actor model settings without mutating learner configuration."""
    if not checkpoint_actor(config):
        return config
    projected = config.model_copy(deep=True)
    actor = config.execution.actor
    if actor.model_name_or_path:
        projected.policy.model_name_or_path = actor.model_name_or_path
        projected.policy.model_revision = actor.model_revision
    elif actor.model_revision:
        projected.policy.model_revision = actor.model_revision
    return projected


def validate_actor_representation(config, tokenizer_factory=None):
    """Independent likelihood ratios require the same token event space.

    The initial adapter accepts identical fast-tokenizer pipelines. Other
    mappings need an explicit model/algorithm adapter rather than guessing
    from vocabulary size or matching sequence lengths.
    """
    if not checkpoint_actor(config) or config.execution.path == "generate":
        return
    actor = config.execution.actor
    name = actor.model_name_or_path or config.policy.model_name_or_path
    revision = (
        actor.model_revision
        if actor.model_name_or_path
        else actor.model_revision or config.policy.model_revision
    )
    if (name, revision) == (
        config.policy.model_name_or_path,
        config.policy.model_revision,
    ):
        return
    if tokenizer_factory is None:
        from transformers import AutoTokenizer

        tokenizer_factory = AutoTokenizer.from_pretrained
    learner = tokenizer_factory(
        config.policy.model_name_or_path,
        revision=config.policy.model_revision,
        trust_remote_code=True,
    )
    generator = tokenizer_factory(name, revision=revision, trust_remote_code=True)

    def signature(tokenizer):
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError(
                "Independent model token alignment requires compatible fast tokenizers"
            )
        return (backend.to_str(), tokenizer.special_tokens_map, tokenizer.chat_template)

    if signature(learner) != signature(generator):
        raise ValueError("Independent actor and learner token representations differ")
