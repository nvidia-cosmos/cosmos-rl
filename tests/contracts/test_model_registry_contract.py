# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model registry / weight-mapper alignment contract.

The rollout worker resolves the weight mapper for a model via two
registries that **must stay in lockstep**:

* ``ModelRegistry._MODEL_REGISTRY``        -- ``model_type`` -> model class
* ``WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY`` -- ``model_type`` -> mapper class

The standard registration path (``ModelRegistry.register_model``)
populates both at once, but a model can also be inserted directly
into ``_MODEL_REGISTRY`` (e.g. via ``register_local_model_config`` +
forgetting to call ``register_model``).  When the registries fall out
of sync, ``cosmos_rl/rollout/worker/rollout_control.py:170-175``
silently falls back to ``HFModelWeightMapper`` -- which is LLM-specific
and rejects any non-LLM ``hf_config`` (this is exactly the bug we hit
wiring up the ``cosmos_rl_gym_mlp`` model).

This file pins three properties on the joint registry state:

1. Every ``model_type`` in ``ModelRegistry`` has a corresponding
   ``WeightMapper`` registered (no silent ``HFModelWeightMapper``
   fallback).
2. Every registered weight-mapper class is a subclass of
   ``WeightMapper`` (catches a registration that snuck in a
   non-conforming class).
3. Every registered model class declares the ``model_type`` in its
   ``supported_model_types()`` (catches the case where someone
   registers ``MyModel`` for ``"my_model"`` but ``MyModel`` claims
   ``["other_model"]``; the rollout worker would then surface the
   wrong key in error messages).
"""

from __future__ import annotations

import unittest


def _registries():
    """Eagerly import the model packages so the registries are populated.

    Cosmos-rl auto-discovers models at ``import cosmos_rl.policy.model``
    time, but in a test process some optional imports may have been
    skipped.  We additionally trigger our gym example registration
    so the gym model is in the snapshot.
    """
    from cosmos_rl.policy.model import base as _model_base  # noqa: F401
    from cosmos_rl.policy.model.base import ModelRegistry, WeightMapper

    try:
        from cosmos_rl.tools.gym_example.gym_policy import register_gym_policy

        register_gym_policy()
    except Exception:
        # Optional: the gym example may not be importable in a CI
        # environment without ``gymnasium`` installed.
        pass

    return (
        dict(ModelRegistry._MODEL_REGISTRY),
        dict(WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY),
        WeightMapper,
    )


class TestModelRegistryWeightMapperAlignment(unittest.TestCase):
    """Both registries must agree on the set of registered model types."""

    def test_every_registered_model_type_has_a_weight_mapper(self):
        models, mappers, _ = _registries()
        self.assertTrue(models, "ModelRegistry is empty after imports")
        missing = sorted(set(models.keys()) - set(mappers.keys()))
        self.assertFalse(
            missing,
            f"Model types in ModelRegistry but not in "
            f"WeightMapper._MODEL_WEIGHT_MAPPER_REGISTRY: {missing}. "
            "These will silently fall back to HFModelWeightMapper "
            "(rollout_control.py:170-175), which is LLM-specific and "
            "will fail on non-LLM hf_configs.  Either call "
            "ModelRegistry.register_model(cls, mapper_cls) (populates "
            "both) or WeightMapper.register_class(model_type, mapper_cls).",
        )

    def test_every_registered_weight_mapper_is_a_weightmapper_subclass(self):
        _, mappers, WeightMapper = _registries()
        for model_type, mapper_cls in mappers.items():
            with self.subTest(model_type=model_type):
                self.assertTrue(
                    isinstance(mapper_cls, type),
                    f"Weight mapper for {model_type!r} is {mapper_cls!r}, "
                    "which is not a class.",
                )
                self.assertTrue(
                    issubclass(mapper_cls, WeightMapper),
                    f"Weight mapper {mapper_cls.__name__} for "
                    f"{model_type!r} does not inherit from WeightMapper.",
                )


class TestRegisteredModelDeclaresItsModelType(unittest.TestCase):
    """A model registered as ``model_type`` must claim ``model_type``."""

    def test_supported_model_types_round_trips(self):
        models, _, _ = _registries()
        for model_type, model_cls in models.items():
            with self.subTest(model_type=model_type):
                supported = getattr(model_cls, "supported_model_types", None)
                self.assertTrue(
                    callable(supported),
                    f"{model_cls.__name__} registered for "
                    f"{model_type!r} does not implement "
                    "``supported_model_types()``; ModelRegistry."
                    "register_model() requires it.",
                )
                claimed = supported()
                if isinstance(claimed, str):
                    claimed = [claimed]
                self.assertIn(
                    model_type,
                    claimed,
                    f"{model_cls.__name__} is registered for "
                    f"{model_type!r} but supported_model_types() "
                    f"claims {claimed!r}.  The registry key and the "
                    "class's claimed types must agree, otherwise the "
                    "rollout worker's error messages name the wrong "
                    "type when resolution fails.",
                )


if __name__ == "__main__":
    unittest.main()
