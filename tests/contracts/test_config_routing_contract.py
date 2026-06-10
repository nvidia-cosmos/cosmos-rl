# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-routing contract for shipped TOMLs.

Two regression cases this guards against, both of which silently
mis-route a launch (the user gets a worker-init error rather than a
config error):

1. **Top-level keys swallowed by the preceding ``[section]`` header.**
   In TOML, a top-level key like ``mode = "colocated"`` placed *after*
   any ``[section]`` header (e.g. after ``[env]``) is parsed as a
   member of that section, not as a top-level field.  Pydantic then
   ignores the misplaced key and ``Config.mode`` falls back to its
   default of ``"disaggregated"``.  ``policy_entry.py`` then routes
   the worker through ``RLPolicyWorker`` instead of
   ``ColocatedRLControlWorker`` and the user sees a confusing
   downstream error.

2. **Heuristic override of the discriminated union ``train_policy.type``.**
   ``Config.preprocess`` (in ``cosmos_rl/policy/config/__init__.py``)
   silently rewrites ``train.train_policy.type`` based on whether any
   GRPO-characteristic field is present.  This means an explicit
   ``type = "grpo"`` can be flipped to ``"sft"`` if the config lacks
   any of ``temperature / epsilon_low / epsilon_high / kl_beta /
   use_remote_reward``.  This caught us when wiring up a non-LLM RL
   trainer (``gym_pg``) which doesn't naturally use those fields.

For each shipped TOML the test asserts:

* If the raw TOML sets a top-level ``mode``, ``Config.from_dict(...)``
  preserves it.
* If the raw TOML's ``[train.train_policy]`` sets ``type``,
  ``Config.from_dict(...)`` preserves it.

Configs that are not full cosmos-rl ``Config`` documents (e.g. the
reward-service rewards toml) are excluded by glob; configs that
intentionally exercise non-default ``mode`` should set ``mode`` at
the very top of the file.
"""

from __future__ import annotations

import glob
import os
import unittest
from typing import Any, Dict, List

import toml


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _shipped_config_paths() -> List[str]:
    """Glob the cosmos-rl config corpus.

    Includes:
      * ``configs/**/*.toml``           -- canonical model configs
      * ``cosmos_rl/tools/**/configs/*.toml`` -- per-tool configs
      * ``tests/configs/*.toml``        -- test fixtures

    Excludes the reward-service rewards.toml (not a ``Config`` doc).
    """
    patterns = [
        os.path.join(REPO_ROOT, "configs", "**", "*.toml"),
        os.path.join(REPO_ROOT, "cosmos_rl", "tools", "**", "configs", "*.toml"),
        os.path.join(REPO_ROOT, "tests", "configs", "*.toml"),
    ]
    paths: List[str] = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))
    # The reward service ships a TOML that is NOT a cosmos-rl Config.
    return sorted(p for p in paths if "reward_service" not in p)


def _raw_top_level_keys(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Return only the top-level scalar keys (not section dicts).

    A ``[section]`` header in TOML produces a nested dict in the raw
    parse; everything else (string / int / bool at the top level) is
    a true top-level key.
    """
    return {k: v for k, v in raw.items() if not isinstance(v, dict)}


class TestShippedConfigRoutingContract(unittest.TestCase):
    """The test corpus is non-empty and every entry round-trips."""

    @classmethod
    def setUpClass(cls):
        from cosmos_rl.policy.config import Config as CosmosConfig

        cls.CosmosConfig = CosmosConfig
        cls.config_paths = _shipped_config_paths()

    def test_corpus_is_nonempty(self):
        # If this fails, either the glob is wrong or someone moved
        # the configs; either way, refusing to silently pass.
        self.assertGreater(
            len(self.config_paths),
            0,
            f"No configs found under {REPO_ROOT}/configs or tools/**/configs",
        )

    def _rel(self, path: str) -> str:
        return os.path.relpath(path, REPO_ROOT)

    def test_top_level_mode_is_preserved(self):
        """If a TOML sets a top-level ``mode``, the parsed Config keeps it.

        Catches the "section-bleed" bug where ``mode = "colocated"``
        placed after a ``[section]`` header is parsed into the section
        instead of the top level, leaving ``Config.mode`` at its
        default and silently mis-routing the launcher.
        """
        for path in self.config_paths:
            rel = self._rel(path)
            with self.subTest(path=rel):
                raw = toml.load(path)
                top = _raw_top_level_keys(raw)
                if "mode" not in top:
                    continue  # default-mode configs are fine
                cfg = self.CosmosConfig.from_dict(raw)
                self.assertEqual(
                    cfg.mode,
                    top["mode"],
                    f"Config.mode={cfg.mode!r} but raw TOML top-level "
                    f"set mode={top['mode']!r}; check that the top-level "
                    "``mode`` key is placed BEFORE any [section] header "
                    "(TOML parses post-section keys into that section).",
                )

    def test_explicit_train_policy_type_is_preserved(self):
        """If ``[train.train_policy]`` sets ``type``, parsing preserves it.

        Catches ``Config.preprocess``'s heuristic override that
        rewrites the discriminator based on the presence of
        GRPO-characteristic fields, silently flipping an explicit
        ``type = "grpo"`` to ``"sft"`` when the config lacks
        ``temperature / epsilon_low / epsilon_high / kl_beta /
        use_remote_reward``.
        """
        for path in self.config_paths:
            rel = self._rel(path)
            with self.subTest(path=rel):
                raw = toml.load(path)
                tp = raw.get("train", {}).get("train_policy", {})
                if "type" not in tp:
                    continue  # default-type configs are fine
                explicit_type = tp["type"]
                cfg = self.CosmosConfig.from_dict(raw)
                self.assertEqual(
                    cfg.train.train_policy.type,
                    explicit_type,
                    f"train_policy.type={cfg.train.train_policy.type!r} "
                    f"but raw TOML set type={explicit_type!r}; "
                    "Config.preprocess may have overridden it via the "
                    "GRPO-characteristic-field heuristic.  Add a "
                    "GRPO-characteristic field (e.g. kl_beta=0.0) or "
                    "fix Config.preprocess to honor explicit type.",
                )


if __name__ == "__main__":
    unittest.main()
