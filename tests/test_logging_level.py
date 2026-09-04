# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``COSMOS_LOG_LEVEL`` must be honoured even when a host app configured logging.

This regression fails SILENTLY -- the only symptom is missing output -- which
is why it needs a test rather than a code comment.  ``logger.hasHandlers()``
walks the ancestor chain, so a handler on the ROOT logger (which any embedding
application installs the moment it calls ``logging.basicConfig``) once made the
whole configuration block a no-op, and the level was never applied.

cosmos-rl's own entry point exports the variable before importing anything, so
it never saw this; only embedders did, and for them every ``logger.debug`` in
the package -- including the ``[Trace]`` payload-transport telemetry the
profiler parser consumes -- was unreachable.

Each case reloads the module in a FRESH interpreter: the configuration runs at
import time, and a reload in-process would inherit the singleton "cosmos"
logger's existing state and the ambient handlers of the test runner.
"""

import subprocess
import sys
import textwrap
import unittest


def _run(body: str, env_level: str = "DEBUG") -> str:
    """Run ``body`` in a fresh interpreter with COSMOS_LOG_LEVEL set."""
    script = textwrap.dedent(f"""
        import io, logging, os
        _lvl = {env_level!r}
        if _lvl:
            os.environ["COSMOS_LOG_LEVEL"] = _lvl
        else:
            os.environ.pop("COSMOS_LOG_LEVEL", None)
        {textwrap.indent(textwrap.dedent(body), " " * 8).strip()}
    """)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{proc.stderr[-2000:]}")
    # The package prints model-discovery noise to stdout on import; the probe
    # marks its own line so the assertion cannot match that instead.
    return proc.stdout


class TestLogLevelWithHostLoggingConfigured(unittest.TestCase):
    """The embedded case: a host handler exists before cosmos-rl is imported."""

    def test_level_is_applied(self):
        out = _run("""
            logging.basicConfig(level=logging.INFO, stream=io.StringIO())
            from cosmos_rl.utils.logging import logger
            print("PROBE", logging.getLevelName(logger.level),
                  logger.isEnabledFor(logging.DEBUG))
        """)
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(
            probe,
            "PROBE DEBUG True",
            "COSMOS_LOG_LEVEL was ignored because a host handler existed",
        )

    def test_debug_records_reach_the_host_handler(self):
        # Setting the level is only useful if records actually arrive: the
        # cosmos logger keeps propagating to the host's handlers, and
        # propagation consults handler levels, not ancestor logger levels.
        out = _run("""
            stream = io.StringIO()
            logging.basicConfig(level=logging.INFO, stream=stream)
            from cosmos_rl.utils.logging import logger
            logger.debug("CANARY-DEBUG")
            print("PROBE", "CANARY-DEBUG" in stream.getvalue())
        """)
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(
            probe, "PROBE True", "a DEBUG record never reached the host handler"
        )


class TestLogLevelStandalone(unittest.TestCase):
    """The path cosmos-rl's own launcher takes must not regress."""

    def test_level_applied_and_own_handler_installed(self):
        out = _run("""
            from cosmos_rl.utils.logging import logger
            print("PROBE", logging.getLevelName(logger.level),
                  len(logger.handlers) >= 1, logger.propagate)
        """)
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        # Own handler installed, and propagate disabled so records are not
        # also emitted by an ancestor.
        self.assertEqual(probe, "PROBE DEBUG True False")


class TestLogLevelFallback(unittest.TestCase):
    def test_non_level_attribute_falls_back_to_info(self):
        # "basic_format" upper-cases to BASIC_FORMAT, which IS an attribute of
        # ``logging`` -- but a str, not a level.  Without the isinstance guard
        # getattr returns it and setLevel raises ValueError at import time,
        # taking down every process that sets the variable to a typo.
        out = _run(
            """
            from cosmos_rl.utils.logging import logger
            print("PROBE", logging.getLevelName(logger.level))
            """,
            env_level="basic_format",
        )
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(probe, "PROBE INFO")


class TestConfigDrivenLevel(unittest.TestCase):
    """[logging].level must reach every component that loads the config.

    Workers on other machines never see the submitting shell's environment
    unless the batch template happens to forward it, so the config -- which is
    distributed to all of them -- is the reliable channel.  It can only be
    applied after parse, because logging is configured at import time.
    """

    def test_config_level_is_applied(self):
        out = _run(
            """
            from cosmos_rl.utils.logging import logger, configure_logging
            print("PROBE-BEFORE", logging.getLevelName(logger.level))
            configure_logging("DEBUG")
            print("PROBE-AFTER", logging.getLevelName(logger.level),
                  logger.isEnabledFor(logging.DEBUG))
            """,
            env_level="",
        )
        before = [ln for ln in out.splitlines() if ln.startswith("PROBE-BEFORE")][-1]
        after = [ln for ln in out.splitlines() if ln.startswith("PROBE-AFTER")][-1]
        self.assertEqual(before, "PROBE-BEFORE INFO")
        self.assertEqual(after, "PROBE-AFTER DEBUG True")

    def test_env_overrides_config(self):
        # The operator's override must win: raising verbosity for one run
        # cannot require editing a config shared by every component.
        out = _run("""
            from cosmos_rl.utils.logging import logger, configure_logging
            configure_logging("WARNING")
            print("PROBE", logging.getLevelName(logger.level))
        """)  # env_level defaults to DEBUG
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(probe, "PROBE DEBUG")

    def test_none_leaves_level_untouched(self):
        out = _run(
            """
            from cosmos_rl.utils.logging import logger, configure_logging
            configure_logging(None)
            print("PROBE", logging.getLevelName(logger.level))
            """,
            env_level="",
        )
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(probe, "PROBE INFO")

    def test_own_handler_level_follows(self):
        # Raising the logger's level is useless if the handler we installed
        # still filters the records out.
        out = _run(
            """
            from cosmos_rl.utils.logging import logger, configure_logging
            configure_logging("DEBUG")
            print("PROBE", [logging.getLevelName(h.level) for h in logger.handlers])
            """,
            env_level="",
        )
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(probe, "PROBE ['DEBUG']")


class TestConfigFileEndToEnd(unittest.TestCase):
    """The whole path: a real job config on disk -> the cosmos logger.

    Exercises the actual choke point (``Config.from_dict``) rather than the
    helper in isolation, because that call is what every component -- including
    workers on other nodes -- goes through.
    """

    def test_level_from_a_real_config_file(self):
        out = _run(
            """
            import glob, toml
            from cosmos_rl.policy.config import Config
            from cosmos_rl.utils.logging import logger
            data = toml.load(sorted(glob.glob("tests/configs/*.toml"))[0])
            data.setdefault("logging", {})["level"] = "DEBUG"
            print("PROBE-BEFORE", logging.getLevelName(logger.level))
            Config.from_dict(data)
            print("PROBE-AFTER", logging.getLevelName(logger.level))
            """,
            env_level="",
        )
        before = [ln for ln in out.splitlines() if ln.startswith("PROBE-BEFORE")][-1]
        after = [ln for ln in out.splitlines() if ln.startswith("PROBE-AFTER")][-1]
        self.assertEqual(before, "PROBE-BEFORE INFO")
        self.assertEqual(after, "PROBE-AFTER DEBUG")

    def test_config_without_level_leaves_logging_alone(self):
        # The field is opt-in: loading a config that does not set it must not
        # touch a level the host or the environment already chose.
        out = _run(
            """
            import glob, toml
            from cosmos_rl.policy.config import Config
            from cosmos_rl.utils.logging import logger
            logger.setLevel(logging.WARNING)
            data = toml.load(sorted(glob.glob("tests/configs/*.toml"))[0])
            data.pop("logging", None)
            Config.from_dict(data)
            print("PROBE", logging.getLevelName(logger.level))
            """,
            env_level="",
        )
        probe = [ln for ln in out.splitlines() if ln.startswith("PROBE")][-1]
        self.assertEqual(probe, "PROBE WARNING")


class TestSlurmTemplateDoesNotForceDebug(unittest.TestCase):
    """The controller block must not pin its own verbosity.

    It used to ``export COSMOS_LOG_LEVEL=DEBUG`` unconditionally, making the
    controller the only component whose level could not be lowered and leaving
    it inconsistent with the policy/rollout/reference blocks in the same file.
    """

    def test_no_hardcoded_level_in_multi_node_template(self):
        import pathlib

        template = (
            pathlib.Path(__file__).resolve().parents[1]
            / "cosmos_rl/tools/slurm/cosmos_rl_job_multi_node.sh"
        )
        body = template.read_text()
        self.assertNotIn(
            "export COSMOS_LOG_LEVEL=DEBUG",
            body,
            "the controller block pins COSMOS_LOG_LEVEL again",
        )


if __name__ == "__main__":
    unittest.main()
