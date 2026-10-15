"""RoboTwin training can read rollout limits without installing LIBERO."""

import builtins
import runpy
from pathlib import Path
from unittest.mock import patch

import cosmos_rl


def test_rollout_limits_without_libero():
    original_import = builtins.__import__

    def reject_libero(name, *args, **kwargs):
        if name == "libero" or name.startswith("libero."):
            raise ModuleNotFoundError("LIBERO is deliberately unavailable")
        return original_import(name, *args, **kwargs)

    source = Path(cosmos_rl.__file__).resolve().parent / "simulators/libero/utils.py"
    with patch("builtins.__import__", side_effect=reject_libero):
        namespace = runpy.run_path(str(source))
        assert namespace["LIBERO_MAX_STEPS_MAP"]["libero_10"] == 512
