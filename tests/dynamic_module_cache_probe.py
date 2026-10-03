# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic diagnostic of a shared Transformers module-copy/load race.

This does not change Cosmos-RL or installed Transformers. A separate process
reproduces the truncate/write publication window of shutil.copy while the real
dynamic loader executes a module. The observed production cause remains an
inference unless the original interleaving is directly captured.
"""

import importlib.util
import multiprocessing
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch


def overwrite(path, contents, begin, empty, restore, restored):
    assert begin.wait(30)
    with open(path, "w") as stream:
        stream.flush()
        empty.set()
        assert restore.wait(30)
        stream.write(contents)
    restored.set()


def main():
    import transformers
    from transformers import dynamic_module_utils as dynamic

    print("TRANSFORMERS", transformers.__version__, flush=True)
    context = multiprocessing.get_context("spawn")
    contents = "class RaceConfig:\n    value = 17\n"
    with tempfile.TemporaryDirectory(prefix="cosmos-dynamic-probe-") as directory:
        root = Path(directory)
        path = root / "configuration_probe.py"
        path.write_text(contents)
        events = [context.Event() for _ in range(4)]
        begin, empty, restore, restored = events
        writer = context.Process(target=overwrite, args=(path, contents, *events))
        writer.start()
        original_spec = importlib.util.spec_from_file_location
        injected = False

        def interrupted_spec(name, *, location):
            nonlocal injected
            spec = original_spec(name, location=location)
            if not injected and Path(location) == path:
                execute = spec.loader.exec_module

                def interrupted_execute(module):
                    nonlocal injected
                    injected = True
                    begin.set()
                    assert empty.wait(30)
                    # get_class_in_module already hashed the complete source.
                    # A competing publisher temporarily truncated it afterward.
                    execute(module)
                    restore.set()
                    assert restored.wait(30)

                spec.loader.exec_module = interrupted_execute
            return spec

        try:
            with patch.object(dynamic, "HF_MODULES_CACHE", directory):
                with patch.object(
                    importlib.util, "spec_from_file_location", interrupted_spec
                ):
                    failures = 0
                    for attempt in range(3):
                        try:
                            dynamic.get_class_in_module("RaceConfig", path.name)
                        except AttributeError as error:
                            assert "RaceConfig" in str(error)
                            failures += 1
                            print(
                                "POISONED_RETRY",
                                attempt,
                                type(error).__name__,
                                flush=True,
                            )
                        else:
                            raise AssertionError("expected missing-class failure")
                    assert failures == 3 and path.read_text() == contents
                # A fresh import of the now-complete immutable source is healthy.
                recovered = dynamic.get_class_in_module(
                    "RaceConfig", path.name, force_reload=True
                )
                assert recovered.value == 17
                # Repeated imports of an already-published module are healthy.
                assert dynamic.get_class_in_module("RaceConfig", path.name) is recovered
                print("PUBLISHED_SOURCE_CONTROL_PASS", flush=True)
        finally:
            restore.set()
            writer.join(5)
            if writer.is_alive():
                writer.terminate()
                writer.join(5)
            sys.modules.pop("configuration_probe", None)
        assert writer.exitcode == 0, writer.exitcode
        print("DYNAMIC_CACHE_POISON_REPRODUCED", flush=True)


if __name__ == "__main__":
    main()
