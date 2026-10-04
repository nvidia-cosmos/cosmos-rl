# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""R-7: execute the actual async backend method with controlled child requests."""

import ast
import asyncio
import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import Mock

import pytest
import cosmos_rl
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.rollout.schema import RolloutResult


@pytest.mark.parametrize("outcome", ["healthy", "failed", "cancelled"])
def test_task_waits_for_every_child_before_returning(outcome):
    source = (
        Path(cosmos_rl.__file__).parent / "rollout/vllm_rollout/vllm_rollout_async.py"
    )
    tree = ast.parse(source.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "vLLMRolloutAsync"
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "rollout_generation"
    )
    namespace = {
        "asyncio": asyncio,
        "copy": copy,
        "List": List,
        "RLPayload": RLPayload,
        "RolloutResult": RolloutResult,
        "DataPacker": object,
        "DataFetcherBase": object,
        "RequestOutputKind": SimpleNamespace(FINAL_ONLY="final"),
        "torch": SimpleNamespace(
            cuda=SimpleNamespace(
                Stream=object,
                current_stream=lambda: None,
                stream=lambda _: nullcontext(),
            )
        ),
        "logger": Mock(),
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"),
        namespace,
    )

    async def run():
        children = []
        entered, finished = asyncio.Event(), asyncio.Event()
        release = asyncio.Event()

        async def child(prompt, params, request_id):
            children.append(asyncio.current_task())
            if request_id == 0:
                if outcome == "failed":
                    raise RuntimeError("one request failed")
                return "first result"
            entered.set()
            try:
                await release.wait()
                return "sibling result"
            finally:
                finished.set()

        worker = SimpleNamespace(
            _engine_initialized=SimpleNamespace(is_set=lambda: True),
            rollout_config=SimpleNamespace(
                multi_turn_config=SimpleNamespace(enable=False)
            ),
            sampling_params=SimpleNamespace(output_kind="final", n=2),
            is_vlm=False,
            _sub_generate_task=child,
            _get_request_id=lambda prompt_idx, child_idx: child_idx,
        )
        packer = SimpleNamespace(
            get_rollout_input=lambda x: x, rollout_collate_fn=lambda x: x
        )
        parent = asyncio.create_task(
            namespace["rollout_generation"](
                worker, [RLPayload(prompt_idx=0, prompt="p")], None, packer, None
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 1)
            await asyncio.sleep(0.01)
            assert not parent.done(), (
                "backend returned while a sibling request was still generating"
            )
            if outcome == "cancelled":
                parent.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(parent, 1)
            else:
                release.set()
                result = await asyncio.wait_for(parent, 1)
                if outcome == "failed":
                    assert result == []
                else:
                    assert result[0].completions == ["first result", "sibling result"]
            assert entered.is_set()
            assert finished.is_set(), (
                "backend returned while a sibling request was still generating"
            )
            assert all(task.done() for task in children)
        finally:
            release.set()
            await asyncio.gather(parent, return_exceptions=True)
            await asyncio.gather(*children, return_exceptions=True)

    asyncio.run(run())
