"""Prefetch setup must use the initialized communicator's actual budget."""

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.policy.trainer import prefetch
from cosmos_rl.utils import constant


def test_constructor_initializes_prefetch_after_communicator():
    tree = ast.parse(textwrap.dedent(inspect.getsource(RLPolicyWorker.__init__)))
    statements = tree.body[0].body
    communicator = next(
        i
        for i, s in enumerate(statements)
        if isinstance(s, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == "inter_policy_nccl"
            for t in s.targets
        )
    )
    setup = next(
        i
        for i, s in enumerate(statements)
        if isinstance(s, ast.Expr)
        and isinstance(s.value, ast.Call)
        and isinstance(s.value.func, ast.Attribute)
        and s.value.func.attr == "_setup_payload_prefetch"
    )
    assert setup > communicator
    # build_runner runs earlier: it must not read this not-yet-created member.
    runner = ast.parse(textwrap.dedent(inspect.getsource(RLPolicyWorker.build_runner)))
    assert not any(
        isinstance(n, ast.Attribute) and n.attr == "inter_policy_nccl"
        for n in ast.walk(runner)
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_setup_uses_actual_communicator_timeout(monkeypatch, enabled):
    monkeypatch.setattr(
        prefetch,
        "prefetch_fallback_reason",
        lambda config: None if enabled else "disabled",
    )
    factory = Mock()
    monkeypatch.setattr(prefetch, "TrainerPayloadPrefetch", factory)
    packer = SimpleNamespace(
        _prefetch_enabled=True, _prefetch_timeout_s=120.0, _setup_prefetch=Mock()
    )
    worker = SimpleNamespace(
        config=SimpleNamespace(train=SimpleNamespace(prefetch_payloads=enabled)),
        data_packer=packer,
        device="cpu",
        inter_policy_nccl=SimpleNamespace(default_timeout_ms=8000),
    )
    RLPolicyWorker._setup_payload_prefetch(worker)
    if enabled:
        factory.assert_called_once_with(packer, device="cpu")
        packer._setup_prefetch.assert_called_once_with(
            prefetch_timeout=min(120.0, constant.COSMOS_GLOO_TIMEOUT / 4, 2.0)
        )
    else:
        factory.assert_not_called()
        packer._setup_prefetch.assert_not_called()
        assert worker.payload_prefetch is None
