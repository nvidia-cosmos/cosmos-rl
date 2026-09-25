# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cosmos_rl.utils.payload_transport.nccl.strategy import NCCLTransportStrategy
from cosmos_rl.utils.payload_transport.ucxx.strategy import UCXXTransportStrategy
from cosmos_rl.utils.transport_failure import TransportUnusableError


def test_portable_canary_accepts_prefetch_mixins_positional_delegation():
    from transfer_contract_canary import Packer

    payload = {"healthy": 1}
    assert Packer().get_policy_input(None, payload, 0) is payload


@pytest.mark.parametrize("backend", ["nccl", "ucxx"])
@pytest.mark.parametrize("terminal", [False, True])
def test_sync_fetch_distinguishes_missing_from_terminal(monkeypatch, backend, terminal):
    def fetch(*args):
        if terminal:
            raise TransportUnusableError("completion unknown")
        return {}, 0, 0

    if backend == "nccl":
        strategy = NCCLTransportStrategy()
        monkeypatch.setattr(
            "cosmos_rl.utils.payload_transport.nccl.strategy._parse_ref",
            lambda *args, **kwargs: {"schema": []},
        )
        strategy._fetch_all = fetch
    else:
        strategy = UCXXTransportStrategy()
        strategy._client = object()

        def run(coroutine):
            coroutine.close()
            return fetch()

        strategy._run_async = run
    if terminal:
        with pytest.raises(TransportUnusableError, match="completion unknown"):
            strategy.sync_fetch({"_nccl": True, "_ucxx": True})
    else:
        assert strategy.sync_fetch({"_nccl": True, "_ucxx": True}) is None
