# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real VLA trainer loops with CPU model/clock substitutes."""

from contextlib import contextmanager, nullcontext
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.vla_trainer.pi_grpo_trainer import PI05GRPOTrainer
from cosmos_rl.policy.trainer.vla_trainer.vla_trainer import OpenVLAGRPOTrainer


class Model(torch.nn.Module):
    action_chunk = 1
    action_env_dim = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def _set_fsdp_reshard_after_forward(self, _):
        pass

    def get_log_prob_value(self, *, state, **kwargs):
        values = (self.weight * state).reshape(-1, 1, 1, 1)
        return values, torch.zeros_like(values)

    def forward_with_trajectory_structure(self, input_ids, *args, **kwargs):
        return SimpleNamespace(logprobs=self.weight * input_ids)


class Packer:
    def get_policy_input(self, rollout, device):
        return rollout

    def policy_logprob_masks(self, policy_input, max_chunks, *, device=None):
        return torch.arange(max_chunks, device=device).reshape(-1, 1) < len(
            policy_input.values
        )

    def policy_collate_fn(self, policy_input, max_chunks):
        values = torch.tensor(
            policy_input.values + [0.0] * (max_chunks - len(policy_input.values)),
            dtype=torch.float64,
        ).reshape(-1, 1)
        mask = torch.arange(max_chunks).reshape(-1, 1) < len(policy_input.values)
        return dict(
            input_ids=values,
            states=values,
            logprob_masks=mask,
            images=values.reshape(-1, 1, 1),
            image_masks=mask,
            pixel_values=values,
            attention_mask=mask,
            responses=values,
            chains=values,
            denoise_inds=values,
            old_log_probs=torch.zeros_like(values),
        )


@pytest.mark.parametrize("weighting", ["sample", "episode"])
@pytest.mark.parametrize("values", [[True, False, True], []])
def test_custom_one_dimensional_mask_preserves_objective_counts(weighting, values):
    from cosmos_rl.policy.trainer.objectives import vla_objective

    trainer = SimpleNamespace(
        config=SimpleNamespace(vla=SimpleNamespace(objective_weighting=weighting)),
        parallel_dims=SimpleNamespace(dp_enabled=False),
        device=torch.device("cpu"),
    )
    comm = Mock()
    comm.world_size.return_value = 1
    objective, count, divisor, chunks = vla_objective(
        trainer, [{"logprob_masks": torch.tensor(values, dtype=torch.bool)}], comm
    )
    expected = sum(values) if weighting == "sample" else int(any(values))
    assert objective.count == count == expected
    assert divisor == 1
    assert chunks == max(len(values), 1)
    if values:
        assert objective.weights == (
            (1.0, 1.0) if weighting == "sample" else (0.5, 0.5)
        )
    comm.allreduce.assert_called_once()


@pytest.mark.parametrize(
    "dimension", ["cp_enabled", "tp_enabled", "pp_enabled", "ep_enabled"]
)
def test_vla_objective_rejects_unsupported_topology_before_collectives(dimension):
    from cosmos_rl.policy.trainer.objectives import vla_objective

    trainer = SimpleNamespace(parallel_dims=SimpleNamespace(**{dimension: True}))
    comm = Mock()
    with pytest.raises(ValueError, match="pure DP"):
        vla_objective(trainer, [], comm)
    comm.wait_comm_ready.assert_not_called()
    comm.allreduce.assert_not_called()


@pytest.mark.parametrize("kind", ["openvla", "pi05"])
@pytest.mark.parametrize("finish_step", [0, 1, 2, 3, 6])
@pytest.mark.parametrize("max_chunks", [3, 5])
def test_builtin_mask_only_matches_training_collation(
    monkeypatch, kind, finish_step, max_chunks
):
    from cosmos_rl.dispatcher.data.packer import vla_data_packer
    from cosmos_rl.dispatcher.data.packer.pi05_data_packer import PI05DataPacker
    from cosmos_rl.policy.trainer.objectives import vla_objective_inputs

    monkeypatch.setattr(vla_data_packer, "_get_vla_constants", lambda: (2, 3, 100))
    if kind == "openvla":
        packer = object.__new__(vla_data_packer.VLADataPacker)
        packer.tokenizer = SimpleNamespace(pad_token_id=0)
        policy_input = SimpleNamespace(
            finish_step=finish_step,
            input_ids=torch.ones(3, 4, dtype=torch.long),
            responses=torch.ones(3, 6, dtype=torch.long),
            pixel_values=torch.ones(3, 2, 2),
            old_log_probs=torch.ones(3, 6),
        )
        metadata = SimpleNamespace(finish_step=finish_step)
        dtype = torch.long
    else:
        packer = object.__new__(PI05DataPacker)
        policy_input = SimpleNamespace(
            finish_step=finish_step,
            chains=torch.ones(3, 2, 2, 3),
            denoise_inds=torch.ones(3, 2, dtype=torch.long),
            images=torch.ones(3, 1, 2, 2),
            image_masks=torch.ones(3, 1, dtype=torch.bool),
            states=torch.ones(3, 3),
            tokenized_prompt=torch.ones(3, 4, dtype=torch.long),
            tokenized_prompt_mask=torch.ones(3, 4, dtype=torch.bool),
            old_log_probs=torch.ones(3, 2, 3),
        )
        # Count preparation needs only shape metadata, never tensor contents.
        metadata = SimpleNamespace(
            finish_step=finish_step, old_log_probs=SimpleNamespace(shape=(3, 2, 3))
        )
        dtype = torch.float32
    original = packer.policy_collate_fn(policy_input, max_chunks)
    # Independent pre-refactor mask formula, including a partial final chunk.
    expected = (
        torch.cat(
            (torch.ones(finish_step, 3), torch.zeros(max_chunks * 2 - finish_step, 3))
        )
        .reshape(max_chunks, 6)
        .to(dtype)
    )
    torch.testing.assert_close(original["logprob_masks"], expected)
    packer.policy_collate_fn = Mock(side_effect=AssertionError("full collation"))
    packer.policy_logprob_masks = Mock(wraps=packer.policy_logprob_masks)
    data = list(vla_objective_inputs(packer, [metadata], max_chunks))[0]
    torch.testing.assert_close(data["logprob_masks"], expected)
    assert data["logprob_masks"].device.type == "cpu"
    packer.policy_collate_fn.assert_not_called()


def test_custom_packer_keeps_collation_fallback():
    from cosmos_rl.policy.trainer.objectives import vla_objective_inputs

    data = {"logprob_masks": torch.tensor([[False, True]])}
    packer = SimpleNamespace(policy_collate_fn=Mock(return_value=data))
    assert list(vla_objective_inputs(packer, ["episode"], 3)) == [data]
    packer.policy_collate_fn.assert_called_once_with("episode", 3)


def test_custom_collation_override_does_not_use_inherited_masks():
    from cosmos_rl.policy.trainer.objectives import vla_objective_inputs

    class CustomPacker(Packer):
        def policy_collate_fn(self, policy_input, max_chunks):
            return {"logprob_masks": torch.zeros(max_chunks, 1, dtype=torch.bool)}

    packer = CustomPacker()
    episode = SimpleNamespace(values=[1.0, 2.0])
    data = list(vla_objective_inputs(packer, [episode], 3))[0]
    assert not data["logprob_masks"].any()


@pytest.mark.parametrize("finish_step,expected_steps", [(-1, 0), (9, 6)])
def test_pi05_mask_preserves_finish_step_clamping(finish_step, expected_steps):
    from cosmos_rl.dispatcher.data.packer.pi05_data_packer import PI05DataPacker

    packer = object.__new__(PI05DataPacker)
    metadata = SimpleNamespace(
        finish_step=finish_step, old_log_probs=SimpleNamespace(shape=(3, 2, 3))
    )
    mask = packer.policy_logprob_masks(metadata, 3, device="cpu")
    assert mask.sum().item() == expected_steps * 3


@pytest.mark.parametrize("cls", [PI05GRPOTrainer, OpenVLAGRPOTrainer])
@pytest.mark.parametrize(
    "weighting,expected", [("sample", 0.3), ("episode", 0.4), (None, 0.4)]
)
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_production_objective_is_chunk_invariant(
    monkeypatch, cls, weighting, expected, chunk_size, empty, partial=False
):
    if weighting is None and empty:
        pytest.skip(
            "Legacy empty-episode behavior is not changed by this opt-in feature"
        )
    if weighting is None and cls is PI05GRPOTrainer:
        expected = {1: 0.6, 2: 0.525, 3: 0.4}[chunk_size]
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda: None, elapsed_time=lambda other: 0.0
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    # Simulator/model dependencies are not involved in this loss-loop test.
    libero = ModuleType("cosmos_rl.simulators.libero.utils")
    libero.LIBERO_MAX_STEPS_MAP = {"test": 3}
    monkeypatch.setitem(sys.modules, "cosmos_rl.simulators.libero.utils", libero)
    from cosmos_rl.dispatcher.data.packer import vla_data_packer

    monkeypatch.setattr(vla_data_packer, "_get_vla_constants", lambda: (1, 1, 1))
    trainer = object.__new__(cls)
    trainer.device = torch.device("cpu")
    trainer.model = Model()
    trainer.data_packer = Packer()
    if partial:
        original = trainer.data_packer.policy_collate_fn

        def partially_valid(policy_input, max_chunks):
            data = original(policy_input, max_chunks)
            for key in ("input_ids", "logprob_masks", "old_log_probs"):
                data[key] = data[key].repeat(1, 2)
            if len(policy_input.values) == 3:
                data["logprob_masks"][2, 1] = False
            return data

        trainer.data_packer.policy_collate_fn = partially_valid
    trainer.data_packer.policy_collate_fn = Mock(
        wraps=trainer.data_packer.policy_collate_fn
    )
    trainer.data_packer.policy_logprob_masks = Mock(
        wraps=trainer.data_packer.policy_logprob_masks
    )
    trainer.parallel_dims = SimpleNamespace(dp_enabled=False)
    trainer.global_rank = 0
    trainer.config = SimpleNamespace(
        vla=SimpleNamespace(
            objective_weighting=weighting, training_chunk_size=chunk_size
        ),
        logging=SimpleNamespace(logger=[]),
        train=SimpleNamespace(
            fsdp_reshard_after_forward="default",
            train_policy=SimpleNamespace(
                dataset=SimpleNamespace(subset="test"),
                temperature=1.0,
                epsilon_low=0.2,
                epsilon_high=0.2,
            ),
        ),
    )
    trainer.optimizers = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_schedulers = torch.optim.lr_scheduler.StepLR(
        trainer.optimizers, 1, gamma=1.0
    )

    scope_active = False

    @contextmanager
    def scope():
        nonlocal scope_active
        assert not scope_active
        scope_active = True
        try:
            yield
        finally:
            scope_active = False

    def step(_):
        assert scope_active == (weighting is not None)
        trainer.optimizers.step()
        return 0.0

    trainer.all_reduce_states = Mock(side_effect=step)
    comm = SimpleNamespace(
        wait_comm_ready=Mock(),
        world_size=lambda: 1,
        allreduce=Mock(),
        operation_scope=Mock(side_effect=scope),
    )
    episodes = [
        SimpleNamespace(
            values=values,
            chains=torch.zeros(3, 1),
            task_id=0,
            trial_id=0,
            weight_version=0,
            advantage=1.0,
            finish_step=len(values),
        )
        for values in ([[], []] if empty else [[1.0, 2.0, 3.0], [6.0]])
    ]
    trainer.step_training(episodes, 1, 2, 0, comm, False)
    assert trainer.model.weight.item() == pytest.approx(0.0 if empty else expected)
    assert trainer.all_reduce_states.call_count == (0 if empty else 1)
    assert trainer.lr_schedulers.last_epoch == (0 if empty else 1)
    assert trainer.data_packer.policy_collate_fn.call_count == (
        0 if empty else len(episodes)
    )
    if weighting is None:
        # Default execution must not pay for the opt-in count/collation pass.
        assert trainer.data_packer.policy_collate_fn.call_count == len(episodes)
        comm.wait_comm_ready.assert_not_called()
        comm.allreduce.assert_not_called()
        comm.operation_scope.assert_not_called()
        trainer.data_packer.policy_logprob_masks.assert_not_called()
    else:
        assert trainer.data_packer.policy_logprob_masks.call_count == len(episodes)
        comm.operation_scope.assert_called_once()
    assert not scope_active


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize(
    "weighting,expected", [(None, 0.39), ("episode", 0.4), ("sample", 0.3)]
)
def test_partial_final_chunk_keeps_legacy_default(
    monkeypatch, chunk_size, weighting, expected
):
    # First episode: component mean (1+1+2+2+3)/5 = 1.8, not chunk mean 2.
    # Second episode: mean 6. Legacy gradient = (1.8+6)/2 = 3.9.
    test_production_objective_is_chunk_invariant(
        monkeypatch,
        OpenVLAGRPOTrainer,
        weighting,
        expected,
        chunk_size,
        False,
        partial=True,
    )


def test_new_objective_requires_explicit_configuration():
    from cosmos_rl.policy.config import VLAConfig

    assert VLAConfig().objective_weighting is None
    assert VLAConfig(objective_weighting="sample").objective_weighting == "sample"
    assert VLAConfig(objective_weighting="episode").objective_weighting == "episode"
