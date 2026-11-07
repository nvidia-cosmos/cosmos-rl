# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Application-owned continuous-action losses through the native data worker."""

import copy
from types import SimpleNamespace as NS

import pytest
import torch

from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.objectives import ObjectiveWindow, clipped_policy_loss
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.source import (
    CorpusSupervisedDataset,
    prepare_supervised_record,
)
from cosmos_rl.policy.config import Config
from cosmos_rl.policy.trainer.base import TrainerRegistry
from test_resume_data_index import _WorkerHarness


class ActionPacker:
    def __init__(self, delegate=None):
        self.delegate = delegate

    def prepare_record(self, record):
        trajectory = {
            "observations": record.inputs["observations"],
            "actions": record.targets,
        }
        if self.delegate is not None:
            trajectory = self.delegate.get_policy_input(rollout_output=trajectory)
        masks = {name: value.clone() for name, value in record.masks.items()}
        if record.annotations.get("empty_contribution"):
            masks = {name: torch.zeros_like(value) for name, value in masks.items()}
        elif record.annotations.get("rl_eligible") is False:
            masks["policy_gradient"] = torch.zeros_like(masks["policy_gradient"])
        return {
            **trajectory,
            "masks": masks,
            "behavior": record.behavior_logprobs,
            "advantages": record.annotations["advantages"],
        }


def configuration(root):
    return Config.model_validate(
        {
            "execution": {
                "path": "train_from_data",
                "source": {"kind": "corpus", "path": str(root)},
                "objectives": {"supervised": 0.6, "policy_gradient": 0.4},
            },
            "train": {
                "epoch": 1,
                "sequence_packing": False,
                "train_batch_per_replica": 4,
                "train_policy": {
                    "type": "sft",
                    "trainer_type": "application_actions",
                    "dataloader_shuffle": False,
                },
            },
            "policy": {
                "model_name_or_path": "application-action-model",
                "parallelism": {"tp_size": 1},
            },
            "logging": {"logger": []},
        }
    )


def qualify_action_trainer(
    root, monkeypatch, model_factory, packer_factory=ActionPacker
):
    """Also callable against an application's actual model/packer factories."""
    torch.manual_seed(29)
    initial = copy.deepcopy(model_factory().state_dict())
    actor = model_factory()
    writer = CorpusWriter(root, run_id="continuous-trajectories")
    for i, length in enumerate((3, 1, 2, 4)):
        observations = torch.randn(length, 4)
        actions = torch.randn(length, 2)
        with torch.no_grad():
            behavior = (
                torch.distributions.Normal(actor(observations), 1)
                .log_prob(actions)
                .sum(-1)
            )
        writer.commit(
            SampleRecord(
                sample_id=str(i),
                occurrence_id=str(i),
                group_id=str(i // 2),
                group_size=2,
                group_slot=i % 2,
                producer=ModelIdentity(
                    role="actor", name="original-actor", behavior_version=7
                ),
                inputs={"observations": observations},
                targets=actions,
                behavior_logprobs=behavior,
                masks={
                    "supervised": torch.ones(length, dtype=torch.bool),
                    "policy_gradient": torch.ones(length, dtype=torch.bool),
                },
                annotations={
                    "advantages": torch.full((length,), 0.5 if i % 2 else -0.5),
                    "quality_trainable": i != 3,
                },
            )
        )

    class ActionTrainer:
        recorded_objectives = {"supervised", "policy_gradient"}

        def __init__(self, config, **kwargs):
            self.model = model_factory()
            self.model.load_state_dict(initial)
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=0.05, momentum=0.9, weight_decay=0.02
            )
            self.config = config

        def load_model(self):
            return 0, 0, {}

        def step_training(self, global_batch, **kwargs):
            coefficients = {"supervised": 0.6, "policy_gradient": 0.4}
            counts = {
                name: sum(int(row["masks"][name].sum()) for row in global_batch)
                for name in coefficients
            }
            window = ObjectiveWindow.prepare(coefficients, counts)
            if window.empty:
                return {"train/empty_objective_window": 1}
            self.optimizer.zero_grad()
            for row in global_batch:
                predicted = self.model(row["observations"])
                logp = (
                    torch.distributions.Normal(predicted, 1)
                    .log_prob(row["actions"])
                    .sum(-1)
                )
                values = {
                    "supervised": (predicted - row["actions"]).square().mean(-1),
                    "policy_gradient": clipped_policy_loss(
                        logp, row["behavior"], row["advantages"]
                    ),
                }
                window.loss(values, row["masks"]).backward()
            self.optimizer.step()
            return {"train/empty_objective_window": 0}

    monkeypatch.setitem(
        TrainerRegistry._TRAINER_REGISTRY, "application_actions", ActionTrainer
    )
    worker = _WorkerHarness(configuration(root))
    worker._packer = packer_factory()
    worker.parallel_dims.mesh = {"dp": NS(get_group=lambda: None)}
    worker.build_runner()
    rows = next(iter(worker.train_data_loader))
    assert all("label_ids" not in row for row in rows)
    assert all(row["_execution_record"].producer.behavior_version == 7 for row in rows)
    actual = worker.trainer
    actual.step_training(rows)

    reference = model_factory()
    reference.load_state_dict(initial)
    optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.05, momentum=0.9, weight_decay=0.02
    )
    mse, policies = [], []
    for row in rows:
        predicted = reference(row["observations"])
        mse.append(
            (predicted - row["actions"]).square().mean(-1)[row["masks"]["supervised"]]
        )
        eligible = row["masks"]["policy_gradient"]
        logp = torch.distributions.Normal(predicted, 1).log_prob(row["actions"]).sum(-1)
        ratio = (logp - row["behavior"]).exp()
        advantage = row["advantages"]
        policies.append(
            -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage)[
                eligible
            ]
        )
    (0.6 * torch.cat(mse).mean() + 0.4 * torch.cat(policies).mean()).backward()
    optimizer.step()
    for name, value in reference.state_dict().items():
        assert torch.allclose(actual.model.state_dict()[name], value, atol=1e-7), name
    expected_state = optimizer.state_dict()["state"]
    for key, state in actual.optimizer.state_dict()["state"].items():
        assert torch.allclose(
            state["momentum_buffer"], expected_state[key]["momentum_buffer"], atol=1e-7
        )

    view = CorpusSupervisedDataset(CorpusDataset(root))
    masked = prepare_supervised_record(view[None], worker._packer)
    assert not any(masked["masks"][name].any() for name in masked["masks"])
    assert actual.step_training([masked])["train/empty_objective_window"] == 1
    return {
        "continuous_action_gradients": True,
        "optimizer_equivalence": True,
        "non_token_padding": True,
    }


def test_registered_application_trainer_keeps_continuous_action_targets(
    tmp_path, monkeypatch
):
    qualify_action_trainer(
        tmp_path,
        monkeypatch,
        lambda: torch.nn.Sequential(
            torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
        ),
    )


def test_unadvertised_objectives_fail_before_source_or_model_setup(
    tmp_path, monkeypatch
):
    from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer

    class Unsupported(SFTTrainer):
        def __init__(self, **kwargs):
            pytest.fail("constructed an unsupported model/trainer")

        def step_training(self, *args, **kwargs):
            pytest.fail("an overridden training loop inherited objective support")

    monkeypatch.setitem(
        TrainerRegistry._TRAINER_REGISTRY, "application_actions", Unsupported
    )
    worker = _WorkerHarness(configuration(tmp_path))
    with pytest.raises(ValueError, match="does not support recorded objectives"):
        worker.build_runner()
