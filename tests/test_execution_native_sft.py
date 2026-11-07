# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the native trainer with real multimodal gradients on CPU."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F

from cosmos_rl.execution.config import ExecutionConfig
from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.execution.source import (
    CorpusSupervisedDataset,
    prepare_supervised_record,
)
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer


class ObservationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = torch.nn.Embedding(3, 3)
        self.camera = torch.nn.Linear(3, 3, bias=False)

    def forward(self, input_ids, image, **kwargs):
        logits = self.tokens(input_ids) + self.camera(image.mean((-1, -2)))[:, None]

        class Output(dict):
            __getattr__ = dict.__getitem__

        return Output(logits=logits)

    def get_position_ids(self, input_ids, **kwargs):
        return torch.arange(input_ids.shape[1]), input_ids, 1

    def step_hook(self, step):
        return {}


class Packer:
    batch_size = staticmethod(len)
    slice_batch = staticmethod(lambda batch, start, end: batch[start:end])
    sft_compute_max_len = staticmethod(
        lambda batch: max(len(row["token_ids"]) for row in batch)
    )

    def prepare_record(self, record):
        return {**record.targets, "image": record.inputs["image"]}

    def sft_collate_fn(self, rows, computed_max_len, ignore_label_id):
        return {
            "input_ids": torch.tensor(
                [
                    row["token_ids"] + [0] * (computed_max_len - len(row["token_ids"]))
                    for row in rows
                ]
            ),
            "label_ids": torch.tensor(
                [
                    row["label_ids"]
                    + [ignore_label_id] * (computed_max_len - len(row["label_ids"]))
                    for row in rows
                ]
            ),
            "image": torch.stack([row["image"] for row in rows]),
        }


def samples(root):
    writer = CorpusWriter(root, run_id="observations")
    for i, tokens in enumerate(([0, 1, 2], [1, 0], [2, 0, 1, 2])):
        n = len(tokens) - 1
        writer.commit(
            SampleRecord(
                sample_id=str(i),
                occurrence_id=str(i),
                group_id="episode",
                producer=ModelIdentity(
                    role="actor",
                    name="behavior",
                    revision="original",
                    behavior_version=4,
                ),
                inputs={"image": torch.full((3, 2, 2), 0.1 * (i + 1))},
                targets={"token_ids": tokens, "label_ids": tokens},
                behavior_logprobs=[-1.4] * n,
                masks={"policy_gradient": [i != 2] * n},
                annotations={"advantages": [0.4] * n},
            )
        )
    dataset = CorpusSupervisedDataset(CorpusDataset(root))
    return [
        prepare_supervised_record(dataset[i], Packer()) for i in range(len(dataset))
    ]


def trainer(model, mini_batch):
    worker = object.__new__(SFTTrainer)
    worker.device = torch.device("cpu")
    worker.config = NS(
        execution=ExecutionConfig(
            path="train_from_data",
            objectives={"supervised": 0.6, "policy_gradient": 0.4},
        ),
        policy=NS(model_max_length=32, enable_liger_fused_cross_entropy=False),
        train=NS(
            train_policy=NS(mini_batch=mini_batch),
            sequence_packing=False,
            optm_grad_norm_clip=0,
        ),
        logging=NS(logger=[]),
    )
    group = NS(get_group=lambda: None)
    worker.parallel_dims = NS(
        pp_enabled=False,
        cp_enabled=False,
        dp_replicate_enabled=False,
        mesh={"dp": group, "loss_parallel": group},
    )
    worker.enable_dp_load_balancing = False
    worker.data_packer = Packer()
    worker.model = model
    worker.model_parts = [model]
    worker.set_model_train = model.train
    worker.seq_len_multiple = 1
    worker.act_offloading_ctx_manager = nullcontext()
    worker.optimizers = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.02
    )
    worker.lr_schedulers = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizers, lambda _: 1
    )
    return worker


@pytest.fixture
def cpu_runtime(monkeypatch):
    class Event:
        def __init__(self, **kwargs):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_native_corpus_training_matches_independent_multimodal_reference(
    tmp_path, cpu_runtime
):
    torch.manual_seed(9)
    model = ObservationModel()
    reference = ObservationModel()
    reference.load_state_dict(model.state_dict())
    rows = samples(tmp_path)
    actual = trainer(model, 1)
    actual.step_training(rows, total_steps=2, train_step=0, save_freq=1)

    ce, policy = [], []
    for row in rows:
        tokens = torch.tensor(row["token_ids"])
        logits = reference(tokens[None], row["image"][None]).logits[0, :-1]
        ce.append(F.cross_entropy(logits, tokens[1:], reduction="none"))
        record = row["_execution_record"]
        if any(record.masks["policy_gradient"]):
            logp = logits.log_softmax(-1).gather(-1, tokens[1:, None]).squeeze(-1)
            ratio = (logp + 1.4).exp()
            policy.append(-torch.minimum(ratio * 0.4, ratio.clamp(0.8, 1.2) * 0.4))
    optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.1, momentum=0.9, weight_decay=0.02
    )
    (0.6 * torch.cat(ce).mean() + 0.4 * torch.cat(policy).mean()).backward()
    optimizer.step()
    assert model.camera.weight.grad.abs().sum() > 0
    for name, value in model.state_dict().items():
        assert torch.allclose(value, reference.state_dict()[name], atol=1e-7), name


def test_native_all_empty_window_preserves_optimizer_and_scheduler(
    tmp_path, cpu_runtime
):
    rows = samples(tmp_path)
    for row in rows:
        row["label_ids"] = [-100] * len(row["label_ids"])
    actual = trainer(ObservationModel(), 1)
    before = {name: value.clone() for name, value in actual.model.state_dict().items()}
    schedule = actual.lr_schedulers.state_dict()
    report = actual.step_training(rows, total_steps=2, train_step=0, save_freq=1)
    assert report["train/empty_objective_window"] == 1
    assert actual.optimizers.state_dict()["state"] == {}
    assert actual.lr_schedulers.state_dict() == schedule
    for name, value in before.items():
        assert torch.equal(value, actual.model.state_dict()[name])


def test_native_corpus_resume_matches_uninterrupted_optimizer_boundary(
    tmp_path, cpu_runtime, monkeypatch
):
    import copy
    from unittest.mock import patch
    from cosmos_rl.policy.config import Config
    from cosmos_rl.policy.trainer.base import TrainerRegistry
    from test_resume_data_index import _WorkerHarness

    corpus_path = tmp_path / "corpus"
    samples(corpus_path)
    config = Config.model_validate(
        {
            "execution": {
                "path": "train_from_data",
                "source": {"kind": "corpus", "path": str(corpus_path)},
                "objectives": {"supervised": 1},
            },
            "train": {
                "epoch": 1,
                "compile": False,
                "sequence_packing": False,
                "train_batch_per_replica": 1,
                "optm_grad_norm_clip": 0,
                "train_policy": {
                    "type": "sft",
                    "mini_batch": 1,
                    "dataloader_shuffle": True,
                    "dataloader_num_workers": 0,
                },
            },
            "policy": {
                "model_name_or_path": "test-model",
                "model_max_length": 32,
                "enable_liger_fused_cross_entropy": False,
                "parallelism": {"tp_size": 1},
            },
            "logging": {"logger": []},
        }
    )
    torch.manual_seed(17)
    initial = copy.deepcopy(ObservationModel().state_dict())
    saved = None

    class NativeTrainer:
        recorded_objectives = {"supervised"}

        def __init__(self, config, **kwargs):
            self.inner = trainer(ObservationModel(), 1)
            self.inner.config = config
            self.inner.model.load_state_dict(initial)
            self.config = config

        def load_model(self):
            if self.config.train.resume:
                self.inner.model.load_state_dict(saved["model"])
                self.inner.optimizers.load_state_dict(saved["optimizer"])
                self.inner.lr_schedulers.load_state_dict(saved["scheduler"])
                return 3, 1, {"execution_source": saved["source"]}
            return 0, 0, {}

    def build(configuration):
        worker = _WorkerHarness(configuration)
        worker._packer = Packer()
        group = NS(get_group=lambda: None)
        worker.parallel_dims.mesh = {"dp": group}
        worker.build_runner()
        return worker

    with patch.object(TrainerRegistry, "get_trainer_cls", lambda _: NativeTrainer):
        uninterrupted = build(config)
        seen = []
        for step, rows in enumerate(uninterrupted.train_data_loader):
            seen.extend(row["_execution_record"].sample_id for row in rows)
            inner = uninterrupted.trainer.inner
            inner.step_training(rows, total_steps=3, train_step=step, save_freq=1)
            if step == 0:
                saved = copy.deepcopy(
                    {
                        "model": inner.model.state_dict(),
                        "optimizer": inner.optimizers.state_dict(),
                        "scheduler": inner.lr_schedulers.state_dict(),
                        "source": uninterrupted.trainer.execution_source_state,
                    }
                )
        resumed_config = config.model_copy(deep=True)
        resumed_config.train.resume = True
        resumed = build(resumed_config)
        resumed_seen = []
        for step, rows in enumerate(resumed.train_data_loader, start=1):
            resumed_seen.extend(row["_execution_record"].sample_id for row in rows)
            resumed.trainer.inner.step_training(
                rows, total_steps=3, train_step=step, save_freq=1
            )
        assert resumed_seen == seen[1:]
        for name, value in uninterrupted.trainer.inner.model.state_dict().items():
            assert torch.equal(value, resumed.trainer.inner.model.state_dict()[name]), (
                name
            )
        expected = uninterrupted.trainer.inner.optimizers.state_dict()
        actual = resumed.trainer.inner.optimizers.state_dict()
        for key, state in expected["state"].items():
            assert torch.equal(
                state["momentum_buffer"], actual["state"][key]["momentum_buffer"]
            )
        assert (
            uninterrupted.trainer.inner.lr_schedulers.state_dict()
            == resumed.trainer.inner.lr_schedulers.state_dict()
        )
        resumed_config.execution.source.reward_weighting = "nonnegative"
        with pytest.raises(ValueError, match="configuration changed"):
            build(resumed_config)
