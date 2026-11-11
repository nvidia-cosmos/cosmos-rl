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


class RandomPreparationPacker(Packer):
    def prepare_record(self, record, *, context):
        import random
        import numpy as np

        row = super().prepare_record(record)
        generator = torch.Generator().manual_seed(context.seed)
        offset = random.Random(context.seed).uniform(-0.1, 0.1)
        scale = np.random.default_rng(context.seed).uniform(0.8, 1.2)
        row["image"] = (
            row["image"] * scale
            + offset
            + torch.rand(row["image"].shape, generator=generator)
        )
        row["label_ids"] = list(row["label_ids"])
        if random.Random(context.seed).random() < 0.5:
            row["label_ids"][1] = -100
        return row


class StochasticObservationModel(ObservationModel):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.25)

    def forward(self, *args, **kwargs):
        result = super().forward(*args, **kwargs)
        result["logits"] = self.dropout(result.logits)
        return result


def test_seeded_preparation_requires_context_adapter_before_model_setup(tmp_path):
    from unittest.mock import patch
    from cosmos_rl.policy.config import Config
    from cosmos_rl.policy.trainer.base import TrainerRegistry
    from test_resume_data_index import _WorkerHarness

    samples(tmp_path)
    config = Config.model_validate(
        {
            "execution": {
                "path": "train_from_data",
                "source": {
                    "kind": "corpus",
                    "path": str(tmp_path),
                    "preparation_seed": 1,
                },
                "objectives": {"supervised": 1},
            },
            "train": {"sequence_packing": False, "train_policy": {"type": "sft"}},
            "policy": {
                "enable_liger_fused_cross_entropy": False,
                "parallelism": {"tp_size": 1},
            },
        }
    )

    class NeverLoaded:
        recorded_objectives = {"supervised"}

        def __init__(self, **kwargs):
            raise AssertionError(
                "Model setup must not start for an incompatible adapter"
            )

    worker = _WorkerHarness(config)
    worker._packer = Packer()
    worker.parallel_dims.mesh = {"dp": NS(get_group=lambda: None)}
    with patch.object(TrainerRegistry, "get_trainer_cls", lambda _: NeverLoaded):
        with pytest.raises(ValueError, match="requires packer.prepare_record"):
            worker.build_runner()


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
    writer.finalize()
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


@pytest.mark.parametrize("mini_batch", [1, 2])
def test_native_corpus_training_matches_independent_multimodal_reference(
    tmp_path, cpu_runtime, mini_batch
):
    torch.manual_seed(9)
    model = ObservationModel()
    reference = ObservationModel()
    reference.load_state_dict(model.state_dict())
    rows = samples(tmp_path)
    actual = trainer(model, mini_batch)
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
    qualify_corpus_resume(tmp_path, stochastic=False, workers=(0, 0), saved_step=1)


def test_microbatch_guard_with_valid_fixed_optimizer_batch(tmp_path, cpu_runtime):
    qualify_corpus_resume(
        tmp_path, stochastic=False, workers=(0, 0), saved_step=1, batch=2
    )


@pytest.mark.parametrize("workers", [(0, 2), (2, 0)])
@pytest.mark.parametrize("saved_step", [1, 4])
def test_random_preparation_and_dropout_resume_across_loader_workers(
    tmp_path, cpu_runtime, workers, saved_step
):
    qualify_corpus_resume(
        tmp_path, stochastic=True, workers=workers, saved_step=saved_step
    )


def qualify_corpus_resume(tmp_path, *, stochastic, workers, saved_step, batch=1):
    import copy
    from unittest.mock import patch
    from cosmos_rl.policy.config import Config
    from cosmos_rl.policy.trainer.base import TrainerRegistry
    from cosmos_rl.utils.checkpoint import CheckpointMananger
    from test_resume_data_index import _WorkerHarness

    corpus_path = tmp_path / "corpus"
    samples(corpus_path)
    config = Config.model_validate(
        {
            "execution": {
                "path": "train_from_data",
                "source": {
                    "kind": "corpus",
                    "path": str(corpus_path),
                    "preparation_seed": 71 if stochastic else None,
                },
                "objectives": {"supervised": 1},
            },
            "train": {
                "epoch": 2,
                "compile": False,
                "sequence_packing": False,
                "train_batch_per_replica": batch,
                "optm_grad_norm_clip": 0,
                "train_policy": {
                    "type": "sft",
                    "mini_batch": 1,
                    "dataloader_shuffle": True,
                    "dataloader_num_workers": workers[0],
                    "dataloader_prefetch_factor": 1 if workers[0] else None,
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
    factory = StochasticObservationModel if stochastic else ObservationModel
    initial = copy.deepcopy(factory().state_dict())
    saved = None

    class NativeTrainer:
        recorded_objectives = {"supervised"}

        def __init__(self, config, **kwargs):
            self.inner = trainer(factory(), 1)
            self.inner.config = config
            self.inner.model.load_state_dict(initial)
            self.config = config

        def load_model(self):
            if self.config.train.resume:
                self.inner.model.load_state_dict(saved["model"])
                self.inner.optimizers.load_state_dict(saved["optimizer"])
                self.inner.lr_schedulers.load_state_dict(saved["scheduler"])
                CheckpointMananger.set_rng_state(saved["rng"])
                return 6, saved_step, {"execution_source": saved["source"]}
            return 0, 0, {}

    def build(configuration):
        worker = _WorkerHarness(configuration)
        worker._packer = RandomPreparationPacker() if stochastic else Packer()
        group = NS(get_group=lambda: None)
        worker.parallel_dims.mesh = {"dp": group}
        worker.build_runner()
        return worker

    with patch.object(TrainerRegistry, "get_trainer_cls", lambda _: NativeTrainer):
        uninterrupted = build(config)
        seen = []
        step = 0
        for epoch in range(2):
            uninterrupted.train_sampler.set_epoch(epoch)
            uninterrupted.train_data_loader.dataset.set_epoch(epoch)
            for rows in uninterrupted.train_data_loader:
                seen.extend(row["_execution_record"].sample_id for row in rows)
                inner = uninterrupted.trainer.inner
                inner.step_training(rows, total_steps=6, train_step=step, save_freq=1)
                step += 1
                if step == saved_step:
                    saved = copy.deepcopy(
                        {
                            "model": inner.model.state_dict(),
                            "optimizer": inner.optimizers.state_dict(),
                            "scheduler": inner.lr_schedulers.state_dict(),
                            "source": uninterrupted.trainer.execution_source_state,
                            "rng": CheckpointMananger.get_rng_state(),
                        }
                    )
        final_rng = torch.get_rng_state().clone()
        resumed_config = config.model_copy(deep=True)
        resumed_config.train.resume = True
        # Relocation and a larger admission budget preserve the same training
        # identity, sample order and optimizer continuation.
        import shutil

        relocated = tmp_path / "relocated"
        shutil.copytree(corpus_path, relocated)
        resumed_config.execution.source.path = str(relocated)
        resumed_config.execution.source.max_prefetch_bytes *= 2
        resumed_config.train.train_policy.dataloader_num_workers = workers[1]
        resumed_config.train.train_policy.dataloader_prefetch_factor = (
            1 if workers[1] else None
        )
        resumed = build(resumed_config)
        resumed_seen = []
        step = saved_step
        for epoch in range(resumed.start_epoch, 2):
            resumed.train_sampler.set_epoch(epoch)
            resumed.train_data_loader.dataset.set_epoch(epoch)
            for rows in resumed.train_data_loader:
                resumed_seen.extend(row["_execution_record"].sample_id for row in rows)
                resumed.trainer.inner.step_training(
                    rows, total_steps=6, train_step=step, save_freq=1
                )
                step += 1
        assert resumed_seen == seen[saved_step * batch :]
        if stochastic:
            assert torch.equal(final_rng, torch.get_rng_state())
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
        assert saved["source"]["mini_batch"] == 1
        changed_microbatch = resumed_config.model_copy(deep=True)
        changed_microbatch.train.train_policy.mini_batch = 2
        with pytest.raises(ValueError, match="microbatch configuration changed"):
            build(changed_microbatch)
        original_source = copy.deepcopy(saved["source"])
        del saved["source"]["mini_batch"]
        with pytest.raises(ValueError, match="lacks microbatch metadata"):
            build(resumed_config)
        saved["source"] = original_source
        saved["source"]["mini_batch"] = True
        with pytest.raises(ValueError, match="microbatch configuration changed"):
            build(resumed_config)
        saved["source"] = original_source
        saved["source"]["mini_batch"] = 1
        resumed_config.execution.source.reward_weighting = "nonnegative"
        with pytest.raises(ValueError, match="configuration changed"):
            build(resumed_config)
        if stochastic:
            resumed_config.execution.source.reward_weighting = "none"
            resumed_config.execution.source.preparation_seed += 1
            with pytest.raises(ValueError, match="configuration changed"):
                build(resumed_config)
