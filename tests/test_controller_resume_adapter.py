# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from unittest.mock import Mock

import pytest
import torch
from pydantic import ValidationError

from cosmos_rl.dispatcher.data.data_fetcher import ControllerDataFetcher
from cosmos_rl.dispatcher.data.resume import ControllerResumeMetadata
from test_resume_data_index import _make_rl_config, _RLPromptDataset


class CursorSampler:
    """Intentionally stateful and has no len; even a probe consumes its cursor."""

    def __init__(self, batch=False):
        self.position = 0
        self.iterations = 0
        self.epochs = []
        self.batch = batch

    def set_epoch(self, epoch):
        self.epochs.append(epoch)
        self.position = 0

    def __iter__(self):
        self.iterations += 1
        while self.position < 10:
            index = self.position
            if self.batch:
                self.position = min(10, index + 3)
                yield list(range(index, self.position))
            else:
                self.position += 1
                yield index


def metadata(**overrides):
    return ControllerResumeMetadata(
        **(
            {
                "checkpoint_path": "/application/checkpoint",
                "completed_training_steps": 20,
                "completed_optimizer_updates": 40,
                "remaining_completions": 16,
                "epoch": 1,
                "sampling_owner": "sampler",
                "sampler_state": {"cursor": 4},
            }
            | overrides
        )
    )


def build(adapter, sampler, *, batch=False, resume=True):
    config = _make_rl_config(resume=resume)
    config.train.train_policy.dataloader_num_workers = 0
    config.train.train_policy.dataloader_prefetch_factor = None
    return ControllerDataFetcher(
        config,
        dataset=_RLPromptDataset(size=10),
        sampler=None if batch else sampler,
        batch_sampler=sampler if batch else None,
        resume_adapter=adapter,
    )


def adapter_for(state):
    def restore(sampler, metadata):
        assert sampler.iterations == 0
        assert sampler.epochs == [metadata.epoch]
        sampler.position = metadata.sampler_state["cursor"]

    return Mock(
        load_metadata=Mock(return_value=state),
        restore_sampler=Mock(side_effect=restore),
    )


@pytest.mark.parametrize("step", [0, 20])
@pytest.mark.parametrize("batch", [False, True])
def test_restore_before_first_iteration_without_probe_or_epoch_reset(step, batch):
    state = metadata(
        completed_training_steps=step,
        sampling_owner="batch_sampler" if batch else "sampler",
    )
    adapter = adapter_for(state)
    sampler = CursorSampler(batch=batch)
    fetcher = build(adapter, sampler, batch=batch)
    assert sampler.iterations == 0
    assert fetcher.ckpt_extra_info["step"] == step
    assert fetcher.ckpt_extra_info["optimizer_updates"] == 40
    assert fetcher.config.train.resume == state.checkpoint_path
    assert fetcher.remain_samples_num == 16
    # Reading metadata repeatedly never touches sampling state.
    assert fetcher.resume_metadata is state
    assert fetcher.resume_metadata is state
    assert sampler.position == 4
    indices, _ = next(fetcher.train_dataloader_iter)
    assert [int(i) for i in indices] == ([4, 5, 6] if batch else [4, 5])
    adapter.load_metadata.assert_called_once()
    adapter.restore_sampler.assert_called_once_with(sampler, state)
    assert sampler.epochs == [1]


def test_partial_batch_tail():
    state = metadata(sampling_owner="batch_sampler", sampler_state={"cursor": 9})
    sampler = CursorSampler(batch=True)
    fetcher = build(adapter_for(state), sampler, batch=True)
    indices, _ = next(fetcher.train_dataloader_iter)
    assert [int(i) for i in indices] == [9]


def test_explicit_missing_path_fails_without_consuming_sampler():
    sampler = CursorSampler()
    with pytest.raises(FileNotFoundError):
        build(adapter_for(None), sampler, resume="/missing")
    assert sampler.iterations == 0


def test_auto_discovery_miss_bootstraps_without_restoration():
    adapter = adapter_for(None)
    sampler = CursorSampler()
    fetcher = build(adapter, sampler)
    assert fetcher.config.train.resume is False
    assert fetcher.ckpt_extra_info == {}
    adapter.restore_sampler.assert_not_called()
    indices, _ = next(fetcher.train_dataloader_iter)
    assert [int(i) for i in indices] == [0, 1]


def test_resume_disabled_does_not_call_adapter():
    adapter = adapter_for(metadata())
    build(adapter, CursorSampler(), resume=False)
    adapter.load_metadata.assert_not_called()
    adapter.restore_sampler.assert_not_called()


def test_ambiguous_sampler_ownership_rejected():
    sampler = CursorSampler(batch=True)
    with pytest.raises(ValueError, match="owner"):
        build(adapter_for(metadata()), sampler, batch=True)
    assert sampler.iterations == 0


@pytest.mark.parametrize("phase", ["load_metadata", "restore_sampler"])
def test_adapter_errors_are_not_swallowed(phase):
    adapter = adapter_for(metadata())
    getattr(adapter, phase).side_effect = ValueError("corrupt checkpoint")
    sampler = CursorSampler()
    with pytest.raises(ValueError, match="corrupt"):
        build(adapter, sampler)
    assert sampler.iterations == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"schema_version": 2},
        {"completed_training_steps": -1},
        {"completed_training_steps": True},
        {"completed_optimizer_updates": "20"},
        {"remaining_completions": -1},
        {"epoch": 0},
        {"sampling_owner": "both"},
        {"checkpoint_path": ""},
    ],
)
def test_metadata_validation(changes):
    with pytest.raises(ValidationError):
        metadata(**changes)


def test_real_checkpoint_resume_matches_next_samples_and_optimizer_updates(tmp_path):
    sampler = CursorSampler()
    uninterrupted = build(adapter_for(None), sampler, resume=False)
    weight = torch.nn.Parameter(torch.tensor([0.25]))
    optimizer = torch.optim.SGD([weight], lr=0.01, momentum=0.9)

    def update(fetcher, parameter, optimizer):
        indices, _ = next(fetcher.train_dataloader_iter)
        inputs = torch.tensor([int(i) + 1 for i in indices], dtype=torch.float32)
        optimizer.zero_grad()
        loss = ((parameter * inputs - 1) ** 2).mean()
        loss.backward()
        optimizer.step()
        return inputs

    for _ in range(2):
        update(uninterrupted, weight, optimizer)
    path = tmp_path / "application.pt"
    state = metadata(
        checkpoint_path=str(path),
        completed_training_steps=2,
        completed_optimizer_updates=2,
        sampler_state={"cursor": sampler.position},
    )
    torch.save(
        {
            "metadata": state.model_dump(),
            "weight": weight.detach(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )

    class FileAdapter:
        def load_metadata(self, config):
            return ControllerResumeMetadata.model_validate(
                torch.load(config.train.resume, weights_only=True)["metadata"]
            )

        def restore_sampler(self, sampler, metadata):
            sampler.position = metadata.sampler_state["cursor"]

    restored = build(FileAdapter(), CursorSampler(), resume=str(path))
    saved = torch.load(path, weights_only=True)
    restored_weight = torch.nn.Parameter(saved["weight"])
    restored_optimizer = torch.optim.SGD([restored_weight], lr=0.01, momentum=0.9)
    restored_optimizer.load_state_dict(saved["optimizer"])
    # These are new updates 3 and 4, not merely a successful metadata read.
    for _ in range(2):
        expected_indices = update(uninterrupted, weight, optimizer)
        actual_indices = update(restored, restored_weight, restored_optimizer)
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(restored_weight, weight, rtol=0, atol=0)
        torch.testing.assert_close(
            restored_optimizer.state[restored_weight]["momentum_buffer"],
            optimizer.state[weight]["momentum_buffer"],
            rtol=0,
            atol=0,
        )
