# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from cosmos_rl.dispatcher.data.packer.hf_vlm_data_packer import HFVLMDataPacker
from cosmos_rl.policy.model.hf_models import HFModel
from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import (
    _enforce_visual_gradient_contract,
    _vlm_component_gradient_metrics,
)


def _processed_sample(input_ids):
    return {
        "input_ids": list(input_ids),
        "label_ids": list(input_ids),
        "logprob_masks": [True] * len(input_ids),
        "pixel_values_videos": None,
        "video_grid_thw": None,
        "second_per_grid_ts": None,
        "pixel_values": None,
        "image_grid_thw": None,
        "pixel_values_videos_lengths_per_sample": None,
        "pixel_values_lengths_per_sample": None,
        "aspect_ratio_ids": None,
        "aspect_ratio_mask": None,
        "image_sizes": None,
        "batch_num_images": None,
    }


def test_hf_vlm_collator_emits_length_based_attention_mask():
    packer = object.__new__(HFVLMDataPacker)
    # Token 0 deliberately appears inside a real sequence. The attention mask
    # must be based on length, not an input_ids != pad_token_id comparison.
    packer.tokenizer = SimpleNamespace(pad_token_id=0)
    samples = [
        _processed_sample([7, 0, 9]),
        _processed_sample([1, 2, 3, 4, 5, 6]),
    ]

    batch = packer._collate_fn(samples, computed_max_len=5)

    assert batch["input_ids"].tolist() == [[7, 0, 9, 0, 0], [1, 2, 3, 4, 5]]
    assert batch["attention_mask"].dtype == torch.long
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == batch["logprob_masks"].shape


def test_hf_model_forwards_attention_mask_to_transformers_model():
    class Recorder:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            return kwargs

    recorder = Recorder()
    wrapper = SimpleNamespace(
        model=recorder,
        model_forward_valid_kwargs={"attention_mask"},
    )
    input_ids = torch.tensor([[1, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 0]])

    HFModel.forward(
        wrapper,
        input_ids=input_ids,
        attention_mask=attention_mask,
        ignored_argument=torch.tensor(1),
    )

    assert recorder.kwargs["attention_mask"] is attention_mask
    assert "ignored_argument" not in recorder.kwargs


class _ToyVisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.merger = nn.Linear(4, 4)


class _ToyVLM(nn.Module):
    is_vlm = True

    def __init__(self):
        super().__init__()
        self.language_model = nn.Linear(4, 4)
        self.vision_model = _ToyVisionModel()
        self.multi_modal_projector = self.vision_model.merger
        self.lm_head = nn.Linear(4, 2)

    def visual_loss(self, inputs):
        hidden = self.vision_model.encoder(inputs)
        hidden = self.multi_modal_projector(hidden)
        hidden = self.language_model(hidden)
        return self.lm_head(hidden).sum()


def test_visual_gradient_contract_reports_non_overlapping_components():
    model = _ToyVLM()
    model.visual_loss(torch.ones(2, 4)).backward()

    metrics = _vlm_component_gradient_metrics(model)
    _enforce_visual_gradient_contract(metrics)

    assert metrics["model/components/visual_gradient_contract"] == "passed"
    assert metrics["model/components/vision_encoder/grad_norm"] > 0.0
    assert metrics["model/components/vision_projector/grad_norm"] > 0.0
    assert metrics["model/components/vision_encoder/total_parameters"] + metrics[
        "model/components/vision_projector/total_parameters"
    ] == sum(parameter.numel() for parameter in model.vision_model.parameters())


def test_visual_gradient_contract_rejects_effectively_frozen_trainable_vision():
    model = _ToyVLM()
    # Produce a valid language-side backward pass with no dependency on vision.
    model.lm_head(model.language_model(torch.ones(2, 4))).sum().backward()

    metrics = _vlm_component_gradient_metrics(model)
    with pytest.raises(RuntimeError, match="Visual-gradient contract failed"):
        _enforce_visual_gradient_contract(metrics)


def test_visual_gradient_contract_rejects_nonfinite_vision_gradient():
    model = _ToyVLM()
    model.visual_loss(torch.ones(2, 4)).backward()
    next(model.vision_model.encoder.parameters()).grad.fill_(float("nan"))

    metrics = _vlm_component_gradient_metrics(model)
    with pytest.raises(RuntimeError, match="Visual-gradient contract failed"):
        _enforce_visual_gradient_contract(metrics)


def test_visual_gradient_contract_accepts_explicitly_frozen_vision():
    model = _ToyVLM()
    for parameter in model.vision_model.parameters():
        parameter.requires_grad_(False)

    metrics = _vlm_component_gradient_metrics(model)
    _enforce_visual_gradient_contract(metrics)

    assert (
        metrics["model/components/visual_gradient_contract"] == "not_applicable_frozen"
    )
    assert metrics["model/components/vision_encoder/trainable_parameters"] == 0
    assert metrics["model/components/vision_projector/trainable_parameters"] == 0
    assert metrics["model/components/vision_encoder/frozen_parameters"] > 0
    assert metrics["model/components/vision_projector/frozen_parameters"] > 0
