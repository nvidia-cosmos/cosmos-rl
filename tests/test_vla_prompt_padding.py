"""Training must select action logits at each unpadded prompt boundary."""

from types import MethodType, SimpleNamespace

import torch
from cosmos_rl.policy.model.vla.openvla_oft.modeling_prismatic import (
    PrismaticForConditionalGeneration,
)


def test_unequal_prompt_lengths_keep_bos_and_action_offsets():
    captured = {}

    class Vision:
        def __call__(self, pixels):
            return torch.zeros(pixels.shape[0], 2, 4)

        def get_num_patches(self):
            return 2

        def get_num_images_in_input(self):
            return 1

    embedding = torch.nn.Embedding(33000, 4)

    def language(**kwargs):
        captured.update(kwargs)
        batch, length, _ = kwargs["inputs_embeds"].shape
        return SimpleNamespace(
            logits=torch.arange(length).view(1, length, 1).expand(batch, -1, -1)
        )

    model = SimpleNamespace(
        vision_backbone=Vision(),
        projector=lambda x: x,
        proprio_projector=None,
        get_input_embeddings=lambda: embedding,
        language_model=language,
    )
    for name in ("_build_multimodal_attention", "_convert_to_bidirectional_4d_mask"):
        setattr(
            model,
            name,
            MethodType(getattr(PrismaticForConditionalGeneration, name), model),
        )
    result = PrismaticForConditionalGeneration.forward(
        model,
        input_ids=torch.tensor([[0, 0, 1, 42], [1, 7, 8, 9]]),
        attention_mask=torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]]),
        pixel_values=torch.zeros(2, 3, 4, 4),
    )
    assert result[0, 0, 0] == 3  # 2 image patches + 2 prompt tokens - 1
    assert result[1, 0, 0] == 5  # 2 image patches + 4 prompt tokens - 1
    torch.testing.assert_close(
        captured["inputs_embeds"][:, 0], embedding(torch.tensor([1, 1]))
    )
    assert captured["attention_mask"][0, :, :, -2:].isneginf().all()
    assert captured["attention_mask"][1].isfinite().all()
