"""Joint-state conditioning must survive trajectory replay and padding."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from cosmos_rl.dispatcher.data.packer.vla_data_packer import VLADataPacker


def test_proprio_survives_replay_and_padding():
    packer = object.__new__(VLADataPacker)
    packer.config = SimpleNamespace(
        train=SimpleNamespace(output_dir="/tmp"), vla=SimpleNamespace(use_proprio=True)
    )
    packer.tokenizer = SimpleNamespace(pad_token_id=0)
    trajectory = dict(
        input_ids=torch.ones(2, 3, dtype=torch.long),
        attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        pixel_values=torch.ones(2, 3, 4, 4),
        responses=torch.ones(2, 350, dtype=torch.long),
        old_log_probs=torch.zeros(2, 350),
        proprio=torch.arange(28).reshape(2, 14).float(),
    )
    sample = SimpleNamespace(
        weight_version=0,
        prompt={"task_id": 0, "trial_id": 1},
        advantage=1.0,
        completion={"finish_step": 40, "complete": True, "trajectory_id": "test"},
    )
    with patch(
        "cosmos_rl.dispatcher.data.packer.vla_data_packer.load_trajectory_from_buffer",
        return_value=trajectory,
    ):
        policy_input = packer.get_policy_input(sample, device="cpu")
    with patch(
        "cosmos_rl.dispatcher.data.packer.vla_data_packer._get_vla_constants",
        return_value=(25, 14, 100),
    ):
        batch = packer.policy_collate_fn(policy_input, max_chunks=3)
    torch.testing.assert_close(batch["proprio"][:2], trajectory["proprio"])
    torch.testing.assert_close(
        batch["attention_mask"][:2], trajectory["attention_mask"]
    )
    assert batch["attention_mask"][2].count_nonzero() == 0
    assert batch["proprio"][2].count_nonzero() == 0
    assert batch["logprob_masks"][2].count_nonzero() == 0
    trajectory.pop("proprio")
    with patch(
        "cosmos_rl.dispatcher.data.packer.vla_data_packer.load_trajectory_from_buffer",
        return_value=trajectory,
    ):
        try:
            packer.get_policy_input(sample, device="cpu")
        except ValueError as error:
            assert "proprio" in str(error)
        else:
            raise AssertionError("Missing required proprio was silently accepted")
        packer.config.vla.use_proprio = False
        policy_input = packer.get_policy_input(sample, device="cpu")
    with patch(
        "cosmos_rl.dispatcher.data.packer.vla_data_packer._get_vla_constants",
        return_value=(25, 14, 100),
    ):
        assert "proprio" not in packer.policy_collate_fn(policy_input, max_chunks=3)
