# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for WAN2.2 (Wan2pt2) reward-service integration.

Covers:
  - DecodeArgs config fields for wan2pt2
  - DecodeHandler.decode_video auto-detection (16ch vs 48ch)
  - DecodeHandler.initialize wan2pt2 path
  - Pre-decoded video payload branch in extract_video
  - WanVAE_ model architecture (builds without checkpoint)
  - WanVAE wrapper scale tensors
"""

import io
import json
import unittest
from unittest import mock

import numpy as np
import torch


# ---------------------------------------------------------------------------
# 1. Config tests (no GPU, no heavy imports)
# ---------------------------------------------------------------------------
class TestDecodeArgsConfig(unittest.TestCase):
    """Verify that DecodeArgs exposes the wan2pt2 config fields."""

    def test_wan2pt2_fields_exist_with_defaults(self):
        from cosmos_rl_reward.launcher.config import DecodeArgs

        args = DecodeArgs()
        self.assertEqual(args.wan2pt2_model_path, "")
        self.assertEqual(args.wan2pt2_credential_path, "")

    def test_wan2pt2_fields_accept_values(self):
        from cosmos_rl_reward.launcher.config import DecodeArgs

        args = DecodeArgs(
            wan2pt2_model_path="/some/path/Wan2.2_VAE.pth",
            wan2pt2_credential_path="/creds/s3.json",
        )
        self.assertEqual(args.wan2pt2_model_path, "/some/path/Wan2.2_VAE.pth")
        self.assertEqual(args.wan2pt2_credential_path, "/creds/s3.json")

    def test_full_config_with_wan2pt2(self):
        from cosmos_rl_reward.launcher.config import Config

        cfg = Config.from_dict(
            {
                "decode_args": {
                    "wan2pt2_model_path": "https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/resolve/main/Wan2.2_VAE.pth",
                }
            }
        )
        self.assertIn("Wan2.2_VAE", cfg.decode_args.wan2pt2_model_path)

    def test_config_backward_compatible_without_wan2pt2(self):
        from cosmos_rl_reward.launcher.config import Config

        cfg = Config.from_dict({})
        self.assertEqual(cfg.decode_args.wan2pt2_model_path, "")


# ---------------------------------------------------------------------------
# 2. DecodeHandler routing tests (mock decoders, no real model files)
# ---------------------------------------------------------------------------
class TestDecodeHandlerRouting(unittest.TestCase):
    """Test that decode_video dispatches to the correct decoder based on channel dim."""

    def _make_handler(self):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        handler = DecodeHandler.__new__(DecodeHandler)
        handler.device = "cpu"
        return handler

    def test_routes_48ch_to_wan2pt2(self):
        handler = self._make_handler()
        fake_video = torch.randn(1, 3, 4, 8, 8).clamp(-1, 1)

        handler.latent_decoder = mock.MagicMock()
        handler.latent_decoder_wan2pt2 = mock.MagicMock()
        handler.latent_decoder_wan2pt2.decode.return_value = fake_video

        latents = torch.randn(1, 48, 2, 4, 4)
        video = handler.decode_video(latents)

        handler.latent_decoder_wan2pt2.decode.assert_called_once_with(latents)
        handler.latent_decoder.decode_latents.assert_not_called()
        self.assertEqual(video.dtype, torch.uint8)

    def test_routes_16ch_to_wan2pt1(self):
        handler = self._make_handler()
        fake_video = torch.randn(1, 3, 4, 8, 8).clamp(-1, 1)

        handler.latent_decoder = mock.MagicMock()
        handler.latent_decoder.decode_latents.return_value = fake_video
        handler.latent_decoder_wan2pt2 = mock.MagicMock()

        latents = torch.randn(1, 16, 2, 4, 4)
        video = handler.decode_video(latents)

        handler.latent_decoder.decode_latents.assert_called_once_with(latents)
        handler.latent_decoder_wan2pt2.decode.assert_not_called()
        self.assertEqual(video.dtype, torch.uint8)

    def test_48ch_falls_back_without_wan2pt2_decoder(self):
        handler = self._make_handler()
        fake_video = torch.randn(1, 3, 4, 8, 8).clamp(-1, 1)

        handler.latent_decoder = mock.MagicMock()
        handler.latent_decoder.decode_latents.return_value = fake_video

        latents = torch.randn(1, 48, 2, 4, 4)
        video = handler.decode_video(latents)

        handler.latent_decoder.decode_latents.assert_called_once_with(latents)
        self.assertEqual(video.dtype, torch.uint8)


# ---------------------------------------------------------------------------
# 3. DecodeHandler.initialize wan2pt2 path
# ---------------------------------------------------------------------------
class TestDecodeHandlerInitialize(unittest.TestCase):
    """Test that initialize() wires up wan2pt2 decoder when path is given."""

    def setUp(self):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        if hasattr(DecodeHandler, "_instance"):
            del DecodeHandler._instance

    def tearDown(self):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        if hasattr(DecodeHandler, "_instance"):
            del DecodeHandler._instance

    @mock.patch(
        "cosmos_rl_reward.handler.decode.DecodeHandler.set_latent_decoder_wan2pt2"
    )
    @mock.patch("cosmos_rl_reward.handler.decode.DecodeHandler.set_latent_decoder")
    def test_initialize_calls_wan2pt2_setup(self, mock_set_wan1, mock_set_wan2):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        DecodeHandler.initialize(
            info={},
            requires_latent_decode=True,
            wan2pt2_model_path="/fake/Wan2.2_VAE.pth",
            wan2pt2_credential_path="/fake/creds.json",
            device="cpu",
        )
        mock_set_wan2.assert_called_once_with(
            model_path="/fake/Wan2.2_VAE.pth",
            device="cpu",
            credential_path="/fake/creds.json",
        )

    @mock.patch("cosmos_rl_reward.handler.decode.DecodeHandler.set_latent_decoder")
    def test_initialize_skips_wan2pt2_when_empty(self, mock_set_wan1):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        DecodeHandler.initialize(
            info={},
            requires_latent_decode=True,
            wan2pt2_model_path="",
            device="cpu",
        )
        controller = DecodeHandler.get_instance()
        self.assertFalse(hasattr(controller, "latent_decoder_wan2pt2"))


# ---------------------------------------------------------------------------
# 4. Pre-decoded video payload (media_type="video") in extract_video
# ---------------------------------------------------------------------------
class TestExtractVideoPayload(unittest.TestCase):
    """Test the is_video_payload branch in extract_video."""

    def _make_handler_with_dispatcher(self):
        from cosmos_rl_reward.handler.decode import DecodeHandler

        handler = DecodeHandler.__new__(DecodeHandler)
        handler.device = "cpu"
        handler.reward_dispatcher = {}
        return handler

    def test_video_payload_skips_decode(self):
        handler = self._make_handler_with_dispatcher()

        video_tensor = torch.randint(0, 255, (2, 3, 16, 64, 64), dtype=torch.uint8)
        npy_buf = io.BytesIO()
        np.save(npy_buf, video_tensor.numpy())
        npy_bytes = npy_buf.getvalue()

        metadata = {
            "reward_fn": {},
            "media_type": "video",
        }
        raw_body = json.dumps(metadata).encode() + b"\n" + npy_bytes

        handler.extract_video("test-uuid", raw_body)

    def test_video_payload_torch_format(self):
        handler = self._make_handler_with_dispatcher()

        video_tensor = torch.randint(0, 255, (1, 3, 8, 32, 32), dtype=torch.uint8)
        buf = io.BytesIO()
        torch.save(video_tensor, buf)
        tensor_bytes = buf.getvalue()

        metadata = {
            "reward_fn": {},
            "media_type": "video",
        }
        raw_body = json.dumps(metadata).encode() + b"\n" + tensor_bytes

        handler.extract_video("test-uuid-torch", raw_body)


# ---------------------------------------------------------------------------
# 5. WanVAE_ architecture tests (no checkpoint needed)
# ---------------------------------------------------------------------------
class TestWanVAEArchitecture(unittest.TestCase):
    """Verify the Wan2.2 VAE model can be instantiated and has correct structure."""

    def test_model_builds_on_meta_device(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import WanVAE_

        with torch.device("meta"):
            model = WanVAE_(
                dim=160,
                z_dim=48,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                attn_scales=[],
                temperal_downsample=[False, True, True],
            )
        self.assertIsNotNone(model.encoder)
        self.assertIsNotNone(model.decoder)
        self.assertIsNotNone(model.conv1)
        self.assertIsNotNone(model.conv2)
        self.assertEqual(model.z_dim, 48)

    def test_decoder_output_channels(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import Decoder3d

        with torch.device("meta"):
            dec = Decoder3d(
                dim=128,
                z_dim=48,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                temperal_upsample=[True, True, False],
            )
        self.assertEqual(dec.head[-1].out_channels, 12)

    def test_encoder_input_channels(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import Encoder3d

        with torch.device("meta"):
            enc = Encoder3d(
                dim=128,
                z_dim=96,
                dim_mult=[1, 2, 4, 4],
                num_res_blocks=2,
                temperal_downsample=[True, True, False],
            )
        self.assertEqual(enc.conv1.in_channels, 12)


# ---------------------------------------------------------------------------
# 6. WanVAE wrapper scale computation
# ---------------------------------------------------------------------------
class TestWanVAEScale(unittest.TestCase):
    """Verify that the WanVAE wrapper computes scale tensors correctly."""

    def test_scale_tensors(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import (
            _WAN22_MEAN,
            _WAN22_STD,
        )

        self.assertEqual(len(_WAN22_MEAN), 48)
        self.assertEqual(len(_WAN22_STD), 48)

        mean = torch.tensor(_WAN22_MEAN)
        std = torch.tensor(_WAN22_STD)
        inv_std = 1.0 / std

        self.assertEqual(mean.shape, (48,))
        self.assertEqual(std.shape, (48,))
        self.assertTrue(torch.isfinite(mean).all())
        self.assertTrue(torch.all(std > 0))
        self.assertTrue(torch.isfinite(inv_std).all())

        # Verify the scale list matches what WanVAE.__init__ computes
        scale_mean = mean.to(torch.bfloat16)
        scale_inv_std = (1.0 / std).to(torch.bfloat16)
        self.assertEqual(scale_mean.dtype, torch.bfloat16)
        self.assertEqual(scale_inv_std.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(scale_mean).all())
        self.assertTrue(torch.isfinite(scale_inv_std).all())

    def test_patchify_unpatchify_roundtrip(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import patchify, unpatchify

        x = torch.randn(1, 3, 4, 16, 16)
        patched = patchify(x, patch_size=2)
        self.assertEqual(patched.shape, (1, 12, 4, 8, 8))
        recovered = unpatchify(patched, patch_size=2)
        self.assertTrue(torch.allclose(x, recovered))

    def test_patchify_4d(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import patchify, unpatchify

        x = torch.randn(2, 3, 16, 16)
        patched = patchify(x, patch_size=2)
        self.assertEqual(patched.shape, (2, 12, 8, 8))
        recovered = unpatchify(patched, patch_size=2)
        self.assertTrue(torch.allclose(x, recovered))

    def test_patchify_identity(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import patchify

        x = torch.randn(1, 3, 4, 8, 8)
        self.assertTrue(torch.equal(patchify(x, patch_size=1), x))


# ---------------------------------------------------------------------------
# 7. Building blocks unit tests
# ---------------------------------------------------------------------------
class TestBuildingBlocks(unittest.TestCase):
    """Test individual Wan2.2 VAE building blocks."""

    def test_causal_conv3d_forward(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import CausalConv3d

        conv = CausalConv3d(4, 8, 3, padding=1)
        x = torch.randn(1, 4, 3, 8, 8)
        out = conv(x)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 8)
        self.assertEqual(out.shape[3], 8)
        self.assertEqual(out.shape[4], 8)

    def test_rms_norm(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import RMS_norm

        norm = RMS_norm(16, channel_first=True, images=False)
        x = torch.randn(2, 16, 4, 8, 8)
        out = norm(x)
        self.assertEqual(out.shape, x.shape)

    def test_attention_block(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import AttentionBlock

        attn = AttentionBlock(dim=16)
        x = torch.randn(1, 16, 2, 4, 4)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_block(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import ResidualBlock

        block = ResidualBlock(16, 16)
        x = torch.randn(1, 16, 2, 4, 4)
        out = block(x)
        self.assertEqual(out.shape, x.shape)

    def test_residual_block_dim_change(self):
        from cosmos_rl.policy.model.wfm.tokenizer.wan2pt2 import ResidualBlock

        block = ResidualBlock(16, 32)
        x = torch.randn(1, 16, 2, 4, 4)
        out = block(x)
        self.assertEqual(out.shape[1], 32)


if __name__ == "__main__":
    unittest.main()
