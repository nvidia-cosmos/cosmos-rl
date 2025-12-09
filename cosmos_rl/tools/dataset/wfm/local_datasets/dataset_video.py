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

"""Generic video dataset loader for Cosmos Predict2."""

import os
import pickle
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
from torchvision import transforms as T

from cosmos_rl.utils.logging import logger
from cosmos_rl.tools.dataset.wfm.local_datasets.dataset_utils import (
    ResizePreprocess,
    ToTensorVideo,
)


class VideoDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        video_size: tuple[int, int],
        offline_text_embedding: bool = True,
        text_encoder_type: str = "t5_xxl",
    ) -> None:
        """Dataset class for loading image-text-to-video generation data.

        Args:
            dataset_dir (str): Base path to the dataset directory
            num_frames (int): Number of frames to load per sequence
            video_size (tuple[int, int]): Target size (H,W) for video frames
            offline_text_embedding (bool): Whether to use pre-processed offline text embeddings, if not, text embeddings will be computed online
            text_encoder_type (str): Type of text encoder to use, ['t5_xxl', 'cosmos_reason1']

        Returns dict with:
            - video: RGB frames tensor [T,C,H,W]
            - video_name: Dict with episode/frame metadata
        """

        super().__init__()
        assert text_encoder_type in ["t5_xxl", "cosmos_reason1"], (
            f"Unsupported text_encoder_type: {text_encoder_type}. "
            "Supported types are: [t5_xxl, cosmos_reason1]"
        )
        self.dataset_dir = dataset_dir
        self.sequence_length = num_frames

        video_dir = os.path.join(self.dataset_dir, "videos")
        self.caption_dir = os.path.join(self.dataset_dir, "metas")

        self.video_paths = [
            os.path.join(video_dir, f)
            for f in os.listdir(video_dir)
            if f.endswith(".mp4")
        ]

        self.offline_text_embedding = offline_text_embedding
        self.text_encoder_type = text_encoder_type
        if self.offline_text_embedding:
            if self.text_encoder_type == "t5_xxl":
                self.text_embedding_dir = os.path.join(self.dataset_dir, "t5_xxl")
                self.text_encoder_num_tokens = 512
                self.text_encoder_embed_dim = 1024
            else:  # cosmos_reason1
                self.text_embedding_dir = os.path.join(
                    self.dataset_dir, "cosmos_reason1"
                )
                self.text_encoder_num_tokens = 512
                self.text_encoder_embed_dim = 100352
                raise NotImplementedError(
                    "cosmos_reason1 offline text embedding is not implemented yet."
                )

            # remove video paths that does not have text_embedding
            self.video_paths = [
                vp
                for vp in self.video_paths
                if os.path.exists(
                    os.path.join(
                        self.text_embedding_dir,
                        os.path.basename(vp).replace(".mp4", ".pickle"),
                    )
                )
            ]

        self.video_paths = sorted(self.video_paths)
        logger.info(f"{len(self.video_paths)} videos in total")

        self.num_failed_loads = 0
        self.preprocess = T.Compose(
            [ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))]
        )

    def __str__(self) -> str:
        return f"{len(self.video_paths)} samples from {self.dataset_dir}"

    def __len__(self) -> int:
        return len(self.video_paths)

    def _load_video(self, video_path: str) -> tuple[np.ndarray, float]:
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        total_frames = len(vr)
        if total_frames < self.sequence_length:
            raise ValueError(
                f"Video {video_path} has only {total_frames} frames, "
                f"at least {self.sequence_length} frames are required."
            )

        # randomly sample a sequence of frames
        max_start_idx = total_frames - self.sequence_length
        start_frame = np.random.randint(0, max_start_idx)
        end_frame = start_frame + self.sequence_length
        frame_ids = np.arange(start_frame, end_frame).tolist()

        frame_data = vr.get_batch(frame_ids).asnumpy()
        vr.seek(0)  # set video reader point back to 0 to clean up cache

        try:
            fps = vr.get_avg_fps()
        except Exception:  # failed to read FPS, assume it is 16
            fps = 16
        del vr  # delete the reader to avoid memory leak
        return frame_data, fps

    def _load_text(self, text_source: Path) -> str:
        """Load text caption from file."""
        try:
            return text_source.read_text().strip()
        except Exception as e:
            logger.warning(f"Failed to read caption file {text_source}: {e}")
            return ""

    def _get_frames(self, video_path: str) -> tuple[torch.Tensor, float]:
        frames, fps = self._load_video(video_path)
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W]
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        return frames, fps

    def __getitem__(self, index: int) -> dict | Any:
        try:
            data = dict()
            video, fps = self._get_frames(self.video_paths[index])
            video = video.permute(
                1, 0, 2, 3
            )  # Rearrange from [T, C, H, W] to [C, T, H, W]
            video_path = self.video_paths[index]
            caption_path = os.path.join(
                self.caption_dir,
                os.path.basename(video_path).replace(".mp4", ".txt"),
            )
            data["video"] = video
            data["ai_caption"] = self._load_text(Path(caption_path))

            if self.offline_text_embedding:
                t5_embedding_path = os.path.join(
                    self.text_embedding_dir,
                    os.path.basename(video_path).replace(".mp4", ".pickle"),
                )
                with open(t5_embedding_path, "rb") as f:
                    t5_embedding = pickle.load(f)[
                        0
                    ]  # [n_tokens, self.text_encoder_embed_dim]
                n_tokens = t5_embedding.shape[0]
                if n_tokens < self.text_encoder_num_tokens:
                    t5_embedding = np.concatenate(
                        [
                            t5_embedding,
                            np.zeros(
                                (
                                    self.text_encoder_num_tokens - n_tokens,
                                    self.text_encoder_embed_dim,
                                ),
                                dtype=np.float32,
                            ),
                        ],
                        axis=0,
                    )
                t5_text_mask = torch.zeros(
                    self.text_encoder_num_tokens, dtype=torch.int64
                )
                t5_text_mask[:n_tokens] = 1

                data["t5_text_embeddings"] = torch.from_numpy(t5_embedding)
                data["t5_text_mask"] = t5_text_mask

            _, _, h, w = video.shape

            data["fps"] = fps
            data["image_size"] = torch.tensor([h, w, h, w])
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, h, w)

            return data
        except Exception as e:
            self.num_failed_loads += 1
            logger.warning(
                f"Failed to load video {self.video_paths[index]} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
            )
            # Randomly sample another video
            return self[np.random.randint(len(self.video_paths))]
