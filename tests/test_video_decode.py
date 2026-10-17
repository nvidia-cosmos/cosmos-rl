# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the video decoder without model downloads or a GPU."""

from pathlib import Path
import unittest

import av
import torchvision
from torchvision.io import read_video, read_video_timestamps


class TestVideoDecode(unittest.TestCase):
    video = Path(__file__).parent / "data" / "test_data_packer.mp4"

    def test_read_video(self):
        frames, _, info = read_video(str(self.video), pts_unit="sec")
        self.assertEqual(frames.ndim, 4)
        self.assertGreater(frames.shape[0], 0)
        self.assertEqual(frames.shape[-1], 3)
        self.assertGreater(info["video_fps"], 0)

    def test_read_video_timestamps(self):
        timestamps, fps = read_video_timestamps(str(self.video), pts_unit="sec")
        self.assertGreater(len(timestamps), 0)
        self.assertGreater(fps, 0)
        self.assertEqual(timestamps, sorted(timestamps))


if __name__ == "__main__":
    print(
        f"Video dependencies: av={av.__version__} torchvision={torchvision.__version__}"
    )
    unittest.main()
