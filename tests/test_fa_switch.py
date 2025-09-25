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

import unittest
import torch

from cosmos_rl.policy.kernel.modeling_utils import FlashAttnMeta


class TestFASwitch(unittest.TestCase):
    def _test_fa_switch(self):
        torch_compile = False
        a_flash_attn_meta = FlashAttnMeta(torch_compile=torch_compile)
        b_flash_attn_meta = FlashAttnMeta(torch_compile=torch_compile)

        assert (
            a_flash_attn_meta is b_flash_attn_meta
        ), "FlashAttnMeta is not a singleton"

        major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
        if major >= 9 and not torch_compile:
            assert a_flash_attn_meta.fa_version == 3
            assert b_flash_attn_meta.fa_version == 3
        else:
            assert a_flash_attn_meta.fa_version == 2
            assert b_flash_attn_meta.fa_version == 2

    def test_fa_switch_no_compile(self):
        self._test_fa_switch()


if __name__ == "__main__":
    unittest.main()
