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


class TestApex(unittest.TestCase):
    def test_fused_adam(self):
        import apex
        import torch

        model = torch.nn.Linear(10, 10).cuda()
        opt = apex.optimizers.FusedAdam(model.parameters(), lr=0.001)

        input_data = torch.randn(5, 10).cuda()
        output = model(input_data)
        loss = output.sum()

        opt.zero_grad()
        loss.backward()
        opt.step()


if __name__ == "__main__":
    # Run the tests
    unittest.main()
