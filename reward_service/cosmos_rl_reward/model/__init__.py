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

try:
    from cosmos_rl_reward.model.dance_grpo import DanceGRPOVideoReward  # noqa: F401
except ImportError:
    pass

try:
    from cosmos_rl_reward.model.cosmos_reason1 import CosmosReason1Reward  # noqa: F401
except ImportError:
    pass

try:
    from cosmos_rl_reward.model.ocr import OcrReward  # noqa: F401
except ImportError as e:
    print("[DEBUG] OcrReward not found")
    print(e)
    pass

try:
    from cosmos_rl_reward.model.image_reward import ImageReward  # noqa: F401
except ImportError as e:
    print("[DEBUG] ImageReward not found")
    print(e)
    pass

try:
    from cosmos_rl_reward.model.gen_eval import GenEvalReward  # noqa: F401
except ImportError as e:
    print("[DEBUG] GenEvalReward not found")
    print(e)
    pass

try:
    from cosmos_rl_reward.model.hpsv2 import HPSv2Reward  # noqa: F401
except ImportError as e:
    print("[DEBUG] HPSv2Reward not found")
    print(e)
    pass

try:
    from cosmos_rl_reward.model.pickscore import PickScoreReward  # noqa: F401
except ImportError as e:
    print("[DEBUG] PickScoreReward not found")
    print(e)
    pass