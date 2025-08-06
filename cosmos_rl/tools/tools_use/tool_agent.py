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

from typing import List, Optional

from cosmos_rl.tools.tools_use.base_tool import BaseTool
from cosmos_rl.tools.tools_use.base_tool_parser import ToolParser
from cosmos_rl.tools.tools_use import ToolResponse


class ToolAgent:
    def __init__(self, tool_parser: ToolParser, tools: List[BaseTool]):
        self.tool_parser = tool_parser
        self.tools = tools

    def __call__(self, text: str, groud_truth: Optional[str] = None) -> ToolResponse:
        """Call tool and return tool response"""
        _, tool_calls = self.tool_parser.extract_tool_calls(text)

        if not tool_calls:
            return None

        assert len(tool_calls) == 1, "Only one tool call is supported for now"
        tool_call = tool_calls[0]

        try:
            tool_name = tool_call.name
            tool = self.tools[tool_name]
            with tool.tool_context(groud_truth):
                return tool(**tool_call.arguments)
        except Exception:
            return None
