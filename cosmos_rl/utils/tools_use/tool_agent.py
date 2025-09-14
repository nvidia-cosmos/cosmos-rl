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

from typing import List, Optional, Dict

from cosmos_rl.utils.tools_use.schema import OpenAIFunctionToolSchema
from cosmos_rl.utils.logging import logger
from .base_tool import BaseTool
from .base_tool_parser import ToolParser
from .schema import ToolResponse


class ToolAgent:
    def __init__(self, tool_parser: ToolParser, tools: List[BaseTool]):
        self.tool_parser = tool_parser
        self.tools: Dict[str, BaseTool] = {tool.name: tool for tool in tools}

    def call_tools(
        self, text: str, groud_truth: Optional[str] = None
    ) -> ToolResponse | None:
        """Call tool and return tool response. Support multiple tool calls."""
        _, tool_calls = self.tool_parser.extract_tool_calls(text)

        if not tool_calls:
            return None

        tool_responses = []
        for tool_call in tool_calls:
            try:
                tool_name = tool_call.name
                tool = self.tools[tool_name]
                with tool.tool_context(groud_truth):
                    response = tool(**tool_call.arguments)
                    tool_responses.append(response)
            except Exception as e:
                logger.error(f"[ToolAgent] Failed to execute tool {tool_name}: {e}")

        return tool_responses

    def tool_schemas(self) -> List[OpenAIFunctionToolSchema]:
        return [tool.tool_schema.model_dump() for tool in self.tools.values()]
