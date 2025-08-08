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

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Optional

from transformers.utils import get_json_schema

from .schema import OpenAIFunctionToolSchema, ToolResponse


class BaseTool(ABC):
    def __init__(self, name: str, schema: Optional[OpenAIFunctionToolSchema] = None):
        self.name = name
        self._schema = schema

    def __call__(self, *args, **kwargs) -> Any:
        return self.function(*args, **kwargs)

    @abstractmethod
    @contextmanager
    def tool_context(self, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        """
        raise NotImplementedError("This method should be implemented by the subclass")

    @property
    def tool_schema(self) -> OpenAIFunctionToolSchema:
        """
        The schema of the tool.
        """
        if self._schema is None:
            # create the default schema for the tool
            _schema = get_json_schema(self.function)
            self._schema = OpenAIFunctionToolSchema.model_validate(_schema)
            self._schema.function.name = self.name
        return self._schema

    @abstractmethod
    def function(self, *args, **kwargs) -> ToolResponse:
        """
        The function to be called when the tool is used.
        """
        raise NotImplementedError("This method should be implemented by the subclass")
