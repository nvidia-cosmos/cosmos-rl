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

import torch
import torch.nn as nn

from typing import Dict, List, Optional, Union

from cosmos_rl.policy.config.wfm import ConditionerConfig
from cosmos_rl.utils.wfm.distributed import batch_mul
from cosmos_rl.utils.wfm.utils import count_params


class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()

        self._is_trainable = None
        self._dropout_rate = None
        self._input_key = None
        # TODO: (qsh 2024-02-14) a cleaner define or we use return dict by default?
        self._return_dict = False

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def dropout_rate(self) -> Union[float, torch.Tensor]:
        return self._dropout_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @property
    def is_return_dict(self) -> bool:
        return self._return_dict

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @dropout_rate.setter
    def dropout_rate(self, value: Union[float, torch.Tensor]):
        self._dropout_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_return_dict.setter
    def is_return_dict(self, value: bool):
        self._return_dict = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @dropout_rate.deleter
    def dropout_rate(self):
        del self._dropout_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

    @is_return_dict.deleter
    def is_return_dict(self):
        del self._return_dict

    def random_dropout_input(
        self,
        in_tensor: torch.Tensor,
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        return batch_mul(
            torch.bernoulli(
                (1.0 - dropout_rate) * torch.ones(in_tensor.shape[0])
            ).type_as(in_tensor),
            in_tensor,
        )

    def details(self) -> str:
        return ""

    def summary(self) -> str:
        input_key = (
            self.input_key
            if self.input_key is not None
            else getattr(self, "input_keys", None)
        )
        return (
            f"{self.__class__.__name__} \n\tinput key: {input_key}"
            f"\n\tParam count: {count_params(self, False)} \n\tTrainable: {self.is_trainable}"
            f"\n\tDropout rate: {self.dropout_rate}"
            f"\n\t{self.details()}"
        )


class ReMapkey(AbstractEmbModel):
    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
        dtype: Optional[str] = None,
    ):
        super().__init__()
        self.output_key = output_key
        self.dtype = {
            None: None,
            "float": torch.float32,
            "bfloat16": torch.bfloat16,
            "half": torch.float16,
            "float16": torch.float16,
            "int": torch.int32,
            "long": torch.int64,
        }[
            dtype
        ]  # TODO: (@yenchenl): Do we need this? Ideally, ReMapkey should just remap and do not change dtype.
        self._input_key = input_key
        self._output_key = output_key
        self._dropout_rate = dropout_rate

    def forward(self, element: torch.Tensor) -> Dict[str, torch.Tensor]:
        key = self.output_key if self.output_key else self.input_key
        if isinstance(element, torch.Tensor):
            element = element.to(dtype=self.dtype)
        return {key: element}

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\tDtype: {self.dtype}"


class BooleanFlag(AbstractEmbModel):
    def __init__(
        self,
        input_key: str,
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
    ):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate
        self.output_key = output_key

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        del args, kwargs
        key = self.output_key if self.output_key else self.input_key
        return {key: self.flag}

    def random_dropout_input(
        self,
        in_tensor: torch.Tensor,
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        del key
        dropout_rate = dropout_rate if dropout_rate is not None else self.dropout_rate
        self.flag = (
            torch.bernoulli((1.0 - dropout_rate) * torch.ones(1))
            .bool()
            .to(device=in_tensor.device)
        )
        return in_tensor

    def details(self) -> str:
        key = self.output_key if self.output_key else self.input_key
        return f"Output key: {key} \n\t This is a boolean flag"


class TextAttr(AbstractEmbModel):
    def __init__(
        self,
        input_key: List[str],
        output_key: Optional[str] = None,
        dropout_rate: Optional[float] = 0.0,
    ):
        super().__init__()
        self._input_key = input_key
        self._dropout_rate = dropout_rate

    def forward(self, token: torch.Tensor):
        return {"crossattn_emb": token}

    def random_dropout_input(
        self,
        in_tensor: torch.Tensor,
        dropout_rate: Optional[float] = None,
        key: Optional[str] = None,
    ) -> torch.Tensor:
        if key is not None and "mask" in key:
            return in_tensor
        return super().random_dropout_input(in_tensor, dropout_rate, key)

    def details(self) -> str:
        return "Output key: [crossattn_emb]"


EMB_CLS_MAPPING = {
    "remap_key": ReMapkey,
    "text_attr": TextAttr,
    "boolean_flag": BooleanFlag,
}


def get_embedder(embed_config: ConditionerConfig) -> AbstractEmbModel:
    """Instantiate an embedder model based on the configuration.

    Args:
        embed_config (Dict): Configuration dictionary for the embedder.

    Returns:
        AbstractEmbModel: An instance of the embedder model.
    """
    embedder_cls = EMB_CLS_MAPPING.get(embed_config.type, None)
    if embedder_cls is None:
        raise ValueError(
            f"Unknown embedder type: {embed_config.type}. "
            f"Available types: {list(EMB_CLS_MAPPING.keys())}"
        )
    return embedder_cls(
        input_key=embed_config.input_key,
        output_key=embed_config.output_key,
        dropout_rate=embed_config.dropout_rate,
    )
