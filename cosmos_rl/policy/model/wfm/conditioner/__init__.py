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

import copy
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from cosmos_rl.policy.config.wfm import ConditionerConfig
from cosmos_rl.utils.wfm.utils import disabled_train
from cosmos_rl.policy.model.wfm.conditioner.emb_models import (
    get_embedder,
    AbstractEmbModel,
    TextAttr,
)
from cosmos_rl.policy.model.wfm.conditioner.condition import Vid2VidCondition


class GeneralConditioner(nn.Module, ABC):
    """
    An abstract module designed to handle various embedding models with conditional and unconditional configurations.
    This abstract base class initializes and manages a collection of embedders that can dynamically adjust
    their dropout rates based on conditioning.

    Attributes:
        KEY2DIM (dict): A mapping from output keys to dimensions used for concatenation.
        embedders (nn.ModuleDict): A dictionary containing all embedded models initialized and configured
                                   based on the provided configurations.

    Parameters:
        emb_models (List]): A List of configurations for initializing the embedders.

    Example:
        See Edify4ConditionerConfig
    """

    KEY2DIM = {"crossattn_emb": 1}

    def __init__(self, emb_models: List[ConditionerConfig]):
        super().__init__()
        self.embedders = nn.ModuleDict()
        for emb_config in emb_models:
            emb_name = emb_config.name
            embedder = get_embedder(emb_config)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = getattr(emb_config, "is_trainable", True)
            embedder.dropout_rate = getattr(emb_config, "dropout_rate", 0.0)
            if not embedder.is_trainable:
                embedder.train = disabled_train
                for param in embedder.parameters():
                    param.requires_grad = False
                embedder.eval()

            # log.info(f"Initialized embedder #{n}-{emb_name}: \n {embedder.summary()}")
            self.embedders[emb_name] = embedder

    @abstractmethod
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Any:
        """Should be implemented in subclasses to handle conditon datatype"""
        raise NotImplementedError

    def _forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Processes the input batch through all configured embedders, applying conditional dropout rates if specified.
        Output tensors for each key are concatenated along the dimensions specified in KEY2DIM.

        Parameters:
            batch (Dict): The input data batch to process.
            override_dropout_rate (Optional[Dict[str, float]]): Optional dictionary to override default dropout rates
                                                                per embedder key.

        Returns:
            Dict: A dictionary of output tensors concatenated by specified dimensions.

        Note:
            In case the network code is sensitive to the order of concatenation, you can either control the order via \
            config file or make sure the embedders return a unique key for each output.
        """
        output = defaultdict(list)
        if override_dropout_rate is None:
            override_dropout_rate = {}

        # make sure emb_name in override_dropout_rate is valid
        for emb_name in override_dropout_rate.keys():
            assert emb_name in self.embedders, f"invalid name found {emb_name}"

        for emb_name, embedder in self.embedders.items():
            embedding_context = nullcontext if embedder.is_trainable else torch.no_grad
            with embedding_context():
                if isinstance(embedder.input_key, str):
                    emb_out = embedder(
                        embedder.random_dropout_input(
                            batch[embedder.input_key],
                            override_dropout_rate.get(emb_name, None),
                        )
                    )
                elif isinstance(embedder.input_key, list):
                    emb_out = embedder(
                        *[
                            embedder.random_dropout_input(
                                batch.get(k),
                                override_dropout_rate.get(emb_name, None),
                                k,
                            )
                            for k in embedder.input_key
                        ]
                    )
                else:
                    raise KeyError(
                        f"Embedder '{embedder.__class__.__name__}' requires an 'input_key' attribute to be defined as either a string or list of strings"
                    )
            for k, v in emb_out.items():
                output[k].append(v)
        # Concatenate the outputs
        return {k: torch.cat(v, dim=self.KEY2DIM.get(k, -1)) for k, v in output.items()}

    def get_condition_uncondition(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Processes the provided data batch to generate two sets of outputs: conditioned and unconditioned. This method
        manipulates the dropout rates of embedders to simulate two scenarios — one where all conditions are applied
        (conditioned), and one where they are removed or reduced to the minimum (unconditioned).

        This method first sets the dropout rates to zero for the conditioned scenario to fully apply the embedders' effects.
        For the unconditioned scenario, it sets the dropout rates to 1 (or to 0 if the initial unconditional dropout rate
        is insignificant) to minimize the embedders' influences, simulating an unconditioned generation.

        Parameters:
            data_batch (Dict): The input data batch that contains all necessary information for embedding processing. The
                            data is expected to match the required format and keys expected by the embedders.

        Returns:
            Tuple[Any, Any]: A tuple containing two condition:
                - The first one contains the outputs with all embedders fully applied (conditioned outputs).
                - The second one contains the outputs with embedders minimized or not applied (unconditioned outputs).
        """
        cond_dropout_rates, dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            dropout_rates[emb_name] = 1.0 if embedder.dropout_rate > 1e-4 else 0.0

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(data_batch, override_dropout_rate=dropout_rates)
        return condition, un_condition

    def get_condition_with_negative_prompt(
        self,
        data_batch: Dict,
    ) -> Tuple[Any, Any]:
        """
        Similar functionality as get_condition_uncondition
        But use negative prompts for unconditon
        """
        cond_dropout_rates, uncond_dropout_rates = {}, {}
        for emb_name, embedder in self.embedders.items():
            cond_dropout_rates[emb_name] = 0.0
            if isinstance(embedder, TextAttr):
                uncond_dropout_rates[emb_name] = 0.0
            else:
                uncond_dropout_rates[emb_name] = (
                    1.0 if embedder.dropout_rate > 1e-4 else 0.0
                )

        data_batch_neg_prompt = copy.deepcopy(data_batch)
        if "neg_t5_text_embeddings" in data_batch_neg_prompt:
            if isinstance(
                data_batch_neg_prompt["neg_t5_text_embeddings"], torch.Tensor
            ):
                data_batch_neg_prompt["t5_text_embeddings"] = data_batch_neg_prompt[
                    "neg_t5_text_embeddings"
                ]

        condition: Any = self(data_batch, override_dropout_rate=cond_dropout_rates)
        un_condition: Any = self(
            data_batch_neg_prompt, override_dropout_rate=uncond_dropout_rates
        )

        return condition, un_condition


class Vid2VidConditioner(GeneralConditioner):
    def forward(
        self,
        batch: Dict,
        override_dropout_rate: Optional[Dict[str, float]] = None,
    ):
        output = super()._forward(batch, override_dropout_rate)
        return Vid2VidCondition(**output)
