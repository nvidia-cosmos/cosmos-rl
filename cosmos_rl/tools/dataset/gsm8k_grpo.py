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

import re
from typing import Optional, Any, List, Dict, Union
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from cosmos_rl.launcher.worker_entry import main as launch_worker
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.dispatcher.algo.reward import gsm8k_reward_fn
from cosmos_rl.dispatcher.data.packer import DecoderOnlyLLMDataPacker, DataPacker
from cosmos_rl.utils.modelscope import modelscope_load_dataset
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.tools_use import (
    ToolAgent,
    BaseTool,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from cosmos_rl.utils.tools_use.hermes_tool_parser import HermesToolParser
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
    add_tool_response_messages,
    add_assistant_message,
)
from cosmos_rl.dispatcher.data import RLPayload
from contextlib import contextmanager
import cosmos_rl.utils.util as util


class GSM8kDataset(Dataset):
    """TODO(zjx): we should refactor it with RLDataset."""

    def setup(self, config: CosmosConfig, *args, **kwargs):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        self.config = config
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)
        self.apply_chat_template = not config.rollout.multi_turn_config.enable
        modelscope_dataset_if_enabled = modelscope_load_dataset(
            config.train.train_policy.dataset.name, subset_name="main", split="train"
        )
        if modelscope_dataset_if_enabled is None:
            self.dataset = load_dataset("openai/gsm8k", "main", split="train")
        else:
            self.dataset = modelscope_dataset_if_enabled

    def __len__(self):
        return 16  # len(self.dataset)

    def __getitem__(self, idx: int) -> RLPayload:
        """
        For DecoderOnlyLLMDataPacker, it should either return:
        - raw text prompt to be converted into input_ids by both rollout and policy models;
        - conversation format:
        ```
        [
            {
                "role": "user",
                "content": f"{question} Let\'s think step by step and output the final answer after \"####\".",
            }
        ]
        ```
        """
        assert hasattr(
            self, "tokenizer"
        ), "`self.tokenizer` should be set by the launcher"
        question = self.dataset[idx]["question"]
        assert isinstance(
            question, str
        ), f"Prompt should be a string, but got {type(question)}, {question}"
        # Convert to templated prompt
        conversation = [
            {
                "role": "user",
                "content": f'{question} Let\'s think step by step and output the final answer after "####".',
            }
        ]

        if not self.apply_chat_template:
            return RLPayload(conversation=conversation)

        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        return RLPayload(prompt=prompt)

    def get_reference_answer(self, idx: int) -> Any:
        """
        This is mandatory for GRPO to get a reference answer for reward computation.
        """
        return self.dataset[idx]["answer"]


class GSM8kValDataset(GSM8kDataset):
    """
    This is a validation dataset for GSM8K, which is used to evaluate the performance of the model.
    It should be used in the launcher to evaluate the model during training.
    """

    def setup(self, config: CosmosConfig, *args, **kwargs):
        if not config.validation.enable:
            logger.warning(
                "Validation is not enabled in the config. Skipping setup for GSM8kValDataset."
            )
            return
        self.apply_chat_template = not config.rollout.multi_turn_config.enable
        self.config = config
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)

        self.dataset = load_dataset(
            config.validation.dataset.name, config.validation.dataset.subset
        )
        if config.validation.dataset.split:
            if isinstance(config.validation.dataset.split, list):
                dataset_list = []
                for split_name in config.validation.dataset.split:
                    dataset_list.append(self.dataset[split_name])
                self.dataset = ConcatDataset(dataset_list)
            else:
                assert isinstance(config.validation.dataset.split, str)
                self.dataset = self.dataset[config.validation.dataset.split]


def custom_reward_fn(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> float:
    assert isinstance(reference, str), "Reference answer should be a string"
    reward = gsm8k_reward_fn(to_be_evaluated, reference, *args, **kwargs)
    # Add more reward functions here
    # ...
    return reward


class GSM8kTool(BaseTool):
    def __init__(self):
        _tool_name = "calc_gsm8k_reward"
        _tool_schema = OpenAIFunctionToolSchema.model_validate(
            {
                "type": "function",
                "function": {
                    "name": _tool_name,
                    "description": "A tool for calculating the reward of gsm8k. (1.0 if parsed answer is correct, 0.0 if parsed answer is incorrect or not correctly parsed)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "The model's answer to the GSM8K math problem, must be a digits",
                            },
                        },
                        "required": ["answer"],
                    },
                },
            }
        )
        super().__init__(_tool_name, _tool_schema)

    @contextmanager
    def tool_context(self, groud_truth: str):
        self.groud_truth = groud_truth
        yield
        self.groud_truth = None

    def _extract_solution(self, solution_str, method="strict"):
        assert method in ["strict", "flexible"]

        if method == "strict":
            # this also tests the formatting of the model
            solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
            if solution is None:
                final_answer = None
            else:
                final_answer = solution.group(0)
                final_answer = (
                    final_answer.split("#### ")[1].replace(",", "").replace("$", "")
                )
        elif method == "flexible":
            answer = re.findall("(\\-?[0-9\\.\\,]+)", solution_str)
            final_answer = None
            if len(answer) == 0:
                # no reward is there is no answer
                pass
            else:
                invalid_str = ["", "."]
                # find the last number that is not '.'
                for final_answer in reversed(answer):
                    if final_answer not in invalid_str:
                        break
        return final_answer

    def function(self, answer: str) -> ToolResponse:
        # model may generate tool call with answer like "### 123" or "123",
        # so we set flexible method to extract the answer.
        answer = self._extract_solution(answer, method="flexible")
        truth_answer = self._extract_solution(self.groud_truth, method="flexible")
        if truth_answer is None:
            truth_answer = ""
        try:
            reward = 1.0 if answer == truth_answer else 0.0
        except Exception:
            reward = 0.0
        return ToolResponse(
            text=f"Current parsed is answer={truth_answer}, reward={reward}"
        )


def custom_logger_fn(data: dict, step: int) -> None:
    """
    Example of custom logger function.
    Please mimic this to implement your own custom logger function.
    Your custom logger function is in the same format with this `custom_logger_fn`.
    It shoule be with two arguments: `data` and `step`.

    Arguments:
        The data argument will always be the follwing format:
            data = {
                "train/loss_avg": total_loss_avg,
                "train/loss_max": total_loss_max,
                "train/learning_rate": total_learning_rate,
                "train/iteration_time": total_iter_time_avg,
                "train/kl_loss_avg": total_kl_loss_avg,
                "train/kl_loss_max": total_kl_loss_max,
                "train/grad_norm": total_grad_norm,
            }
        The data argument includes the infomation could be logged.
        The step argument is the number for the current training step.

    Usage:
        After implementing this `custom_logger_fn` function, you can pass it through the `custom_logger_fns` argument of the `launch_worker`.
        The `custom_logger_fns` argument of the `launch_worker` is a list of Callable, which can includes multiple custom logger functions.
        Then the given custom logger functions will be called with the appropriate arguments during the training process.
    """
    # total_loss_avg = data.get("train/loss_avg", None)
    # total_loss_max = data.get("train/loss_max", None)
    # total_learning_rate = data.get("train/learning_rate", None)
    # total_iter_time_avg = data.get("train/iteration_time", None)
    # total_kl_loss_avg = data.get("train/kl_loss_avg", None)
    # total_kl_loss_max = data.get("train/kl_loss_max", None)
    # total_grad_norm = data.get("train/grad_norm", None)
    # This is only an example and no actual operations done for the logging.
    pass


class GSM8kDataPacker(DataPacker):
    """
    This is a demo data packer that wraps the underlying data packer of the selected model.
    This is meaningless for this example, but useful for explaining:
        - how dataset data is processed and collated into a mini-batch for rollout engine;
        - how rollout output is processed and collated into a mini-batch for policy model;
    """

    def __init__(self, tool_agent: Optional[ToolAgent] = None, *args, **kwargs):
        super().__init__(tool_agent=tool_agent, *args, **kwargs)
        # Check source code of DecoderOnlyLLMDataPacker to see how it's implemented
        self.underlying_data_packer = DecoderOnlyLLMDataPacker(tool_agent=tool_agent)

    def setup(
        self,
        config: CosmosConfig,
        *args,
        tool_agent: Optional[ToolAgent] = None,
        **kwargs,
    ):
        """
        This method is optional and get called by launcher after being mounted
        `config`: config;
        """
        super().setup(config, *args, tool_agent=tool_agent, **kwargs)
        self.underlying_data_packer.setup(
            config, *args, tool_agent=tool_agent, **kwargs
        )

    def get_rollout_input(self, item: Any) -> Any:
        """
        Convert dataset item into what rollout engine (e.g. vllm) expects
        """
        return self.underlying_data_packer.get_rollout_input(item)

    def rollout_collate_fn(self, items: List[Any]) -> Any:
        """
        Collate the rollout inputs into a mini-batch for rollout engine
        """
        return self.underlying_data_packer.rollout_collate_fn(items)

    def get_policy_input(
        self,
        item: Any,
        rollout_output: Union[str, List[int]],
        n_ignore_prefix_tokens: int = 0,
    ) -> Any:
        """
        Process samples & rollout output before collating them into a mini-batch
        """
        return self.underlying_data_packer.get_policy_input(
            item, rollout_output, n_ignore_prefix_tokens
        )

    def policy_compute_max_len(self, processed_samples: List[Any]) -> int:
        """
        Compute the maximum sequence length of the mini-batch
        """
        return self.underlying_data_packer.policy_compute_max_len(processed_samples)

    def policy_collate_fn(
        self, processed_samples: List[Any], computed_max_len: int
    ) -> Dict[str, Any]:
        """
        Collate the mini-batch into the kwargs required by the policy model
        """
        return self.underlying_data_packer.policy_collate_fn(
            processed_samples, computed_max_len
        )

    def extend_conversation(
        self,
        conversation: ConversationType,
        responses: List[str],
        ground_truth: Optional[str] = None,
    ) -> ConversationType:
        """
        Extend the conversation by models response.
        """
        assert self.tool_agent is not None, "Tool agent is not set"

        real_response = ""
        for response in responses:
            if response:
                real_response = response
                break

        if real_response == "":
            return conversation

        # 1. add real_response as assistant
        conversation = add_assistant_message(conversation, real_response)

        # 2. check if the response contains tool call instruction
        tool_responses = self.tool_agent.call_tools(real_response, ground_truth)
        if tool_responses:
            for tr in tool_responses:
                conversation = add_tool_response_messages(conversation, tr.text)

        return conversation


if __name__ == "__main__":

    def get_dataset(config: CosmosConfig) -> Dataset:
        return GSM8kDataset()

    def get_val_dataset(config: CosmosConfig) -> Dataset:
        return GSM8kValDataset()

    # build the tool agent for multi-turn conversation
    tool_agent = ToolAgent(HermesToolParser(), [GSM8kTool()])

    # It is best practice to pass the dataset as a factory function
    launch_worker(
        dataset=get_dataset,
        val_dataset=get_val_dataset,
        # Override the reward functions defined in toml
        reward_fns=[custom_reward_fn],
        # Optional: if not provided, the default data packer of the selected model will be used
        data_packer=GSM8kDataPacker(tool_agent=tool_agent),
        val_data_packer=GSM8kDataPacker(tool_agent=tool_agent),
        custom_logger_fns=[custom_logger_fn],
    )
