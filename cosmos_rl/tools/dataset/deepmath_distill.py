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

import argparse
import re
from typing import Any, Optional
import toml
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.tools.dataset.math_grpo import (
    MathDataPacker,
)
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset
from datasets import concatenate_datasets
import cosmos_rl.utils.util as util
from cosmos_rl.launcher.worker_entry import main as launch_dispatcher
from cosmos_rl.policy.config import Config


class DeepMathDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def setup(
        self,
        config: Config,
    ):
        self.config = config

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        converted_item = item["question"]
        messages = [
            {"role": "user", "content": converted_item},
            {"role": "assistant", "content": ""},
        ]
        """
        Format like this:
            <|im_start|>system
            You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
            <|im_start|>user
            What can you help me with?<|im_end|>
            <|im_start|>assistant
            <think>

            </think>
            I can help you with...<|im_end|>
        """
        munual_converted = ""
        for idx, message in enumerate(messages):
            assert (
                message.get("thinking") is None
            ), "TODO: support CoT in Qwen3 renderer"
            assert isinstance(
                message["content"], str
            ), "Qwen3Renderer only supports message with string content"
            maybe_newline = "\n" if idx > 0 else ""
            ob_str = f"{maybe_newline}<|im_start|>{message['role']}\n"
            ac_content = message["content"]
            if message["role"] == "assistant" and "</think>" in ac_content:
                # Multi-turn conversation, we remove the thinking section from the assistant message.
                # This matches how Qwen3 models were trained - they only see their own thinking
                # during the current turn, not from previous turns.
                ac_content = ac_content.split("</think>")[1].lstrip()
            elif message["role"] == "assistant" and "<think>" not in ac_content:
                # Matching the paper, we force the assistant to start with <think>. Some SFT datasets include
                # <think> in the assistant messages, we so don't need to re-add it in those cases.
                ob_str += "<think>\n"
            # Observation (prompt) part
            assert (
                "tool_calls" not in message
            ), "Tool calls not supported in Qwen3 renderer"
            ac_content += "<|im_end|>"
            munual_converted += ob_str
            if message["role"] != "assistant":
                munual_converted += ac_content
        # logger.info(f"Converted prompt for idx {idx}:\n{munual_converted}")
        return RLPayload(prompt=munual_converted)


def custom_reward_fn(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> float:
    # Always return 0.0 reward for Distillation tasks
    return 0.0


def evaluate_amc23_or_aime24_zeroshot(
    to_be_evaluated: str, reference: Optional[Any] = None, *args, **kwargs
) -> float:
    """Evaluate AMC23 or AIME24/25 zero-shot performance.

    Args:
        input_datapath: Path to model output JSONL file
        test_datapath: Path to AMC23/AIME24/AIME25 test JSONL file

    Returns:
        float: Accuracy score
    """
    from cosmos_rl.tools.eval_help.grader import (
        math_equal,
        math_answer_cleaning,
        round_number,
        is_equal_after_calculation,
    )

    score = 0.0
    try:
        pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        pattern2 = r"\*\*(.*?)\*\*"
        pattern3 = r"\\\[\n(.*?)\n\\\]"
        pattern4 = r"is \\\((.*?)\\\)"
        pattern5 = r"\\\[\\n(.*?)\\n\\\]"

        pattern1_re = re.compile(pattern1, re.DOTALL)
        pattern2_re = re.compile(pattern2, re.DOTALL)
        pattern3_re = re.compile(pattern3, re.DOTALL)
        pattern4_re = re.compile(pattern4, re.DOTALL)
        pattern5_re = re.compile(pattern5, re.DOTALL)

        # logger.info(
        #     f"Evaluating model output: {to_be_evaluated} against reference: {reference}"
        # )
        line = to_be_evaluated
        gold = reference

        matches1 = pattern1_re.findall(line)
        matches2 = pattern2_re.findall(line)
        matches3 = pattern3_re.findall(line)
        matches4 = pattern4_re.findall(line)
        matches5 = pattern5_re.findall(line)

        if len(matches1) >= 1:
            extracted_answer = matches1[-1]
        elif len(matches2) >= 1:
            extracted_answer = matches2[-1]
        elif len(matches3) >= 1:
            extracted_answer = matches3[-1]
        elif len(matches4) >= 1:
            extracted_answer = matches4[-1]
        elif len(matches5) >= 1:
            extracted_answer = matches5[-1]
        else:
            extracted_answer = None

        if extracted_answer is None:
            score = 0.0
        if gold is None:
            score = 0.0

        extracted_answer = math_answer_cleaning(extracted_answer)
        gold = math_answer_cleaning(gold)

        # logger.info(f"Extracted answer: {extracted_answer}, Gold answer: {gold}")
        if math_equal(extracted_answer, gold):
            score = 1.0
        elif round_number(extracted_answer) == round_number(gold):
            score = 1.0
        elif is_equal_after_calculation(extracted_answer, gold):
            score = 1.0
        # logger.info(f"Evaluation score: {score}")
    except Exception:
        # logger.warning(f"Error in evaluate_amc23_or_aime24_zeroshot: {e}")
        score = 0.0
    return score


class AIMEDataSet(Dataset):
    def setup(
        self,
        config: Config,
    ):
        self.config = config
        logger.info(
            f"Loading AIME validation dataset... {config.validation.dataset.name}"
        )
        dataset = util.load_data_from_disk_or_hf(
            config.validation.dataset.name,
            config.validation.dataset.subset,
            config.validation.dataset.revision or None,
        )
        dataset_list = []
        for split_name in config.validation.dataset.split:
            print(
                f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
            )
            dataset_list.append(dataset[split_name])
        self.dataset = concatenate_datasets(dataset_list)
        self.tokenizer = util.setup_tokenizer(config.policy.model_name_or_path)
        logger.info(f"Final AIME validation dataset size = {len(self.dataset)}")

    def __len__(self):
        if isinstance(self.config.validation.dataset.test_size, float):
            return int(len(self.dataset) * self.config.validation.dataset.test_size)
        elif isinstance(self.config.validation.dataset.test_size, int):
            return min(self.config.validation.dataset.test_size, len(self.dataset))
        return len(self.dataset)

    def __getitem__(self, idx: int) -> RLPayload:
        question = self.dataset[idx]["problem"].strip()
        assert isinstance(
            question, str
        ), f"Prompt should be a string, but got {type(question)}, {question}"
        # Convert to templated prompt
        final_prompt = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.".format(
            question=question
        )
        conversation = [
            {
                "role": "user",
                "content": final_prompt,
            }
        ]
        prompt = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        if "<think>" not in prompt:
            # Ensure the model starts with <think>
            prompt = prompt + "<think>\n"
        # logger.info(f"AIME Prompt for idx {idx}:\n{prompt}")
        return RLPayload(prompt=prompt)

    def get_reference_answer(self, idx: int) -> Any:
        """
        This is mandatory for GRPO to get a reference answer for reward computation.
        """
        response = self.dataset[idx]["solution"]
        pattern1 = r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
        pattern1_re = re.compile(pattern1, re.DOTALL)
        matches1 = pattern1_re.findall(response)
        if len(matches1) >= 1:
            response = matches1[-1]
        # if "boxed" not in response:
        #     response = "$\\boxed{" + response + "}$"
        return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_known_args()[0]
    with open(args.config, "r") as f:
        config = toml.load(f)
    config = Config.from_dict(config)
    # Download HF dataset only on launcher worker
    dataset = util.load_data_from_disk_or_hf(
        config.train.train_policy.dataset.name,
        config.train.train_policy.dataset.subset,
        config.train.train_policy.dataset.revision or None,
    )
    dataset_list = []
    for split_name in config.train.train_policy.dataset.split:
        print(
            f"Appending split {split_name}, dataset size = {len(dataset[split_name])}"
        )
        dataset_list.append(dataset[split_name])
    train_dataset = concatenate_datasets(dataset_list)
    logger.info(f"Final training dataset size = {len(train_dataset)}")

    launch_dispatcher(
        dataset=DeepMathDataset(dataset=train_dataset),
        val_dataset=AIMEDataSet(),
        reward_fns=[custom_reward_fn],
        val_reward_fns=[evaluate_amc23_or_aime24_zeroshot],
        val_data_packer=MathDataPacker(),
    )
