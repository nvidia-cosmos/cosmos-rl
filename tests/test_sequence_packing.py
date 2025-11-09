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
import os
import sys
import subprocess
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
    print("Install transformers >= 4.57.0 to test Qwen3VLForConditionalGeneration")

try:
    from qwen_vl_utils import process_vision_info as qwen_vl_process_vision_info
except ImportError:
    qwen_vl_process_vision_info = None
    print("Install qwen_vl_utils >= 0.0.14 to test Qwen3VLForConditionalGeneration")


def get_inputs(hf_processor, conversations, pad_token_id):
    inputs = []
    valid_input_len = []
    max_input_len = 0
    for conversation in conversations:
        image_inputs, video_inputs, video_kwargs = qwen_vl_process_vision_info(
            conversation,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        if video_inputs is not None:
            video_inputs, video_metadatas = zip(*video_inputs)
            video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
        else:
            video_metadatas = None
        kwarg = {
            "return_tensors": "pt",
            "images": image_inputs,
            "videos": video_inputs,
            "video_metadata": video_metadatas,
            "do_resize": False,
        }
        text = hf_processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False,
        )
        input = hf_processor(
            text=[text],
            **kwarg,
            **video_kwargs,
        ).to(device="cuda")
        input.pop("attention_mask")
        inputs.append(input)
        valid_input_len.append(input["input_ids"].shape[1])

    max_input_len = max(valid_input_len)
    valid_input_len_tensor = torch.tensor(
        valid_input_len, dtype=torch.int32, device=inputs[0]["input_ids"].device
    )
    input0 = inputs[0]
    if len(inputs) > 1:
        for input in inputs[1:]:
            for key, value in input.items():
                if key == "input_ids":
                    if input0[key].shape[1] < max_input_len:
                        input0[key] = torch.cat(
                            [
                                input0[key],
                                torch.full(
                                    (
                                        input0[key].shape[0],
                                        max_input_len - input0[key].shape[1],
                                    ),
                                    pad_token_id,
                                    device=input0[key].device,
                                ),
                            ],
                            dim=1,
                        )
                    if value.shape[1] < max_input_len:
                        value = torch.cat(
                            [
                                value,
                                torch.full(
                                    (value.shape[0], max_input_len - value.shape[1]),
                                    pad_token_id,
                                    device=value.device,
                                ),
                            ],
                            dim=1,
                        )
                input0[key] = torch.cat([input0[key], value], dim=0)

        input0["valid_input_len"] = valid_input_len_tensor
    return input0


class SeqPackingTest(unittest.TestCase):
    def run_train_for_sequence_packing(self, fsdp, tp, cp):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 4
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",  # Use -1 to indicate no need for shared memory
            "--shm_size",
            "-1",  # Use -1 to indicate no need for shared memory size
            "--mode",
            "sft_for_sequence_packing",
            "--parallel_config",
            f"fsdp:{fsdp};tp:{tp};cp:{cp}",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        processes = [process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert (
                process.returncode == 0
            ), f"Process failed with code: {process.returncode}"

    def test_train_for_sequence_packing(self):
        self.run_train_for_sequence_packing(4, 1, 1)
        self.run_train_for_sequence_packing(2, 2, 1)
        self.run_train_for_sequence_packing(1, 2, 2)

    def test_hfmodel_sequence_packing(self):
        for model_id in [
            "Qwen/Qwen3-VL-8B-Instruct",
            "microsoft/phi-4",
            # "google/gemma-3-1b-pt", # Need access to it.
        ]:
            if model_id in ["Qwen/Qwen3-VL-8B-Instruct"]:
                from cosmos_rl.policy.model.hf_models.patch import (
                    sequence_packing_forward_qwen3_vl_patch,
                )

                if (
                    Qwen3VLForConditionalGeneration is not None
                    and qwen_vl_process_vision_info is not None
                ):
                    hf_processor = AutoProcessor.from_pretrained(
                        model_id, trust_remote_code=True
                    )
                    model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_id,
                        dtype=torch.bfloat16,
                        trust_remote_code=True,
                        device_map="cuda:0",
                    ).eval()
                    # Patch the model for sequence packing
                    sequence_packing_forward_qwen3_vl_patch(model)

                    conversation1 = [
                        {"content": "Answer the questions.", "role": "system"},
                        {
                            "content": [
                                {
                                    "type": "video",
                                    "video": "tests/data/test_data_packer.mp4",
                                    "max_pixels": 1024 * 128,
                                    "fps": 6,
                                },
                                {
                                    "type": "text",
                                    "text": "What are the objects and actions of the robot arm in the video?",
                                },
                            ],
                            "role": "user",
                        },
                        {
                            "content": "On the left side of the table, there is a black metal dish rack, empty and standing upright. Center-left on the table, there is a silver metallic thermos with a black lid. Left side of the table near the dish rack, there is a yellow and green sponge with a white handle. Left side of the table near the sponge, there is a clear plastic bottle with a white cap, partially filled with liquid. Right side of the table, there is a black wire basket containing bags of chips. Basket contains the red and blue chip bags being arranged. Inside the basket and on the table, there is a red bag labeled 'Backyard Barbecue Potato Chips' with gold text. Inside the basket and on the table, there is a blue bag labeled 'Sea Salt & Vinegar Potato Chips' with yellow text.\n\nReaching into the basket, there is a human hand with a bracelet on the wrist, interacting with the chips bags. Hand places a blue chip bag into the basket. Subsequently, hand adjusts the position of the red chip bag inside the basket.",
                            "role": "assistant",
                        },
                    ]
                    conversation2 = [
                        {
                            "content": "You are a helpful assistant. Answer the questions.",
                            "role": "system",
                        },
                        {
                            "content": [
                                {
                                    "type": "video",
                                    "video": "tests/data/test_data_packer.mp4",
                                    "max_pixels": 1024 * 128,
                                    "fps": 6,
                                },
                                {
                                    "type": "text",
                                    "text": "What are the objects and actions of the robot arm in the video?",
                                },
                            ],
                            "role": "user",
                        },
                        {
                            "content": "On the left side of the table, there is a black metal dish rack, empty and standing upright. Center-left on the table, there is a silver metallic thermos with a black lid. Left side of the table near the dish rack, there is a yellow and green sponge with a white handle. Left side of the table near the sponge, there is a clear plastic bottle with a white cap, partially filled with liquid. Right side of the table, there is a black wire basket containing bags of chips. Basket contains the red and blue chip bags being arranged. Inside the basket and on the table, there is a red bag labeled 'Backyard Barbecue Potato Chips' with gold text. Inside the basket and on the table, there is a blue bag labeled 'Sea Salt & Vinegar Potato Chips' with yellow text.\n\nReaching into the basket, there is a human hand with a bracelet on the wrist, interacting with the chips bags. Hand places a blue chip bag into the basket. Subsequently, hand adjusts the position of the red chip bag inside the basket.",
                            "role": "assistant",
                        },
                    ]
                    packed_inputs = get_inputs(
                        hf_processor,
                        [conversation1, conversation2],
                        pad_token_id=hf_processor.tokenizer.pad_token_id,
                    )
                    single_inputs = get_inputs(
                        hf_processor,
                        [conversation2],
                        pad_token_id=hf_processor.tokenizer.pad_token_id,
                    )
                    packed_valid_input_len = packed_inputs["valid_input_len"]
                    accumulated_valid_input_len = torch.cumsum(
                        packed_valid_input_len, dim=0
                    )
                    with torch.no_grad():
                        packed_output = model(**packed_inputs).logits
                        assert packed_output.shape[0] == 1
                        assert (
                            packed_output.shape[1]
                            == accumulated_valid_input_len[-1].item()
                        )
                        conv2_output = packed_output[
                            :,
                            accumulated_valid_input_len[
                                0
                            ].item() : accumulated_valid_input_len[1].item(),
                            :,
                        ]
                        single_output = model(**single_inputs).logits
                        assert single_output.shape == conv2_output.shape

                        conv2_output_max_index = conv2_output[0, -1, :].argmax(dim=-1)
                        single_output_max_index = single_output[0, -1, :].argmax(dim=-1)
                        conv2_output_max_logit = (
                            conv2_output[0, -1, :].max(dim=-1).values
                        )
                        single_output_max_logit = (
                            single_output[0, -1, :].max(dim=-1).values
                        )
                        print(
                            f"conv2_output_max_index: {conv2_output_max_index} | single_output_max_index: {single_output_max_index}"
                        )
                        print(
                            f"conv2_output_max_logit: {conv2_output_max_logit} | single_output_max_logit: {single_output_max_logit}"
                        )
                        assert conv2_output_max_index == single_output_max_index
                        assert (
                            conv2_output_max_logit - single_output_max_logit
                        ).abs() < 0.5

                    del model
                    del packed_inputs
                    del single_inputs
                    del packed_output
                    del single_output
                    del conv2_output
                    del single_output_max_index
                    del single_output_max_logit
                    torch.cuda.empty_cache()
            elif model_id in ["microsoft/phi-4", "google/gemma-3-1b-pt"]:
                from cosmos_rl.policy.model.hf_models.patch import (
                    sequence_packing_forward_llm_patch,
                )

                hf_processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    dtype=torch.bfloat16,
                    trust_remote_code=True,
                    device_map="cuda:0",
                ).eval()
                # Patch the model for sequence packing forward
                sequence_packing_forward_llm_patch(model)

                conversation1 = [
                    {
                        "role": "system",
                        "content": "You are a pirate chatbot who always responds in pirate speak!",
                    },
                    {"role": "user", "content": "Who are you?"},
                ]

                conversation2 = [
                    {
                        "role": "system",
                        "content": "You are a Chinese-English translator!",
                    },
                    {"role": "user", "content": "你好，我是谁？"},
                ]
                if model_id == "google/gemma-3-1b-pt":
                    if isinstance(conversation1, list):
                        input1_text = "\n".join(
                            [
                                f"{turn['role']}: {turn['content']}"
                                for turn in conversation1
                            ]
                        )
                    if isinstance(conversation2, list):
                        input2_text = "\n".join(
                            [
                                f"{turn['role']}: {turn['content']}"
                                for turn in conversation2
                            ]
                        )
                else:
                    input1_text = hf_processor.apply_chat_template(
                        conversation1, tokenize=False
                    )
                    input2_text = hf_processor.apply_chat_template(
                        conversation2, tokenize=False
                    )
                input1 = hf_processor(
                    text=input1_text,
                    return_tensors="pt",
                ).to("cuda")
                input1.pop("attention_mask")

                input2 = hf_processor(
                    text=input2_text,
                    return_tensors="pt",
                ).to("cuda")
                input2.pop("attention_mask")
                for key, value in input1.items():
                    print(f"input1 {key}: {value.shape}")

                for key, value in input2.items():
                    print(f"input2 {key}: {value.shape}")
                input1_ids = input1["input_ids"]
                input2_ids = input2["input_ids"]
                input1_ids_len = input1_ids.shape[1]
                input2_ids_len = input2_ids.shape[1]
                if input1_ids_len < input2_ids_len:
                    input1_ids = torch.cat(
                        [
                            input1_ids,
                            torch.full(
                                (1, input2_ids_len - input1_ids_len),
                                hf_processor.pad_token_id,
                                device=input1_ids.device,
                            ),
                        ],
                        dim=1,
                    )
                elif input1_ids_len > input2_ids_len:
                    input2_ids = torch.cat(
                        [
                            input2_ids,
                            torch.full(
                                (1, input1_ids_len - input2_ids_len),
                                hf_processor.pad_token_id,
                                device=input2_ids.device,
                            ),
                        ],
                        dim=1,
                    )
                merged_input_ids = torch.cat([input1_ids, input2_ids], dim=0)
                packed_inputs = {
                    "input_ids": merged_input_ids,
                    "valid_input_len": torch.tensor(
                        [input1_ids_len, input2_ids_len],
                        dtype=torch.int32,
                        device=input1_ids.device,
                    ),
                }
                packed_valid_input_len = packed_inputs["valid_input_len"]
                accumulated_valid_input_len = torch.cumsum(
                    packed_valid_input_len, dim=0
                )
                with torch.no_grad():
                    packed_output = model(**packed_inputs).logits
                    assert packed_output.shape[0] == 1
                    assert (
                        packed_output.shape[1] == accumulated_valid_input_len[-1].item()
                    )
                    conv2_output = packed_output[
                        :,
                        accumulated_valid_input_len[
                            0
                        ].item() : accumulated_valid_input_len[1].item(),
                        :,
                    ]
                    single_output = model(**input2).logits
                    assert single_output.shape == conv2_output.shape

                    conv2_output_max_index = conv2_output[0, -1, :].argmax(dim=-1)
                    single_output_max_index = single_output[0, -1, :].argmax(dim=-1)
                    conv2_output_max_logit = conv2_output[0, -1, :].max(dim=-1).values
                    single_output_max_logit = single_output[0, -1, :].max(dim=-1).values

                    print(
                        f"conv2_output_max_index: {conv2_output_max_index} | single_output_max_index: {single_output_max_index}"
                    )
                    print(
                        f"conv2_output_max_logit: {conv2_output_max_logit} | single_output_max_logit: {single_output_max_logit}"
                    )
                    assert conv2_output_max_index == single_output_max_index
                    assert conv2_output_max_logit == single_output_max_logit


if __name__ == "__main__":
    unittest.main()
