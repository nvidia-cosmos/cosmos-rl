from cosmos_rl.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Union
import torch
import copy
from cosmos_rl.utils.logging import logger
from cosmos_rl.dispatcher.data.packer.multi_turn import (
    ConversationType,
    check_chat_template_schema,
    process_conversation_with_chat_template,
)

IGNORE_LABEL_ID = -100


class DecoderOnlyLLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for the decoder only LLM for SFT and RL training.
    """

    class RLPolicyInput:
        input_ids: List[int]
        logprob_masks: List[int]

        def __init__(self, input_ids: List[int], logprob_masks: List[int]):
            self.input_ids = input_ids
            self.logprob_masks = logprob_masks

    def get_rollout_input(self, sample: Union[str, ConversationType]) -> str:
        """
        This is the default implementation for decoder only LLM data packer.
        It assumes that each sample is either a raw text or a conversation format list.
        """
        # 1. if item is a string, then assume it is a raw text
        if isinstance(sample, str):
            return sample

        # 2. if item is a list, check the conversation format
        check_chat_template_schema(sample)
        if not self.config.rollout.multi_turn_config.enable:
            # Apply template to each item
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
            )
            return prompt
        else:
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tools=self.tools,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=self.config.rollout.multi_turn_config.enable_thinking,
            )
            return prompt

    def get_policy_input(
        self,
        sample: Union[str, ConversationType],
        completion: str,
        n_ignore_prefix_tokens: int = 0,
    ) -> RLPolicyInput:
        """
        Default text policy input packer.
        Only support raw text input.
        """
        assert isinstance(completion, str), "Completion should be a string"

        if not self.config.rollout.multi_turn_config.enable:
            # Reuse the same logic as get_rollout_input to get raw text prompts
            prompt = self.get_rollout_input(sample)
            assert isinstance(prompt, str), "Prompt should be a string"

            input_ids = self.tokenizer(
                prompt, add_special_tokens=False
            ).input_ids  # not padded yet

            completion_ids = self.tokenizer(
                completion, add_special_tokens=False
            ).input_ids

            return DecoderOnlyLLMDataPacker.RLPolicyInput(
                input_ids=input_ids + completion_ids,
                logprob_masks=[0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
                + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
                + [0],
            )
        else:
            # TODO(zjx): here we just simple add the completion to the sample
            # later, we need 1. add_function_call_message, 2. add_assistant_message
            sample += [{"role": "assistant", "content": completion}]
            input_ids, loss_mask = process_conversation_with_chat_template(
                self.tokenizer,
                sample,
                enable_thinking=self.config.rollout.multi_turn_config.enable_thinking,
                tools=self.tools,
            )

            full_prompt = self.get_rollout_input(sample)
            full_prompt_ids = self.tokenizer(
                full_prompt, add_special_tokens=False
            ).input_ids

            if len(full_prompt_ids) != len(input_ids) or not all(
                a == b for a, b in zip(full_prompt_ids, input_ids, strict=True)
            ):
                logger.error(
                    "Token mismatch detected! Full tokenization length: {}, Concatenated tokens length: {}. Using concatenated version.".format(
                        len(full_prompt_ids), len(input_ids)
                    )
                )

            return DecoderOnlyLLMDataPacker.RLPolicyInput(
                input_ids=input_ids, logprob_masks=loss_mask
            )

    def policy_compute_max_len(self, processed_samples: List[RLPolicyInput]) -> int:
        return max([len(x.input_ids) for x in processed_samples])

    def policy_collate_fn(
        self, processed_samples: List[RLPolicyInput], computed_max_len: int
    ) -> Dict[str, Any]:
        input_ids = [x.input_ids for x in processed_samples]
        logprob_masks = [x.logprob_masks for x in processed_samples]
        assert len(input_ids) == len(
            logprob_masks
        ), "The length of input_ids, and logprob_masks should be the same"
        device = torch.cuda.current_device()

        collated_dict = {}
        collated_dict["input_ids"] = torch.tensor(
            [
                x[:computed_max_len]
                + [self.tokenizer.pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in input_ids
            ],
            dtype=torch.long,
        ).to(device)
        collated_dict["logprob_masks"] = torch.tensor(
            [
                x[:computed_max_len] + [0] * (max(0, computed_max_len - len(x)))
                for x in logprob_masks
            ],
            dtype=torch.bool,
        ).to(device)

        return collated_dict

    def _replace_assistant_content(
        self,
        token_ids: List[int],
        label_ids: List[int],
        pad_token_id: int,
        eos_token_id: int,
        replacement_ids: List[int],
        pad_run_length: int = 10,
    ) -> List[int]:
        """
        Find the first run of exactly `pad_run_length` pad_token_id's in token_ids,
        replace that run with replacement_ids, and return the new list.
        If no such run is found, returns the original list unchanged.
        """
        n = len(token_ids)
        target_run = [pad_token_id] * pad_run_length

        # find the start index of the first matching run
        for i in range(n - pad_run_length + 1):
            if token_ids[i : i + pad_run_length] == target_run:
                # splice in the replacement
                if (
                    len(token_ids) > i + pad_run_length
                    and token_ids[i + pad_run_length] == eos_token_id
                ):
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + [eos_token_id]
                        + label_ids[i + pad_run_length + 1 :]
                    )
                else:
                    label_ids = (
                        label_ids[:i]
                        + replacement_ids
                        + label_ids[i + pad_run_length :]
                    )
                return (
                    True,
                    token_ids[:i] + replacement_ids + token_ids[i + pad_run_length :],
                    label_ids,
                )
        # no match found
        return False, token_ids, label_ids

    def sft_process_sample(self, sample: Union[str, List[Dict[str, str]]]) -> List[int]:
        """
        Process the sample into the format required by the SFT model.
        Accepts either raw text or conversation format.
        """
        # 1. if item is a string, then assume it is a raw text
        if isinstance(sample, str):
            token_ids = self.tokenizer(sample, add_special_tokens=False).input_ids
            label_ids = token_ids.copy()
        # 2. if item is a list, then assume it is in conversation format:
        else:
            check_chat_template_schema(sample)

            original_sample = copy.deepcopy(sample)

            try:
                if (
                    self.tokenizer.pad_token is None
                    or self.tokenizer.pad_token_id is None
                ):
                    raise ValueError("pad_token and pad_token_id should be set")

                assistant_contents = []
                pad_token = self.tokenizer.pad_token
                pad_token_id = self.tokenizer.pad_token_id
                eos_token_id = self.tokenizer.eos_token_id
                pad_run_length = 10

                for x in sample:
                    if x["role"] == "assistant":
                        assistant_contents.append(x["content"])
                        x["content"] = pad_token * pad_run_length

                token_ids = self.tokenizer.apply_chat_template(
                    sample,
                    return_dict=True,
                    add_generation_prompt=False,
                )["input_ids"]
                label_ids = [IGNORE_LABEL_ID] * len(token_ids)

                for assistant_content in assistant_contents:
                    replaced, token_ids, label_ids = self._replace_assistant_content(
                        token_ids,
                        label_ids,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        replacement_ids=self.tokenizer.encode(
                            assistant_content, add_special_tokens=False
                        ),
                        pad_run_length=pad_run_length,
                    )
                    if not replaced:
                        raise ValueError("No assistant content to replace")
                    if len(token_ids) != len(label_ids):
                        raise ValueError(
                            f"token_ids and label_ids should have the same length, but got {len(token_ids)} and {len(label_ids)}"
                        )
            except Exception:
                # Fall back to the non-assistant-masking
                token_ids = self.tokenizer.apply_chat_template(
                    original_sample,
                    return_assistant_tokens_mask=False,
                    return_dict=True,
                    add_generation_prompt=False,
                )["input_ids"]
                label_ids = token_ids.copy()
        assert isinstance(token_ids, list), "token_ids should be a list"
        assert isinstance(token_ids[0], int), "Each item in token_ids should be an int"
        return {
            "token_ids": token_ids,
            "label_ids": label_ids,
        }

    def sft_compute_max_len(self, processed_samples: List[List[int]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        max_len = max([len(x["token_ids"]) for x in processed_samples])
        return max_len

    def sft_collate_fn(
        self,
        processed_samples: List[Dict[str, Any]],
        computed_max_len: int,
        pad_token_id: int,
        ignore_label_id: int,
    ) -> Dict[str, Any]:
        """
        Collate the processed samples into a minibatch dictionary passed to the SFT model.
        """
        # First truncate the samples to the computed_max_len
        list_of_input_ids = [
            x["token_ids"][:computed_max_len] for x in processed_samples
        ]
        list_of_label_ids = [
            x["label_ids"][:computed_max_len] for x in processed_samples
        ]

        # Then pad the samples to the computed_max_len
        input_ids = torch.tensor(
            [
                x[:computed_max_len]
                + [pad_token_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_input_ids
            ],
            dtype=torch.long,
        )
        # Model accept unshifted label_ids for loss computation
        label_ids = torch.tensor(
            [
                x[:computed_max_len]
                + [ignore_label_id] * (max(0, computed_max_len - len(x)))
                for x in list_of_label_ids
            ],
            dtype=torch.long,
        )

        # valid_label_mask = label_ids != ignore_label_id
        # assert torch.all(input_ids[valid_label_mask] == label_ids[valid_label_mask]), "input_ids and label_ids should have the same value"
        # print(f"input_ids: {self.tokenizer.decode(input_ids[0])}")
        # print(f"label_ids: {self.tokenizer.decode(label_ids[0][label_ids[0] != ignore_label_id])}")

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
        }
