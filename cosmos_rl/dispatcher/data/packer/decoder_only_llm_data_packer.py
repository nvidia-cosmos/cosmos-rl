from cosmos_rl.dispatcher.data.packer.base import DataPacker
from typing import List, Any, Dict, Union
import torch


IGNORE_LABEL_ID = -100


class DecoderOnlyLLMDataPacker(DataPacker):
    """
    Data protocol & processing logic for the decoder only LLM for SFT and RL training.
    """

    ConversationType = List[Dict[str, str]]

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
        # 2. if item is a list, then assume it is in conversation format of:
        # [
        #     {
        #         "role": "user",
        #         "content": "..."
        #     },
        #     {
        #         "role": "assistant",
        #         "content": "..."
        #     }
        # ]
        else:
            assert isinstance(sample, list), "All items should be list"
            # Check `role` and `content` in each item
            for x in sample:
                assert isinstance(x, dict), "Each item should be a dict"
                assert "role" in x, "Each item should have 'role'"
                assert "content" in x, "Each item should have 'content'"

            # Apply template to each item
            prompt = self.tokenizer.apply_chat_template(
                sample,
                tokenize=False,
                add_generation_prompt=True,
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

        # Reuse the same logic as get_rollout_input to get raw text prompts
        prompt = self.get_rollout_input(sample)
        assert isinstance(prompt, str), "Prompt should be a string"

        input_ids = self.tokenizer(
            prompt, add_special_tokens=False
        ).input_ids  # not padded yet

        completion_ids = self.tokenizer(completion, add_special_tokens=False).input_ids

        return DecoderOnlyLLMDataPacker.RLPolicyInput(
            input_ids=input_ids + completion_ids,
            logprob_masks=[0] * (len(input_ids) - 1 + n_ignore_prefix_tokens)
            + [1] * (len(completion_ids) - n_ignore_prefix_tokens)
            + [0],
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
        assert isinstance(sample, dict), "Sample should be a dict"
        jpg = sample["jpg"]
        caption = sample["json"]["prompt"]

        chat = f"""<|im_start|>system
You are a helpful assistant to generate image given the following prompt: {caption}.<|im_end|>
<|im_start|>assistant
This is the generated image: <|vision_start|>{"<|image_pad|>" * 2048}<|vision_end|><|im_end|>
"""

        token_ids = self.tokenizer(chat, add_special_tokens=False).input_ids
        label_ids = token_ids.copy()
        return {
            "token_ids": token_ids,
            "label_ids": label_ids,
            "jpg": jpg,
        }

    def sft_compute_max_len(self, processed_samples: List[List[int]]) -> int:
        """
        Compute the maximum sequence length of the processed samples
        """
        return max([len(x["token_ids"]) for x in processed_samples])

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
        list_of_jpg = [x["jpg"] for x in processed_samples]

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

        images = torch.cat(list_of_jpg, dim=0)
        assert (
            images.shape[0] == input_ids.shape[0]
        ), "The number of images should be the same as the number of input_ids"
        assert images.shape[1] == 3, "The number of channels should be 3"
        assert images.shape[2] == 384, "The height should be 384"
        assert images.shape[3] == 384, "The width should be 384"
        assert images.ndim == 4, "The number of dimensions should be 4"

        return {
            "input_ids": input_ids,
            "label_ids": label_ids,
            "imgs": images,
        }
