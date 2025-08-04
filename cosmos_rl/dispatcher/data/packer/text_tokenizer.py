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

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer

from cosmos_rl.utils.logging import logger as log


class TextTokenizer:
    """
    Text tokenizer class built on HuggingFace's Fast Tokenizer (Rust based).
    """

    def __init__(
        self,
        tokenizer_type: str,
        is_instruct_model: bool,
        cache_dir: str = os.getenv("IMAGINAIRE_CACHE_DIR", "~/.cache/imaginaire"),
        tokenizer_local_path: Optional[str] = None,
    ):
        """
        Initialize the TextTokenizer.
        Args:
            tokenizer_type (str): The tokenizer type.
            is_instruct_model (bool): Whether the model is an instruct model.
            cache_dir (str): The cache directory. Defaults to "~/.cache/imaginaire".
            tokenizer_local_path (str): The local path to the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_local_path, use_fast=True
        )
        self.stop_tokens = {
            self.tokenizer.eos_token_id,
        }
        self.tokenizer_type = tokenizer_type
        self.is_instruct_model = is_instruct_model
        self.eos_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            if tokenizer_type.startswith("llama"):
                self.pad_id = 128004  # "<|finetune_right_pad_id|>"
            elif tokenizer_type.startswith("mistral"):
                self.pad_id = 10  # "<pad>"
            elif tokenizer_type == "pixtral":
                self.pad_id = 11  # "<pad>"
            elif tokenizer_type == "nemotron5_8b":
                self.pad_id = 10  # "<pad>", read from `s3://checkpoints/edify_tokenizer/text_tokenizers/nemotron5_8b-instruct/tokenizer.json` in pbss_dir
            elif tokenizer_type == "nemotron5_56b":
                self.pad_id = 10
            else:
                raise ValueError(
                    f"pad_id not defined for tokenizer_type {tokenizer_type}"
                )
        else:
            self.pad_id = self.tokenizer.pad_token_id
        log.info(f"pad_id = {self.pad_id}, tokenizer_type = {tokenizer_type}")

    def tokenize(
        self, text: str, *, add_special_tokens: bool = False, **kwargs
    ) -> List[str]:
        """
        Converts a string into a sequence of tokens, replacing unknown tokens with the `unk_token`.

        Args:
            text (`str`):
                The sequence to be encoded.
            add_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to add the special tokens associated with the corresponding model.
        Returns:
            `List[str]`: The list of tokens.
        """
        return self.tokenizer.tokenize(
            text, add_special_tokens=add_special_tokens, **kwargs
        )

    def encode(
        self,
        text: Union[str, List[str], List[int]],
        *,  # Enforce keyword-only arguments
        add_special_tokens: bool = True,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        return_tensors: Optional[str] = None,
        **kwargs,
    ) -> List[int]:
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers (tokenized string ids using the `convert_tokens_to_ids`
                method).
            add_special_tokens (`bool`, *optional*, defaults to `True`):
                Whether or not to add special tokens when encoding the sequences. This will use the underlying
                `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are
                automatically added to the input ids. This is usefull if you want to add `bos` or `eos` tokens
                automatically.
            padding (`bool`, `str`, *optional*, defaults to `False`):
                Activates and controls padding. Accepts the following values:

                - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
                  sequence if provided).
                - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
                  acceptable input length for the model if that argument is not provided.
                - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
                  lengths).
            truncation (`bool`, `str`, *optional*, defaults to `False`):
                Activates and controls truncation. Accepts the following values:

                - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or
                  to the maximum acceptable input length for the model if that argument is not provided. This will
                  truncate token by token, removing a token from the longest sequence in the pair if a pair of
                  sequences (or a batch of pairs) is provided.
                - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the
                  maximum acceptable input length for the model if that argument is not provided. This will only
                  truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
                - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths
                  greater than the model maximum admissible input size).
            max_length (`int`, *optional*):
                Controls the maximum length to use by one of the truncation/padding parameters.

                If left unset or set to `None`, this will use the predefined model maximum length if a maximum length
                is required by one of the truncation/padding parameters. If the model has no specific maximum input
                length (like XLNet) truncation/padding to a maximum length will be deactivated.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing tokens returned when
                `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping tokens.
            is_split_into_words (`bool`, *optional*, defaults to `False`):
                Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the
                tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace)
                which it will tokenize. This is useful for NER or token classification.
            pad_to_multiple_of (`int`, *optional*):
                If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated.
                This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability
                `>= 7.5` (Volta).
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return Numpy `np.ndarray` objects.
        """
        return self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            return_tensors=return_tensors,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], "np.ndarray", "torch.Tensor"],
        *,  # Enforce keyword-only arguments
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor, tf.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def apply_chat_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        padding: bool = False,
        truncation: bool = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_dict: bool = False,
        return_assistant_tokens_mask: bool = False,
        generation_prefix: str = "",
        tokenizer_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Converts a list of dictionaries with `"role"` and `"content"` keys to a list of token
        ids. This method is intended for use with chat models, and will read the tokenizer's chat_template attribute to determine the format and control tokens to use when converting.

        More details can be found at https://huggingface.co/docs/transformers/main/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template

        Args:
            conversation (Union[List[Dict[str, str]], List[List[Dict[str, str]]]]): A list of dicts
                with "role" and "content" keys, representing the chat history so far.
            add_generation_prompt (bool, *optional*):
                If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output. This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
            continue_final_message (bool, *optional*):
                If this is set, the chat will be formatted so that the final
                message in the chat is open-ended, without any EOS tokens. The model will continue this message
                rather than starting a new one. This allows you to "prefill" part of
                the model's response for it. Cannot be used at the same time as `add_generation_prompt`.
            tokenize (`bool`, defaults to `True`):
                Whether to tokenize the output. If `False`, the output will be a string.
            padding (`bool`, defaults to `False`):
                Whether to pad sequences to the maximum length. Has no effect if tokenize is `False`.
            truncation (`bool`, defaults to `False`):
                Whether to truncate sequences at the maximum length. Has no effect if tokenize is `False`.
            max_length (`int`, *optional*):
                Maximum length (in tokens) to use for padding or truncation. Has no effect if tokenize is `False`. If
                not specified, the tokenizer's `max_length` attribute will be used as a default.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Has no effect if tokenize is `False`. Acceptable
                values are:
                - `'tf'`: Return TensorFlow `tf.Tensor` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.
            return_dict (`bool`, defaults to `False`):
                Whether to return a dictionary with named outputs. Has no effect if tokenize is `False`.
            generation_prefix (str): Prefix to add before asking model to generate. Helpful to guide the generation. Defaults to "".
            tokenizer_kwargs (`Dict[str: Any]`, *optional*): Additional kwargs to pass to the tokenizer.
            return_assistant_tokens_mask (`bool`, defaults to `False`):
                Whether to return a mask of the assistant generated tokens. For tokens generated by the assistant,
                the mask will contain 1. For user and system tokens, the mask will contain 0.
                This functionality is only available for chat templates that support it via the `{% generation %}` keyword.
            **kwargs: Additional kwargs to pass to the template renderer. Will be accessible by the chat template.

        Returns:
            `Union[List[int], Dict]`: A list of token ids representing the tokenized chat so far, including control tokens. This
            output is ready to pass to the model, either directly or via methods like `generate()`. If `return_dict` is
            set, will return a dict of tokenizer outputs instead.
        """
        if not self.is_instruct_model:
            raise ValueError(
                "apply_chat_template is only supported for instruct models. You should pass argument is_instruct_model=True to the TextTokenizer constructor."
            )
        # Since generation_prefix is added to the text in the end, ensure that the setting is correct
        if generation_prefix:
            assert (
                not tokenize
            ), "tokenize must be False when generation_prefix is provided."
            assert (
                add_generation_prompt
            ), "add_generation_prompt must be set when generation_prefix is provided."
        formatted_text: Union[str, List[int]] = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            return_dict=return_dict,
            return_assistant_tokens_mask=return_assistant_tokens_mask,
            tokenizer_kwargs=tokenizer_kwargs,
            **kwargs,
        )
        if generation_prefix:
            formatted_text: str = formatted_text + generation_prefix
            log.info(
                f"Adding generation prefix: {generation_prefix} to the formatted text\n"
                f"Formatted text: {formatted_text}"
            )
        return formatted_text

    def add_assistant_tokens_mask(self, tokens):
        """
        Add a mask to the assistant tokens.
        This is used to mask out tokens that are not generated by the assistant (e.g.,  system prompts, user prompts, chat templates), such that in the loss computation, only the tokens generated by the assistant are used.
        If there are multiple turns in the conversation, the mask will mask all the assistant tokens in each turn.

        Args:
            tokens (Union[List[int], torch.Tensor]): The tokens to add the mask to.
        Returns:
            Union[List[bool], torch.Tensor]: The mask. True for tokens generated by the assistant (i.e. should apply loss on), False for tokens not generated by the assistant.
        """
        assert (
            self.is_instruct_model
        ), "`return_assistant_tokens_mask=True` is only supported for instruct models."
        if isinstance(tokens, torch.Tensor) and tokens.ndim == 2:
            mask = torch.stack(
                [
                    self.add_assistant_tokens_mask(tokens[i])
                    for i in range(tokens.shape[0])
                ]
            )
            assert mask.shape == tokens.shape
            return mask
        np_tokens = (
            tokens.cpu().numpy()
            if isinstance(tokens, torch.Tensor)
            else np.array(tokens)
        )
        assert np_tokens.ndim == 1
        if self.tokenizer_type.startswith("mistral") or self.tokenizer_type.startswith(
            "pixtral"
        ):
            """
            mistral will give [/INST] to each user message
            whatever is after [/INST] and before </s> is the assistant message
            """

            boundary_token = 4  # "[/INST]", the last token of each user message
            eos_token = self.tokenizer.eos_token_id  # </s>

            # Find the first position of the boundary token from right to left
            eot_appearances = np.argwhere(np_tokens == boundary_token).flatten()
            assert (
                len(eot_appearances) > 0
            ), f"Boundary token {boundary_token} not found in the tokens"
            eos_appearances = np.argwhere(np_tokens == eos_token).flatten()
            masks = np.zeros_like(np_tokens, dtype=bool)

            for i in range(len(eot_appearances)):
                start_pos = (
                    eot_appearances[i] + 1
                )  # +1 beacause we want to train on the first token after the boundary token
                # find the next eos position after this boundary token
                end_pos = eos_appearances[eos_appearances > start_pos]
                if len(end_pos) == 0:
                    # log.warning(
                    #     f"No end position found for assistant token at position {start_pos}, text: {self.tokenizer.decode(np_tokens)}, eos_appearances={eos_appearances}, start_pos={start_pos}"
                    # )
                    return None
                end_pos = end_pos[0]
                masks[start_pos : end_pos + 1] = (
                    True  # +1 beacause we also train on eos_token
                )

        elif self.tokenizer_type == "llama3.1":
            """
            llama will give <|start_header_id|>role to start a turn
            and the turn will always end with <|eot_id|>, regardless of the role
            """

            boundary_token = (
                self.tokenizer.eos_token_id
            )  # 128009 for '<|eot_id|>', the last token of each message
            header_start_token = self.tokenizer.added_tokens_encoder[
                "<|start_header_id|>"
            ]
            assistant_token = 78191  # "assistant" role

            eot_appearances = np.argwhere(np_tokens == boundary_token).flatten()
            assert (
                len(eot_appearances) > 0
            ), f"Boundary token {boundary_token} not found in the tokens"
            header_start_appearances = np.argwhere(
                np_tokens == header_start_token
            ).flatten()
            roles_tokens = np.array(
                [np_tokens[pos + 1] for pos in header_start_appearances]
            )
            masks = np.zeros_like(np_tokens, dtype=bool)

            # iteratively find the assistant tokens to set True
            for i in range(len(roles_tokens)):
                if roles_tokens[i] == assistant_token:
                    start_pos = header_start_appearances[i]
                    # skip the next four tokens:
                    # <|start_header_id|>, role, <|end_header_id|>, \n\n
                    start_pos += 4
                    # find the next boundary position after this header start token
                    end_pos = eot_appearances[eot_appearances > start_pos]
                    if len(end_pos) == 0:
                        log.warning(
                            f"No end position found for assistant token at position {start_pos}, text: {self.tokenizer.decode(np_tokens)}"
                        )
                        return None
                    end_pos = end_pos[0]
                    masks[start_pos : end_pos + 1] = (
                        True  # +1 beacause we also train on boundary_token
                    )
        elif self.tokenizer_type.startswith("deepseek_r1"):
            # This is to create the loss mask, we only want to apply loss for the assistant reply, and dont compute loss on user prompt or special tokens
            assistant_token = 128804  # "<｜Assistant｜>" token
            eos_token = 1  # "<｜end▁of▁sentence｜>" token
            masks = np.zeros_like(np_tokens, dtype=bool)
            eos_appearances = np.argwhere(np_tokens == eos_token).flatten()
            if len(eos_appearances) == 0:
                # log.warning(
                #     f"No eos token found in the tokens, set the last position as eos, text: {self.tokenizer.decode(np_tokens)}"
                # )
                eos_appearances = np.array(
                    [len(np_tokens) - 1]
                )  # set the last position as eos
            assistant_appearances = np.argwhere(np_tokens == assistant_token).flatten()
            # assert assistant token exists
            if len(assistant_appearances) == 0:
                log.warning(
                    f"No assistant token found in the tokens, set the first position as assistant, text: {self.tokenizer.decode(np_tokens)}"
                )
                assistant_appearances = np.array([0])
            assert (
                len(assistant_appearances) == len(eos_appearances)
                or len(assistant_appearances) == len(eos_appearances) + 1
            ), f"Assistant token should appear before eos token, the number of assistant tokens should be greater than or equal to the number of eos tokens. get assistant_appearances={assistant_appearances}, eos_appearances={eos_appearances}, np_tokens={np_tokens}, text={self.tokenizer.decode(np_tokens)}"
            if len(assistant_appearances) == len(eos_appearances) + 1:
                eos_appearances = np.concatenate(
                    [eos_appearances, [len(np_tokens) - 1]]
                )
            for i in range(len(assistant_appearances)):
                assert (
                    assistant_appearances[i] < eos_appearances[i]
                ), f"Assistant token should appear before eos token, get assistant_appearances[{i}]={assistant_appearances[i]}, eos_appearances[{i}]={eos_appearances[i]}, np_tokens={np_tokens}, text={self.tokenizer.decode(np_tokens)}"
                masks[assistant_appearances[i] + 1 : eos_appearances[i] + 1] = True

        elif self.tokenizer_type.startswith("nemotron5"):
            if self.tokenizer_type == "nemotron5_8b":
                # Encode special tokens and patterns
                start_sequence = "[PREFIX]Assistant\n"
                start_tokens = self.tokenizer.encode(
                    start_sequence, add_special_tokens=False
                )
                end_token = self.tokenizer.encode("[PREFIX]", add_special_tokens=False)
                assert len(end_token) == 1, "Expected [PREFIX] to be a single token"
            elif self.tokenizer_type == "nemotron5_56b":
                # Encode special tokens and patterns
                start_sequence = "<SPECIAL_14>assistant\n"
                start_tokens = self.tokenizer.encode(
                    start_sequence, add_special_tokens=False
                )
                end_token = self.tokenizer.encode(
                    "<SPECIAL_15>", add_special_tokens=False
                )
                assert len(end_token) == 1, "Expected <SPECIAL_15> to be a single token"
            else:
                raise ValueError(
                    f"add_assistant_tokens_mask is not implemented for tokenizer_type {self.tokenizer_type}"
                )
            end_token = end_token[0]
            newline_token = self.tokenizer.encode("\n", add_special_tokens=False)
            assert len(newline_token) == 1, "Expected newline to be a single token"
            newline_token = newline_token[0]

            masks = np.zeros_like(np_tokens, dtype=bool)

            # Find all occurrences of the assistant start pattern
            start_indices = []
            for i in range(len(np_tokens) - len(start_tokens) + 1):
                if np.array_equal(np_tokens[i : i + len(start_tokens)], start_tokens):
                    start_indices.append(i)

            # Process each assistant section
            for start_idx in start_indices:
                content_start = start_idx + len(start_tokens)
                # Find next [PREFIX] token after assistant content
                post_content = np_tokens[content_start:]
                end_positions = np.where(post_content == end_token)[0]
                if not len(end_positions):
                    continue
                end_pos = content_start + end_positions[0]

                # Set mask for assistant-generated tokens
                if end_pos > content_start:
                    end_pos += 1  # Include the [PREFIX] token as the end-of-turn
                    masks[content_start:end_pos] = True
        else:
            raise ValueError(
                f"add_assistant_tokens_mask is not implemented for tokenizer_type {self.tokenizer_type}"
            )

        assert masks.shape == np_tokens.shape
        if isinstance(tokens, torch.Tensor):
            return torch.from_numpy(masks)
        else:
            return masks.tolist()
