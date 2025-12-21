import os
import shutil
import logging

import numpy as np
import sentencepiece as sentencepiece
from transformers import PreTrainedTokenizer


class PaligemmaTokenizer(PreTrainedTokenizer):
    """
    A HF-compatible tokenizer that also exposes `tokenize_openpi(prompt, state=None)`
    returning (tokens, mask) exactly like Cosmos-RL local PaligemmaTokenizer.
    """

    vocab_files_names = {"vocab_file": "tokenizer.model"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, vocab_file: str, max_len: int = 48, **kwargs):
        super().__init__(**kwargs)

        self.vocab_file = vocab_file
        self._max_len = int(max_len)
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(vocab_file))
        self.pad_token_id = 0

    # ---- minimal HF plumbing ----
    @property
    def vocab_size(self) -> int:  # type: ignore[override]
        return int(self._tokenizer.get_piece_size())

    def get_vocab(self) -> dict[str, int]:
        return {self._tokenizer.id_to_piece(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> list[str]:
        return list(self._tokenizer.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self._tokenizer.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self._tokenizer.id_to_piece(int(index)))

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return self._tokenizer.decode(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        # OpenPI builds prompts explicitly; don't inject extra special tokens here.
        if token_ids_1 is None:
            return token_ids_0
        return token_ids_0 + token_ids_1

    def save_vocabulary(
        self, save_directory: str, filename_prefix: str | None = None
    ) -> tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        out_name = (filename_prefix + "-" if filename_prefix else "") + "tokenizer.model"
        out_path = os.path.join(save_directory, out_name)
        if os.path.abspath(self.vocab_file) != os.path.abspath(out_path):
            shutil.copyfile(self.vocab_file, out_path)
        return (out_path,)

    def tokenize_openpi(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logging.warning(f"Token length ({tokens_len}) exceeds max ({self._max_len}), truncating.")
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


