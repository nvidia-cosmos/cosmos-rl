# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Minimal PaligemmaTokenizer, same as official_openpi/src/openpi/models/tokenizer.py.
Requires: sentencepiece, requests, filelock (for multi-process safe download).
"""

import logging
import os
import pathlib
import shutil

import filelock
import numpy as np
import requests
import sentencepiece

logger = logging.getLogger(__name__)

_TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
_CACHE_DIR = (
    pathlib.Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi"))
    .expanduser()
    .resolve()
)


def _maybe_download(url: str) -> pathlib.Path:
    """Download file to cache if not exists, return local path. Multi-process safe."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _CACHE_DIR / pathlib.Path(url).name

    if local_path.exists():
        return local_path

    lock_path = local_path.with_suffix(".lock")
    with filelock.FileLock(lock_path):
        # Double-check after acquiring lock
        if local_path.exists():
            return local_path

        logger.info(f"Downloading {url} -> {local_path}")
        scratch_path = local_path.with_suffix(".partial")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        scratch_path.write_bytes(resp.content)
        shutil.move(scratch_path, local_path)

    return local_path


class PaligemmaTokenizer:
    """Exact replica of official_openpi PaligemmaTokenizer."""

    def __init__(self, max_len: int = 48):
        self._max_len = max_len
        path = _maybe_download(_TOKENIZER_URL)
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))

    def tokenize(
        self, prompt: str, state: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            discretized_state = (
                np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
            )
            state_str = " ".join(map(str, discretized_state))
            full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
            tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            tokens = self._tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._tokenizer.encode("\n")

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if tokens_len > self._max_len:
                logging.warning(
                    f"Token length ({tokens_len}) exceeds max ({self._max_len}), truncating."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)
