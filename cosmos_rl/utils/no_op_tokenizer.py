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

"""Tokenizer-shaped object for non-text (e.g. tensor / Gymnasium) RL tasks.

Cosmos-RL's data-packer / controller code paths assume a HuggingFace
tokenizer is available even for tasks where the rollout output is not
text (e.g. trajectory tensors from a ``gymnasium.Env``).  Rather than
forcing every non-LLM caller to monkey-patch ``setup_tokenizer``, this
module ships a minimal duck-typed object that satisfies the few tokenizer
methods cosmos-rl actually calls during non-text RL.

It is designed to be returned from a custom tokenizer loader registered
via :func:`cosmos_rl.utils.util.register_tokenizer_loader`:

.. code-block:: python

    from cosmos_rl.utils.util import register_tokenizer_loader
    from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer

    register_tokenizer_loader(
        predicate=lambda path: path.endswith(".toml"),
        loader=lambda path: NoOpTokenizer(),
    )

The object only implements the methods that are reached on the non-text
code path; calling tokenization-specific methods (e.g. ``batch_decode``)
returns empty/identity results that should never be used by a correctly
configured non-text trainer.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


# Field name used by tensor data packers to record the (already-known)
# trajectory length.  Kept here so the tokenizer can derive a stand-in
# ``input_ids`` length without importing the data-packer module.
_EPISODE_LENGTH_KEY = "episode_length"


def _episode_length(item: Any) -> int:
    """Best-effort extraction of an episode length from a trajectory dict
    or scalar tensor.  Returns ``1`` for unknown / unparseable inputs."""
    if isinstance(item, dict):
        value = item.get(_EPISODE_LENGTH_KEY)
        if value is None:
            return 1
        if hasattr(value, "item"):
            try:
                return max(1, int(value.item()))
            except Exception:
                return 1
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1
    return 1


class _BatchEncoding:
    """A minimal stand-in for :class:`transformers.BatchEncoding`.

    Provides ``input_ids`` and ``attention_mask`` as zero tensors of the
    requested length so downstream code that only inspects shapes (e.g.
    completion-length statistics) operates correctly.
    """

    def __init__(self, length: int):
        self.input_ids = torch.zeros((1, length), dtype=torch.long)
        self.attention_mask = torch.ones((1, length), dtype=torch.long)

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    def keys(self) -> List[str]:
        return ["input_ids", "attention_mask"]


class NoOpTokenizer:
    """A duck-typed tokenizer for non-text RL tasks.

    Satisfies the subset of the HuggingFace tokenizer interface that
    cosmos-rl actually invokes when the model is not text-based:

    * Standard token-id / token attributes (``eos_token_id``,
      ``pad_token_id``, etc.) so config validation does not crash.
    * :meth:`encode` returns a list of zeros sized by the trajectory's
      ``episode_length`` field, allowing completion-length statistics
      to be computed correctly.
    * ``__call__`` returns a :class:`_BatchEncoding`-shaped object.
    * :meth:`decode` / :meth:`batch_decode` return empty strings (no
      real text to render).

    Attributes are kept on the instance (not the class) so multiple
    callers can independently mutate them without surprising each other.
    """

    def __init__(
        self,
        *,
        vocab_size: int = 4,
        model_max_length: int = 512,
    ):
        self.eos_token = "</s>"
        self.eos_token_id = 0
        self.bos_token = "<s>"
        self.bos_token_id = 1
        self.pad_token = "<pad>"
        self.pad_token_id = 2
        self.unk_token = "<unk>"
        self.unk_token_id = 3
        self.vocab_size = vocab_size
        self.model_max_length = model_max_length

    def encode(self, item: Any, **_kwargs) -> List[int]:
        """Return a list of zero token-ids sized by ``episode_length``.

        For trajectory dicts, this lets the controller compute meaningful
        ``completion_length`` statistics without real tokenization.  For
        plain strings (e.g. an empty completion placeholder), returns a
        single-element list.
        """
        if isinstance(item, dict):
            return [0] * _episode_length(item)
        return [0]

    def decode(self, _token_ids, **_kwargs) -> str:
        return ""

    def batch_decode(self, batches, **_kwargs) -> List[str]:
        try:
            return [""] * len(batches)
        except TypeError:
            return [""]

    def __call__(self, item: Any, **_kwargs) -> _BatchEncoding:
        return _BatchEncoding(_episode_length(item))

    def convert_tokens_to_ids(self, _tokens) -> List[int]:
        return []

    def convert_ids_to_tokens(self, _ids) -> List[str]:
        return []

    def __repr__(self) -> str:
        return "NoOpTokenizer(for non-text RL tasks)"


__all__ = ["NoOpTokenizer"]
