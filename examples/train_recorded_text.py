# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Train a tokenized text corpus through the native learner-only SFT path.

cosmos-rl --config configs/execution/train-text-throughput.toml examples/train_recorded_text.py
Records contain token_ids and optional label_ids, inline or in a DataReference.
Explicit labels preserve assistant/prompt masks; absent labels supervise all tokens.
"""

from cosmos_rl.dispatcher.data.packer.decoder_only_llm_data_packer import (
    DecoderOnlyLLMDataPacker,
)
from cosmos_rl.execution.storage import DataReference, FileSystemDataStore
from cosmos_rl.launcher.worker_entry import main


class TokenCorpusPacker(DecoderOnlyLLMDataPacker):
    def prepare_record(self, record, *, context=None):
        row = record.targets
        if isinstance(row, DataReference):
            source = self.config.execution.source
            if not hasattr(self, "reader"):
                self.reader = FileSystemDataStore(source.path)
            row = self.reader.load(row, max_bytes=source.max_prefetch_bytes)
        tokens = row["token_ids"]
        labels = row.get("label_ids", tokens)
        return {
            "token_ids": tokens.tolist() if hasattr(tokens, "tolist") else list(tokens),
            "label_ids": labels.tolist() if hasattr(labels, "tolist") else list(labels),
        }

    def record_preparation_state(self):
        return {"token_corpus_schema": 1}


if __name__ == "__main__":
    main(data_packer=TokenCorpusPacker())
