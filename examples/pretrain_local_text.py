# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Minimal learner-only continued pretraining through the normal launcher.

cosmos-rl --config configs/execution/pretrain.toml examples/pretrain_local_text.py
Replace the illustrative paragraphs with your own dataset. Plain strings cause
the normal decoder packer to supervise all tokens rather than assistant turns.
"""

from torch.utils.data import Dataset

from cosmos_rl.launcher.worker_entry import main


class TextDataset(Dataset):
    texts = (
        "A training dataset can feed the learner without starting a generator.",
        "Checkpointed optimizer state and source progress describe one training boundary.",
        "Generation and supervised training can exchange an immutable recorded corpus.",
        "A separate teacher can provide targets without receiving student weights.",
    )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        return self.texts[index]


if __name__ == "__main__":
    main(dataset=lambda config: TextDataset())
