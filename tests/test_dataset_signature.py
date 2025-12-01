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
from unittest.mock import patch, MagicMock
from torch.utils.data import Dataset

from cosmos_rl.utils.util import call_setup


class _DatasetWithoutSetup(Dataset):
    pass


class _NewStyleDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.seen_config = None

    def setup(self, config):
        self.seen_config = config


class _OldStyleDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.calls = []

    def setup(self, config, tokenizer):
        self.calls.append((config, tokenizer))


class _InvalidSignatureDataset(Dataset):
    def setup(self, config, tokenizer, extra):
        self.config = config
        self.tokenizer = tokenizer
        self.extra = extra


class _StubPolicy:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path


class _StubConfig:
    def __init__(self, model_name_or_path: str = "test/model"):
        self.policy = _StubPolicy(model_name_or_path)


class TestallDatasetSetup(unittest.TestCase):
    def setUp(self):
        self.config = _StubConfig()
        patcher = patch("transformers.AutoTokenizer.from_pretrained")
        self.mock_from_pretrained = patcher.start()
        self.addCleanup(patcher.stop)
        self.fake_tokenizer = MagicMock()
        self.mock_from_pretrained.return_value = self.fake_tokenizer

    def test_no_setup_attribute_short_circuits(self):
        dataset = _DatasetWithoutSetup()
        call_setup(dataset, self.config)
        self.mock_from_pretrained.assert_not_called()

    def test_new_signature_invokes_setup_with_config_only(self):
        dataset = _NewStyleDataset()
        call_setup(dataset, self.config)
        self.assertIs(dataset.seen_config, self.config)
        self.mock_from_pretrained.assert_not_called()

    def test_old_signature_warns_and_injects_tokenizer(self):
        dataset = _OldStyleDataset()
        with self.assertWarns(DeprecationWarning):
            call_setup(dataset, self.config)
        self.assertEqual(dataset.calls, [(self.config, self.fake_tokenizer)])

    def test_invalid_signature_raises_type_error(self):
        dataset = _InvalidSignatureDataset()
        with self.assertRaises(TypeError):
            call_setup(dataset, self.config)


if __name__ == "__main__":
    unittest.main()
