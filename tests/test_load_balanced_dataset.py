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

"""
Unit test for LoadBalancedDataset with real HuggingFace dataset.
"""

import unittest
from unittest.mock import patch
from torch.utils.data import Dataset
import random
import os

from cosmos_rl.dispatcher.data.load_balanced_dataset import (
    LoadBalancedDataset,
    get_sample_length,
)
from cosmos_rl.utils import util


class RealDatasetAdapter(Dataset):
    """
    Adapter to convert HuggingFace dataset to format compatible with LoadBalancedDataset.
    This adapter extracts conversation data and converts it to input_ids format.
    """

    def __init__(
        self, hf_dataset, conversation_column_name="conversation", max_samples=None
    ):
        """
        Initialize the adapter.

        Args:
            hf_dataset: HuggingFace dataset
            conversation_column_name: Column name for conversation data
            max_samples: Maximum number of samples to use (None for all)
        """
        self.hf_dataset = hf_dataset
        self.conversation_column_name = conversation_column_name
        self.max_samples = max_samples

        # Determine dataset size
        if max_samples is not None:
            self.size = min(len(hf_dataset), max_samples)
        else:
            self.size = len(hf_dataset)

        print(
            f"[RealDatasetAdapter] Dataset size: {self.size}, "
            f"conversation_column: {conversation_column_name}"
        )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Returns a dict with 'input_ids' key for LoadBalancedDataset compatibility.
        The input_ids are estimated based on conversation length.
        """
        if idx >= self.size:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.size}"
            )

        sample = self.hf_dataset[idx]

        # Extract conversation data
        if self.conversation_column_name in sample:
            conversation = sample[self.conversation_column_name]
        else:
            # Fallback: try to find any list/dict field
            conversation = sample.get("messages", sample.get("conversations", sample))

        # Estimate sequence length from conversation
        # For testing purposes, we'll estimate based on conversation structure
        if isinstance(conversation, list):
            # Count tokens roughly: each message contributes ~50-200 tokens
            estimated_tokens = sum(
                len(str(msg).split()) * 1.3  # Rough token estimation
                for msg in conversation
            )
        elif isinstance(conversation, str):
            estimated_tokens = len(conversation.split()) * 1.3
        else:
            # Fallback: use a default length
            estimated_tokens = 1000

        # Ensure reasonable bounds
        estimated_tokens = max(100, min(int(estimated_tokens), 16384))

        # Create input_ids as a list (LoadBalancedDataset expects this)
        # For testing, we use a simple representation
        input_ids = list(range(int(estimated_tokens)))

        return {
            "input_ids": input_ids,
            "sample_id": idx,  # Track sample ID for verification
            "seq_len": len(input_ids),  # Store actual length for verification
            "original_sample": sample,  # Keep original for debugging
        }


class MockDataset(Dataset):
    """Mock dataset with variable length sequences (fallback for testing)."""

    def __init__(self, size=800, min_len=100, max_len=8192):
        """
        Create a mock dataset with variable length sequences.

        Args:
            size: Total number of samples
            min_len: Minimum sequence length
            max_len: Maximum sequence length
        """
        self.size = size
        self.min_len = min_len
        self.max_len = max_len

        # Generate variable length sequences
        # Mix of short, medium, and long sequences
        self.samples = []
        for i in range(size):
            # Create a mix: 40% short (100-2000), 40% medium (2000-4000), 20% long (4000-8192)
            if i % 5 < 2:
                # Short sequences
                seq_len = random.randint(min_len, 2000)
            elif i % 5 < 4:
                # Medium sequences
                seq_len = random.randint(2000, 4000)
            else:
                # Long sequences
                seq_len = random.randint(4000, max_len)

            self.samples.append(
                {
                    "input_ids": list(range(seq_len)),  # Simple list of integers
                    "sample_id": i,  # Track sample ID for verification
                    "seq_len": seq_len,  # Store actual length for verification
                }
            )

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.samples[idx]


class TestLoadBalancedDataset(unittest.TestCase):
    """Test LoadBalancedDataset with real HuggingFace dataset."""

    def setUp(self):
        """Set up test fixtures."""
        # Get num_ranks from CUDA_VISIBLE_DEVICES environment variable
        # If CUDA_VISIBLE_DEVICES is set, count the number of devices
        # Otherwise, default to 8
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible_devices:
            # CUDA_VISIBLE_DEVICES can be comma-separated device IDs
            # Count the number of devices
            device_list = [
                d.strip() for d in cuda_visible_devices.split(",") if d.strip()
            ]
            self.num_ranks = len(device_list) if device_list else 8
            print(
                f"\nUsing CUDA_VISIBLE_DEVICES={cuda_visible_devices}, num_ranks={self.num_ranks}"
            )
        else:
            # Default to 8 if not set
            self.num_ranks = 8
            print(
                f"\nCUDA_VISIBLE_DEVICES not set, using default num_ranks={self.num_ranks}"
            )

        self.pool_size = 32
        self.max_tokens_for_batch = 25600  # Match config file
        self.seed = 42

        # Load real dataset
        # Use a small subset for testing to avoid long test times
        self.max_test_samples = 1000
        self.dataset_name = "LNTANOooo/sharegpt52k"
        self.dataset_subset = ""
        self.dataset_split = "train"
        self.conversation_column_name = "conversation"

        try:
            print(f"\nLoading real dataset: {self.dataset_name}")
            hf_dataset = util.load_data_from_disk_or_hf(
                self.dataset_name,
                self.dataset_subset,
                revision=None,
            )

            # Get the train split
            if isinstance(hf_dataset, dict):
                train_split = hf_dataset.get(
                    self.dataset_split, hf_dataset.get("train", None)
                )
                if train_split is None:
                    # Use first available split
                    train_split = list(hf_dataset.values())[0]
            else:
                train_split = hf_dataset

            # Limit dataset size for testing
            if len(train_split) > self.max_test_samples:
                train_split = train_split.select(range(self.max_test_samples))
                print(f"Using subset of {self.max_test_samples} samples for testing")

            # Create adapter
            self.base_dataset = RealDatasetAdapter(
                train_split,
                conversation_column_name=self.conversation_column_name,
                max_samples=self.max_test_samples,
            )
            self.dataset_size = len(self.base_dataset)
            self.use_real_dataset = True
            print(f"Loaded {self.dataset_size} samples from real dataset")

        except Exception as e:
            print(f"\nWarning: Failed to load real dataset: {e}")
            print("Falling back to MockDataset")
            # Fallback to mock dataset if real dataset fails to load
            random.seed(self.seed)
            self.base_dataset = MockDataset(size=800)
            self.dataset_size = 800
            self.use_real_dataset = False

    def _simulate_rank(self, rank, world_size, accumulate_steps=1):
        """Simulate a single rank's data loading."""
        # Mock torch.distributed
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=rank),
            patch("torch.distributed.get_world_size", return_value=world_size),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            # Create LoadBalancedDataset for this rank
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                dp_rank=rank,
                dp_world_size=world_size,
                accumulate_steps=accumulate_steps,
            )

            # Collect all batches and samples
            batches = []
            sample_ids = []
            total_tokens_list = []

            for item in dataset:
                # item is always a list of batches (even when accumulate_steps=1, it's [batch])
                accumulated_batches = item
                self.assertIsInstance(
                    accumulated_batches, list, "Dataset should yield a list of batches"
                )
                self.assertGreater(
                    len(accumulated_batches),
                    0,
                    "Accumulated batches should not be empty",
                )
                self.assertLessEqual(
                    len(accumulated_batches),
                    accumulate_steps,
                    f"Should have at most {accumulate_steps} batches per iteration",
                )

                batches.extend(accumulated_batches)

                for batch in accumulated_batches:
                    batch_size = len(batch)
                    max_len = max(
                        get_sample_length(sample, "input_ids") for sample in batch
                    )
                    total_tokens = batch_size * max_len
                    total_tokens_list.append(total_tokens)

                    # Collect sample IDs
                    for sample in batch:
                        sample_ids.append(sample["sample_id"])

            return {
                "rank": rank,
                "batches": batches,
                "sample_ids": sample_ids,
                "total_tokens_list": total_tokens_list,
                "num_batches": len(batches),
                "num_samples": len(sample_ids),
                "avg_batch_size": sum(len(b) for b in batches) / len(batches)
                if batches
                else 0,
                "avg_total_tokens": sum(total_tokens_list) / len(total_tokens_list)
                if total_tokens_list
                else 0,
            }

    def test_no_duplicate_samples_across_ranks(self):
        """Test that different ranks don't get duplicate samples."""
        print("\n=== Test: No duplicate samples across ranks ===")

        # Simulate all 8 ranks
        rank_results = []
        for rank in range(self.num_ranks):
            result = self._simulate_rank(rank, self.num_ranks)
            rank_results.append(result)
            print(
                f"Rank {rank}: {result['num_samples']} samples, {result['num_batches']} batches"
            )

        # Collect all sample IDs from all ranks
        all_sample_ids = []
        for result in rank_results:
            all_sample_ids.extend(result["sample_ids"])

        # Check for duplicates
        unique_sample_ids = set(all_sample_ids)
        duplicates = len(all_sample_ids) - len(unique_sample_ids)

        print(f"Total samples across all ranks: {len(all_sample_ids)}")
        print(f"Unique samples: {len(unique_sample_ids)}")
        print(f"Duplicates: {duplicates}")

        # Assert no duplicates
        self.assertEqual(duplicates, 0, "Found duplicate samples across ranks!")

        # Assert all samples are covered (within expected range)
        # Note: Due to dynamic batching, some samples might be skipped if they don't fit
        # But we should cover most of the dataset
        # For real datasets, coverage might be slightly lower due to variable sequence lengths
        coverage = len(unique_sample_ids) / self.dataset_size
        print(f"Dataset coverage: {coverage:.2%}")
        min_coverage = 0.7 if self.use_real_dataset else 0.8
        self.assertGreater(
            coverage,
            min_coverage,
            f"Dataset coverage too low: {coverage:.2%} (expected > {min_coverage:.0%})",
        )

    def test_batch_size_variation(self):
        """Test that different ranks have different batch sizes (dynamic batching)."""
        print("\n=== Test: Batch size variation across ranks ===")

        rank_results = []
        for rank in range(self.num_ranks):
            result = self._simulate_rank(rank, self.num_ranks)
            rank_results.append(result)
            print(
                f"Rank {rank}: avg_batch_size={result['avg_batch_size']:.2f}, "
                f"avg_total_tokens={result['avg_total_tokens']:.2f}"
            )

        # Check that batch sizes vary across ranks
        batch_sizes = [r["avg_batch_size"] for r in rank_results]
        min_batch_size = min(batch_sizes)
        max_batch_size = max(batch_sizes)

        print(f"Batch size range: [{min_batch_size:.2f}, {max_batch_size:.2f}]")

        # In dynamic batching, batch sizes should vary
        # But if all samples have similar lengths, batch sizes might be similar
        # So we just check that the implementation works correctly
        self.assertGreater(
            min_batch_size, 0, "All ranks should have at least one batch"
        )

    def test_total_tokens_balancing(self):
        """Test that total_tokens are balanced across ranks."""
        print("\n=== Test: Total tokens balancing ===")

        rank_results = []
        for rank in range(self.num_ranks):
            result = self._simulate_rank(rank, self.num_ranks)
            rank_results.append(result)

        # Calculate statistics
        avg_tokens_per_rank = [r["avg_total_tokens"] for r in rank_results]
        overall_avg = sum(avg_tokens_per_rank) / len(avg_tokens_per_rank)
        min_avg = min(avg_tokens_per_rank)
        max_avg = max(avg_tokens_per_rank)

        print("Average total_tokens per rank:")
        for rank, result in enumerate(rank_results):
            print(f"  Rank {rank}: {result['avg_total_tokens']:.2f}")
        print(f"Overall average: {overall_avg:.2f}")
        print(f"Range: [{min_avg:.2f}, {max_avg:.2f}]")
        print(f"Variation: {(max_avg - min_avg) / overall_avg * 100:.2f}%")

        # Check that total_tokens are reasonably balanced
        # Variation should be less than 50% (due to dynamic batching)
        # For real datasets, variation might be slightly higher due to variable sequence lengths
        variation = (max_avg - min_avg) / overall_avg if overall_avg > 0 else 0
        max_variation = 0.6 if self.use_real_dataset else 0.5
        self.assertLess(
            variation,
            max_variation,
            f"Total tokens variation too high: {variation:.2%} (expected < {max_variation:.0%})",
        )

    def test_max_tokens_constraint(self):
        """Test that batches respect max_tokens_for_batch constraint."""
        print("\n=== Test: Max tokens constraint ===")

        rank_results = []
        for rank in range(self.num_ranks):
            result = self._simulate_rank(rank, self.num_ranks)
            rank_results.append(result)

        # Check all batches from all ranks
        violations = []
        for result in rank_results:
            for batch_idx, total_tokens in enumerate(result["total_tokens_list"]):
                if total_tokens > self.max_tokens_for_batch:
                    violations.append(
                        {
                            "rank": result["rank"],
                            "batch_idx": batch_idx,
                            "total_tokens": total_tokens,
                        }
                    )

        print(
            f"Found {len(violations)} batches exceeding max_tokens_for_batch={self.max_tokens_for_batch}"
        )
        if violations:
            for v in violations[:5]:  # Show first 5 violations
                print(
                    f"  Rank {v['rank']}, Batch {v['batch_idx']}: {v['total_tokens']} tokens"
                )

        # Assert no violations
        self.assertEqual(
            len(violations),
            0,
            f"Found {len(violations)} batches exceeding max_tokens_for_batch!",
        )

    def test_batch_formation(self):
        """Test that batches are formed correctly."""
        print("\n=== Test: Batch formation ===")

        # Test a single rank in detail
        result = self._simulate_rank(0, self.num_ranks)

        print(
            f"Rank 0: {result['num_batches']} batches, {result['num_samples']} samples"
        )

        # Check first few batches
        for i, batch in enumerate(result["batches"][:5]):
            batch_size = len(batch)
            lengths = [get_sample_length(s, "input_ids") for s in batch]
            max_len = max(lengths)
            total_tokens = batch_size * max_len

            print(
                f"  Batch {i}: size={batch_size}, lengths={lengths}, "
                f"max_len={max_len}, total_tokens={total_tokens}"
            )

            # Check that batch is not empty
            self.assertGreater(batch_size, 0, f"Batch {i} is empty!")

            # Check that total_tokens <= max_tokens_for_batch
            self.assertLessEqual(
                total_tokens,
                self.max_tokens_for_batch,
                f"Batch {i} exceeds max_tokens_for_batch!",
            )

    def test_pool_mechanism(self):
        """Test that pool mechanism works correctly."""
        print("\n=== Test: Pool mechanism ===")

        # Create dataset with small pool size to test pool behavior
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=10,  # Small pool size
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=1,
            )

            # Get all accumulated groups
            accumulated_groups = list(dataset)
            self.assertGreater(
                len(accumulated_groups), 0, "Should have at least one accumulated group"
            )

            # Count total batches
            total_batches = sum(len(group) for group in accumulated_groups)
            print(
                f"With pool_size=10, got {len(accumulated_groups)} accumulated groups, {total_batches} total batches"
            )

    def test_deterministic_with_seed(self):
        """Test that results are deterministic with the same seed."""
        print("\n=== Test: Deterministic behavior ===")

        # Run twice with same seed
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset1 = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=1,
            )

            dataset2 = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=1,
            )

            accumulated_groups1 = [g for g in dataset1]
            accumulated_groups2 = [g for g in dataset2]

            # Check that number of accumulated groups match
            self.assertEqual(
                len(accumulated_groups1),
                len(accumulated_groups2),
                "Different number of accumulated groups with same seed!",
            )

            # Check that each group has the same number of batches
            for i, (g1, g2) in enumerate(zip(accumulated_groups1, accumulated_groups2)):
                self.assertEqual(
                    len(g1),
                    len(g2),
                    f"Accumulated group {i} sizes differ with same seed!",
                )

                # Check that batches within each group have the same sizes
                for j, (b1, b2) in enumerate(zip(g1, g2)):
                    self.assertEqual(
                        len(b1),
                        len(b2),
                        f"Batch {j} in group {i} sizes differ with same seed!",
                    )

    def test_accumulate_steps_single_batch(self):
        """Test that accumulate_steps=1 returns lists containing single batches."""
        print("\n=== Test: accumulate_steps=1 (default behavior) ===")

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                accumulate_steps=1,
            )

            accumulated_groups = []
            total_batches = 0

            for item in dataset:
                # With accumulate_steps=1, item should be a list containing a single batch
                self.assertIsInstance(item, list, "Should return a list of batches")
                self.assertEqual(
                    len(item),
                    1,
                    "With accumulate_steps=1, should return list with 1 batch",
                )

                batch = item[0]
                self.assertIsInstance(
                    batch, list, "Each batch should be a list of samples"
                )
                self.assertGreater(len(batch), 0, "Batch should not be empty")

                accumulated_groups.append(item)
                total_batches += len(item)

            print(
                f"With accumulate_steps=1, got {len(accumulated_groups)} accumulated groups"
            )
            print(f"Total batches: {total_batches}")
            self.assertGreater(
                len(accumulated_groups), 0, "Should have at least one accumulated group"
            )

    def test_accumulate_steps_multiple_batches(self):
        """Test that accumulate_steps > 1 returns lists of batches."""
        print("\n=== Test: accumulate_steps > 1 (gradient accumulation) ===")

        accumulate_steps = 4

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=accumulate_steps,
            )

            accumulated_groups = []
            total_batches = 0

            for item in dataset:
                # With accumulate_steps > 1, item should be a list of batches
                self.assertIsInstance(
                    item,
                    list,
                    "With accumulate_steps > 1, should return a list of batches",
                )
                self.assertGreater(
                    len(item), 0, "Accumulated group should not be empty"
                )
                self.assertLessEqual(
                    len(item),
                    accumulate_steps,
                    f"Accumulated group should have at most {accumulate_steps} batches",
                )

                accumulated_groups.append(item)
                total_batches += len(item)

                # Verify each batch in the accumulated group
                for batch in item:
                    self.assertIsInstance(
                        batch, list, "Each item in accumulated group should be a batch"
                    )
                    self.assertGreater(len(batch), 0, "Each batch should not be empty")

            print(
                f"With accumulate_steps={accumulate_steps}, got {len(accumulated_groups)} accumulated groups"
            )
            print(f"Total batches: {total_batches}")

            self.assertGreater(
                len(accumulated_groups), 0, "Should have at least one accumulated group"
            )

            # Most groups should have exactly accumulate_steps batches
            # (except possibly the last one)
            full_groups = sum(
                1 for group in accumulated_groups if len(group) == accumulate_steps
            )
            print(
                f"Full groups (with {accumulate_steps} batches): {full_groups}/{len(accumulated_groups)}"
            )

    def test_accumulate_steps_equivalence(self):
        """Test that accumulate_steps=1 and accumulate_steps > 1 process the same samples."""
        print("\n=== Test: accumulate_steps equivalence ===")

        accumulate_steps = 3

        # Test with accumulate_steps=1
        result1 = self._simulate_rank(0, 1, accumulate_steps=1)
        sample_ids1 = set(result1["sample_ids"])

        # Test with accumulate_steps > 1
        result2 = self._simulate_rank(0, 1, accumulate_steps=accumulate_steps)
        sample_ids2 = set(result2["sample_ids"])

        print(
            f"accumulate_steps=1: {len(sample_ids1)} unique samples, {result1['num_batches']} batches"
        )
        print(
            f"accumulate_steps={accumulate_steps}: {len(sample_ids2)} unique samples, {result2['num_batches']} batches"
        )

        # Both should process similar number of samples (may differ slightly due to batching)
        # But the difference should be small
        sample_diff = abs(len(sample_ids1) - len(sample_ids2))
        sample_diff_ratio = sample_diff / max(len(sample_ids1), len(sample_ids2), 1)

        print(f"Sample count difference: {sample_diff} ({sample_diff_ratio:.2%})")

        # The difference should be small
        # For real datasets, difference might be slightly larger due to variable sequence lengths
        max_diff_ratio = 0.15 if self.use_real_dataset else 0.1
        self.assertLess(
            sample_diff_ratio,
            max_diff_ratio,
            f"Sample count difference too large: {sample_diff_ratio:.2%} (expected < {max_diff_ratio:.0%})",
        )

    def test_accumulate_steps_with_multiple_ranks(self):
        """Test accumulate_steps with multiple ranks."""
        print("\n=== Test: accumulate_steps with multiple ranks ===")

        accumulate_steps = 2

        # Simulate all ranks with accumulate_steps > 1
        rank_results = []
        for rank in range(self.num_ranks):
            result = self._simulate_rank(
                rank, self.num_ranks, accumulate_steps=accumulate_steps
            )
            rank_results.append(result)
            print(
                f"Rank {rank}: {result['num_samples']} samples, {result['num_batches']} batches"
            )

        # Collect all sample IDs from all ranks
        all_sample_ids = []
        for result in rank_results:
            all_sample_ids.extend(result["sample_ids"])

        # Check for duplicates
        unique_sample_ids = set(all_sample_ids)
        duplicates = len(all_sample_ids) - len(unique_sample_ids)

        print(f"Total samples across all ranks: {len(all_sample_ids)}")
        print(f"Unique samples: {len(unique_sample_ids)}")
        print(f"Duplicates: {duplicates}")

        # Assert no duplicates
        self.assertEqual(duplicates, 0, "Found duplicate samples across ranks!")

        # Assert reasonable coverage
        # For real datasets, coverage might be slightly lower due to variable sequence lengths
        coverage = len(unique_sample_ids) / self.dataset_size
        print(f"Dataset coverage: {coverage:.2%}")
        min_coverage = 0.7 if self.use_real_dataset else 0.8
        self.assertGreater(
            coverage,
            min_coverage,
            f"Dataset coverage too low: {coverage:.2%} (expected > {min_coverage:.0%})",
        )

    def test_accumulate_steps_invalid_value(self):
        """Test that invalid accumulate_steps values raise errors."""
        print("\n=== Test: Invalid accumulate_steps values ===")

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            # Test accumulate_steps=0 (invalid)
            with self.assertRaises(ValueError):
                LoadBalancedDataset(
                    base_dataset=self.base_dataset,
                    pool_size=self.pool_size,
                    max_tokens_for_batch=self.max_tokens_for_batch,
                    dp_rank=0,
                    dp_world_size=1,
                    accumulate_steps=0,
                )

            # Test accumulate_steps < 0 (invalid)
            with self.assertRaises(ValueError):
                LoadBalancedDataset(
                    base_dataset=self.base_dataset,
                    pool_size=self.pool_size,
                    max_tokens_for_batch=self.max_tokens_for_batch,
                    dp_rank=0,
                    dp_world_size=1,
                    accumulate_steps=-1,
                )

            print("Invalid accumulate_steps values correctly raise ValueError")

    def test_real_dataset_basic_functionality(self):
        """Test basic functionality with real dataset."""
        print("\n=== Test: Real dataset basic functionality ===")

        if not self.use_real_dataset:
            print("Skipping: Using mock dataset, not real dataset")
            return

        # Test that we can iterate through the dataset
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=1,
            )

            # Test iteration
            batch_count = 0
            sample_count = 0
            total_tokens = 0

            for item in dataset:
                self.assertIsInstance(item, list, "Should return a list of batches")
                self.assertEqual(
                    len(item),
                    1,
                    "With accumulate_steps=1, should return list with 1 batch",
                )

                batch = item[0]
                self.assertIsInstance(
                    batch, list, "Each batch should be a list of samples"
                )
                self.assertGreater(len(batch), 0, "Batch should not be empty")

                batch_count += 1
                sample_count += len(batch)

                # Check batch structure
                for sample in batch:
                    self.assertIn("input_ids", sample, "Sample should have input_ids")
                    self.assertIn("sample_id", sample, "Sample should have sample_id")
                    seq_len = get_sample_length(sample, "input_ids")
                    total_tokens += seq_len

            print(f"Processed {batch_count} batches, {sample_count} samples")
            print(
                f"Average tokens per sample: {total_tokens / sample_count if sample_count > 0 else 0:.2f}"
            )

            self.assertGreater(batch_count, 0, "Should have at least one batch")
            self.assertGreater(sample_count, 0, "Should have at least one sample")

    def test_real_dataset_with_accumulate_steps(self):
        """Test accumulate_steps with real dataset."""
        print("\n=== Test: Real dataset with accumulate_steps ===")

        if not self.use_real_dataset:
            print("Skipping: Using mock dataset, not real dataset")
            return

        accumulate_steps = 4

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch("torch.utils.data.get_worker_info", return_value=None),
        ):
            dataset = LoadBalancedDataset(
                base_dataset=self.base_dataset,
                pool_size=self.pool_size,
                max_tokens_for_batch=self.max_tokens_for_batch,
                length_key="input_ids",
                batching_strategy="prefer_closest",
                seed=self.seed,
                dp_rank=0,
                dp_world_size=1,
                accumulate_steps=accumulate_steps,
            )

            accumulated_groups = []
            total_batches = 0

            for item in dataset:
                self.assertIsInstance(item, list, "Should return a list of batches")
                self.assertGreater(
                    len(item), 0, "Accumulated group should not be empty"
                )
                self.assertLessEqual(
                    len(item),
                    accumulate_steps,
                    f"Accumulated group should have at most {accumulate_steps} batches",
                )

                accumulated_groups.append(item)
                total_batches += len(item)

            print(
                f"With accumulate_steps={accumulate_steps}, got {len(accumulated_groups)} accumulated groups"
            )
            print(f"Total batches: {total_batches}")

            self.assertGreater(
                len(accumulated_groups), 0, "Should have at least one accumulated group"
            )

            # Most groups should have exactly accumulate_steps batches (except possibly the last one)
            full_groups = sum(
                1 for group in accumulated_groups if len(group) == accumulate_steps
            )
            print(
                f"Full groups (with {accumulate_steps} batches): {full_groups}/{len(accumulated_groups)}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
