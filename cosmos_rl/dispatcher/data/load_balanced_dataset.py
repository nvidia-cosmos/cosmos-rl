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
Load-balanced data loading for FSDP training.

This module implements a pool-based dynamic batching strategy to balance
the number of tokens across different data parallel ranks, reducing padding
waste and improving training efficiency.
"""

from collections import deque
from typing import Dict, List, Optional, Any, Deque, Union
import torch
from torch.utils.data import Dataset, IterableDataset
from cosmos_rl.utils.logging import logger


def get_sample_length(sample: Dict[str, Any], length_key: str = "input_ids") -> int:
    """
    Get the sequence length of a sample.

    Args:
        sample: A sample dictionary, typically from SFTDataset
        length_key: Key to access the sequence length (default: "input_ids")

    Returns:
        Length of the sequence
    """
    if length_key in sample:
        seq = sample[length_key]
        if isinstance(seq, (list, torch.Tensor)):
            return len(seq)
        elif hasattr(seq, "__len__"):
            return len(seq)

    # Fallback: try to find any list/tensor field
    for key, value in sample.items():
        if isinstance(value, (list, torch.Tensor)):
            if isinstance(value, torch.Tensor) and value.dim() > 0:
                return value.shape[0]
            elif isinstance(value, list) and len(value) > 0:
                return len(value)

    raise ValueError(
        f"Could not determine sequence length for sample. Available keys: {list(sample.keys())}"
    )


class ShardedIterableDataset(IterableDataset):
    """
    An IterableDataset that shards a base dataset according to dp_rank and dp_world_size.
    This ensures each rank only sees its portion of the data.

    This class converts a regular Dataset to an IterableDataset by implementing
    __iter__ that yields samples based on sharding logic.
    """

    def __init__(
        self,
        base_dataset: Union[Dataset, IterableDataset],
        dp_rank: int,
        dp_world_size: int,
        seed: int = 0,
    ):
        super().__init__()
        self.base_dataset = base_dataset
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.seed = seed
        logger.debug(
            f"[ShardedIterableDataset] base_dataset type: {type(base_dataset)}, "
            f"dp_rank={dp_rank}, dp_world_size={dp_world_size}"
        )

    def __len__(self) -> int:
        """Return approximate length for progress tracking."""
        total_len = len(self.base_dataset)
        return (total_len + self.dp_world_size - 1) // self.dp_world_size

    def __iter__(self):
        """
        Iterate over sharded samples.

        Converts Dataset to IterableDataset by yielding samples based on sharding.
        For Dataset: uses index-based access with stride sampling.
        For IterableDataset: uses counter-based sharding.
        """
        import random

        # Get worker info (for multi-process data loading within a rank)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        num_workers = 1
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Calculate total shards: dp_world_size * num_workers
        # Each shard is identified by (dp_rank, worker_id)
        total_shards = self.dp_world_size * num_workers
        shard_id = self.dp_rank * num_workers + worker_id
        # For Dataset: convert to IterableDataset by yielding samples via index access
        # Use stride sampling: shard_id, shard_id+total_shards, shard_id+2*total_shards, ...
        total_samples = len(self.base_dataset)
        indices = list(range(shard_id, total_samples, total_shards))

        logger.info(
            f"[ShardedIterableDataset] Converting Dataset to IterableDataset: "
            f"total_samples={total_samples}, shard_id={shard_id}, "
            f"total_shards={total_shards}, indices_count={len(indices)}"
        )

        # Shuffle deterministically based on shard_id for reproducibility
        rng = random.Random(self.seed + shard_id)
        rng.shuffle(indices)

        # Yield samples by index - this converts Dataset to IterableDataset
        for idx in indices:
            yield self.base_dataset[idx]


class LoadBalancedDataset(IterableDataset):
    """
    An IterableDataset that implements load-balanced dynamic batching.

    This dataset maintains a pool of samples and dynamically creates batches
    that maximize batch_size * max_input_len while staying within max_tokens_for_batch
    constraint, minimizing padding waste.

    This class doesn't explicitly handle dp_rank.
    Instead, it relies on IterableDataset's natural behavior where each worker/rank
    gets its own independent iterator, ensuring data distribution across ranks.

    Example:
        >>> base_dataset = SFTDataset(...)
        >>> balanced_dataset = LoadBalancedDataset(
        ...     base_dataset=base_dataset,
        ...     pool_size=32,
        ...     max_tokens_for_batch=4 * 8192,
        ... )
        >>> dataloader = DataLoader(balanced_dataset, batch_size=None, ...)
    """

    def __init__(
        self,
        base_dataset: Union[Dataset, IterableDataset],
        pool_size: int = 32,
        max_tokens_for_batch: int = 32768,
        length_key: str = "input_ids",
        batching_strategy: str = "prefer_closest",  # "prefer_first" or "prefer_closest"
        max_tokens_len: Optional[
            int
        ] = None,  # If sample length >= this, emit as singleton
        seq_packing_enabled: bool = False,
        seed: int = 0,
        dp_rank: int = 0,
        dp_world_size: int = 1,
    ):
        """
        Initialize the load-balanced dataset.

        Args:
            base_dataset: The underlying dataset (Dataset or IterableDataset)
            pool_size: Size of the sample pool maintained by each rank
            max_tokens_for_batch: Maximum tokens per batch (batch_size * max_seq_len)
            length_key: Key to access sequence length in samples
            batching_strategy: "prefer_first" (FIFO) or "prefer_closest" (minimize padding)
            max_tokens_len: If a sample's length >= this, emit it alone
            seq_packing_enabled: Whether sequence packing is enabled
            seed: Random seed for sampling (used if base_dataset is Dataset)
            dp_rank: Data parallel rank for sharding (default: 0)
            dp_world_size: Data parallel world size for sharding (default: 1)
        """
        super().__init__()
        self.pool_size = pool_size
        self.max_tokens_for_batch = max_tokens_for_batch
        self.length_key = length_key
        self.batching_strategy = batching_strategy
        self.max_tokens_len = max_tokens_len or max_tokens_for_batch
        self.seq_packing_enabled = seq_packing_enabled
        self.seed = seed
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size

        # Shard base_dataset according to dp_rank and convert to IterableDataset
        self.base_dataset = ShardedIterableDataset(
            base_dataset=base_dataset,
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            seed=seed,
        )

        # Initialize pool
        self._pool: Deque[Dict[str, Any]] = deque()

        # Iterator will be created in __iter__
        self._dataset_iterator: Optional[Any] = None
        self.accumulated_tokens_this_rank = 0
        self.accumulated_samples_this_rank = 0

        logger.info(
            f"[LoadBalancedDataset] pool_size={pool_size}, max_tokens_for_batch={max_tokens_for_batch}, "
            f"batching_strategy={batching_strategy}, seq_packing_enabled={seq_packing_enabled}, "
            f"dp_rank={dp_rank}, dp_world_size={dp_world_size}"
        )

    def __len__(self) -> int:
        """Return approximate length for progress tracking."""
        return len(self.base_dataset)

    def _get_next_sample(self) -> Optional[Dict[str, Any]]:
        """
        Get the next sample from the base dataset.

        For IterableDataset: uses the iterator directly, recreates if exhausted
        For Dataset: uses the iterator created in __iter__

        Returns:
            Sample dictionary or None if dataset exhausted
        """
        if self._dataset_iterator is None:
            return None

        try:
            # Both IterableDataset and Dataset use the iterator created in __iter__
            # which already handles sharding and StopIteration
            return next(self._dataset_iterator)
        except StopIteration:
            return None
        except (IndexError, KeyError) as e:
            logger.warning(f"Error getting sample: {e}")
            return None

    def _fill_pool(self):
        """Fill the pool with samples up to pool_size."""
        while len(self._pool) < self.pool_size:
            sample = self._get_next_sample()
            if sample is None:
                break
            self._pool.append(sample)

    def _find_best_candidate_prefer_first(
        self, cur_max: int, num_chosen: int
    ) -> Optional[int]:
        """
        Find best candidate by iterating from first to last in pool (FIFO).
        Returns the first candidate that fits within budget and minimizes padding.
        """
        best_idx = None
        best_new_total_tokens = None

        for idx, cand in enumerate(self._pool):
            cand_token_len = get_sample_length(cand, self.length_key)
            new_max_token_len = max(cur_max, cand_token_len)
            new_total_tokens = new_max_token_len * (num_chosen + 1)

            if new_total_tokens <= self.max_tokens_for_batch:
                if (
                    best_new_total_tokens is None
                    or new_total_tokens < best_new_total_tokens
                ):
                    best_new_total_tokens = new_total_tokens
                    best_idx = idx

        return best_idx

    def _find_best_candidate_prefer_closest(
        self, cur_max: int, num_chosen: int
    ) -> Optional[int]:
        """
        Find best candidate by selecting the one with closest length to current max.
        Among valid candidates, picks the one that minimizes padding waste.
        """
        best_idx = None
        best_new_total_tokens = None
        smallest_length_diff = None

        for idx, cand in enumerate(self._pool):
            cand_token_len = get_sample_length(cand, self.length_key)
            new_max_token_len = max(cur_max, cand_token_len)
            new_total_tokens = new_max_token_len * (num_chosen + 1)

            if new_total_tokens <= self.max_tokens_for_batch:
                length_diff = abs(cand_token_len - cur_max)
                # Prefer candidates with closer length to current max
                if (
                    best_new_total_tokens is None
                    or new_total_tokens < best_new_total_tokens
                    or (
                        new_total_tokens == best_new_total_tokens
                        and length_diff < smallest_length_diff
                    )
                ):
                    best_new_total_tokens = new_total_tokens
                    best_idx = idx
                    smallest_length_diff = length_diff

        return best_idx

    def _find_best_candidate_seq_packing(self, chosen_samples: list) -> Optional[int]:
        """
        Find the best candidate index based on the sequence packing strategy.
        """
        if len(self._pool) == 0:
            return None

        total_tokens = sum(
            get_sample_length(sample, self.length_key) for sample in chosen_samples
        )

        # TODO(huik): optimize this search algorithm
        for idx, cand in enumerate(self._pool):
            cand_token_len = get_sample_length(cand, self.length_key)
            new_total_tokens = total_tokens + cand_token_len
            if new_total_tokens > self.max_tokens_for_batch:
                continue
            return idx

        return None

    def _find_best_candidate(
        self, cur_max: int, chosen_samples: list = []
    ) -> Optional[int]:
        """
        Find the best candidate index based on the batching strategy.

        Args:
            cur_max: Current maximum sequence length in the batch
            num_chosen: Number of items already chosen

        Returns:
            Index of best candidate in pool, or None if no candidate fits
        """
        if self.seq_packing_enabled:
            return self._find_best_candidate_seq_packing(chosen_samples)
        elif self.batching_strategy == "prefer_closest":
            return self._find_best_candidate_prefer_closest(
                cur_max, len(chosen_samples)
            )
        else:
            # prefer_first
            return self._find_best_candidate_prefer_first(cur_max, len(chosen_samples))

    def _remove_from_pool(self, idx: int) -> Dict[str, Any]:
        """
        Efficiently remove item at index from pool.

        Args:
            idx: Index of item to remove

        Returns:
            Removed item
        """
        if idx == 0:
            return self._pool.popleft()
        elif idx == len(self._pool) - 1:
            return self._pool.pop()
        else:
            # rotate trick for O(1) pops from either end
            self._pool.rotate(-idx)
            item = self._pool.popleft()
            self._pool.rotate(idx)
            return item

    def _best_fit_batch(self) -> List[Dict[str, Any]]:
        """
        Build one batch using the specified batching strategy.

        Returns:
            List of samples forming a batch
        """
        self._fill_pool()

        if not self._pool:
            raise StopIteration

        # Seed with oldest item (FIFO)
        seed = self._pool.popleft()
        seed_token_len = get_sample_length(seed, self.length_key)

        # Emit alone if seed sample is too long
        if (
            seed_token_len >= self.max_tokens_len
            or seed_token_len >= self.max_tokens_for_batch
        ):
            logger.debug(
                f"[LoadBalancedDataset] Single long sample: length={seed_token_len}"
            )
            return [seed]

        chosen = [seed]
        cur_max = seed_token_len

        # Greedy best-fit: pick items that minimally increase cost
        while self._pool:
            # TODO(huik): If seq_packing_enabled, return a list of indices for the best candidates
            best_idx = self._find_best_candidate(cur_max, chosen)

            # No candidate fits within budget
            if best_idx is None:
                break

            if not isinstance(best_idx, list):
                indices = [best_idx]
            else:
                indices = best_idx

            # Take the best-fitting candidate and remove from pool
            for idx in indices:
                cand = self._remove_from_pool(idx)
                chosen.append(cand)
                cur_max = max(cur_max, get_sample_length(cand, self.length_key))

        if self.seq_packing_enabled:
            total_tokens = sum(
                get_sample_length(sample, self.length_key) for sample in chosen
            )
        else:
            total_tokens = cur_max * len(chosen)

        self.accumulated_tokens_this_rank += total_tokens
        self.accumulated_samples_this_rank += len(chosen)
        # Log batch info (including rank info for distributed training)
        logger.info(
            f"[LoadBalancedDataset] Rank {self.dp_rank}: "
            f"Batch: size={len(chosen)}, max_len={cur_max}, total_tokens={total_tokens} Accumulated: tokens={self.accumulated_tokens_this_rank}, samples={self.accumulated_samples_this_rank}"
        )

        return chosen

    def __iter__(self):
        """
        Iterate over batches.

        The base_dataset is already sharded according to dp_rank in __init__,
        so we can directly iterate over it.
        """
        # Create iterator from the already-sharded base_dataset
        self._dataset_iterator = iter(self.base_dataset)

        # Reset pool
        # Note: If pool has remaining samples from previous iteration, they will be discarded.
        # This is intentional - each epoch should start fresh. However, if you need to preserve
        # remaining samples across iterator recreations, consider yielding them before clearing.
        if self._pool:
            logger.warning(
                f"[LoadBalancedDataset] Clearing pool with {len(self._pool)} remaining samples. "
                "These samples will be discarded. This is normal if starting a new epoch, "
                "but may indicate premature iterator recreation."
            )
        self._pool.clear()

        # Iterate and yield batches
        while True:
            try:
                batch = self._best_fit_batch()
                yield batch
            except StopIteration:
                # Check if we can refill pool
                self._fill_pool()
                if not self._pool:
                    # Dataset exhausted
                    break
                # Try again with refilled pool
                try:
                    batch = self._best_fit_batch()
                    yield batch
                except StopIteration:
                    break

        logger.info(
            f"[LoadBalancedDataset] Iteration completed. "
            f"dp_rank={self.dp_rank}/{self.dp_world_size}"
        )
