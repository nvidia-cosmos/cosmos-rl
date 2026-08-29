import ast
from collections import Counter
from collections import OrderedDict
import math
from pathlib import Path
from typing import Iterator

import pytest
import torch


class _Logger:
    def info(self, *args, **kwargs):
        pass


def _load_sampler_class():
    source_path = (
        Path(__file__).parents[1] / "cosmos_rl/tools/custom_hooks/tao_sft_example.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "MediaGroupedDistributedSampler"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "Iterator": Iterator,
        "OrderedDict": OrderedDict,
        "logger": _Logger(),
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["MediaGroupedDistributedSampler"]


MediaGroupedDistributedSampler = _load_sampler_class()


class _MediaDataset:
    def __init__(self, keys):
        self.keys = list(keys)

    def __len__(self):
        return len(self.keys)

    def media_key(self, index):
        return (self.keys[index],)


class _WrappedDataset:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)


def test_media_grouped_sampler_preserves_distributed_padding_multiset():
    source = _MediaDataset(["a"] * 4 + ["b"] * 3 + ["c"] * 3)
    wrapped = _WrappedDataset(source)
    replicas = 3
    samplers = [
        MediaGroupedDistributedSampler(wrapped, replicas, rank)
        for rank in range(replicas)
    ]

    expected = list(range(len(source)))
    per_rank = math.ceil(len(source) / replicas)
    expected += expected[: per_rank * replicas - len(expected)]
    actual = [index for sampler in samplers for index in sampler]

    assert [len(sampler) for sampler in samplers] == [per_rank] * replicas
    assert Counter(actual) == Counter(expected)
    assert [index for index, count in Counter(actual).items() if count > 1] == [0, 1]

    # Equal contiguous cuts can split at most one media group at each rank
    # boundary, so global decoder exposures stay near the unique-media floor.
    exposures = sum(
        len({source.media_key(index) for index in sampler}) for sampler in samplers
    )
    assert exposures <= len(set(source.keys)) + replicas - 1


def test_media_grouped_sampler_is_deterministic_and_rejects_shuffle():
    wrapped = _WrappedDataset(_MediaDataset(["a", "b", "a", "c", "b"]))
    first = list(MediaGroupedDistributedSampler(wrapped, 2, 0))
    second = list(MediaGroupedDistributedSampler(wrapped, 2, 0))
    assert first == second

    with pytest.raises(ValueError, match="shuffle"):
        MediaGroupedDistributedSampler(wrapped, 2, 0, shuffle=True)


def test_media_grouped_sampler_frontloads_one_record_per_rank_media_group():
    source = _MediaDataset(["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d"] * 2)
    wrapped = _WrappedDataset(source)

    for rank in range(2):
        indices = list(MediaGroupedDistributedSampler(wrapped, 2, rank))
        unique_keys = {source.media_key(index) for index in indices}
        frontloaded = indices[: len(unique_keys)]

        assert len({source.media_key(index) for index in frontloaded}) == len(
            unique_keys
        )


def test_media_grouped_sampler_stages_new_media_without_changing_multiset():
    source = _MediaDataset(
        [key for key in ("a", "b", "c", "d", "e", "f") for _ in range(5)]
    )
    wrapped = _WrappedDataset(source)
    original_batch_size = MediaGroupedDistributedSampler.cache_frontload_batch_size
    original_unique = MediaGroupedDistributedSampler.cache_frontload_unique_per_batch
    try:
        MediaGroupedDistributedSampler.cache_frontload_batch_size = 6
        MediaGroupedDistributedSampler.cache_frontload_unique_per_batch = 2
        indices = list(MediaGroupedDistributedSampler(wrapped, 1, 0))
    finally:
        MediaGroupedDistributedSampler.cache_frontload_batch_size = original_batch_size
        MediaGroupedDistributedSampler.cache_frontload_unique_per_batch = original_unique

    assert Counter(indices) == Counter(range(len(source)))
    first_seen = {}
    for position, index in enumerate(indices):
        first_seen.setdefault(source.media_key(index), position)
    assert sorted(position // 6 for position in first_seen.values()) == [0, 0, 1, 1, 2, 2]


def test_media_grouped_sampler_rejects_partial_staged_frontload_config():
    wrapped = _WrappedDataset(_MediaDataset(["a", "a", "b", "b"]))
    original_batch_size = MediaGroupedDistributedSampler.cache_frontload_batch_size
    original_unique = MediaGroupedDistributedSampler.cache_frontload_unique_per_batch
    try:
        MediaGroupedDistributedSampler.cache_frontload_batch_size = 4
        MediaGroupedDistributedSampler.cache_frontload_unique_per_batch = 0
        with pytest.raises(ValueError, match="staged validation cache frontloading"):
            MediaGroupedDistributedSampler(wrapped, 1, 0)
    finally:
        MediaGroupedDistributedSampler.cache_frontload_batch_size = original_batch_size
        MediaGroupedDistributedSampler.cache_frontload_unique_per_batch = original_unique
