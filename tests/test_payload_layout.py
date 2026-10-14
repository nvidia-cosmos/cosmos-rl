# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from cosmos_rl.utils.payload_transport.layout import (
    LayoutCache,
    TensorLayout,
    WIRE_VERSION,
    deserialize_layout_descriptor,
    validate_layout_id,
)
from cosmos_rl.utils.trajectory import TensorSpec


def _schema():
    return [
        TensorSpec(name="flag", shape=(3,), dtype=np.bool_),
        TensorSpec(name="scalar", shape=(), dtype=np.int64),
        TensorSpec(name="empty", shape=(0, 7), dtype=np.float32),
        TensorSpec(name="values", shape=(2, 3), dtype=np.float32),
    ]


def test_layout_aligns_every_typed_view_and_round_trips_descriptor():
    layout = TensorLayout.compile(_schema())

    assert all(field.offset % field.dtype.itemsize == 0 for field in layout.fields)
    assert [field.name for field in layout.fields] == [
        "flag",
        "scalar",
        "empty",
        "values",
    ]
    assert layout.payload_nbytes == sum(field.nbytes for field in layout.fields) + sum(
        nbytes for _, nbytes in layout.padding
    )
    assert layout.payload_nbytes % layout.alignment == 0
    assert TensorLayout.from_descriptor(layout.descriptor()) == layout
    assert layout.descriptor()["wire_version"] == WIRE_VERSION


def test_views_share_one_storage_without_per_field_clones():
    layout = TensorLayout.compile(_schema())
    storage = torch.empty(layout.payload_nbytes, dtype=torch.uint8)
    views = layout.views(storage)

    assert views["scalar"].shape == torch.Size([])
    assert views["empty"].shape == torch.Size([0, 7])
    pointers = {value.untyped_storage().data_ptr() for value in views.values()}
    assert pointers == {storage.untyped_storage().data_ptr()}

    views["scalar"].fill_(17)
    views["values"].copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
    rebound = layout.views(storage)
    assert rebound["scalar"].item() == 17
    assert torch.equal(rebound["values"], views["values"])


def test_views_reject_a_misaligned_backing_slice():
    layout = TensorLayout.compile(_schema())
    allocation = torch.empty(layout.payload_nbytes + 1, dtype=torch.uint8)
    with pytest.raises(ValueError, match="alignment"):
        layout.views(allocation[1:])


def test_only_explicit_padding_is_initialized():
    layout = TensorLayout.compile(_schema())
    storage = torch.full((layout.payload_nbytes,), 0xA5, dtype=torch.uint8)
    layout.zero_padding(storage)
    for offset, nbytes in layout.padding:
        assert torch.count_nonzero(storage[offset : offset + nbytes]) == 0
    for field in layout.fields:
        assert torch.all(storage[field.offset : field.offset + field.nbytes] == 0xA5)


@pytest.mark.parametrize("alignment", [0, 3, -8, True, 128])
def test_layout_rejects_invalid_alignment(alignment):
    with pytest.raises(ValueError, match="alignment"):
        TensorLayout.compile(_schema(), alignment=alignment)


def test_layout_rejects_alignment_smaller_than_a_field_dtype():
    with pytest.raises(ValueError, match="smaller than"):
        TensorLayout.compile(_schema(), alignment=4)


def test_layout_rejects_duplicate_names_and_unsupported_dtype():
    with pytest.raises(ValueError, match="unique"):
        TensorLayout.compile(
            [
                TensorSpec(name="same", shape=(1,), dtype=np.float32),
                TensorSpec(name="same", shape=(1,), dtype=np.float32),
            ]
        )
    with pytest.raises(ValueError, match="unsupported"):
        TensorLayout.compile([TensorSpec(name="text", shape=(1,), dtype="U4")])
    with pytest.raises(ValueError, match="strings"):
        TensorLayout.compile([TensorSpec(name=7, shape=(1,), dtype=np.float32)])


def test_layout_descriptor_rejects_noncanonical_offsets_and_identity():
    layout = TensorLayout.compile(_schema())
    descriptor = layout.descriptor()
    descriptor["fields"][1]["offset"] += 1
    with pytest.raises(ValueError, match="canonical"):
        TensorLayout.from_descriptor(descriptor)

    descriptor = layout.descriptor()
    descriptor["layout_id"] = "0" * 32
    with pytest.raises(ValueError, match="identity"):
        TensorLayout.from_descriptor(descriptor)

    descriptor = layout.descriptor()
    descriptor["layout_id"] = None
    with pytest.raises(ValueError, match="32 lowercase hex"):
        TensorLayout.from_descriptor(descriptor)

    descriptor = layout.descriptor()
    descriptor["ignored"] = True
    with pytest.raises(ValueError, match="canonical"):
        TensorLayout.from_descriptor(descriptor)


def test_layout_descriptor_rejects_boolean_integer_substitution():
    layout = TensorLayout.compile(
        [TensorSpec(name="empty", shape=(0,), dtype=np.float32)]
    )
    descriptor = layout.descriptor()
    descriptor["fields"][0]["offset"] = False
    descriptor["fields"][0]["nbytes"] = False

    with pytest.raises(ValueError, match="canonical"):
        TensorLayout.from_descriptor(descriptor)


def test_descriptor_json_rejects_duplicate_and_nonfinite_values():
    with pytest.raises(ValueError, match="repeats JSON field"):
        deserialize_layout_descriptor('{"layout_id":"first","layout_id":"second"}')
    with pytest.raises(ValueError, match="nonfinite JSON value"):
        deserialize_layout_descriptor('{"payload_nbytes":NaN}')


@pytest.mark.parametrize(
    "layout_id",
    [None, "", "0" * 31, "0" * 33, "G" * 32, "A" * 32, 0],
)
def test_layout_identity_must_be_canonical(layout_id):
    with pytest.raises(ValueError, match="32 lowercase hex"):
        validate_layout_id(layout_id)


def test_layout_cache_reuses_validated_layout_and_is_bounded():
    first = TensorLayout.compile(_schema())
    second = TensorLayout.compile(
        [TensorSpec(name="other", shape=(4,), dtype=np.int32)]
    )
    cache = LayoutCache(capacity=1)

    resolved = cache.resolve(first.layout_id, first.descriptor())
    assert cache.resolve(first.layout_id, None) is resolved
    cache.register(second)
    assert len(cache) == 1
    with pytest.raises(KeyError, match="unknown"):
        cache.resolve(first.layout_id, None)


def test_layout_cache_resolve_is_safe_for_concurrent_publishers():
    layout = TensorLayout.compile(_schema())
    cache = LayoutCache(capacity=2)
    cache.register(layout)

    with ThreadPoolExecutor(max_workers=16) as executor:
        resolved = list(
            executor.map(
                lambda _: cache.resolve(layout.layout_id, None),
                range(10_000),
            )
        )

    assert all(item is layout for item in resolved)
