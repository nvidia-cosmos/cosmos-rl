import ast
from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch


class _Logger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


def _load_cache_class():
    source_path = (
        Path(__file__).parents[1] / "cosmos_rl/policy/model/hf_models/__init__.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "_ValidationVideoFeatureCache"
    )
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "Callable": Callable,
        "OrderedDict": OrderedDict,
        "logger": _Logger(),
        "torch": torch,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_ValidationVideoFeatureCache"]


def _load_video_key_extractor():
    source_path = (
        Path(__file__).parents[1]
        / "cosmos_rl/dispatcher/data/packer/hf_vlm_data_packer.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    class_node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "HFVLMDataPacker"
    )
    node = next(
        item
        for item in class_node.body
        if isinstance(item, ast.FunctionDef)
        and item.name == "_extract_video_cache_keys"
    )
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {"os": __import__("os")}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_extract_video_cache_keys"]


def test_validation_cache_deduplicates_within_and_across_batches():
    cache_cls = _load_cache_class()
    cache = cache_cls(capacity=4)
    calls = []

    def encode(pixel_values, grids):
        split_sizes = grids.prod(-1).tolist()
        chunks = torch.split(pixel_values, split_sizes)
        calls.append(len(chunks))
        main = tuple(chunk.sum().reshape(1, 1) for chunk in chunks)
        merged = torch.cat(main)
        return main, [merged + 10, merged + 20]

    grids = torch.tensor([[1, 2, 2], [1, 2, 2], [1, 2, 2]])
    a = torch.full((4, 1), 1.0)
    b = torch.full((4, 1), 2.0)
    main, deepstack = cache.get_or_encode(
        ["a", "a", "b"],
        torch.cat([a, a, b]),
        grids,
        encode,
        spatial_merge_size=2,
    )

    assert calls == [2]
    assert [tensor.item() for tensor in main] == [4.0, 4.0, 8.0]
    assert deepstack[0].flatten().tolist() == [14.0, 14.0, 18.0]
    assert cache.misses == 2

    second_main, second_deepstack = cache.get_or_encode(
        ["a", "b"],
        torch.cat([a, b]),
        grids[:2],
        encode,
        spatial_merge_size=2,
    )
    assert calls == [2]
    assert [tensor.item() for tensor in second_main] == [4.0, 8.0]
    assert second_deepstack[1].flatten().tolist() == [24.0, 28.0]
    assert cache.hits == 2

    stats = cache.clear()
    assert stats == {
        "calls": 2,
        "hits": 2,
        "misses": 2,
        "entries": 2,
        "sync_dummy_encodes": 0,
        "global_all_hit_calls": 1,
        "bypassed_calls": 0,
    }
    assert not cache.entries


def test_validation_cache_uses_grid_in_identity_and_obeys_lru_capacity():
    cache_cls = _load_cache_class()
    cache = cache_cls(capacity=1)
    calls = []

    def encode(pixel_values, grids):
        calls.append(tuple(tuple(row) for row in grids.tolist()))
        split_sizes = grids.prod(-1).tolist()
        main = tuple(chunk.clone() for chunk in torch.split(pixel_values, split_sizes))
        return main, []

    cache.get_or_encode(
        ["same"], torch.ones(4, 1), torch.tensor([[1, 2, 2]]), encode, 1
    )
    cache.get_or_encode(
        ["same"], torch.ones(8, 1), torch.tensor([[1, 2, 4]]), encode, 1
    )

    assert len(calls) == 2
    assert len(cache.entries) == 1
    assert next(iter(cache.entries))[1] == (1, 2, 4)


def test_validation_cache_keeps_fsdp_encoder_calls_rank_synchronous(monkeypatch):
    cache_cls = _load_cache_class()
    cache = cache_cls(capacity=2)
    calls = []

    def encode(pixel_values, grids):
        calls.append(len(grids))
        split_sizes = grids.prod(-1).tolist()
        main = tuple(
            chunk.sum().reshape(1, 1)
            for chunk in torch.split(pixel_values, split_sizes)
        )
        return main, []

    grids = torch.tensor([[1, 2, 2]])
    pixels = torch.ones(4, 1)

    # Populate the local rank cache without a distributed process group.
    first, _ = cache.get_or_encode(["a"], pixels, grids, encode, 2)
    assert first[0].item() == 4.0

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def fake_all_reduce(flag, op):
        # Eligibility uses MIN and remains true.  The MAX miss reduction
        # reports that another rank has a miss, forcing one local dummy encode.
        if op == torch.distributed.ReduceOp.MAX:
            flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)
    second, _ = cache.get_or_encode(["a"], pixels, grids, encode, 2)

    assert second[0].item() == 4.0
    assert calls == [1, 1]
    assert cache.sync_dummy_encodes == 1
    assert cache.global_all_hit_calls == 0


def test_video_key_extractor_accepts_paths_and_rejects_urls():
    extract = _load_video_key_extractor()
    sample = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": "/lustre/data/a.mp4"},
                {"type": "text", "text": "question"},
            ],
        }
    ]
    assert extract(sample) == ["/lustre/data/a.mp4"]

    sample[0]["content"][0]["video"] = "https://example.invalid/a.mp4"
    assert extract(sample) is None


def test_video_key_extractor_accepts_pydantic_style_chat_messages():
    extract = _load_video_key_extractor()

    class Message:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self):
            return self.payload

    sample = [
        Message(
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/lustre/data/a.mp4"},
                    {"type": "text", "text": "question"},
                ],
            }
        )
    ]
    assert extract(sample) == ["/lustre/data/a.mp4"]
