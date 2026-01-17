import os
import pickle
from types import SimpleNamespace
from typing import Any, List, Tuple

import torch


def _is_simple_ns(obj: Any) -> bool:
    return isinstance(obj, SimpleNamespace)


def _as_dict(obj: Any):
    if _is_simple_ns(obj):
        return vars(obj)
    return obj


def _compare(a: Any, b: Any, path: Tuple, mismatches: List[str], rtol=1e-4, atol=1e-6):
    # Stop early to avoid printing enormous diffs (e.g., full model state_dict).
    if len(mismatches) >= 50:
        return

    # Tensor comparison
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape:
            mismatches.append(f"{path}: shape mismatch {a.shape} vs {b.shape}")
            return
        if a.dtype != b.dtype:
            mismatches.append(f"{path}: dtype mismatch {a.dtype} vs {b.dtype}")
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = (a - b).abs()
            max_abs = diff.max().item()
            max_rel = (diff / (b.abs() + 1e-12)).max().item()
            mismatches.append(f"{path}: values differ (max_abs={max_abs:.3e}, max_rel={max_rel:.3e})")
        return

    # Dict comparison
    if isinstance(a, dict) and isinstance(b, dict):
        keys_a = set(a.keys())
        keys_b = set(b.keys())
        if keys_a != keys_b:
            mismatches.append(f"{path}: key mismatch {keys_a ^ keys_b}")
        for k in keys_a & keys_b:
            _compare(a[k], b[k], path + (k,), mismatches, rtol=rtol, atol=atol)
        return

    # List/Tuple comparison
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            mismatches.append(f"{path}: length mismatch {len(a)} vs {len(b)}")
        for i, (av, bv) in enumerate(zip(a, b, strict=False)):
            _compare(av, bv, path + (i,), mismatches, rtol=rtol, atol=atol)
        return

    # SimpleNamespace or objects with __dict__
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        _compare(_as_dict(a), _as_dict(b), path, mismatches, rtol=rtol, atol=atol)
        return

    # Fallback equality
    if a != b:
        mismatches.append(f"{path}: value mismatch {a} vs {b}")


def _load_any(path: str) -> Any:
    """Load either a pickle (.pkl) or a torch checkpoint (.pt/.pth)."""
    if path.endswith(".pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if path.endswith(".pt") or path.endswith(".pth"):
        # map_location=cpu keeps comparison deterministic and avoids GPU requirements.
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            # Commonly caused by interrupted/partial writes of large checkpoints.
            raise RuntimeError(f"Failed to torch.load({path}): {e}") from e
    raise ValueError(f"Unsupported file type: {path}")


def compare_pair(left_path: str, right_path: str, rtol=1e-4, atol=1e-6) -> List[str]:
    if not os.path.exists(left_path):
        return [f"{left_path} not found"]
    if not os.path.exists(right_path):
        return [f"{right_path} not found"]

    try:
        left = _load_any(left_path)
    except Exception as e:
        return [str(e)]
    try:
        right = _load_any(right_path)
    except Exception as e:
        return [str(e)]

    mismatches: List[str] = []
    _compare(left, right, path=(), mismatches=mismatches, rtol=rtol, atol=atol)
    if len(mismatches) >= 50:
        mismatches.append("(truncated after 50 mismatches)")
    return mismatches


def main():
    base = "/workspace/fix_input"
    pairs = [
        ("cosmos_preproc.pkl", "openpi_preproc.pkl"),
        ("cosmos_forward.pkl", "openpi_forward.pkl"),
        ("cosmos_outputs.pkl", "openpi_outputs.pkl"),
        # Optional debug dumps (only present if enabled in code)
        ("cosmos_prefix_parts.pkl", "openpi_prefix_parts.pkl"),
        ("cosmos_prefix_debug.pkl", "openpi_prefix_debug.pkl"),
        ("cosmos_rope_debug.pkl", "openpi_rope_debug.pkl"),
        ("cosmos_siglip_output.pkl", "openpi_siglip_output.pkl"),
        ("cosmos_visual_embeddings.pkl", "openpi_visual_embeddings.pkl"),
    ]

    any_mismatch = False
    for l, r in pairs:
        l_path = os.path.join(base, l)
        r_path = os.path.join(base, r)
        mismatches = compare_pair(l_path, r_path)
        if mismatches:
            any_mismatch = True
            print(f"[DIFF] {l} vs {r}:")
            for m in mismatches:
                print(f"  - {m}")
        else:
            print(f"[OK] {l} vs {r} are identical within tolerance")

    if any_mismatch:
        exit(1)
    else:
        exit(0)


if __name__ == "__main__":
    main()


