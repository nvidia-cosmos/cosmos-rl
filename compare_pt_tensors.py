#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import math
import os
from typing import Any

import torch


@dataclasses.dataclass(frozen=True)
class DiffItem:
    path: str
    message: str
    a: str | None = None
    b: str | None = None


@dataclasses.dataclass(frozen=True)
class EqualAfterCastItem:
    path: str
    message: str


def _fmt_path(path: str) -> str:
    return path or "<root>"

def _summarize(x: Any, *, tensor_max_elems: int) -> str:
    if torch.is_tensor(x):
        t = x.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        base = f"tensor(shape={tuple(t.shape)}, dtype={t.dtype})"
        n = int(t.numel())
        if n == 0:
            return base + " []"
        if n <= tensor_max_elems:
            flat = t.flatten()
            vals = flat.tolist()
            return base + f" {vals!r}"
        flat = t.flatten()[:tensor_max_elems]
        return base + f" head({tensor_max_elems})={flat.tolist()!r}"
    if isinstance(x, dict):
        return f"dict(keys={len(x)})"
    if isinstance(x, (list, tuple)):
        return f"{type(x).__name__}(len={len(x)})"
    return repr(x)


def _cmp(
    a: Any,
    b: Any,
    path: str,
    *,
    name_a: str,
    name_b: str,
    rtol: float,
    atol: float,
    nan_eq: bool,
    tensor_max_elems: int,
    equal: list[str],
    equal_after_cast: list[EqualAfterCastItem],
    not_equal: list[DiffItem],
) -> None:
    p = _fmt_path(path)

    if a is None or b is None:
        if a is b:
            equal.append(p)
        else:
            not_equal.append(DiffItem(p, "one is None", a=repr(a), b=repr(b)))
        return

    if torch.is_tensor(a) or torch.is_tensor(b):
        if not (torch.is_tensor(a) and torch.is_tensor(b)):
            not_equal.append(
                DiffItem(
                    p,
                    "type mismatch (tensor vs non-tensor)",
                    a=f"{name_a}: {type(a).__name__}",
                    b=f"{name_b}: {type(b).__name__}",
                )
            )
            return
        if a.shape != b.shape:
            not_equal.append(
                DiffItem(
                    p,
                    "shape mismatch",
                    a=f"{name_a}: {tuple(a.shape)} {a.dtype}",
                    b=f"{name_b}: {tuple(b.shape)} {b.dtype}",
                )
            )
            return
        aa = a.detach()
        bb = b.detach()
        if aa.device.type != "cpu":
            aa = aa.cpu()
        if bb.device.type != "cpu":
            bb = bb.cpu()

        dtype_mismatch = aa.dtype != bb.dtype
        if dtype_mismatch:
            # Compare values by casting A to B's dtype (treat B as reference / openpi dtype).
            try:
                aa_cmp = aa.to(dtype=bb.dtype)
            except Exception as e:
                not_equal.append(
                    DiffItem(
                        p,
                        f"dtype mismatch and cast {name_a} -> {name_b} failed: {e}",
                        a=f"{name_a}: {aa.dtype}",
                        b=f"{name_b}: {bb.dtype}",
                    )
                )
                return
            bb_cmp = bb
        else:
            aa_cmp = aa
            bb_cmp = bb

        if aa_cmp.dtype.is_floating_point or aa_cmp.dtype.is_complex:
            fa = aa_cmp.to(torch.float64) if aa_cmp.dtype.is_floating_point else aa_cmp
            fb = bb_cmp.to(torch.float64) if bb_cmp.dtype.is_floating_point else bb_cmp
            ok = torch.allclose(fa, fb, rtol=rtol, atol=atol, equal_nan=nan_eq)
            if ok:
                if dtype_mismatch:
                    equal_after_cast.append(
                        EqualAfterCastItem(
                            p,
                            f"{name_a}.dtype={aa.dtype} cast_to {name_b}.dtype={bb.dtype} -> values equal",
                        )
                    )
                else:
                    equal.append(p)
                return
            d = (fa - fb).abs()
            max_abs = float(d.max().item()) if d.numel() else 0.0
            mean_abs = float(d.mean().item()) if d.numel() else 0.0
            not_equal.append(
                DiffItem(
                    p,
                    (
                        (f"dtype mismatch ({name_a}={aa.dtype} -> {name_b}={bb.dtype}); " if dtype_mismatch else "")
                        + f"mean_abs={mean_abs:.6g} max_abs={max_abs:.6g}"
                    ),
                    a=f"{name_a}: {_summarize(aa_cmp, tensor_max_elems=tensor_max_elems)}",
                    b=f"{name_b}: {_summarize(bb_cmp, tensor_max_elems=tensor_max_elems)}",
                )
            )
            return

        if torch.equal(aa_cmp, bb_cmp):
            if dtype_mismatch:
                equal_after_cast.append(
                    EqualAfterCastItem(
                        p,
                        f"{name_a}.dtype={aa.dtype} cast_to {name_b}.dtype={bb.dtype} -> values equal",
                    )
                )
            else:
                equal.append(p)
            return
        not_equal.append(
            DiffItem(
                p,
                (
                    (f"dtype mismatch ({name_a}={aa.dtype} -> {name_b}={bb.dtype}); " if dtype_mismatch else "")
                    + "tensor values differ (exact)"
                ),
                a=f"{name_a}: {_summarize(aa_cmp, tensor_max_elems=tensor_max_elems)}",
                b=f"{name_b}: {_summarize(bb_cmp, tensor_max_elems=tensor_max_elems)}",
            )
        )
        return

    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            not_equal.append(
                DiffItem(
                    p,
                    "type mismatch (dict vs non-dict)",
                    a=f"{name_a}: {type(a).__name__}",
                    b=f"{name_b}: {type(b).__name__}",
                )
            )
            return
        ka, kb = set(a.keys()), set(b.keys())
        for k in sorted(kb - ka):
            not_equal.append(DiffItem(f"{p}.{k}" if path else str(k), f"missing key in {name_a}"))
        for k in sorted(ka - kb):
            not_equal.append(DiffItem(f"{p}.{k}" if path else str(k), f"missing key in {name_b}"))
        for k in sorted(ka & kb):
            _cmp(
                a[k],
                b[k],
                f"{path}.{k}" if path else str(k),
                name_a=name_a,
                name_b=name_b,
                rtol=rtol,
                atol=atol,
                nan_eq=nan_eq,
                tensor_max_elems=tensor_max_elems,
                equal=equal,
                equal_after_cast=equal_after_cast,
                not_equal=not_equal,
            )
        return

    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not (isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))):
            not_equal.append(
                DiffItem(
                    p,
                    "type mismatch (sequence)",
                    a=f"{name_a}: {type(a).__name__}",
                    b=f"{name_b}: {type(b).__name__}",
                )
            )
            return
        if len(a) != len(b):
            not_equal.append(
                DiffItem(p, "len mismatch", a=f"{name_a}: {len(a)}", b=f"{name_b}: {len(b)}")
            )
        for i, (xa, xb) in enumerate(zip(a, b)):
            _cmp(
                xa,
                xb,
                f"{path}[{i}]" if path else f"[{i}]",
                name_a=name_a,
                name_b=name_b,
                rtol=rtol,
                atol=atol,
                nan_eq=nan_eq,
                tensor_max_elems=tensor_max_elems,
                equal=equal,
                equal_after_cast=equal_after_cast,
                not_equal=not_equal,
            )
        return

    # Scalars / fallbacks
    if isinstance(a, float) or isinstance(b, float):
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            not_equal.append(
                DiffItem(
                    p,
                    "type mismatch (float vs non-number)",
                    a=f"{name_a}: {type(a).__name__}",
                    b=f"{name_b}: {type(b).__name__}",
                )
            )
            return
        if math.isnan(float(a)) and math.isnan(float(b)) and nan_eq:
            equal.append(p)
            return
        if math.isclose(float(a), float(b), rel_tol=rtol, abs_tol=atol):
            equal.append(p)
            return
        not_equal.append(
            DiffItem(
                p,
                f"float mismatch (rtol={rtol} atol={atol})",
                a=f"{name_a}: {a!r}",
                b=f"{name_b}: {b!r}",
            )
        )
        return

    if a == b:
        equal.append(p)
        return
    not_equal.append(DiffItem(p, "value mismatch", a=f"{name_a}: {a!r}", b=f"{name_b}: {b!r}"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="~/siglip_io.cosmos.pt", help="Path to A .pt (default: %(default)s)")
    ap.add_argument("--b", default="~/siglip_io.openpi.pt", help="Path to B .pt (default: %(default)s)")
    ap.add_argument("--a-name", default="", help="Label for --a in output (default: basename of --a)")
    ap.add_argument("--b-name", default="", help="Label for --b in output (default: basename of --b)")
    ap.add_argument("--rtol", type=float, default=0.0)
    ap.add_argument("--atol", type=float, default=0.0)
    ap.add_argument("--nan-eq", action="store_true", help="Treat NaNs as equal for floats")
    ap.add_argument("--max-equal", type=int, default=50, help="Max equal paths to print")
    ap.add_argument("--max-not-equal", type=int, default=50, help="Max not-equal items to print")
    ap.add_argument("--tensor-max-elems", type=int, default=16, help="Max tensor elements to print in summaries")
    args = ap.parse_args()

    a_path = os.path.expanduser(args.a)
    b_path = os.path.expanduser(args.b)
    name_a = args.a_name.strip() or os.path.basename(a_path)
    name_b = args.b_name.strip() or os.path.basename(b_path)
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)

    equal: list[str] = []
    equal_after_cast: list[EqualAfterCastItem] = []
    not_equal: list[DiffItem] = []
    _cmp(
        a,
        b,
        "",
        name_a=name_a,
        name_b=name_b,
        rtol=args.rtol,
        atol=args.atol,
        nan_eq=args.nan_eq,
        tensor_max_elems=max(0, int(args.tensor_max_elems)),
        equal=equal,
        equal_after_cast=equal_after_cast,
        not_equal=not_equal,
    )

    if equal:
        print(f"EQUAL: {len(equal)}")
        for p in equal[: max(0, int(args.max_equal))]:
            print(" +", p)
        if len(equal) > int(args.max_equal):
            print(f" + ... ({len(equal) - int(args.max_equal)} more)")
    else:
        print("EQUAL: 0")

    if equal_after_cast:
        print(f"EQUAL AFTER CAST: {len(equal_after_cast)} (cast {name_a} -> {name_b} dtype)")
        for it in equal_after_cast[: max(0, int(args.max_equal))]:
            print(" ~", it.path)
            print("   ", it.message)
        if len(equal_after_cast) > int(args.max_equal):
            print(f" ~ ... ({len(equal_after_cast) - int(args.max_equal)} more)")
    else:
        print("EQUAL AFTER CAST: 0")

    if not_equal:
        print(f"NOT EQUAL: {len(not_equal)}")
        for it in not_equal[: max(0, int(args.max_not_equal))]:
            print(" -", it.path + ":", it.message)
            if it.a is not None or it.b is not None:
                if it.a is not None:
                    print("   ", it.a)
                if it.b is not None:
                    print("   ", it.b)
        if len(not_equal) > int(args.max_not_equal):
            print(f" - ... ({len(not_equal) - int(args.max_not_equal)} more)")
        return 1

    print("OK: all compared leaves are equal under the given rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


