#!/usr/bin/env python3
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

"""Check UCXX installation, IB/RDMA transport, and UCX device info.

Verifies the entire UCXX stack: Python imports, UCX transport plugins,
RDMA libraries, InfiniBand devices, permissions, and network config.
Uses ``ucx_info`` from the pip ``libucx-cu12`` package for accurate
runtime transport reporting.

Usage::

    python -m cosmos_rl.tools.profiler.check_ucxx          # standard check
    python -m cosmos_rl.tools.profiler.check_ucxx -v       # verbose

Returns 0 on a healthy stack and 1 if any required component is
missing, so it is suitable as a pre-launch SBATCH check.
"""

import ctypes
import ctypes.util
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


# -- helpers -----------------------------------------------------------------


def _get_libucx_pkg_dir():
    """Return the root of the pip libucx package (e.g. site-packages/libucx/)."""
    try:
        import libucx

        return Path(libucx.__file__).parent
    except ImportError:
        return None


def _find_libucx_so(name):
    """Find a UCX shared library from the pip libucx package or system."""
    pkg = _get_libucx_pkg_dir()
    if pkg:
        pkg_lib = pkg / "lib"
        candidates = sorted(pkg_lib.glob(f"{name}.so*"), key=lambda p: len(p.name))
        if candidates:
            return str(candidates[0])
    found = ctypes.util.find_library(name.removeprefix("lib"))
    if found:
        return found
    return None


def _run_ucx_info(*args, _extra_env=None):
    """Run ucx_info from the pip libucx package with correct LD_LIBRARY_PATH."""
    pkg = _get_libucx_pkg_dir()
    if not pkg:
        return None, "libucx package not installed"

    ucx_info = pkg / "bin" / "ucx_info"
    if not ucx_info.exists():
        return None, f"ucx_info not found at {ucx_info}"

    lib_dir = str(pkg / "lib")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")
    env.setdefault("UCX_PLUGIN_DIR", str(pkg / "lib" / "ucx"))
    if _extra_env:
        env.update(_extra_env)

    try:
        proc = subprocess.run(
            [str(ucx_info)] + list(args),
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return proc.stdout + proc.stderr, None
    except FileNotFoundError:
        return None, f"cannot execute {ucx_info}"
    except subprocess.TimeoutExpired:
        return None, "ucx_info timed out"
    except Exception as e:
        return None, str(e)


def _parse_ucx_info_version(output):
    """Parse version, lib path, and configure flags from 'ucx_info -v'."""
    version = config_flags = lib_path = None
    for line in output.splitlines():
        m = re.match(r"#\s*Library version:\s*(.+)", line)
        if m:
            version = m.group(1).strip()
        m = re.match(r"#\s*Library path:\s*(.+)", line)
        if m:
            lib_path = m.group(1).strip()
        m = re.match(r"#\s*Configured with:\s*(.+)", line)
        if m:
            config_flags = m.group(1).strip()
    return version, lib_path, config_flags


def _parse_ucx_info_devices(output):
    """Parse transport/device list from 'ucx_info -d' output.

    Returns list of (component, transport, device) tuples.
    """
    results = []
    component = transport = device = None
    for line in output.splitlines():
        m = re.match(r"#\s+Component:\s*(.+)", line)
        if m:
            component = m.group(1).strip()
        m = re.match(r"#\s+Transport:\s*(.+)", line)
        if m:
            transport = m.group(1).strip()
        m = re.match(r"#\s+Device:\s*(.+)", line)
        if m:
            device = m.group(1).strip()
            if transport:
                results.append((component or "?", transport, device))
    return results


def _get_ib_port_info(device):
    """Read IB device port state/link-layer/rate from sysfs."""
    ports_dir = Path(f"/sys/class/infiniband/{device}/ports")
    if not ports_dir.exists():
        return []
    results = []
    for port_dir in sorted(ports_dir.iterdir()):
        info = {}
        for field in ("state", "link_layer", "rate"):
            f = port_dir / field
            info[field] = f.read_text().strip() if f.exists() else "?"
        results.append((port_dir.name, info))
    return results


def _read_ib_port_counters():
    """Read port_xmit_data and port_rcv_data from all IB devices.

    Returns dict: {device: {"xmit": int, "rcv": int}} or empty if unavailable.
    """
    ib_sys = Path("/sys/class/infiniband")
    if not ib_sys.exists():
        return {}
    counters = {}
    for dev_dir in sorted(ib_sys.iterdir()):
        for port_dir in (
            sorted((dev_dir / "ports").iterdir())
            if (dev_dir / "ports").exists()
            else []
        ):
            counter_dir = port_dir / "counters"
            if not counter_dir.exists():
                continue
            xmit_f = counter_dir / "port_xmit_data"
            rcv_f = counter_dir / "port_rcv_data"
            try:
                xmit = int(xmit_f.read_text().strip()) if xmit_f.exists() else 0
                rcv = int(rcv_f.read_text().strip()) if rcv_f.exists() else 0
                counters[f"{dev_dir.name}:{port_dir.name}"] = {"xmit": xmit, "rcv": rcv}
            except (ValueError, OSError):
                pass
    return counters


def check_multirail():
    """Snapshot IB port counters for multi-rail verification.

    Run this before and after a UCXX transfer batch, then diff the two
    snapshots.  Non-zero deltas on multiple mlx5_* devices confirm
    multi-rail is active.

    Usage::

        python3 scripts/check_ucxx.py --multirail-snapshot before.json
        # ... run transfers ...
        python3 scripts/check_ucxx.py --multirail-snapshot after.json
        python3 scripts/check_ucxx.py --multirail-diff before.json after.json
    """

    counters = _read_ib_port_counters()
    if not counters:
        print("No IB port counters found — multi-rail check unavailable.")
        return 1

    # Filter to mlx5_* devices only (skip virtual/management ports)
    ib_counters = {k: v for k, v in counters.items() if k.startswith("mlx5_")}
    print(f"IB port counters ({len(ib_counters)} ports):")
    for dev, vals in sorted(ib_counters.items()):
        print(f"  {dev:<16s}  xmit={vals['xmit']:>16,d}  rcv={vals['rcv']:>16,d}")
    return ib_counters


def diff_multirail_snapshots(before_path, after_path):
    """Diff two IB port counter snapshots and report which devices transferred data."""
    import json as _json

    with open(before_path) as f:
        before = _json.load(f)
    with open(after_path) as f:
        after = _json.load(f)

    print("=== Multi-Rail Verification ===")
    print(f"  Before: {before_path}")
    print(f"  After:  {after_path}")
    print()

    active_devices = []
    for dev in sorted(set(before.keys()) | set(after.keys())):
        if not dev.startswith("mlx5_"):
            continue
        b = before.get(dev, {"xmit": 0, "rcv": 0})
        a = after.get(dev, {"xmit": 0, "rcv": 0})
        dx = a["xmit"] - b["xmit"]
        dr = a["rcv"] - b["rcv"]
        if dx > 0 or dr > 0:
            active_devices.append(dev)
            # IB counters are in 4-byte units; convert to bytes
            dx_bytes = dx * 4
            dr_bytes = dr * 4
            print(
                f"  {dev:<16s}  xmit_delta={dx_bytes:>16,d} B  rcv_delta={dr_bytes:>16,d} B"
            )

    print()
    if len(active_devices) > 1:
        print(f"MULTI-RAIL CONFIRMED: {len(active_devices)} IB devices active")
    elif len(active_devices) == 1:
        print(f"SINGLE-RAIL: only {active_devices[0]} showed traffic")
    else:
        print("NO TRAFFIC detected on any IB device")

    print(
        f"  Active devices: {', '.join(active_devices) if active_devices else 'none'}"
    )
    return 0 if len(active_devices) > 1 else 1


# -- main check --------------------------------------------------------------


def check_ucxx(verbose=False):
    ok = True

    # 1. UCX version and build config (via ucx_info -v)
    print("=== UCX Libraries ===")
    ver_output, ver_err = _run_ucx_info("-v")
    config_flags = None
    if ver_err:
        print(f"  ucx_info -v failed: {ver_err}")
        for name in ("libucs", "libuct", "libucp"):
            path = _find_libucx_so(name)
            print(f"  {name + ':':<10s} {path or 'NOT FOUND'}")
    else:
        version, lib_path, config_flags = _parse_ucx_info_version(ver_output)
        print(f"  version:    {version or 'unknown'}")
        print(f"  lib path:   {lib_path or 'unknown'}")
        if config_flags:
            has_with_verbs = (
                "--with-verbs" in config_flags and "--without-verbs" not in config_flags
            )
            has_with_rdmacm = (
                "--with-rdmacm" in config_flags
                and "--without-rdmacm" not in config_flags
            )
            if verbose:
                print(f"  configured: {config_flags}")
            if has_with_verbs:
                print("  verbs:      built-in (--with-verbs)")
            else:
                print("  verbs:      plugin-only (core built --without-verbs)")
            if has_with_rdmacm:
                print("  rdmacm:     built-in (--with-rdmacm)")
            else:
                print("  rdmacm:     plugin-only (core built --without-rdmacm)")

    # 2. ucxx Python import
    print("\n=== UCXX Import ===")
    try:
        import ucxx

        print(f"  ucxx: {ucxx.__file__}")
    except ImportError as e:
        print(f"  FAIL: {e}")
        ok = False

    # 3. Transport plugin files (on disk)
    print("\n=== UCT Transport Plugins (on disk) ===")
    pkg = _get_libucx_pkg_dir()
    plugin_dir = pkg / "lib" / "ucx" if pkg else None

    has_ib_file = False
    if plugin_dir and plugin_dir.exists():
        plugins = sorted(p.name for p in plugin_dir.glob("libuct_*.so"))
        labels = {
            "libuct_ib.so": "IB/RDMA/RoCE",
            "libuct_rdmacm.so": "RDMA CM",
            "libuct_cma.so": "cross-memory attach (same-node)",
            "libuct_cuda.so": "CUDA memory",
        }
        for p in plugins:
            label = labels.get(p, "")
            tag = f"  <-- {label}" if label else ""
            print(f"  {p}{tag}")
            if p == "libuct_ib.so":
                has_ib_file = True
        if not has_ib_file:
            print("\n  ** libuct_ib.so NOT FOUND -- RDMA transport unavailable **")
            # Don't set ok=False here; whether this is a real failure
            # depends on whether IB devices are present (checked later).
    else:
        print(f"  plugin dir not found: {plugin_dir}")
        ok = False

    # 3b. dlopen libuct_ib.so to verify dependencies resolve at runtime
    if has_ib_file and plugin_dir:
        ib_so = plugin_dir / "libuct_ib.so"
        try:
            ctypes.CDLL(str(ib_so))
            print("\n  libuct_ib.so dlopen: OK")
        except OSError as e:
            print(f"\n  libuct_ib.so dlopen: FAILED -- {e}")
            print("  (plugin exists but a dependency is missing or ABI-incompatible)")
            ok = False

    # 4. Runtime transports (via ucx_info -d) -- the definitive check
    ib_runtime = False
    print("\n=== UCT Runtime Transports (ucx_info -d) ===")
    dev_output, dev_err = _run_ucx_info("-d")
    if dev_err:
        print(f"  ucx_info -d failed: {dev_err}")
    else:
        entries = _parse_ucx_info_devices(dev_output)
        if entries:
            seen_components = {}
            for comp, transport, device in entries:
                seen_components.setdefault(comp, []).append((transport, device))
            ib_labels = {"ib": "IB/RDMA", "rdmacm": "RDMA-CM"}
            for comp in sorted(seen_components):
                devs = seen_components[comp]
                tag = f"  <-- {ib_labels[comp]}" if comp in ib_labels else ""
                if verbose:
                    for transport, device in devs:
                        print(f"  {comp:<12s} {transport:<14s} {device}{tag}")
                else:
                    dev_names = [d for _, d in devs]
                    print(f"  {comp:<12s} devices: {', '.join(dev_names)}{tag}")
            if "ib" in seen_components:
                ib_runtime = True
                print("\n  IB transport: LOADED (runtime-verified)")
            elif "rdmacm" in seen_components:
                ib_runtime = True
                print("\n  IB transport: LOADED via rdmacm (runtime-verified)")
            else:
                print("\n  ** IB transport: NOT LOADED at runtime **")
                if has_ib_file:
                    # Don't set ok=False here -- no IB devices means UCX
                    # correctly skips the transport.  The summary will
                    # distinguish "no devices" from "devices present but
                    # transport failed to load".
                    print(
                        "  libuct_ib.so on disk + dlopen OK, but UCX did not load 'ib'."
                    )
                    print(
                        "  (Normal if no IB devices on this host.  On IB nodes this is a FAIL.)"
                    )
                    # Use LD_DEBUG to trace whether UCX even tries to dlopen libuct_ib.so.
                    # UCX's own logging is compiled out (--disable-logging), so we use
                    # the Linux dynamic linker's debug facility instead.
                    print("\n  --- LD_DEBUG trace (dlopen calls) ---")
                    dbg_out, dbg_err = _run_ucx_info(
                        "-d", _extra_env={"LD_DEBUG": "libs", "UCX_LOG_LEVEL": "warn"}
                    )
                    if dbg_err:
                        print(f"  (trace failed: {dbg_err})")
                    elif dbg_out:
                        ib_lines = [
                            line.rstrip()
                            for line in dbg_out.splitlines()
                            if "libuct_ib" in line or "libuct_rdma" in line
                        ]
                        if ib_lines:
                            for line in ib_lines:
                                print(f"  {line}")
                        else:
                            print("  libuct_ib.so was NEVER dlopened by UCX")
                            print("  (UCX does not scan the plugin dir for this file)")
                    print("  --- end trace ---")
        else:
            print("  (no transports returned)")

    # 5. RDMA runtime libraries
    print("\n=== RDMA Libraries ===")
    for lib, required in [("libibverbs.so.1", True), ("librdmacm.so.1", False)]:
        try:
            ctypes.CDLL(lib)
            print(f"  {lib:<24s} OK")
        except OSError:
            status = "MISSING (required)" if required else "MISSING (optional)"
            print(f"  {lib:<24s} {status}")
            if required:
                ok = False

    # 6. IB devices and permissions
    print("\n=== InfiniBand Devices ===")
    ib_dev = Path("/dev/infiniband")
    ib_sys = Path("/sys/class/infiniband")

    if ib_dev.exists():
        uverbs = sorted(ib_dev.glob("uverbs*"))
        print(f"  /dev/infiniband:        {len(uverbs)} uverbs devices")
        if uverbs:
            try:
                fd = os.open(str(uverbs[0]), os.O_RDWR)
                os.close(fd)
                print(f"  {uverbs[0].name}:                 readable (OK)")
            except OSError as e:
                print(f"  {uverbs[0].name}:                 PERMISSION DENIED ({e})")
                ok = False
    else:
        print("  /dev/infiniband:        not found")

    devices = []
    if ib_sys.exists():
        devices = sorted(p.name for p in ib_sys.iterdir())
        if devices:
            print(f"  /sys/class/infiniband:  {', '.join(devices)}")
        else:
            print("  /sys/class/infiniband:  empty")
    else:
        print("  /sys/class/infiniband:  not found")

    # 7. IB port details (verbose)
    if verbose and devices:
        print("\n=== IB Port Details ===")
        for dev in devices:
            ports = _get_ib_port_info(dev)
            for port, info in ports:
                print(
                    f"  {dev}:{port}  {info['state']:<20s}  {info['link_layer']:<12s}  {info['rate']}"
                )

    # 8. Network — use the same _get_local_ip() that UCXX uses at runtime
    print("\n=== Network ===")
    print(f"  hostname:          {socket.getfqdn()}")
    try:
        from rl_gym.common.ucxx.mixins import _get_local_ip

        ip = _get_local_ip()
        print(f"  _get_local_ip():   {ip}  (from rl_gym.common.ucxx.mixins)")
    except ImportError:
        ip = socket.gethostbyname(socket.getfqdn())
        print(f"  _get_local_ip():   {ip}  (fallback — could not import mixins)")

    # 9. UCX env vars
    ucx_vars = sorted((k, v) for k, v in os.environ.items() if k.startswith("UCX_"))
    if ucx_vars or verbose:
        print("\n=== UCX Environment ===")
        if ucx_vars:
            for k, v in ucx_vars:
                print(f"  {k}={v}")
        else:
            print("  (none set)")

    # Summary
    print("\n" + "=" * 50)
    if ok and ib_runtime and devices:
        print("PASS -- UCXX with IB/RDMA: transport loads + devices present")
    elif ok and ib_runtime and not devices:
        print("PASS -- UCXX IB transport loads (no IB devices on this host)")
    elif ok and has_ib_file and not ib_runtime and devices:
        print("FAIL -- libuct_ib.so on disk but IB NOT loaded at runtime")
        print("        Run with -v and check dlopen / ucx_info output above")
        ok = False
    elif ok and has_ib_file and not ib_runtime:
        # Plugin present + dlopen OK, but no IB hardware -> expected, not a failure
        print("OK   -- libuct_ib.so ready (no IB devices on this host)")
    elif not has_ib_file and devices:
        print("FAIL -- IB devices visible but libuct_ib.so missing")
        print("        Rebuild base image: ./scripts/build_docker.sh --base")
        ok = False
    elif not has_ib_file and not devices:
        print("OK   -- no IB plugin or devices (TCP-only, expected for local dev)")
    else:
        print("FAIL -- see issues above")
    print("=" * 50)

    return 0 if ok else 1


if __name__ == "__main__":
    import argparse
    import json as _json

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="IB port details, configure flags, env vars",
    )

    sub = parser.add_subparsers(dest="subcmd")

    snap_parser = sub.add_parser(
        "multirail-snapshot",
        help="Save IB port counter snapshot to a JSON file",
    )
    snap_parser.add_argument("output", help="Output JSON path")

    diff_parser = sub.add_parser(
        "multirail-diff",
        help="Diff two IB port counter snapshots to verify multi-rail",
    )
    diff_parser.add_argument("before", help="Before-snapshot JSON")
    diff_parser.add_argument("after", help="After-snapshot JSON")

    args = parser.parse_args()

    if args.subcmd == "multirail-snapshot":
        result = check_multirail()
        if isinstance(result, dict):
            with open(args.output, "w") as f:
                _json.dump(result, f, indent=2)
            print(f"\nSaved to {args.output}")
            sys.exit(0)
        sys.exit(result)
    elif args.subcmd == "multirail-diff":
        sys.exit(diff_multirail_snapshots(args.before, args.after))
    else:
        sys.exit(check_ucxx(verbose=args.verbose))
