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

"""Self-describing header prefixed to every NCCL payload buffer.

A cached 2-rank communicator is an ORDERED stream: the k-th ``nccl_send`` on
it is matched by the k-th ``nccl_recv``, and NCCL carries no tag with which a
receiver could tell which transfer it just got.  So if the two ends ever
disagree about how many transfers have crossed the pair -- one extra send, one
dropped recv, two sends launched out of accept order -- every subsequent
payload lands in the *previous* transfer's buffer, and nothing in the data
plane says so.

For a producer whose payloads all share one fixed schema that mispairing is
invisible but harmless (same layout, same size, wrong episode).  For a producer
with a **per-payload schema** -- rl-gym packs a pickled structure blob whose
length varies per trajectory -- the receiver slices a foreign buffer at its own
payload's offsets and hands the caller decoded garbage, which surfaces much
later as an ``UnpicklingError`` deep in the trainer.

This header makes that impossible to miss.  Every transfer carries, ahead of
the schema-defined region, the payload size and a digest of the ``transfer_id``
it belongs to.  The receiver checks both before it unpacks anything, so a
mispairing becomes an immediate, accurate, *attributable* error on the pair
that desynced instead of silent corruption downstream.

Wire format (little-endian, :data:`HEADER_NBYTES` bytes)::

    offset  size  field
    0       4     magic (:data:`_MAGIC`)
    4       4     version (:data:`HEADER_VERSION`)
    8       8     payload_nbytes -- the schema entry size that follows
    16      8     transfer_key -- 64-bit digest of the transfer id
    24      8     reserved (zero)

Both ends of the transport live in this package, so the format is internal:
a producer and a consumer are always deployed from the same cosmos-rl build.
"""

from __future__ import annotations

import hashlib
import struct

__all__ = [
    "HEADER_NBYTES",
    "HEADER_VERSION",
    "PayloadHeaderMismatch",
    "build_header",
    "parse_header",
    "transfer_key",
    "verify_header",
]

#: Size of the header region prefixed to every payload buffer.  8-byte aligned
#: so the schema region that follows keeps the alignment the packer assumes.
HEADER_NBYTES = 32

#: Bumped only if the layout above changes.  A mismatch is reported rather than
#: tolerated: the two ends ship together, so it means a corrupt buffer.
HEADER_VERSION = 1

_MAGIC = 0x4E43_5031  # 'NCP1'
_STRUCT = struct.Struct("<IIQQQ")


class PayloadHeaderMismatch(RuntimeError):
    """A received buffer's header does not describe the expected transfer.

    Raised by :func:`verify_header`.  Always means the pair's ordered stream
    desynced (or the buffer is corrupt); the caller must tear the pair's
    communicator down so both halves rebuild rather than keep decoding.
    """


def transfer_key(transfer_id: str) -> int:
    """Stable 64-bit digest of ``transfer_id``.

    ``hash()`` is salted per process and would differ on the two ends, so use
    an explicit digest.
    """
    digest = hashlib.blake2b(transfer_id.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def build_header(*, transfer_id: str, payload_nbytes: int) -> bytes:
    """Serialize the header a producer prefixes to ``transfer_id``'s payload."""
    return _STRUCT.pack(
        _MAGIC,
        HEADER_VERSION,
        int(payload_nbytes),
        transfer_key(transfer_id),
        0,
    )


def parse_header(raw: bytes) -> dict:
    """Decode a header buffer into its fields.

    Raises:
        PayloadHeaderMismatch: if ``raw`` is too short to be a header.
    """
    if len(raw) < HEADER_NBYTES:
        raise PayloadHeaderMismatch(
            f"payload header truncated: got {len(raw)} bytes, need {HEADER_NBYTES}"
        )
    magic, version, payload_nbytes, key, _reserved = _STRUCT.unpack(
        bytes(raw[:HEADER_NBYTES])
    )
    return {
        "magic": magic,
        "version": version,
        "payload_nbytes": payload_nbytes,
        "transfer_key": key,
    }


def verify_header(raw: bytes, *, transfer_id: str, payload_nbytes: int) -> None:
    """Check that ``raw`` heads the payload of ``transfer_id``.

    Args:
        raw: The first :data:`HEADER_NBYTES` bytes of the received buffer.
        transfer_id: The transfer this buffer was allocated and posted for.
        payload_nbytes: The schema entry size the receiver allocated for.

    Raises:
        PayloadHeaderMismatch: on any disagreement -- a foreign magic (the
            buffer never held a header, e.g. it was overwritten by a
            differently-sized payload), a foreign transfer key (the pair's
            stream is off by one or more transfers), or a size the receiver's
            schema does not describe.
    """
    fields = parse_header(raw)
    if fields["magic"] != _MAGIC:
        raise PayloadHeaderMismatch(
            f"payload for {transfer_id} has no valid header "
            f"(magic=0x{fields['magic']:08x}, expected 0x{_MAGIC:08x}); the "
            "pair's send/recv stream is desynced or the buffer is corrupt"
        )
    if fields["version"] != HEADER_VERSION:
        raise PayloadHeaderMismatch(
            f"payload for {transfer_id} carries header version "
            f"{fields['version']}, expected {HEADER_VERSION}"
        )
    expected_key = transfer_key(transfer_id)
    if fields["transfer_key"] != expected_key:
        raise PayloadHeaderMismatch(
            f"payload delivered for {transfer_id} belongs to a DIFFERENT "
            f"transfer (key=0x{fields['transfer_key']:016x}, expected "
            f"0x{expected_key:016x}); the pair's send/recv stream is desynced"
        )
    if fields["payload_nbytes"] != int(payload_nbytes):
        raise PayloadHeaderMismatch(
            f"payload for {transfer_id} declares {fields['payload_nbytes']} "
            f"bytes but the receiver's schema describes {int(payload_nbytes)}"
        )
