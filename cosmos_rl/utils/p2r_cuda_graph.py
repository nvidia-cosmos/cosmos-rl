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

"""CUDA graph managers for Policy-to-Rollout (P2R) weight synchronisation.

Two classes are provided:

  P2RSendCudaGraphManager  – used by the policy worker (rl_worker.py) to send
                             weight shards to rollout ranks via pre-captured
                             CUDA graphs.

  P2RRecvCudaGraphManager  – used by the rollout control worker
                             (rollout_control.py) to receive weight shards
                             from policy ranks via pre-captured CUDA graphs.
"""

from __future__ import annotations

from queue import Queue
from typing import Callable, Dict, List, NamedTuple, Optional

import torch

from cosmos_rl.utils import constant
from cosmos_rl.utils.logging import logger


class P2RGraphStats(NamedTuple):
    total_bytes: int
    transferred_params_cnt: int
    skipped_params_cnt: int
    transferred_groups_cnt: int
    skipped_groups_cnt: int


# ---------------------------------------------------------------------------
# Send-side manager (Policy Worker)
# ---------------------------------------------------------------------------


class P2RSendCudaGraphManager:
    """Manages pre-allocated send buffers and CUDA graph capture/replay for
    P2R NCCL sends on the policy side.

    Parameters
    ----------
    policy_to_rollout_insts:
        Fully-populated list of ``WeightSyncInstructionsGroup`` objects
        describing which parameters this rank sends to which rollout ranks.
    trainable_params:
        Set of trainable parameter names; used to filter slots when
        ``trainable_only=True``.
    map_w_from_policy_to_rollout:
        ``trainer.map_w_from_policy_to_rollout`` — maps param name to a
        ``torch.Tensor`` (stable view) or a zero-argument ``Callable``
        returning one.
    p2r_collective_manager:
        The ``P2RCollectiveManager`` used to issue NCCL ``send`` calls.
    transfer_dtype:
        Dtype to cast tensors to before sending.
    """

    def __init__(
        self,
        policy_to_rollout_insts: list,
        trainable_params: set,
        map_w_from_policy_to_rollout: dict,
        p2r_collective_manager,
        transfer_dtype: torch.dtype,
        global_rank: int,
    ) -> None:
        self._insts = policy_to_rollout_insts
        self._trainable_params = trainable_params
        self._weight_map = map_w_from_policy_to_rollout
        self._collective = p2r_collective_manager
        self._transfer_dtype = transfer_dtype
        self._global_rank = global_rank

        # Non-default stream required for CUDA graph capture.
        self.stream = torch.cuda.Stream()

        # Keyed by trainable_only bool.
        # Value: List[(CUDAGraph, chunk_slots)]
        # chunk_slots: List[(dest_name, slice_strategy, send_buffer, r_rank, needs_fill)]
        self._graphs: Dict[bool, List] = {}

        # Single shared storage for needs_fill send buffers, sized to the
        # largest max-chunk footprint seen across all trainable_only cases.
        # Both cases replay sequentially (never concurrently), so sharing is safe.
        # Allocated once on first build(); subsequent builds reuse it via the
        # over_storage chunk-split guard.
        self._buffer_storage: Optional[torch.Tensor] = None
        self._buffer_storage_bytes: int = 0

    def is_built(self, trainable_only: bool) -> bool:
        return trainable_only in self._graphs

    @torch.no_grad()
    def build(
        self,
        pre_P2R_collected_tensors: Dict[str, torch.Tensor],
        base_mesh_key: str,
        trainable_only: bool,
    ) -> None:
        """Pre-allocate stable send buffers and capture CUDA graphs.

        When ``COSMOS_P2R_CUDA_GRAPH_CHUNK_MB > 0`` the slot list is split
        into sequential chunks capped at that extra-memory limit.  Each chunk
        becomes its own ``CUDAGraph``.  ``needs_fill=False`` slots (live views
        into parameter storage) are zero-cost and do not count toward the limit.

        Per-chunk ordering: warm-up send → capture send (must match the
        rollout side's warm-up recv → capture recv).

        Slot tuple: ``(dest_name, slice_strategy, send_buffer, r_rank, needs_fill)``
        """
        transfer_dtype = self._transfer_dtype

        # ── First pass: collect metadata; defer shared-storage allocation ──
        # For needs_fill=True, the raw slot stores a temporary contiguous view
        # used only for shape/numel; the real send_buffer is carved below.
        raw_slots: List = []
        for insts_group in self._insts:
            for insts_for_per_param in insts_group.param_instructions:
                dest_name = insts_for_per_param.param_name
                if dest_name not in self._trainable_params and trainable_only:
                    continue
                for inst in insts_for_per_param.instructions:
                    assert inst.policy_rank == self._global_rank
                    r_rank = inst.rollout_rank
                    slice_strategy = inst.slice_strategy

                    raw_entry = self._weight_map.get(dest_name)
                    is_stable_source = (
                        dest_name not in pre_P2R_collected_tensors
                        and not isinstance(raw_entry, Callable)
                    )

                    if is_stable_source:
                        local_view = raw_entry
                    elif dest_name in pre_P2R_collected_tensors:
                        local_view = pre_P2R_collected_tensors[dest_name]
                    else:
                        local_view = raw_entry()  # Callable

                    view = (
                        local_view.to(transfer_dtype)
                        .cosmos_slice(slice_strategy)
                        .contiguous()
                        .cuda()
                    )

                    # needs_fill=False: view shares storage with the live param;
                    # graph replay reads the latest weights with zero extra memory.
                    # needs_fill=True: requires a stable copy via shared storage.
                    needs_fill = not (
                        is_stable_source
                        and view.storage().data_ptr() == local_view.storage().data_ptr()
                    )
                    raw_slots.append(
                        (dest_name, slice_strategy, view, r_rank, needs_fill)
                    )

        # ── Chunk by extra (needs_fill) memory ──────────────────────────────
        # Two split conditions:
        #   1. COSMOS_P2R_CUDA_GRAPH_CHUNK_MB byte limit (if set).
        #   2. Accumulated needs_fill numel would exceed the already-allocated
        #      shared storage — ensures subsequent builds never need reallocation.
        chunk_limit_bytes = constant.COSMOS_P2R_CUDA_GRAPH_CHUNK_MB * 1024 * 1024
        raw_chunks: List[list] = []
        cur_chunk: list = []
        cur_bytes = 0
        for slot in raw_slots:
            _, _, view, _, needs_fill = slot
            slot_bytes = view.numel() * view.element_size() if needs_fill else 0
            over_byte_limit = (
                cur_chunk
                and chunk_limit_bytes > 0
                and cur_bytes + slot_bytes > chunk_limit_bytes
            )
            over_storage = (
                cur_chunk
                and self._buffer_storage_bytes > 0
                and cur_bytes + slot_bytes > self._buffer_storage_bytes
            )
            if over_byte_limit or over_storage:
                raw_chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append(slot)
            cur_bytes += slot_bytes
        if cur_chunk:
            raw_chunks.append(cur_chunk)

        # Allocate shared storage once (first build); subsequent builds reuse it.
        # The storage constraint above ensures every chunk fits within the allocated size.
        max_chunk_numel = max(
            (sum(v.numel() for _, _, v, _, nf in chunk if nf) for chunk in raw_chunks),
            default=0,
        )
        if self._buffer_storage is None:
            self._buffer_storage = torch.empty(
                max_chunk_numel,
                dtype=transfer_dtype,
                device=torch.cuda.current_device(),
            )
            self._buffer_storage_bytes = self._buffer_storage.nbytes
        shared_storage = self._buffer_storage

        # ── Build final slot lists, carving needs_fill buffers from storage ─
        all_slots: List = []
        chunks: List[list] = []
        for raw_chunk in raw_chunks:
            offset = 0
            chunk_slots: list = []
            for dest_name, ss, view, r_rank, needs_fill in raw_chunk:
                if needs_fill:
                    numel = view.numel()
                    send_buffer = shared_storage.narrow(0, offset, numel).view(
                        view.shape
                    )
                    offset += numel
                else:
                    send_buffer = view
                slot = (dest_name, ss, send_buffer, r_rank, needs_fill)
                all_slots.append(slot)
                chunk_slots.append(slot)
            chunks.append(chunk_slots)

        # ── Per-chunk warm-up + capture ──────────────────────────────────────
        torch.cuda.synchronize()
        graph_chunks: List = []
        for chunk_slots in chunks:
            with torch.cuda.stream(self.stream):
                for _, _, send_buf, r_rank, _ in chunk_slots:
                    self._collective.send(base_mesh_key, send_buf, r_rank)
            self.stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(self.stream):
                graph.capture_begin()
                for _, _, send_buf, r_rank, _ in chunk_slots:
                    self._collective.send(base_mesh_key, send_buf, r_rank)
                graph.capture_end()
            graph_chunks.append((graph, chunk_slots))

        self._graphs[trainable_only] = graph_chunks

        total_mb = sum(s[2].numel() * s[2].element_size() for s in all_slots) / (
            1024 * 1024
        )
        shared_mb = (
            shared_storage.numel() * shared_storage.element_size() / (1024 * 1024)
        )
        logger.info(
            f"[Policy] Captured {len(graph_chunks)} P2R CUDA graph(s): "
            f"{len(all_slots)} sends, {total_mb:.1f} MB total, "
            f"{shared_mb:.1f} MB shared storage (max-chunk needs_fill), "
            f"trainable_only={trainable_only}."
        )

    @torch.no_grad()
    def fill_and_replay(
        self,
        trainable_only: bool,
        pre_P2R_collected_tensors: Dict[str, torch.Tensor],
        train_stream: torch.cuda.Stream,
    ) -> P2RGraphStats:
        """Fill ``needs_fill`` buffers with current param data and replay graphs.

        The send stream waits for ``train_stream`` first so training kernels
        complete before the fill+send sequence begins.
        """
        graph_chunks = self._graphs[trainable_only]

        self.stream.wait_stream(train_stream)
        with torch.cuda.stream(self.stream):
            for chunk_graph, chunk_slots in graph_chunks:
                self._fill_buffers(chunk_slots, pre_P2R_collected_tensors)
                chunk_graph.replay()

        total_bytes = sum(
            s[2].numel() * s[2].element_size()
            for _, chunk_slots in graph_chunks
            for s in chunk_slots
        )
        transferred = sum(len(chunk_slots) for _, chunk_slots in graph_chunks)
        return P2RGraphStats(
            total_bytes=total_bytes,
            transferred_params_cnt=transferred,
            skipped_params_cnt=0,
            transferred_groups_cnt=transferred,
            skipped_groups_cnt=0,
        )

    @torch.no_grad()
    def _fill_buffers(
        self,
        send_slots: list,
        pre_P2R_collected_tensors: Dict[str, torch.Tensor],
    ) -> None:
        """Copy current param data into pre-allocated send buffers.

        Slots where ``needs_fill=False`` point into live parameter storage
        and are skipped.
        """
        transfer_dtype = self._transfer_dtype
        for dest_name, slice_strategy, send_buf, _, needs_fill in send_slots:
            if not needs_fill:
                continue
            if dest_name in pre_P2R_collected_tensors:
                local_view = pre_P2R_collected_tensors[dest_name]
            elif isinstance(self._weight_map[dest_name], Callable):
                local_view = self._weight_map[dest_name]()
            else:
                local_view = self._weight_map[dest_name]
            view = (
                local_view.to(transfer_dtype)
                .cosmos_slice(slice_strategy)
                .contiguous()
                .cuda()
            )
            send_buf.copy_(view)


# ---------------------------------------------------------------------------
# Recv-side manager (Rollout Control Worker)
# ---------------------------------------------------------------------------


class P2RRecvCudaGraphManager:
    """Manages pre-allocated recv buffers and CUDA graph capture/replay for
    P2R NCCL recvs on the rollout side.

    Parameters
    ----------
    policy_to_rollout_recv_insts:
        Fully-populated list of ``WeightSyncInstructionsGroup`` objects.
    trainable_params:
        Set of trainable parameter names.
    weight_inplace_view_map:
        Maps param name → current target tensor view in the rollout model.
    p2r_collective_manager:
        The ``P2RCollectiveManager`` used to issue NCCL ``recv`` calls.
    weight_mapper:
        Rollout ``WeightMapper``; used by ``_apply_post_ops`` via
        ``update_tensor_view``.
    parallel_dims:
        Passed through to ``update_tensor_view``.
    transfer_dtype:
        Dtype that incoming weight tensors arrive in.
    """

    def __init__(
        self,
        policy_to_rollout_recv_insts: list,
        trainable_params: set,
        weight_inplace_view_map: dict,
        p2r_collective_manager,
        weight_mapper,
        parallel_dims,
        transfer_dtype: torch.dtype,
        global_rank: int,
    ) -> None:
        self._insts = policy_to_rollout_recv_insts
        self._trainable_params = trainable_params
        self._view_map = weight_inplace_view_map
        self._collective = p2r_collective_manager
        self._weight_mapper = weight_mapper
        self._parallel_dims = parallel_dims
        self._transfer_dtype = transfer_dtype
        self._global_rank = global_rank

        # Non-default stream required for CUDA graph capture.
        self.stream = torch.cuda.Stream()

        # Keyed by trainable_only bool.
        # Value: (graph_chunks, all_post_ops, all_check_info)
        self._graphs: Dict[bool, tuple] = {}

        # Single shared storage for not-inplace recv buffers, sized to the
        # largest max-chunk footprint seen across all trainable_only cases.
        # Both cases replay sequentially (never concurrently), so sharing is safe.
        # Allocated once on first build(); subsequent builds reuse it via the
        # over_storage chunk-split guard.
        self._buffer_storage: Optional[torch.Tensor] = None
        self._buffer_storage_bytes: int = 0

    def is_built(self, trainable_only: bool) -> bool:
        return trainable_only in self._graphs

    @torch.no_grad()
    def build(self, base_mesh_key: str, trainable_only: bool) -> None:
        """Pre-allocate stable recv buffers and capture CUDA graphs.

        Mirrors the send side exactly: per-chunk ordering is warm-up recv →
        capture recv, matching the policy side's warm-up send → capture send.

        Slot tuple:
        ``(param_name, slice_strategy, recv_buffer, p_rank, is_inplace, target_view)``

        ``is_inplace=True``  → ``recv_buffer`` IS ``target_view``; direct recv.
        ``is_inplace=False`` → ``recv_buffer`` is a carved view from shared storage;
                               data is copied to ``target_view`` in ``_apply_post_ops``.
        """
        target_dtype = self._transfer_dtype

        # ── First pass: collect metadata ────────────────────────────────────
        raw_slots: List = []
        transferred_params_cnt = 0
        transferred_groups_cnt = 0
        for insts_group in self._insts:
            group_params = 0
            for insts_for_per_param in insts_group.param_instructions:
                inst_dest_name = insts_for_per_param.param_name
                if inst_dest_name not in self._trainable_params and trainable_only:
                    continue

                target_tensor = self._view_map[inst_dest_name]
                if isinstance(target_tensor, torch.distributed.tensor.DTensor):
                    target_tensor = target_tensor.to_local()

                for inst in insts_for_per_param.instructions:
                    assert inst.rollout_rank == self._global_rank
                    p_rank = inst.policy_rank
                    slice_strategy = inst.slice_strategy
                    underlying_view = target_tensor.cosmos_slice(slice_strategy)

                    is_inplace = (
                        underlying_view.is_cuda
                        and underlying_view.is_contiguous()
                        and underlying_view.dtype == target_dtype
                    )
                    raw_slots.append(
                        (
                            inst_dest_name,
                            slice_strategy,
                            None,
                            p_rank,
                            is_inplace,
                            underlying_view,
                        )
                    )
                group_params += 1
            if group_params > 0:
                transferred_params_cnt += group_params
                # Mirror the groups/split counting from the original recv path:
                # if the unsplit name differs from the param name the group
                # represents one original (pre-split) parameter → one group.
                if (
                    self._weight_mapper.get_unsplited_weight_name(
                        insts_group.param_instructions[0].param_name
                    )
                    != insts_group.param_instructions[0].param_name
                ):
                    transferred_groups_cnt += 1
                else:
                    transferred_groups_cnt += group_params

        # ── Chunk by extra (not-inplace) memory ─────────────────────────────
        # Two split conditions:
        #   1. COSMOS_P2R_CUDA_GRAPH_CHUNK_MB byte limit (if set).
        #   2. Accumulated not-inplace numel would exceed the already-allocated
        #      shared storage — ensures subsequent builds never need reallocation.
        chunk_limit_bytes = constant.COSMOS_P2R_CUDA_GRAPH_CHUNK_MB * 1024 * 1024
        raw_chunks: List[list] = []
        cur_chunk: list = []
        cur_bytes = 0
        for slot in raw_slots:
            _, _, _, _, is_inplace, target_view = slot
            slot_bytes = (
                target_view.numel() * target_view.element_size()
                if not is_inplace
                else 0
            )
            over_byte_limit = (
                cur_chunk
                and chunk_limit_bytes > 0
                and cur_bytes + slot_bytes > chunk_limit_bytes
            )
            over_storage = (
                cur_chunk
                and self._buffer_storage_bytes > 0
                and cur_bytes + slot_bytes > self._buffer_storage_bytes
            )
            if over_byte_limit or over_storage:
                raw_chunks.append(cur_chunk)
                cur_chunk = []
                cur_bytes = 0
            cur_chunk.append(slot)
            cur_bytes += slot_bytes
        if cur_chunk:
            raw_chunks.append(cur_chunk)

        # Allocate shared storage once (first build); subsequent builds reuse it.
        # The storage constraint above ensures every chunk fits within the allocated size.
        max_chunk_numel = max(
            (
                sum(tv.numel() for _, _, _, _, ip, tv in chunk if not ip)
                for chunk in raw_chunks
            ),
            default=0,
        )
        if self._buffer_storage is None:
            self._buffer_storage = torch.empty(
                max_chunk_numel,
                dtype=target_dtype,
                device=torch.cuda.current_device(),
            )
            self._buffer_storage_bytes = self._buffer_storage.nbytes
        shared_storage = self._buffer_storage

        # ── Build final slot lists ───────────────────────────────────────────
        all_slots: List = []
        chunks: List[list] = []
        all_post_ops: List = []
        all_check_info: List = []
        for raw_chunk in raw_chunks:
            offset = 0
            chunk_slots: list = []
            for pname, ss, _, p_rank, is_inplace, target_view in raw_chunk:
                if is_inplace:
                    recv_buffer = target_view
                else:
                    numel = target_view.numel()
                    recv_buffer = shared_storage.narrow(0, offset, numel).view(
                        target_view.shape
                    )
                    offset += numel
                    all_post_ops.append((target_view, recv_buffer, pname))
                all_check_info.append((target_view, recv_buffer, pname))
                slot = (pname, ss, recv_buffer, p_rank, is_inplace, target_view)
                all_slots.append(slot)
                chunk_slots.append(slot)
            chunks.append(chunk_slots)

        # ── Per-chunk warm-up + capture ──────────────────────────────────────
        # inference_mode allows inplace writes into inference tensors (e.g. FSDP
        # param views created under torch.inference_mode()).
        torch.cuda.synchronize()
        graph_chunks: List = []
        with torch.inference_mode():
            for chunk_slots in chunks:
                with torch.cuda.stream(self.stream):
                    for _, _, recv_buf, p_rank, _, _ in chunk_slots:
                        self._collective.recv(base_mesh_key, recv_buf, p_rank)
                self.stream.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(self.stream):
                    graph.capture_begin()
                    for _, _, recv_buf, p_rank, _, _ in chunk_slots:
                        self._collective.recv(base_mesh_key, recv_buf, p_rank)
                    graph.capture_end()

                graph_chunks.append(
                    (
                        graph,
                        [
                            (recv_buf, p_rank)
                            for _, _, recv_buf, p_rank, _, _ in chunk_slots
                        ],
                    )
                )

        self._graphs[trainable_only] = (
            graph_chunks,
            all_post_ops,
            all_check_info,
            transferred_params_cnt,
            transferred_groups_cnt,
        )

        total_mb = sum(s[2].numel() * s[2].element_size() for s in all_slots) / (
            1024 * 1024
        )
        shared_mb = (
            shared_storage.numel() * shared_storage.element_size() / (1024 * 1024)
        )
        logger.info(
            f"[Rollout] Captured {len(graph_chunks)} P2R recv CUDA graph(s): "
            f"{len(all_slots)} recvs, {total_mb:.1f} MB total, "
            f"{shared_mb:.1f} MB shared storage (max-chunk not-inplace), "
            f"trainable_only={trainable_only}."
        )

    @torch.no_grad()
    def replay(
        self,
        trainable_only: bool,
        inference_stream: torch.cuda.Stream,
        do_weight_sync_check: bool,
        temp_recv_tensor_queue: Queue,
    ) -> P2RGraphStats:
        """Replay recv CUDA graphs and apply not-inplace post-ops.

        Flow:
        1. If ``do_weight_sync_check``: clone target views and zero recv bufs
           on ``inference_stream`` (before switching streams).
        2. Wait for ``inference_stream``, replay all chunk graphs on
           ``self.stream``, apply post-ops, clear queue.
        3. Record completion event; signal ``inference_stream`` to wait.
        4. If ``do_weight_sync_check``: compare clones vs updated params
           CPU-side; raises ``ValueError`` on mismatch.
        """
        (
            graph_chunks,
            all_post_ops,
            all_check_info,
            transferred_params_cnt,
            transferred_groups_cnt,
        ) = self._graphs[trainable_only]

        cloned_check_tensors: List[torch.Tensor] = []
        with torch.cuda.stream(inference_stream):
            if do_weight_sync_check:
                for target_view, recv_buf, _ in all_check_info:
                    cloned_check_tensors.append(target_view.clone().cpu())
                    target_view.zero_()

        if do_weight_sync_check:
            inference_stream.synchronize()
            for target_view, _, pname in all_check_info:
                if target_view.any():
                    raise ValueError(
                        f"Weight sync check: target_view for {pname} is not all-zero before CUDA graph replay."
                    )

        self.stream.wait_stream(inference_stream)
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for chunk_graph, _ in graph_chunks:
                chunk_graph.replay()
            self._apply_post_ops(all_post_ops)
            temp_recv_tensor_queue.queue.clear()

        recv_finished = torch.cuda.Event()
        recv_finished.record(self.stream)
        inference_stream.wait_event(recv_finished)

        if do_weight_sync_check:
            target_dtype = self._transfer_dtype
            checked = 0
            for (target_view, _, pname), cloned in zip(
                all_check_info, cloned_check_tensors
            ):
                cloned = cloned.to(target_dtype).to(cloned.dtype)
                # logger.info(
                #     f"[Rollout] Checking weights for {pname} in CUDA graph P2R path..."
                # )
                if not torch.allclose(cloned, target_view.cpu()):
                    raise ValueError(
                        f"Weight sync check failed for {pname} in CUDA graph P2R path."
                    )
                checked += 1
            logger.info(
                f"[Rollout] CUDA graph P2R weight sync check passed: {checked} tensor(s) verified."
            )

        total_bytes = sum(
            rb.numel() * rb.element_size()
            for _, chunk_recv_ops in graph_chunks
            for rb, _ in chunk_recv_ops
        )
        return P2RGraphStats(
            total_bytes=total_bytes,
            transferred_params_cnt=transferred_params_cnt,
            skipped_params_cnt=0,
            transferred_groups_cnt=transferred_groups_cnt,
            skipped_groups_cnt=0,
        )

    @torch.no_grad()
    def _apply_post_ops(self, all_post_ops: list) -> None:
        """Copy not-inplace recv buffers into model params.

        Must be called inside ``torch.cuda.stream(self.stream)`` after graph
        replay.  Quantization is not supported in the CUDA graph path and is
        handled by the original recv path instead.
        """
        for target_view, recv_buf, param_name in all_post_ops:
            self._weight_mapper.update_tensor_view(
                target_view, recv_buf, param_name, parallel_dims=self._parallel_dims
            )
