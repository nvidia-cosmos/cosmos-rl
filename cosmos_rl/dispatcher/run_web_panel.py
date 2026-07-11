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

import os
import argparse
import signal
import socket
import time
import uvicorn
import toml
from fastapi import FastAPI
from contextlib import asynccontextmanager
from torch.utils.data import Dataset
import asyncio
import threading


from fastapi.responses import HTMLResponse, JSONResponse
from typing import Dict, List, Optional, Callable, Union, Iterable
from cosmos_rl.dispatcher.controller import Controller
from cosmos_rl.dispatcher.command import StopCommand
from cosmos_rl.dispatcher.status import should_broadcast_stop
import cosmos_rl.utils.constant as constant
from cosmos_rl.dispatcher.protocol import MESH_NAMES
from cosmos_rl.dispatcher.replica import Atom, Replica
from cosmos_rl.dispatcher.protocol import (
    RegisterRequest,
    ErrorResponse,
    RolloutRequest,
    ValidationReportRequest,
    HandshakeInitiatorRequest,
    HandshakeAcceptorRequest,
    UnregisterRequest,
    TrainAckRequest,
    HeartbeatRequest,
    SetProfileRequest,
    SetTracePathRequest,
    NcclErrRequest,
    NcclStoreClearRequest,
    GetShardSendRecvInstsRequest,
    IpcInfoRequest,
    QueryIpcInfoRequest,
    ResumeInfoRequest,
    Role,
)
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.utils.network_util import bind_available_port
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.constant import (
    COSMOS_HEARTBEAT_TIMEOUT,
    COSMOS_ROLLOUT_SCAN_INTERVAL,
    COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS,
)
from cosmos_rl.utils.api_suffix import (
    COSMOS_API_PANEL_SUFFIX,
    COSMOS_API_STATUS_SUFFIX,
    COSMOS_API_META_SUFFIX,
    COSMOS_API_REGISTER_SUFFIX,
    COSMOS_API_SET_PROFILE_SUFFIX,
    COSMOS_API_SET_TRACE_PATH_SUFFIX,
    COSMOS_API_UNREGISTER_SUFFIX,
    COSMOS_API_HEARTBEAT_SUFFIX,
    COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX,
    COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX,
    COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX,
    COSMOS_API_NCCL_COMM_ERROR_SUFFIX,
    COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX,
    COSMOS_API_NEXT_PROMPT_SUFFIX,
    COSMOS_API_ROLLOUT_SUFFIX,
    COSMOS_API_VALIDATION_REPORT_SUFFIX,
    COSMOS_API_POLICY_TRAIN_ACK_SUFFIX,
    COSMOS_API_POLICY_SHARD_INFOS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX,
    COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX,
    COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX,
    COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX,
    COSMOS_API_IPC_INFO_SUFFIX,
    COSMOS_API_QUERY_IPC_INFO_SUFFIX,
    COSMOS_API_RESUME_INFO_SUFFIX,
)
from cosmos_rl.dispatcher.data.packer.base import BaseDataPacker, worker_entry_parser
from cosmos_rl.utils.payload import extract_rollouts
from fastapi.responses import Response
from fastapi import Request
from concurrent.futures import ThreadPoolExecutor


def create_error_response(
    code: int, message: str, status_code: Optional[int] = None
) -> JSONResponse:
    if status_code is None:
        status_code = code // 100
    return JSONResponse(
        ErrorResponse(message=message, code=code).model_dump(), status_code=status_code
    )


controller = Controller()
server = None
# Set True on the first successful /register.  Guards heartbeat-reap finalize
# against the startup window where replica managers are empty but workers have
# not connected yet (notably SFT controllers where ``not is_rl`` is always true).
_replicas_were_registered = False
# Set True on the first successful POLICY /register.  Unlike the monitor
# thread's old local flag, this cannot miss short-lived policy replicas that
# register, finish, and unregister between scan intervals.
_policy_replicas_were_registered = False


def _all_replicas_gone() -> bool:
    return (
        len(controller.policy_status_manager) == 0
        and len(controller.rollout_status_manager) == 0
        and len(controller.teacher_result_manager) == 0
    )


def _maybe_finalize(reason: str) -> bool:
    """Shut the controller down once every replica is gone.

    This must trigger regardless of *how* a replica left -- a clean HTTP
    /unregister OR a heartbeat-timeout reap. Historically the finalize check
    lived only in the /unregister handler, so a replica that died ungracefully
    (no clean unregister) left the controller running forever, hanging the
    process-flow test until the outer job timeout. Centralizing it here and
    calling it from both paths makes liveness robust to any ungraceful exit.

    The heartbeat-reap path must *not* finalize at startup (empty managers,
    no replica has registered yet).  The /unregister path is safe without that
    guard because it only runs after a replica actually departs.
    """
    if not _all_replicas_gone():
        return False

    if reason == "heartbeat-death reap":
        # All replicas gone after having been live at least once (reaped or
        # unregistered).  Do not require training_finished / is_rl here --
        # that would re-introduce the stranded-controller hang when the last
        # replica dies without a clean HTTP unregister.
        ready = _replicas_were_registered
    else:
        ready = (
            controller.policy_status_manager.training_finished() or not controller.is_rl
        )

    if ready:
        global server
        if server is not None and not server.should_exit:
            data_fetcher = getattr(
                controller.policy_status_manager, "data_fetcher", None
            )
            config = getattr(controller, "config", None)
            train = getattr(config, "train", None)
            train_policy = getattr(train, "train_policy", None)
            train_policy_type = getattr(train_policy, "type", None)
            if (
                train_policy_type != "sft"
                and getattr(data_fetcher, "activated_val_iter", None) is not None
            ):
                logger.error(
                    "[Controller] Finalizing while validation is still active "
                    "(reason=%s). This violates strict final-validation "
                    "completion and should only occur after abnormal worker "
                    "loss.",
                    reason,
                )
            logger.info(
                f"[Controller] All replicas are finished ({reason}); finalizing -- "
                "shutting down controller."
            )
            server.should_exit = True
        return True
    return False


def _await_rollout_checkout(
    controller,
    shutdown_event,
    timeout_s: float = COSMOS_HEARTBEAT_TIMEOUT,
    scan_interval_s: float = COSMOS_ROLLOUT_SCAN_INTERVAL,
) -> bool:
    """Block until every rollout replica has checked out, or shutdown is forced.

    Gates the ``COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS`` self-SIGTERM: the policy
    being gone does not mean the rollout side is done -- replicas still need to
    receive end-of-data, finish the final R2R broadcast (a collective over the
    whole rollout set), post their end signal, and tear down.  SIGTERM-ing the
    HTTP server before that strands stragglers on a dead controller and orphans
    the broadcast peer of the replicas that did check out.

    ``maintain_life_status`` is pumped each iteration so a genuinely crashed
    rollout (one that stopped heartbeating) is reaped and drops out of
    ``all_rollouts_ended`` on its own -- which is why no separate timeout knob
    is needed for that case.  The ``timeout_s`` deadline (default
    ``COSMOS_HEARTBEAT_TIMEOUT``, the same timescale used to declare a replica
    dead) is the backstop for the abnormal wedged-but-still-heartbeating case.

    Returns True if all rollouts checked out cleanly, False if the deadline or
    an external shutdown signal forced shutdown first.
    """
    rsm = controller.rollout_status_manager
    if len(rsm.rollout_replicas) == 0:
        logger.info(
            "[Controller] No rollout replicas registered; rollout checkout gate "
            "has nothing to wait for."
        )
        return True
    deadline = time.monotonic() + timeout_s
    while not rsm.all_rollouts_ended():
        rsm.maintain_life_status(controller.policy_status_manager)
        if time.monotonic() > deadline:
            n_total = len(rsm.rollout_replicas)
            n_ended = sum(1 for r in rsm.rollout_replicas.values() if r.status.ended)
            logger.warning(
                "[ABNORMAL shutdown] Rollout checkout gate timed out after %ss: "
                "only %d/%d rollout replicas posted end.  A rollout likely wedged "
                "while still heartbeating; forcing controller shutdown.",
                timeout_s,
                n_ended,
                n_total,
            )
            return False
        if shutdown_event.wait(timeout=scan_interval_s):
            return False
    logger.info(
        "[Controller] All rollout replicas checked out; proceeding with "
        "coordinated controller shutdown."
    )
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    shutdown_event = threading.Event()
    executor = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

    def monitor_replica_status():
        stop_broadcast_sent = False
        while not shutdown_event.is_set():
            # Run in separate process
            controller.policy_status_manager.maintain_life_status()
            controller.rollout_status_manager.maintain_life_status(
                controller.policy_status_manager
            )
            # A replica reaped via heartbeat timeout is just as "gone" as one
            # that unregistered cleanly -- finalize the controller here too so a
            # dead/ungracefully-exiting replica can never strand it.
            _maybe_finalize("heartbeat-death reap")

            # Authoritative end-of-job stop for the rollouts.  Tracked
            # independently of the opt-in self-SIGTERM below.  A normal
            # non-validation run ends with every policy replica
            # unregistering immediately after its main loop completes, and a
            # crashed policy is reaped by ``maintain_life_status``; either
            # way ``n_policy == 0`` after we have seen policy replicas means
            # no policy will read this run's rollout output again.  This is
            # the correctly-ordered moment to stop rollouts: stopping on
            # ``training_finished()`` instead would fire at dispatch time,
            # before the policy has pulled the final outputs over UCXX.
            n_policy = len(controller.policy_status_manager)
            if should_broadcast_stop(
                n_policy=n_policy,
                had_policy_replicas=_policy_replicas_were_registered,
                stop_broadcast_sent=stop_broadcast_sent,
                validation_enabled=controller.config.validation.enable,
                training_finished=controller.policy_status_manager.training_finished(),
                all_rollouts_ended=controller.rollout_status_manager.all_rollouts_ended(),
            ):
                # Publish STOP over the redis command channel (NCCL-free) so
                # it reaches every rollout via ``consume_command`` -- including
                # a rank wedged on the weight-version gate that never
                # re-fetches and so never observes the prompt-stream
                # ``is_end``.  Validation runs are excluded: they keep the
                # final weight sync and shut down via the existing R2R
                # ``replica_should_stop`` broadcast.
                #
                # The ``training_finished()`` guard distinguishes genuine
                # end-of-job from a transient ``n_policy == 0`` during
                # dynamic policy rescaling (scale-to-zero / rolling restart),
                # where ``current_step < total_steps`` -- exactly the
                # legitimate-transient case that keeps the SIGTERM escalation
                # below opt-in.  ``training_finished()`` stays True once set
                # (``recompute_total_steps`` early-returns), so it remains
                # valid after the policy has unregistered.
                rollout_replicas = list(
                    controller.rollout_status_manager.rollout_replicas.values()
                )
                # Guard the publish: this monitor thread also drives
                # ``maintain_life_status`` (rollout reaping) and the
                # no-policy SIGTERM escalation below.  An unguarded redis
                # error here would kill the thread and take both backstops
                # with it.  On failure we leave ``stop_broadcast_sent``
                # False so the next tick retries.
                try:
                    if rollout_replicas:
                        logger.info(
                            "[Controller] Policy side finished; broadcasting "
                            "STOP to %d rollout replica(s).",
                            len(rollout_replicas),
                        )
                        StopCommand.trigger(
                            rollout_replicas,
                            redis_handler=controller.rollout_status_manager.redis_handler,
                        )
                    stop_broadcast_sent = True
                except Exception:
                    logger.exception(
                        "[Controller] Failed to broadcast STOP to rollouts; "
                        "will retry next scan."
                    )

            # Opt-in escalation of "all policy replicas dead" to a
            # controller-wide shutdown.  Default OFF because
            # cosmos-rl supports dynamic replica scaling -- intentional
            # scale-to-zero (model swap, maintenance) and rolling
            # restart (old replica unregisters before new one
            # registers) both transit ``len(policy_replicas) == 0`` as
            # a legitimate state.  Treating it as fatal there would
            # kill the controller during normal operation and prevent
            # the orchestrator from bringing replicas back.
            #
            # The escalation IS appropriate in deployments without
            # auto-respawn (one trainer process per job, no
            # replacement on death).  Without it, a trainer crash that
            # the heartbeat thread correctly reports leaves the
            # controller idle until the orchestrator's wall-clock
            # timeout, which can burn significant cluster time.  Those
            # deployments set ``COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS=1``
            # to enable the SIGTERM-self path below; FastAPI then runs
            # its lifespan shutdown cleanly and the process exits,
            # freeing the allocation immediately.
            if COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS:
                if n_policy == 0 and _policy_replicas_were_registered:
                    logger.warning(
                        "[Controller] All policy replicas are dead and "
                        "COSMOS_SHUTDOWN_ON_NO_POLICY_REPLICAS is set.  "
                        "Initiating controller shutdown so the scheduling "
                        "layer (SLURM, etc.) can release the job instead "
                        "of waiting for the wall-clock timeout."
                    )
                    # Coordinated exit: 'no policy replicas' does NOT mean the
                    # rollout side is finished.  Keep the HTTP API serving until
                    # every rollout has checked out before self-terminating, so
                    # stragglers don't wedge on a dead controller and the final
                    # R2R broadcast isn't left with an orphaned peer.
                    _await_rollout_checkout(controller, shutdown_event)
                    shutdown_event.set()
                    os.kill(os.getpid(), signal.SIGTERM)
                    break

            if shutdown_event.wait(timeout=COSMOS_ROLLOUT_SCAN_INTERVAL):
                break  # Exit early if shutdown signaled during sleep

    task = loop.run_in_executor(executor, monitor_replica_status)
    yield
    # Signal shutdown
    shutdown_event.set()
    await task


app = FastAPI(lifespan=lifespan)


@app.get(COSMOS_API_PANEL_SUFFIX)
async def panel():
    # HTML template with JavaScript for auto-refresh
    with open(
        os.path.join(
            os.path.dirname(__file__), "config/frontend", "dispatcher_status.html"
        ),
        "r",
        encoding="utf-8",
    ) as file:
        html = file.read()
    return HTMLResponse(html)


"""
API for replica-controller communication
"""


@app.get(COSMOS_API_STATUS_SUFFIX)
async def get_status():
    return {
        "mesh_names": MESH_NAMES,
        "policy_replicas": _serialize_replicas(
            controller.policy_status_manager.policy_replicas
        ),
        "rollout_replicas": _serialize_replicas(
            controller.rollout_status_manager.rollout_replicas
        ),
    }


@app.get(COSMOS_API_META_SUFFIX)
async def meta():
    meta = {
        "config": controller.config,
    }
    return meta


@app.post(COSMOS_API_REGISTER_SUFFIX)
async def register(request: RegisterRequest):
    global _replicas_were_registered, _policy_replicas_were_registered
    try:
        await controller.register(
            Atom.from_register_request(request),
            role=request.role,
        )
        _replicas_were_registered = True
        if request.role == Role.POLICY:
            _policy_replicas_were_registered = True
        return {"message": "Registered"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post(COSMOS_API_UNREGISTER_SUFFIX)
async def unregister(request: UnregisterRequest):
    try:
        await controller.unregister(request.replica_name)
    except Exception as e:
        logger.error(f"[Controller] Unregister failed: {e}")
    finally:
        _maybe_finalize(f"clean unregister of {request.replica_name}")
        return {"message": "Unregistered"}


@app.post(COSMOS_API_SET_PROFILE_SUFFIX)
async def set_profile(request: SetProfileRequest):
    logger.info(f"[Dispatcher] set profile request: {request}")
    msg = await controller.set_profile(request)
    return msg


@app.post(COSMOS_API_SET_TRACE_PATH_SUFFIX)
async def set_trace_path(request: SetTracePathRequest):
    atom = await controller.set_trace_path(
        request.replica_name, request.trace_path, request.global_rank
    )
    if atom is not None:
        return {"message": f"Trace path set for atom: {atom}"}
    else:
        return {"message": "Ignore the trace path request!"}


@app.post(COSMOS_API_HEARTBEAT_SUFFIX)
async def heartbeat(request: HeartbeatRequest):
    # Set the replica timestamp to the current time for heartbeat
    controller.replica_heartbeat(request.replica_name)
    return {"message": "Heartbeat received"}


@app.post(COSMOS_API_POLICY_SHARD_INFOS_SUFFIX)
async def policy_shard_infos(request: Request):
    content_type = request.headers.get("Content-Type")
    if content_type != "application/msgpack":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST,
            "Invalid Content-Type, expected application/msgpack",
        )

    raw_bytes = await request.body()
    await controller.policy_to_rollout_shard_mapper.set_shard_infos_of_policy(
        raw_bytes,
        controller.policy_status_manager.n_atoms_per_replica(),
    )
    return {"message": "Policy shard infos set"}


@app.post(COSMOS_API_ROLLOUT_SHARD_INFOS_SUFFIX)
async def rollout_shard_infos(request: Request):
    content_type = request.headers.get("Content-Type")
    if content_type != "application/msgpack":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST,
            "Invalid Content-Type, expected application/msgpack",
        )

    raw_bytes = await request.body()
    await controller.policy_to_rollout_shard_mapper.set_shard_infos_of_rollout(
        raw_bytes,
        controller.rollout_status_manager.n_atoms_per_replica(),
    )
    return {"message": "Rollout shard infos set"}


@app.post(COSMOS_API_POLICY_SHARD_SEND_INSTS_SUFFIX)
async def policy_shard_send_insts(request: GetShardSendRecvInstsRequest):
    """
    Get the send instructions for policy.
    :return: A list of send instructions for policy.
    """
    logger.debug(
        f"[Dispatcher] Get policy shard send instructions for rank {request.rank}"
    )
    await controller.policy_to_rollout_shard_mapper.scheme_generation_done.wait()
    # Get the send instructions for policy
    send_insts = (
        await controller.policy_to_rollout_shard_mapper.get_send_insts_for_policy(
            request.rank
        )
    )
    # If the send instructions are not found, return an error response
    if send_insts is None:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Policy shard send instructions not found",
        )
    logger.debug(
        f"[Dispatcher] Received policy shard send instructions for rank {request.rank}"
    )
    return Response(content=send_insts, media_type="application/msgpack")


@app.post(COSMOS_API_ROLLOUT_SHARD_RECV_INSTS_SUFFIX)
async def rollout_shard_recv_insts(request: GetShardSendRecvInstsRequest):
    """
    Get the receive instructions for rollout.
    :return: A list of receive instructions for rollout.
    """
    logger.debug(
        f"[Dispatcher] Get rollout shard receive instructions for rank {request.rank}"
    )
    # Wait for the scheme generation to be done
    await controller.policy_to_rollout_shard_mapper.scheme_generation_done.wait()
    # Get the receive instructions for rollout
    recv_insts = (
        await controller.policy_to_rollout_shard_mapper.get_recv_insts_for_rollout(
            request.rank
        )
    )
    # If the receive instructions are not found, return an error response
    if recv_insts is None:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Rollout shard receive instructions not found",
        )

    logger.debug(
        f"[Dispatcher] Received rollout shard receive instructions for rank {request.rank}"
    )

    return Response(content=recv_insts, media_type="application/msgpack")


@app.get(COSMOS_API_GET_TRAINABLE_PARAMS_SUFFIX)
async def get_trainable_params():
    try:
        return {
            "trainable_params": list(
                controller.policy_to_rollout_shard_mapper.trainable_params
            )
        }
    except Exception:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR,
            "Error getting trainable params",
        )


@app.post(COSMOS_API_RESUME_INFO_SUFFIX)
async def resume_info(request: ResumeInfoRequest):
    logger.info(f"[Dispatcher] Validate resume info: {request.ckpt_extra_info}")
    controller.data_fetcher.validate_after_resume(request.ckpt_extra_info)
    return {"message": "Resume info received and processed"}


"""
NCCL Handshake API
"""


@app.post(COSMOS_API_NCCL_COMM_INITIATOR_SUFFIX)
async def comm_initiator(request: HandshakeInitiatorRequest):
    if request.handle_base64 is None or request.handle_base64 == "":
        return create_error_response(
            constant.ErrorCode.INVALID_REQUEST, "Handle is required"
        )

    await controller.update_kv_store(request.unique_pair_name, request.handle_base64)
    return {"message": "Handshake initiator received"}


@app.post(COSMOS_API_NCCL_COMM_ACCEPTOR_SUFFIX)
async def comm_acceptor(request: HandshakeAcceptorRequest):
    if request.unique_pair_name not in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR, "Unique pair name not found"
        )
    return {"handle_base64": controller.temp_kv_store.get(request.unique_pair_name)}


@app.post(COSMOS_API_IPC_INFO_SUFFIX)
async def ipc_info(request: IpcInfoRequest):
    await controller.update_kv_store(request.mesh_key, request.ipc_addr)
    return {"message": "IPC info received"}


@app.post(COSMOS_API_QUERY_IPC_INFO_SUFFIX)
async def query_ipc_info(request: QueryIpcInfoRequest):
    if request.mesh_key not in controller.temp_kv_store:
        return create_error_response(
            constant.ErrorCode.INTERNAL_ERROR, f"Mesh key {request.mesh_key} not found"
        )
    return {"ipc_addr": controller.temp_kv_store.get(request.mesh_key)}


@app.post(COSMOS_API_NCCL_COMM_ERROR_SUFFIX)
async def comm_error(request: NcclErrRequest):
    await controller.set_replica_ncclerror(request.replica_name, request.error)
    return {"message": "DetectTimeout received"}


@app.post(COSMOS_API_NCCL_COMM_STORE_CLEAR_SUFFIX)
async def comm_store_clear(request: NcclStoreClearRequest):
    try:
        await controller.clear_temp_kv_store(request.unique_pair_name)
    except Exception as e:
        logger.error(f"[Controller] Error clearing store: {e}")
    return {"message": "Store cleared"}


@app.get(COSMOS_API_NCCL_COMM_GET_ALL_SUFFIX)
async def comm_get_all():
    return {"comm_info": controller.temp_kv_store}


"""
Rollout API
"""


@app.get(COSMOS_API_NEXT_PROMPT_SUFFIX)
async def get_batched_prompt(
    n: int, validation_step: Optional[int] = None, rank_in_mesh: Optional[int] = None
):
    payloads_list, is_end = await controller.get_batched_prompt(
        n, validation_step, rank_in_mesh
    )
    return {
        "payloads_list": payloads_list,
        "is_end": is_end,
    }


@app.post(COSMOS_API_VALIDATION_REPORT_SUFFIX)
async def validation_report(request: ValidationReportRequest):
    rollouts_list = extract_rollouts(request.payloads, True, is_validation=True)
    controller.policy_status_manager.validation_report_validation_results(
        request.validation_step, rollouts_list, controller.rollout_status_manager
    )
    return {"message": "Validation rollout put"}


@app.post(COSMOS_API_ROLLOUT_SUFFIX)
async def put_rollout_group(rollout: RolloutRequest):
    try:
        if rollout.is_end:
            logger.info(
                f"[Controller] Received rollout end signal from {rollout.src_replica_name}"
            )
            controller.rollout_status_manager.rollout_end(rollout.src_replica_name)
            controller.policy_status_manager.on_rollout_is_end(
                controller.rollout_status_manager
            )

            return {"message": "Rollout end signal received"}

        rollouts_list = extract_rollouts(rollout.payloads, rollout.is_end)
        # Update the statistics for dynamic sampling used for metrics collection
        if controller.config.train.train_policy.variant == "dapo":
            controller.policy_status_manager.update_dynamic_sampling_statistics(
                rollout.metrics
            )
        # Flatten the rollouts into a single list
        rollouts = [
            rollout
            for rollouts_group in rollouts_list
            for rollout in rollouts_group  # rollouts_group: all rollouts of the same prompt.
        ]
        # Filter out outdated rollouts
        rollouts = controller.policy_status_manager.filter_outdated_rollouts(rollouts)
        if len(rollouts) > 0:
            logger.debug(
                f"[RolloutGroup] from replica: {rollout.src_replica_name} with {len(rollout.payloads)} samples:"
                f"example: rollouts[0]\n{rollouts[0]}"
            )

        await controller.put_rollouts(rollouts)
        return {"message": "Rollout put"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


@app.post(COSMOS_API_POLICY_TRAIN_ACK_SUFFIX)
async def train_ack(request: TrainAckRequest):
    try:
        replicaname = request.replica_name
        step = request.weight_step
        total_steps = request.total_steps
        profile_finished = request.profile_finished
        report_data = request.report_data
        controller.policy_status_manager.train_ack(
            replicaname,
            step,
            total_steps,
            profile_finished,
            report_data,
            controller.rollout_status_manager,
        )
        return {"message": "Ack completed"}
    except Exception as e:
        import traceback

        traceback.print_exc()
        return create_error_response(constant.ErrorCode.INTERNAL_ERROR, str(e))


def _serialize_replicas(replicas: Dict[str, Replica]) -> List[Dict]:
    result = []
    for name, replica in replicas.items():
        result.append(replica.to_dict())
    return result


def main(
    dataset: Optional[Union[Dataset, Callable[[CosmosConfig], Dataset]]] = None,
    dataloader: Optional[Callable[[CosmosConfig], Iterable]] = None,
    data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
    reward_fns: Optional[List[Callable]] = None,
    filter_reward_fns: Optional[List[Callable]] = None,
    val_dataset: Optional[Dataset] = None,
    val_reward_fns: Optional[List[Callable]] = None,
    val_data_packer: Optional[Union[BaseDataPacker, Callable]] = None,
    custom_logger_fns: Optional[List[Callable]] = None,
    hook_fns: Optional[Dict[str, Callable]] = None,
    sampler: Optional[Callable] = None,
    batch_sampler: Optional[Callable] = None,
    val_sampler: Optional[Callable] = None,
    val_batch_sampler: Optional[Callable] = None,
    args: Optional[argparse.Namespace] = None,
    **kwargs,
):
    if kwargs:
        logger.warning(
            f"Params: {list(kwargs.keys())} are not being used in controller initialization."
        )
    if dataloader is not None:
        raise NotImplementedError(
            "Customized dataloader is not supported inside controller now."
        )

    # Deprecated: The following code is to ensure backward compatibility:
    # where `dispatcher` is always launched in custom script
    role = os.environ.get("COSMOS_ROLE")
    assert role in ["Policy", "Rollout", "Controller"], f"Invalid role: {role}"
    if role == "Controller":
        pass
    else:
        logger.warning(
            "Deprecated: Please update your script to use `cosmos_rl.launcher.launch()` instead of `cosmos_rl.dispatcher.run_web_panel.main`"
        )
        if role == "Policy":
            from cosmos_rl.policy.train import main as policy_main

            policy_main(
                args=args,
                dataset=dataset,
                data_packer=data_packer,
                val_dataset=val_dataset,
                val_data_packer=val_data_packer,
                sampler=sampler,
                hook_fns=hook_fns,
                batch_sampler=batch_sampler,
                val_sampler=val_sampler,
                val_batch_sampler=val_batch_sampler,
            )
        else:
            from cosmos_rl.rollout.rollout_entry import run_rollout

            run_rollout(
                args=args,
                dataset=dataset,
                reward_fns=reward_fns,
                filter_reward_fns=filter_reward_fns,
                hook_fns=hook_fns,
                val_dataset=val_dataset,
                val_reward_fns=val_reward_fns,
                data_packer=data_packer,
                val_data_packer=val_data_packer,
            )
        return

    if args is None:
        # This means that args are not parsed in dataset entry script
        # So we need to parse the args manually
        parser = worker_entry_parser()
        try:
            args = parser.parse_args()
        except SystemExit as e:
            logger.error(
                "Error when parsing args. Did you use custom arguments in your script? If so, please check your custom script and pass `args` to this main function."
            )
            raise e
        assert args.config is not None, (
            "Config file path is required. Please provide --config argument."
        )

    # Load config from file if provided
    loaded_config = None
    assert os.path.exists(args.config), f"Config file {args.config} does not exist."

    try:
        logger.info(f"Attempting to load configuration from {args.config}")
        with open(args.config, "r") as f:
            config_dict = toml.load(f)

        # Ensure CosmosConfig is available (it's imported at the top now)
        # from cosmos_rl.policy.config import Config as CosmosConfig
        # Need SFTDataConfig and GrpoConfig for from_dict

        loaded_config = CosmosConfig.from_dict(config_dict)
        # Use redis port from config if available, otherwise use arg/default
        if hasattr(loaded_config, "redis") and loaded_config.redis:
            try:
                redis_port_from_config = int(loaded_config.redis)
                args.redis_port = redis_port_from_config
                logger.info(f"Using Redis port {args.redis_port} from config file.")
            except (ValueError, TypeError):
                logger.warning(
                    f"Invalid redis port format in config file: {loaded_config.redis}. Using default/arg: {args.redis_port}"
                )

        if data_packer is not None:
            assert isinstance(data_packer, BaseDataPacker) or callable(data_packer), (
                "data_packer should be a BaseDataPacker instance or a Callable"
            )
        controller.setup(
            loaded_config,
            redis_port=args.redis_port,
            redis_logfile_path=args.redis_logfile_path,
            dataset=dataset,
            val_dataset=val_dataset,
            custom_logger_fns=custom_logger_fns,
            hook_fns=hook_fns,
            sampler=sampler,
            batch_sampler=batch_sampler,
            val_sampler=val_sampler,
            val_batch_sampler=val_batch_sampler,
        )
        logger.info(f"Successfully loaded configuration from {args.config}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {args.config}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load or parse config file {args.config}: {e}.",
        )

    # Serve on the port every worker was told to reach us at. Re-probing for a
    # different port here would silently strand the workers (and the launcher,
    # which polls the advertised URL for readiness). Prefer the pre-bound
    # listening socket inherited from the launcher — that reservation is
    # race-free; otherwise bind exactly args.port and fail fast if it is taken.
    listen_fd = os.environ.get("COSMOS_CONTROLLER_LISTEN_FD")
    if listen_fd is not None:
        listen_sock = socket.socket(fileno=int(listen_fd))
        port = listen_sock.getsockname()[1]
    else:
        try:
            listen_sock = bind_available_port(args.port, args.port + 1)
        except RuntimeError:
            raise RuntimeError(
                f"Controller port {args.port} is already in use. It cannot be "
                "substituted because workers connect to this exact endpoint."
            )
        port = args.port
    config = uvicorn.Config(app, host="0.0.0.0", port=port, access_log=False)
    global server
    server = uvicorn.Server(config)
    server.run(sockets=[listen_sock])


if __name__ == "__main__":
    main()
