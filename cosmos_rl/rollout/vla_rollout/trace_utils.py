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
Chrome Tracing Utilities for VLA Rollout Performance Analysis.

This module provides utilities for recording performance events during VLA rollout
in Chrome tracing format, which can be visualized in chrome://tracing or Perfetto.

Verbosity Levels:
    - 0: Tracing disabled
    - 1: Trace validation phase only
    - 2: Trace both validation and rollout phases

Event Categories:
    - "rollout": Overall rollout generation calls
    - "task_execution": Individual task execution spans
    - "simulation": Environment simulation steps
    - "init": Environment initialization/reset
    - "inference": Model inference calls

Thread ID Convention:
    - thread_id = env_id: Environment-specific events (0 to num_envs-1)
    - thread_id = 0: Rollout-level aggregated events

Available Colors:
    - "good": Green (success/positive)
    - "bad": Red (failure/negative)
    - "thread_state_running": Bright green
    - "thread_state_runnable": Light blue
    - "rail_response": Orange
    - "rail_idle": Grey
"""

import os
import json
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch


# Event type configuration: maps event type to (name, category, color)
EVENT_CONFIGS = {
    "sim_step": ("Sim_Step", "simulation", "thread_state_running"),
    "env_reset": ("Env_Reset", "init", "rail_idle"),
    "inference": ("Inference", "inference", "rail_response"),
    "rollout_generation": ("Rollout_Generation", "rollout", "thread_state_runnable"),
    "task_execution": (
        "Task",
        "task_execution",
        None,
    ),  # Color depends on success/failure
}


class TracingManager:
    """
    Manager for Chrome tracing format event recording.

    Each rollout_generation call produces one trace file when enabled.
    """

    class TraceEventContext:
        """Context manager for trace events that automatically captures timing."""

        def __init__(
            self, manager, name, category, thread_id, color, env_ids, **metadata
        ):
            self.manager = manager
            self.name = name
            self.category = category
            self.thread_id = thread_id
            self.color = color
            self.env_ids = env_ids
            self.metadata = metadata
            self.start_time = None
            self.end_time = None

        def __enter__(self):
            self.start_time = time.time()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.end_time = time.time()

            # If env_ids provided, record event for each env
            if self.env_ids:
                for env_id in self.env_ids:
                    self.manager.add_event(
                        name=self.name,
                        category=self.category,
                        start_time=self.start_time,
                        end_time=self.end_time,
                        thread_id=env_id,
                        color=self.color,
                        is_validation=self.manager._current_is_validation,
                        **self.metadata,
                    )
            else:
                # Otherwise, record single event with specified thread_id
                self.manager.add_event(
                    name=self.name,
                    category=self.category,
                    start_time=self.start_time,
                    end_time=self.end_time,
                    thread_id=self.thread_id,
                    color=self.color,
                    is_validation=self.manager._current_is_validation,
                    **self.metadata,
                )
            return False  # Don't suppress exceptions

        def add_sibling_event(self, name=None, thread_id=None, **override_metadata):
            """
            Add a sibling event with the same timing but different name/thread_id.
            Useful for recording the same operation for multiple environments.

            Args:
                name: Override name (defaults to original name)
                thread_id: Override thread_id (defaults to original thread_id)
                **override_metadata: Override or add metadata
            """
            combined_metadata = {**self.metadata, **override_metadata}
            self.manager.add_event(
                name=name or self.name,
                category=self.category,
                start_time=self.start_time,
                end_time=self.end_time,
                thread_id=thread_id if thread_id is not None else self.thread_id,
                color=self.color,
                is_validation=self.manager._current_is_validation,
                **combined_metadata,
            )

    def __init__(
        self,
        rank: int,
        output_dir: str,
        verbosity: int = 0,
    ):
        """
        Initialize the tracing manager.

        Args:
            rank: Process rank for multi-process training
            output_dir: Directory to save trace files
            verbosity: Tracing verbosity level (0=disabled, 1=validation only, 2=all)
        """
        self.rank = rank
        self.verbosity = verbosity

        if self.verbosity > 0:
            self.dump_dir = os.path.join(output_dir, "task_timing_logs")
            os.makedirs(self.dump_dir, exist_ok=True)

            self.current_rollout_events = []  # Events for current rollout call
            self.rollout_call_count = 0

        self._current_is_validation = False

    def should_trace(self, is_validation: bool = None) -> bool:
        """
        Check if tracing should be enabled for this rollout.

        Args:
            is_validation: Whether this is a validation rollout (defaults to self._current_is_validation)

        Returns:
            True if tracing should be enabled, False otherwise
        """
        if is_validation is None:
            is_validation = self._current_is_validation

        if self.verbosity == 0:
            return False
        elif self.verbosity == 1:
            return is_validation
        else:  # verbosity >= 2
            return True

    def start_rollout(self, is_validation: bool):
        """
        Start a new rollout call, clearing current events and setting validation state.

        Args:
            is_validation: Whether this is a validation rollout
        """
        self._current_is_validation = is_validation
        if self.should_trace():
            self.current_rollout_events = []
            self.rollout_call_count += 1

    def add_event(
        self,
        name: str,
        category: str,
        start_time: float,
        end_time: Optional[float] = None,
        duration: Optional[float] = None,
        thread_id: int = 0,
        color: Optional[str] = None,
        is_validation: bool = False,
        **metadata,
    ):
        """
        Add a trace event in Chrome tracing format.

        Args:
            name: Event name (e.g., "Inference", "Simulation")
            category: Event category (e.g., "inference", "simulation", "init")
            start_time: Start timestamp in seconds (from time.time())
            end_time: End timestamp in seconds. If provided, duration is computed
            duration: Duration in seconds. If end_time is provided, this is ignored
            thread_id: Thread ID for tracing (e.g., env_id, batch_id)
            color: Chrome tracing color name (e.g., "good", "bad", "rail_response")
            is_validation: Whether this is a validation rollout
            **metadata: Additional metadata to include in event args
        """
        if not self.should_trace(is_validation):
            return

        # Calculate duration
        if end_time is not None:
            dur_seconds = end_time - start_time
        elif duration is not None:
            dur_seconds = duration
        else:
            raise ValueError("Either end_time or duration must be provided")

        # Convert to microseconds for Chrome tracing format
        ts_us = int(start_time * 1_000_000)
        dur_us = int(dur_seconds * 1_000_000)

        # Create Chrome tracing event
        event = {
            "name": name,
            "cat": category,
            "ph": "X",  # Complete event (duration event)
            "ts": ts_us,
            "dur": dur_us,
            "pid": self.rank,  # Process ID (rank)
            "tid": thread_id,  # Thread ID
            "args": {
                "rank": self.rank,
                **{k: self._serialize_value(v) for k, v in metadata.items()},
            },
        }

        if color:
            event["cname"] = color

        self.current_rollout_events.append(event)

    def trace(
        self,
        event_type: str,
        thread_id: int = 0,
        env_ids: Optional[List[int]] = None,
        **metadata,
    ):
        """
        Create a context manager for tracing a predefined event type.
        Uses EVENT_CONFIGS for name/category/color configuration.

        Usage:
            # Simple usage for single event
            with manager.trace("inference", thread_id=0, batch_size=8):
                result = model.generate()

            # For batch operations affecting multiple envs
            with manager.trace("sim_step", env_ids=active_env_ids):
                step_results = env.step(actions)

        Args:
            event_type: Event type key from EVENT_CONFIGS
                       ("sim_step", "env_reset", "inference", "rollout_generation")
            thread_id: Thread ID for tracing (e.g., env_id) - used only if env_ids is None
            env_ids: Optional list of environment IDs to record events for (one per env)
            **metadata: Additional metadata

        Returns:
            Context manager that captures timing automatically
        """
        if event_type not in EVENT_CONFIGS:
            raise ValueError(
                f"Unknown event_type '{event_type}'. "
                f"Valid types: {list(EVENT_CONFIGS.keys())}"
            )

        name, category, color = EVENT_CONFIGS[event_type]

        return self.TraceEventContext(
            manager=self,
            name=name,
            category=category,
            thread_id=thread_id,
            color=color,
            env_ids=env_ids,
            **metadata,
        )

    def add_task_execution_events(self, task_records: List[Dict[str, Any]]):
        """
        Extract timing information from task records and add task execution events.

        Args:
            task_records: List of task record dictionaries
        """
        if not self.should_trace():
            return

        for task in task_records:
            start_time = task.get("start_time")
            end_time = task.get("end_time")

            if start_time is None or end_time is None:
                continue

            task_id = int(task.get("task_id", -1))
            trial_id = int(task.get("trial_id", -1))
            env_id = int(task.get("env_id", -1))
            complete = bool(task.get("complete", False))
            finish_step = int(task.get("finish_step", -1))

            task_name = f"Task_{task_id}.{trial_id}"

            # Add task execution event
            self.add_event(
                name=task_name,
                category="task_execution",
                start_time=start_time,
                end_time=end_time,
                thread_id=env_id,
                color="good" if complete else "bad",
                is_validation=self._current_is_validation,
                task_id=task_id,
                trial_id=trial_id,
                task_suite_name=str(task.get("task_suite_name", "")),
                env_id=env_id,
                complete=complete,
                finish_step=finish_step,
                status="success" if complete else "failed",
            )

    def finalize_rollout(
        self,
        task_records: List[Dict[str, Any]],
        rollout_start_time: float,
        rollout_end_time: float,
        continuous: bool = False,
    ):
        """
        Finalize the current rollout, record rollout-level event, and dump trace file.
        Uses the validation state set by set_validation_state().

        Args:
            task_records: List of task record dictionaries from rollout
            rollout_start_time: Start time of the rollout (from time.time())
            rollout_end_time: End time of the rollout (from time.time())
            continuous: Whether continuous mode was enabled
        """
        if not self.should_trace():
            return

        # Add task execution events
        self.add_task_execution_events(task_records)

        # Calculate rollout-level metrics
        total_sim_frames = sum(task.get("finish_step", 0) for task in task_records)
        rollout_duration = rollout_end_time - rollout_start_time
        sim_fps = total_sim_frames / rollout_duration if rollout_duration > 0 else 0.0

        # Add rollout-level trace event
        name, category, color = EVENT_CONFIGS["rollout_generation"]
        self.add_event(
            name=name,
            category=category,
            start_time=rollout_start_time,
            end_time=rollout_end_time,
            thread_id=0,
            color=color,
            is_validation=self._current_is_validation,
            continuous=continuous,
            num_tasks=len(task_records),
            total_sim_frames=total_sim_frames,
            sim_fps=sim_fps,
        )

        # Dump traces for this rollout immediately
        self.dump_traces()

    def dump_traces(self):
        """
        Dump current rollout trace events to disk in Chrome tracing format.
        Gathers events from all ranks and merges them into a single trace file on rank 0.
        Can be visualized in chrome://tracing or https://ui.perfetto.dev/
        """
        if not self.should_trace():
            return

        if not self.current_rollout_events:
            return

        from cosmos_rl.utils.logging import logger

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        phase = "validation" if self._current_is_validation else "rollout"

        # Prepare metadata for this rank
        local_metadata = {
            "rank": int(self.rank),
            "num_events": len(self.current_rollout_events),
            "rollout_call_count": int(self.rollout_call_count),
        }

        # Check if distributed training is initialized
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )

        # Gather data from all ranks to rank 0
        if world_size > 1:
            all_events = [None] * world_size
            all_metadata = [None] * world_size

            torch.distributed.gather_object(
                self.current_rollout_events,
                all_events if self.rank == 0 else None,
                dst=0,
            )
            torch.distributed.gather_object(
                local_metadata, all_metadata if self.rank == 0 else None, dst=0
            )

            if self.rank == 0:
                # Merge events from all ranks
                merged_events = []
                for rank_events in all_events:
                    if rank_events:
                        merged_events.extend(rank_events)

                combined_metadata = {
                    "dump_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "world_size": world_size,
                    "total_events": sum(m["num_events"] for m in all_metadata if m),
                    "per_rank_info": all_metadata,
                    "trace_verbosity": self.verbosity,
                    "rollout_call_count": self.rollout_call_count,
                    "is_validation": self._current_is_validation,
                }
            else:
                # Non-rank-0 processes just clear and return
                self.current_rollout_events = []
                return
        else:
            # Single rank case
            merged_events = self.current_rollout_events
            combined_metadata = {
                "dump_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "world_size": 1,
                "total_events": len(self.current_rollout_events),
                "per_rank_info": [local_metadata],
                "trace_verbosity": self.verbosity,
                "rollout_call_count": self.rollout_call_count,
                "is_validation": self._current_is_validation,
            }

        # Only rank 0 writes the merged trace file
        if self.rank == 0 or world_size == 1:
            trace_file = os.path.join(
                self.dump_dir,
                f"trace_{phase}_r{self.rollout_call_count}_{timestamp}.json",
            )

            # Add process and thread name metadata events for better visualization
            metadata_events = []
            for r in range(world_size):
                metadata_events.append(
                    {
                        "name": "process_name",
                        "ph": "M",  # Metadata event
                        "pid": r,
                        "args": {"name": f"Rank {r}"},
                    }
                )

            # Add thread names for environment IDs
            seen_threads = set()
            for event in merged_events:
                thread_key = (event["pid"], event["tid"])
                if thread_key not in seen_threads:
                    seen_threads.add(thread_key)
                    metadata_events.append(
                        {
                            "name": "thread_name",
                            "ph": "M",  # Metadata event
                            "pid": event["pid"],
                            "tid": event["tid"],
                            "args": {"name": f"Env {event['tid']}"},
                        }
                    )

            # Write in Chrome tracing format
            trace_data = {
                "traceEvents": metadata_events + merged_events,
                "displayTimeUnit": "ms",
                "metadata": combined_metadata,
            }

            with open(trace_file, "w") as f:
                json.dump(trace_data, f, indent=2)

            logger.info(
                f"Dumped {len(merged_events)} trace events for {phase} "
                f"rollout {self.rollout_call_count} from {world_size} rank(s) to: {trace_file}"
            )

        # Clear events after dumping
        self.current_rollout_events = []

    def _serialize_value(self, value: Any) -> Any:
        """
        Convert non-JSON-serializable types to serializable format.

        Args:
            value: Value to serialize

        Returns:
            JSON-serializable value
        """
        if isinstance(value, (np.ndarray, torch.Tensor)):
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            return value.item()
        elif hasattr(value, "__dict__"):
            return str(value)
        return value


def create_tracing_manager(
    rank: int,
    output_dir: str,
    trace_verbosity: int = 0,
) -> TracingManager:
    """
    Factory function to create a tracing manager based on configuration.

    Args:
        rank: Process rank
        output_dir: Output directory for traces
        trace_verbosity: Tracing verbosity level
            - 0: Disabled
            - 1: Validation phase only
            - 2: All rollouts (validation + training)

    Returns:
        TracingManager instance
    """
    return TracingManager(
        rank=rank,
        output_dir=output_dir,
        verbosity=trace_verbosity,
    )
