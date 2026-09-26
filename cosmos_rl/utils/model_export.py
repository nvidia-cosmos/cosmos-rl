# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned model-export tasks: completion and failures belong to the trainer."""

import threading
import os
import uuid


def staged_export_path(destination, token=None):
    """A sibling on the same filesystem; never expose a half-written export."""
    return f"{destination}.incomplete-{token or uuid.uuid4().hex}"


def publish_export_directory(staged, destination):
    """Publish a complete directory; preserve an overwritten export for recovery.

    Replacement has a short missing-path window, never a partial-directory
    window. A process crash in that window leaves the previous sibling intact.
    Concurrent writers to the same destination are not supported.
    """
    previous = None
    if os.path.lexists(destination):
        previous = f"{destination}.previous-{uuid.uuid4().hex}"
        os.replace(destination, previous)
    try:
        os.replace(staged, destination)
    except BaseException:
        if previous is not None and not os.path.lexists(destination):
            os.replace(previous, destination)
        raise
    return destination


class ModelExportThread(threading.Thread):
    """A join observes writer/upload failures instead of reporting clean exit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.error = None

    def run(self):
        try:
            super().run()
        except BaseException as error:
            self.error = error

    def join(self, timeout=None):
        super().join(timeout)
        if not self.is_alive() and self.error is not None:
            raise RuntimeError("Model export failed") from self.error


def finish_model_export(trainer):
    thread = getattr(trainer, "upload_thread", None)
    if thread is not None:
        thread.join()
        trainer.upload_thread = None


def finish_checkpoint_writes(trainer):
    """Drain both independent owners, even when one of their writers failed."""
    errors = []
    try:
        finish_model_export(trainer)
    except Exception as error:
        errors.append(error)
    manager = getattr(trainer, "ckpt_manager", None)
    if manager is not None and hasattr(manager, "finalize"):
        try:
            manager.finalize()
        except Exception as error:
            errors.append(error)
    if errors:
        raise RuntimeError(
            "Checkpoint/export finalization failed: " + "; ".join(map(str, errors))
        ) from errors[0]
