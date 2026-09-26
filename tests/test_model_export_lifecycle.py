# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Writer ownership, truthful failure, and publication through actual exporters."""

import threading
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import cosmos_rl
from safetensors.torch import load_file, save_file

from cosmos_rl.utils.model_export import (
    ModelExportThread,
    finish_checkpoint_writes,
    finish_model_export,
    publish_export_directory,
)
from cosmos_rl.utils.checkpoint import CheckpointMananger
from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.policy.worker.base import PolicyWorkerBase
from cosmos_rl.policy.worker.rl_worker import RLPolicyWorker
from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker


def test_join_observes_export_failure_and_retains_owner():
    def write():
        raise OSError("injected disk failure")

    trainer = SimpleNamespace(upload_thread=ModelExportThread(target=write))
    trainer.upload_thread.start()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="Model export failed") as exc:
            finish_model_export(trainer)
        assert isinstance(exc.value.__cause__, OSError)
        assert trainer.upload_thread is not None


@pytest.mark.parametrize("case", ["healthy", "shard-failure", "previous-failure"])
def test_two_rank_pipeline_export_canary(tmp_path, case):
    repo = Path(__file__).resolve().parents[1]
    package_root = Path(cosmos_rl.__file__).resolve().parent
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(repo / "tests/model_export_canary.py"),
            "--expected-package-root",
            str(package_root),
            "--device",
            "cpu",
            "--case",
            case,
            "--output",
            str(tmp_path / case),
        ],
        cwd=repo,
        env={**os.environ, "PYTHONPATH": str(package_root.parent)},
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert f"rank={rank} case={case} pipeline_export=PASS" in result.stdout


def test_finalize_drains_checkpoint_writers_and_propagates_failure():
    manager = CheckpointMananger.__new__(CheckpointMananger)
    manager.save_mode = "async"
    manager.executor = ThreadPoolExecutor(2)
    failed = Future()
    failed.set_exception(OSError("failed checkpoint"))
    entered, finish, drained = threading.Event(), threading.Event(), threading.Event()

    def write():
        entered.set()
        assert finish.wait(2)
        drained.set()

    manager.pre_save_futures = [failed, manager.executor.submit(write)]
    assert entered.wait(1)
    timer = threading.Timer(0.05, finish.set)
    timer.start()
    with pytest.raises(OSError, match="failed checkpoint"):
        manager.finalize()
    assert drained.is_set()
    timer.join()


def make_trainer(tmp_path, monkeypatch):
    import cosmos_rl.policy.trainer.llm_trainer.llm_trainer as module

    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.fill_(3)
    model.weight_mapper = SimpleNamespace(
        policy_map_local_key_to_hf_key=lambda name: name,
        policy_map_local_key_for_export_tensor=lambda name, tensor: [(name, tensor)],
    )
    trainer = SimpleNamespace(
        model=model,
        global_rank=0,
        upload_thread=None,
        data_packer=None,
        hf_config=SimpleNamespace(save_pretrained=lambda path: None),
        parallel_dims=SimpleNamespace(
            dp_replicate_coord=(0, 1),
            dp_replicate_enabled=False,
            dp_shard_coord=(0, 1),
            cp_coord=(0, 1),
            tp_coord=(0, 1),
            pp_coord=(0, 1),
            pp_enabled=False,
        ),
        config=SimpleNamespace(
            policy=SimpleNamespace(
                lora=None, model_name_or_path=str(tmp_path), model_revision=None
            ),
            train=SimpleNamespace(
                ckpt=SimpleNamespace(upload_hf=False, upload_s3=False)
            ),
        ),
    )
    monkeypatch.setattr(module.torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(
        module.GenerationConfig,
        "from_pretrained",
        Mock(side_effect=FileNotFoundError("no config")),
    )
    monkeypatch.setattr(
        module.util, "resolve_model_path", lambda *args, **kwargs: str(tmp_path)
    )
    return trainer, module


def test_actual_export_snapshot_survives_mutation_while_writer_is_paused(
    tmp_path, monkeypatch
):
    trainer, module = make_trainer(tmp_path, monkeypatch)
    entered, finish = threading.Event(), threading.Event()

    def delayed(tensors, path):
        entered.set()
        assert finish.wait(2)
        save_file(tensors, path)

    monkeypatch.setattr(module, "save_file", delayed)
    LLMTrainer.export_safetensors(trainer, str(tmp_path), "export")
    assert entered.wait(1)
    assert not (tmp_path / "export").exists()
    trainer.model.weight.data.fill_(42)
    finish.set()
    finish_model_export(trainer)
    saved = load_file(str(tmp_path / "export/00000.safetensors"))
    torch.testing.assert_close(saved["weight"], torch.full((1, 2), 3.0))
    assert trainer.upload_thread is None
    assert (tmp_path / "export/model.safetensors.index.json").is_file()


def test_actual_export_refuses_manifest_for_missing_shard(tmp_path, monkeypatch):
    trainer, module = make_trainer(tmp_path, monkeypatch)
    monkeypatch.setattr(module, "save_file", lambda *args: None)
    LLMTrainer.export_safetensors(trainer, str(tmp_path), "export")
    with pytest.raises(RuntimeError, match="Model export failed") as exc:
        finish_model_export(trainer)
    assert "missing shards" in str(exc.value.__cause__)
    assert not (tmp_path / "export/model.safetensors.index.json").exists()


def test_pipeline_lora_rejected_before_writes(tmp_path, monkeypatch):
    trainer, module = make_trainer(tmp_path, monkeypatch)
    trainer.parallel_dims.pp_enabled = True
    trainer.config.policy.lora = object()
    write = Mock()
    monkeypatch.setattr(module, "save_file", write)
    with pytest.raises(
        ValueError, match="Pipeline-parallel LoRA export is unsupported"
    ):
        LLMTrainer.export_safetensors(trainer, str(tmp_path), "export")
    write.assert_not_called()


def test_diffusers_actual_export_snapshot_and_publication(tmp_path, monkeypatch):
    from cosmos_rl.policy.trainer.diffusers_trainer import diffusers_trainer as module

    trainer, _ = make_trainer(tmp_path, monkeypatch)
    model = trainer.model
    trainer.model = SimpleNamespace(transformer=model)
    trainer.is_lora = False
    trainer.max_size_bytes = 1024**3
    entered, finish = threading.Event(), threading.Event()

    def delayed(values, path, **kwargs):
        entered.set()
        assert finish.wait(2)
        save_file(values, path, **kwargs)

    monkeypatch.setattr(module, "save_file", delayed)
    module.DiffusersTrainer.export_safetensors(
        trainer, str(tmp_path), "export", trainable_only=True
    )
    assert entered.wait(1)
    assert not (tmp_path / "export").exists()
    model.weight.data.fill_(42)
    finish.set()
    finish_model_export(trainer)
    files = list((tmp_path / "export/transformer").glob("*.safetensors"))
    assert len(files) == 1
    torch.testing.assert_close(
        load_file(str(files[0]))["weight"], torch.full((1, 2), 3.0)
    )


def test_failed_rewrite_preserves_previous_complete_export(tmp_path, monkeypatch):
    trainer, module = make_trainer(tmp_path, monkeypatch)
    LLMTrainer.export_safetensors(trainer, str(tmp_path), "export")
    finish_model_export(trainer)
    original = (tmp_path / "export/model.safetensors.index.json").read_bytes()
    trainer.model.weight.data.fill_(42)
    monkeypatch.setattr(module, "save_file", Mock(side_effect=OSError("write failed")))
    LLMTrainer.export_safetensors(trainer, str(tmp_path), "export")
    with pytest.raises(RuntimeError, match="Model export failed"):
        finish_model_export(trainer)
    assert (tmp_path / "export/model.safetensors.index.json").read_bytes() == original
    saved = load_file(str(tmp_path / "export/00000.safetensors"))
    torch.testing.assert_close(saved["weight"], torch.full((1, 2), 3.0))


@pytest.mark.parametrize("failed", [False, True])
def test_directory_replacement_keeps_previous_export_recoverable(
    tmp_path, monkeypatch, failed
):
    destination, staged = tmp_path / "export", tmp_path / "staged"
    destination.mkdir()
    staged.mkdir()
    (destination / "old").write_text("old")
    (staged / "new").write_text("new")
    replace = os.replace

    def injected(source, target):
        if failed and source == staged:
            raise OSError("publish failed")
        replace(source, target)

    monkeypatch.setattr("cosmos_rl.utils.model_export.os.replace", injected)
    if failed:
        with pytest.raises(OSError, match="publish failed"):
            publish_export_directory(staged, destination)
        assert (destination / "old").read_text() == "old"
        assert (staged / "new").is_file()
    else:
        publish_export_directory(staged, destination)
        assert (destination / "new").read_text() == "new"
        backups = list(tmp_path.glob("export.previous-*"))
        assert len(backups) == 1 and (backups[0] / "old").read_text() == "old"


@pytest.mark.parametrize("failed", [False, True])
def test_rl_shutdown_waits_for_trainer_owner_before_stopping_liveness(
    monkeypatch, failed
):
    order = []
    entered, finish = threading.Event(), threading.Event()
    shutdown = threading.Event()

    def write():
        entered.set()
        assert finish.wait(2)
        assert not shutdown.is_set()
        order.append("writer")
        if failed:
            raise OSError("export failed")

    trainer = SimpleNamespace(upload_thread=ModelExportThread(target=write))
    trainer.upload_thread.start()
    assert entered.wait(1)
    worker = SimpleNamespace(
        trainer=trainer,
        shutdown_signal=shutdown,
        shutdown_mp_signal=threading.Event(),
        inter_policy_nccl=SimpleNamespace(shutdown=lambda: order.append("nccl")),
        fetch_rollouts_thread=None,
        fetch_command_thread=None,
        teacher_interact_thread=None,
        heartbeat_thread=None,
        _shutdown_payload_data_packers=lambda: order.append("payload"),
        destroy_worker=lambda: order.append("destroy"),
        unregister_from_controller=lambda: order.append("unregister"),
    )
    monkeypatch.setattr(
        "cosmos_rl.policy.worker.rl_worker.nccl_abort_all", lambda: None
    )
    monkeypatch.setattr("cosmos_rl.policy.worker.rl_worker.time.sleep", lambda _: None)
    timer = threading.Timer(0.05, finish.set)
    timer.start()
    if failed:
        with pytest.raises(RuntimeError, match="finalization failed"):
            RLPolicyWorker.handle_shutdown(worker)
    else:
        RLPolicyWorker.handle_shutdown(worker)
    timer.join()
    assert order == ["writer", "payload", "nccl", "destroy", "unregister"]


def test_execute_error_path_still_joins_actual_export_and_destroys_worker():
    order = []
    thread = ModelExportThread(target=lambda: order.append("writer"))
    thread.start()
    worker = SimpleNamespace(
        trainer=SimpleNamespace(upload_thread=thread),
        main_loop=Mock(side_effect=ValueError("training failed")),
        close_payload_transports=lambda: order.append("payload"),
        destroy_worker=lambda: order.append("destroy"),
    )
    with pytest.raises(ValueError, match="training failed"):
        PolicyWorkerBase.execute(worker)
    assert order == ["writer", "payload", "destroy"]
    assert worker.trainer.upload_thread is None


def test_both_finalizers_attempted_and_sft_still_unregisters_on_export_error():
    thread = ModelExportThread(
        target=lambda: (_ for _ in ()).throw(OSError("write failed"))
    )
    thread.start()
    manager = SimpleNamespace(finalize=Mock())
    trainer = SimpleNamespace(upload_thread=thread, ckpt_manager=manager)
    with pytest.raises(RuntimeError, match="finalization failed"):
        finish_checkpoint_writes(trainer)
    manager.finalize.assert_called_once()
    worker = SimpleNamespace(trainer=trainer, unregister_from_controller=Mock())
    with pytest.raises(RuntimeError, match="finalization failed"):
        SFTPolicyWorker.handle_shutdown(worker)
    worker.unregister_from_controller.assert_called_once()
