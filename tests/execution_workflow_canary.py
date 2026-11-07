# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real GPU workflows through the public launcher and installed package.

Uses a tiny random Qwen2 checkpoint with a cached production tokenizer. This
qualifies execution and optimizer plumbing; it measures no model quality.
Run on >=3 GPUs: python tests/execution_workflow_canary.py
"""

import copy
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import toml
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from cosmos_rl.execution.corpus import CorpusDataset, CorpusWriter
from cosmos_rl.execution.records import ModelIdentity, SampleRecord
from cosmos_rl.policy.config import Config
from cosmos_rl.utils.network_util import find_available_port


def checkpoint(path, **overrides):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    settings = dict(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        tie_word_embeddings=True,
    )
    settings.update(overrides)
    model = Qwen2ForCausalLM(Qwen2Config(**settings))
    model.to(torch.bfloat16).save_pretrained(path)
    return len(tokenizer)


def base(root, model):
    return {
        "mode": "disaggregated",
        "execution": {"path": "train_from_data"},
        "train": {
            "output_dir": str(root),
            "epoch": 1,
            "max_num_steps": 2,
            "compile": False,
            "sequence_packing": False,
            "train_batch_per_replica": 2,
            "optm_lr": 1e-4,
            "optm_warmup_steps": 0,
            "optm_grad_norm_clip": 1.0,
            "train_policy": {
                "type": "sft",
                "mini_batch": 1,
                "dataloader_num_workers": 0,
                "dataloader_shuffle": False,
                "dataloader_drop_last": False,
            },
            "ckpt": {"enable_checkpoint": True, "save_freq": 1, "save_mode": "sync"},
        },
        "policy": {
            "model_name_or_path": str(model),
            "model_revision": "canary-local-checkpoint",
            "model_max_length": 256,
            "enable_liger_fused_cross_entropy": False,
            "parallelism": {"tp_size": 1, "dp_shard_size": 1},
        },
        "rollout": {
            "parallelism": {"tp_size": 1},
            "backend": "vllm",
            "gpu_memory_utilization": 0.2,
            "max_response_length": 16,
            "n_generation": 2,
            "batch_size": 1,
        },
        "logging": {"logger": ["console"]},
    }


def launch(root, name, config, *, corpus=False, training=True, profile=False):
    config = copy.deepcopy(config)
    config["train"]["output_dir"] = str(root / name)
    config["train"]["timestamp"] = name
    config["redis"] = str(find_available_port(19000))
    # Validate the exact file submitted to the launcher before starting roles.
    Config.model_validate(copy.deepcopy(config))
    path = root / f"{name}.toml"
    path.write_text(toml.dumps(config))
    log = root / f"{name}.log"
    env = dict(
        os.environ,
        EXECUTION_CANARY_CORPUS="1" if corpus else "0",
        TOKENIZERS_PARALLELISM="false",
        WANDB_MODE="disabled",
        EXECUTION_CANARY_PROFILE="1"
        if profile
        else os.environ.get("EXECUTION_CANARY_PROFILE", "0"),
    )
    command = [
        sys.executable,
        "-m",
        "cosmos_rl.launcher.launch_all",
        "--config",
        str(path),
        "--port",
        str(find_available_port(18000)),
        str(Path(__file__).with_name("execution_workflow_app.py").resolve()),
    ]
    with log.open("w") as output:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            result = process.wait(timeout=360)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise AssertionError(f"{name} timed out; see {log}")
    text = log.read_text()
    assert result == 0, f"{name} exited {result}; see {log}\n{text[-6000:]}"
    if training:
        steps = config["train"]["max_num_steps"]
        assert f"Step: {steps}/{steps}" in text or f"step {steps}/{steps}" in text, (
            f"{name}: no completed optimizer schedule; see {log}"
        )
        assert (root / name / "safetensors" / f"step_{steps}").is_dir(), (
            f"{name}: no final checkpoint"
        )
    print(json.dumps({"scenario": name, "passed": True, "log": str(log)}), flush=True)


def run(root):
    assert torch.cuda.device_count() >= 3, "The execution canary requires three GPUs"
    model = root / "actor"
    vocab = checkpoint(model)
    common = base(root / "unused", model)

    generation = copy.deepcopy(common)
    generation["execution"] = {
        "path": "generate",
        "output": {
            "path": str(root / "generated-corpus"),
            "run_id": "generation-canary",
        },
    }
    generation["train"]["train_policy"].update(
        type="grpo",
        prompt_column_name="prompt",
        response_column_name="reference",
        bypass_reward=True,
    )
    generation["train"]["ckpt"]["enable_checkpoint"] = False
    generation["rollout"].update(max_prompts=2, max_inflight_prompts=2)
    actor = copy.deepcopy(generation)
    actor["execution"] = {
        "path": "train_from_generation",
        "actor": {"weight_source": "checkpoint", "model_name_or_path": str(model)},
    }
    actor["rollout"].pop("max_prompts")
    actor["rollout"].pop("max_inflight_prompts")
    actor["train"]["train_batch_per_replica"] = 4
    actor["train"]["ckpt"]["enable_checkpoint"] = True
    actor["train"]["train_policy"].update(
        bypass_reward=False,
        on_policy=False,
        use_rollout_logprobs_for_loss=True,
        max_retry_for_on_policy=-1,
    )
    launch(root, "independent-actor", actor)

    opd = copy.deepcopy(actor)
    opd["execution"] = {"path": "train_from_generation"}
    opd["distillation"] = {
        "enable": True,
        "model_name_or_path": str(model),
        "compile": False,
        "batch_size_per_replica": 2,
        "mini_batch": 1,
        "top_k": 16,
        "trainer_token_ids_from_teacher": True,
        "parallelism": {"tp_size": 1, "dp_shard_size": 1},
    }
    launch(root, "on-policy-distillation", opd)
    launch(root, "direct-pretraining", common)
    launch(root, "generation", generation, training=False)
    dataset = CorpusDataset(root / "generated-corpus")
    assert len(dataset) == 4
    assert all(dataset[i].producer.name == str(model) for i in range(len(dataset)))
    manifests = {p.name: p.read_bytes() for p in dataset.paths}
    launch(root, "generation-restart", generation, training=False)
    assert manifests == {p.name: p.read_bytes() for p in dataset.paths}
    sft = copy.deepcopy(common)
    sft["execution"].update(
        source={"kind": "corpus", "path": str(root / "generated-corpus")},
        objectives={"supervised": 1},
    )
    launch(root, "generated-corpus-sft", sft, corpus=True)

    writer = CorpusWriter(root / "mixed-corpus", run_id="recorded-behavior")
    for i in range(8):
        ids = [100 + i, 200 + i, 300 + i, 400 + i, 500 + i][: (4, 2, 3, 5)[i % 4]]
        n = len(ids) - 1
        writer.commit(
            SampleRecord(
                sample_id=str(i),
                occurrence_id=str(i),
                group_id=str(i // 2),
                group_size=2,
                group_slot=i % 2,
                producer=ModelIdentity(
                    role="actor",
                    name="uniform-token-sampler",
                    revision="fixture-v1",
                    behavior_version=0,
                ),
                inputs={"prompt": "recorded trajectory"},
                targets={"token_ids": ids, "label_ids": ids},
                masks={"policy_gradient": [True] * n},
                behavior_logprobs=[-math.log(vocab)] * n,
                annotations={"advantages": [0.5 if i % 2 else -0.5] * n},
            )
        )
    mixed = copy.deepcopy(common)
    mixed["policy"]["parallelism"]["dp_shard_size"] = 2
    mixed["train"]["train_batch_per_replica"] = 1
    mixed["execution"].update(
        source={"kind": "corpus", "path": str(root / "mixed-corpus")},
        objectives={"supervised": 0.5, "policy_gradient": 0.5},
    )
    launch(root, "mixed-recorded-objectives", mixed, corpus=True)
    packed = copy.deepcopy(mixed)
    packed["train"].update(sequence_packing=True, train_batch_per_replica=2)
    packed["train"]["train_policy"]["mini_batch"] = 2
    launch(root, "packed-recorded-objectives", packed, corpus=True, profile=True)
    assert (
        '"train/packing_active": 1'
        in (root / "packed-recorded-objectives.log").read_text()
    )
    checkpoints = list((root / "packed-recorded-objectives").glob("**/step_1/policy"))
    assert len(checkpoints) == 1, checkpoints
    resumed = copy.deepcopy(packed)
    resumed["train"]["resume"] = str(checkpoints[0])
    launch(root, "packed-recorded-resume", resumed, corpus=True, profile=True)
    from safetensors.torch import load_file

    def final_weights(name):
        weights = {}
        for path in (root / name / "safetensors" / "step_2").rglob("*.safetensors"):
            weights.update(load_file(path))
        assert weights
        return weights

    before, after = (
        final_weights("packed-recorded-objectives"),
        final_weights("packed-recorded-resume"),
    )
    assert before.keys() == after.keys()
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)

    (root / "results.json").write_text(
        json.dumps({"passed": True, "scenarios": 9}, indent=2)
    )


if __name__ == "__main__":
    if os.environ.get("TEST_LOG_DIR"):
        Path(os.environ["TEST_LOG_DIR"]).mkdir(parents=True, exist_ok=True)
    directory = Path(
        tempfile.mkdtemp(prefix="execution-canary-", dir=os.environ.get("TEST_LOG_DIR"))
    )
    print(f"Execution workflow artifacts: {directory}", flush=True)
    run(directory)
