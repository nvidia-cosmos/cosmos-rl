# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Full-launcher SFT stop fixture with a tiny locally generated Llama model.

Prepare: python tests/requested_stop_sft_live.py --prepare /tmp/stop-assets
Launch: cosmos-rl --config /tmp/stop-assets/sft.toml --policy 2 --rollout 0
        tests/requested_stop_sft_live.py
STOP_CANARY_AFTER=0/2, STOP_CANARY_DURING_VALIDATION=1 and
STOP_CANARY_FAIL_SAVE=1 select the scenarios. Uses real trainer/checkpoint code.
"""

import os
from pathlib import Path
import sys


def prepare_assets(root):
    import toml
    from datasets import Dataset, DatasetDict
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    root.mkdir(parents=True, exist_ok=True)
    model = root / "model"
    vocab = {
        word: i
        for i, word in enumerate(
            ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "a", "small", "training", "sample"]
        )
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
        model_max_length=128,
    )
    fast.save_pretrained(model)
    LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            pad_token_id=0,
            bos_token_id=2,
            eos_token_id=3,
        )
    ).save_pretrained(model)
    data = root / "data"
    DatasetDict(
        {
            "train": Dataset.from_dict({"text": ["a small training sample " * 4] * 64}),
            "val": Dataset.from_dict({"text": ["a small training sample " * 4] * 4}),
        }
    ).save_to_disk(data)
    config = {
        "redis": "12808",
        "train": {
            "resume": False,
            "epoch": 1,
            "max_num_steps": 8,
            "output_dir": str(root / "checkpoints"),
            "train_batch_per_replica": 2,
            "compile": False,
            "force_use_hf": True,
            "param_dtype": "bfloat16",
            "optm_name": "AdamW",
            "optm_lr": 1e-4,
            "optm_warmup_steps": 0,
            "train_policy": {
                "type": "sft",
                "trainer_type": "stop_live_sft",
                "conversation_column_name": "text",
                "dataset": {"name": str(data), "split": ["train"]},
            },
            "ckpt": {"enable_checkpoint": True, "save_freq": 100, "save_mode": "sync"},
        },
        "policy": {
            "model_name_or_path": str(model),
            "model_max_length": 128,
            "parallelism": {
                "n_init_replicas": 2,
                "tp_size": 1,
                "cp_size": 1,
                "pp_size": 1,
                "dp_shard_size": 1,
                "dp_replicate_size": 1,
            },
        },
        "validation": {
            "enable": True,
            "val_before_train": True,
            "freq": 1,
            "batch_size": 1,
            "dataset": {"name": str(data), "split": ["val"]},
        },
        "logging": {"logger": ["console"]},
    }
    (root / "sft.toml").write_text(toml.dumps(config))


def run():
    from cosmos_rl.policy.trainer.base import TrainerRegistry
    from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer
    from cosmos_rl.dispatcher.api.client import APIClient

    @TrainerRegistry.register("stop_live_sft")
    class StopLiveSFT(SFTTrainer):
        def step_validation(self, *args, **kwargs):
            result = super().step_validation(*args, **kwargs)
            if os.environ.get("STOP_CANARY_DURING_VALIDATION") == "1" and not getattr(
                self, "_stop_canary_requested", False
            ):
                self._stop_canary_requested = True
                accepted = APIClient(role="POLICY").request_stop(
                    "live SFT active validation"
                )
                print(
                    f"[STOP-LIVE] request during validation accepted={accepted}",
                    flush=True,
                )
            print("[STOP-LIVE] validation minibatch completed", flush=True)
            return result

        def step_training(self, *args, **kwargs):
            result = super().step_training(*args, **kwargs)
            step = kwargs["train_step"] if "train_step" in kwargs else args[2]
            if os.environ.get(
                "STOP_CANARY_DURING_VALIDATION"
            ) != "1" and step + 1 >= int(os.environ.get("STOP_CANARY_AFTER", "2")):
                accepted = APIClient(role="POLICY").request_stop("live SFT budget")
                print(
                    f"[STOP-LIVE] request after update={step + 1} accepted={accepted}",
                    flush=True,
                )
            return result

        def _checkpointing(self, *args, **kwargs):
            final = kwargs.get("is_last_step", args[3] if len(args) > 3 else False)
            if final and os.environ.get("STOP_CANARY_FAIL_SAVE") == "1":
                print("[STOP-LIVE] injected final-save failure", flush=True)
                raise RuntimeError("STOP-LIVE injected final-save failure")
            result = super()._checkpointing(*args, **kwargs)
            if final:
                print(
                    f"[STOP-LIVE] final checkpoint step={args[1]} horizon={args[0]}",
                    flush=True,
                )
            return result

    if os.environ["COSMOS_ROLE"].lower() == "controller":
        from cosmos_rl.dispatcher import run_web_panel as panel

        @panel.app.middleware("http")
        async def observe(request, call_next):
            controller = panel.controller
            manager = controller.policy_status_manager
            if (
                manager.policy_init_done
                and os.environ.get("STOP_CANARY_AFTER") == "0"
                and os.environ.get("STOP_CANARY_DURING_VALIDATION") != "1"
                and manager.stop_reason is None
            ):
                await controller.request_stop("live SFT zero budget")
            response = await call_next(request)
            if manager.stop_reason is not None:
                print(
                    f"[STOP-LIVE] terminal={manager.terminal_complete} acks={len(manager.step_boundary.completed)}/{len(manager.step_boundary.participants)}",
                    flush=True,
                )
            return response

        panel.main()
    else:
        from cosmos_rl.policy.policy_entry import policy_entry

        policy_entry()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--prepare":
        prepare_assets(Path(sys.argv[2]))
    else:
        run()
