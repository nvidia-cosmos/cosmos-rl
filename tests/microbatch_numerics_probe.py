# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Compare native BF16 GPT/FSDP2/SFT updates with an independent FP64 oracle.

The reference uses explicit causal attention, FP64 RMSNorm/RoPE/MLP and one
global masked-token mean. Native runs retain FP32 master weights/AdamW state.
This is a small-model arithmetic qualification, not a model-quality benchmark.
"""

from cosmos_rl.dispatcher.data.schema import Rollout
import argparse
import copy
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers import Qwen2Config

from cosmos_rl.execution.config import ExecutionConfig
from cosmos_rl.execution.records import SampleRecord
from cosmos_rl.policy.kernel.modeling_utils import init_flash_attn_meta
from cosmos_rl.policy.model.gpt import GPT, GPTArgs
from cosmos_rl.policy.model.gpt.parallelize import apply_fsdp
from test_execution_native_sft import trainer


LIMITS = dict(
    loss_relative_error=0.02,
    gradient_relative_l2=0.05,
    gradient_cosine_min=0.999,
    update_relative_l2=0.15,
)
ADAM = dict(lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def flatten(values):
    return torch.cat([value.detach().double().flatten().cpu() for value in values])


def relative(actual, expected):
    return ((actual - expected).norm() / expected.norm().clamp_min(1e-30)).item()


def reference_forward(weights, tokens, *, layers=2, heads=2, kv_heads=1, dim=64):
    """Independent functional Qwen-style decoder, with no native model/loss calls."""

    def linear(x, prefix):
        return F.linear(x, weights[prefix + ".weight"], weights.get(prefix + ".bias"))

    def norm(x, name):
        return (
            x
            * (x.square().mean(-1, keepdim=True) + 1e-6).rsqrt()
            * weights[name + ".weight"]
        )

    def rotate(x):
        left, right = x.chunk(2, dim=-1)
        return torch.cat((-right, left), -1)

    batch, length = tokens.shape
    x = F.embedding(tokens, weights["embed_tokens.weight"])
    frequencies = torch.arange(length, dtype=torch.float64)[:, None] / (
        10000 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)
    )
    angles = torch.cat((frequencies, frequencies), -1)[None, :, None]
    for index in range(layers):
        name = f"layers.{index}."
        normalized = norm(x, name + "input_layernorm")
        q = linear(normalized, name + "self_attn.q_proj").reshape(
            batch, length, heads, dim
        )
        k = linear(normalized, name + "self_attn.k_proj").reshape(
            batch, length, kv_heads, dim
        )
        v = linear(normalized, name + "self_attn.v_proj").reshape(
            batch, length, kv_heads, dim
        )
        q = (q * angles.cos() + rotate(q) * angles.sin()).transpose(1, 2)
        k = (k * angles.cos() + rotate(k) * angles.sin()).transpose(1, 2)
        k = k.repeat_interleave(heads // kv_heads, dim=1)
        v = v.transpose(1, 2).repeat_interleave(heads // kv_heads, dim=1)
        scores = q @ k.transpose(-1, -2) / math.sqrt(dim)
        scores = scores.masked_fill(
            torch.ones(length, length, dtype=torch.bool).triu(1), -torch.inf
        )
        attention = (
            (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, heads * dim)
        )
        x = x + linear(attention, name + "self_attn.o_proj")
        normalized = norm(x, name + "post_attention_layernorm")
        hidden = F.silu(linear(normalized, name + "mlp.gate_proj")) * linear(
            normalized, name + "mlp.up_proj"
        )
        x = x + linear(hidden, name + "mlp.down_proj")
    return linear(norm(x, "norm"), "lm_head")


def examples(seed):
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for index, length in enumerate((31, 23, 17, 9)):
        tokens = torch.randint(0, 128, (length,), generator=generator).tolist()
        labels = [token if i % 4 else -100 for i, token in enumerate(tokens)]
        mask = [i % 6 != 0 and index != 3 for i in range(length - 1)]
        record = SampleRecord(
            sample_id=str(index),
            occurrence_id=str(index),
            masks={"supervised": mask},
            rollout=Rollout(prompt={}, completion={}),
        )
        rows.append(
            dict(
                token_ids=tokens,
                label_ids=labels,
                image=torch.zeros(3, 2, 2),
                _execution_record=record,
            )
        )
    return rows


class TokenPacker:
    batch_size = staticmethod(len)
    slice_batch = staticmethod(lambda rows, start, end: rows[start:end])
    sft_compute_max_len = staticmethod(
        lambda rows: max(len(r["token_ids"]) for r in rows)
    )

    def sft_collate_fn(self, rows, computed_max_len, ignore_label_id):
        return {
            "input_ids": torch.tensor(
                [
                    r["token_ids"] + [0] * (computed_max_len - len(r["token_ids"]))
                    for r in rows
                ]
            ),
            "label_ids": torch.tensor(
                [
                    r["label_ids"] + [-100] * (computed_max_len - len(r["label_ids"]))
                    for r in rows
                ]
            ),
        }


def qualify(output):
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    init_flash_attn_meta(deterministic=True, compile=False)
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
    results = []
    for seed in (13, 41):
        torch.manual_seed(seed)
        hf = Qwen2Config(
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=128,
            tie_word_embeddings=False,
            rope_theta=10000.0,
        )
        args = GPTArgs(
            dim=128,
            ffn_dim=256,
            n_layers=2,
            n_heads=2,
            n_kv_heads=1,
            head_dim=64,
            vocab_size=128,
            max_seq_len=128,
            biases=["q_proj", "k_proj", "v_proj"],
            hf_config=hf,
        )
        original = GPT(args)
        initial = copy.deepcopy(original.state_dict())
        reference = {
            name: value.double().clone().requires_grad_()
            for name, value in initial.items()
            if name in dict(original.named_parameters())
        }
        rows = examples(seed)
        token_losses = []
        for row in rows:
            logits = reference_forward(reference, torch.tensor([row["token_ids"]]))[
                0, :-1
            ]
            labels = torch.tensor(row["label_ids"][1:])
            mask = torch.tensor(row["_execution_record"].masks["supervised"]) & (
                labels != -100
            )
            loss = F.cross_entropy(logits, labels, reduction="none", ignore_index=-100)
            token_losses.append(loss[mask])
        reference_loss = 0.7 * torch.cat(token_losses).mean()
        optimizer = torch.optim.AdamW(reference.values(), **ADAM, foreach=False)
        reference_loss.backward()
        reference_gradient = flatten(p.grad for p in reference.values())
        optimizer.step()
        reference_update = flatten(
            reference[name] - initial[name].double() for name in reference
        )
        native_gradients = []
        for mini_batch in (1, 2, 4):
            model = GPT(args).cuda()
            model.load_state_dict(initial)
            apply_fsdp(model, mesh, torch.bfloat16, torch.float32, pp_enabled=False)
            actual = trainer(model, mini_batch)
            actual.device = torch.device("cuda", 0)
            actual.data_packer = TokenPacker()
            actual.config.execution = ExecutionConfig(
                path="train_from_data", objectives={"supervised": 0.7}
            )
            actual.config.policy.model_max_length = 128
            actual.config.logging.logger = ["console"]
            actual.config.logging.report_mfu = False
            actual.parallel_dims.dp_shard_enabled = False
            actual.global_rank = 0
            actual.optimizers = torch.optim.AdamW(
                model.parameters(), **ADAM, fused=True
            )
            actual.lr_schedulers = torch.optim.lr_scheduler.LambdaLR(
                actual.optimizers, lambda _: 1
            )
            arrival = torch.cuda.Event(enable_timing=True)
            arrival.record()
            report = actual.step_training(
                rows,
                total_steps=1,
                train_step=0,
                save_freq=1,
                data_arrival_event=arrival,
            )
            native_loss = float(report["train/loss_avg"])
            params = dict(model.named_parameters())
            gradient = flatten(local(params[name].grad) for name in reference)
            update = flatten(
                local(params[name]).cpu() - initial[name] for name in reference
            )
            cosine = F.cosine_similarity(gradient, reference_gradient, dim=0).item()
            row = dict(
                seed=seed,
                mini_batch=mini_batch,
                native_loss=native_loss,
                reference_loss=reference_loss.item(),
                loss_relative_error=abs(native_loss - reference_loss.item())
                / reference_loss.item(),
                gradient_relative_l2=relative(gradient, reference_gradient),
                gradient_cosine=cosine,
                update_relative_l2=relative(update, reference_update),
            )
            row["passed"] = (
                row["loss_relative_error"] <= LIMITS["loss_relative_error"]
                and row["gradient_relative_l2"] <= LIMITS["gradient_relative_l2"]
                and cosine >= LIMITS["gradient_cosine_min"]
                and row["update_relative_l2"] <= LIMITS["update_relative_l2"]
            )
            native_gradients.append(gradient)
            results.append(row)
            print(json.dumps(row), flush=True)
            del actual, model, params
        for row, gradient in zip(results[-3:], native_gradients, strict=True):
            row["gradient_relative_l2_vs_microbatch1"] = relative(
                gradient, native_gradients[0]
            )
    negative_control = relative(reference_gradient * 2, reference_gradient)
    assert negative_control > LIMITS["gradient_relative_l2"]
    result = dict(
        results=results,
        limits=LIMITS,
        negative_control_double_gradient_error=negative_control,
        passed=all(r["passed"] for r in results),
        reference="Independent FP64 functional GPT and masked global-token mean; FP64 AdamW",
        native="GPT + FSDP2 BF16 computation, FP32 master/reduction + native SFTTrainer + fused AdamW",
        device=torch.cuda.get_device_name(),
        torch_version=torch.__version__,
    )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    assert result["passed"], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as directory:
        dist.init_process_group(
            "nccl", init_method="file://" + directory + "/init", rank=0, world_size=1
        )
        try:
            qualify(args.output)
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
