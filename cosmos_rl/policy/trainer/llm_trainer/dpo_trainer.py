# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# DPO (Direct Preference Optimization) Trainer.
# Supports TRL-style loss combinations as in MPO paper:
#   - sigmoid: preference loss -log(sigmoid(β * (log π(chosen) - log π(rejected))))
#   - bco_pair: quality loss -logsigmoid(β*chosen) - logsigmoid(-β*rejected)
#   - sft: cross-entropy on chosen response tokens
#
# MPO default: loss_type=["sigmoid", "bco_pair", "sft"], loss_weights=[0.8, 0.2, 1.0]
#
# Batch format: list of {"chosen": model_input_dict, "rejected": model_input_dict}
# Each dict must have: input_ids, logprob_masks (1=response tokens), pixel_values, etc.
# Requires a data_packer with dpo_collate_fn / dpo_process_sample for DPO format.

import torch
import torch.nn.functional as F
import torch.distributed as dist
from functools import partial
from typing import Optional, List, Dict, Any

from cosmos_rl.utils.parallelism import ParallelDims
from cosmos_rl.policy.config import Config as CosmosConfig
from cosmos_rl.policy.trainer.optm import build_lr_schedulers
from cosmos_rl.utils.logging import logger
import cosmos_rl.utils.util as util
import cosmos_rl.utils.distributed as dist_util
from cosmos_rl.dispatcher.data.packer import BaseDataPacker

from cosmos_rl.policy.trainer.llm_trainer.llm_trainer import LLMTrainer
from cosmos_rl.policy.trainer.base import TrainerRegistry


def dpo_loss(
    chosen_logps: torch.Tensor,  # [batch_size]
    rejected_logps: torch.Tensor,  # [batch_size]
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss: -log(sigmoid(β * (log π(chosen) - log π(rejected))))
    """
    logits = beta * (chosen_logps - rejected_logps)
    loss = -F.logsigmoid(logits).mean()
    return loss


@TrainerRegistry.register(trainer_type="dpo")
class DPOTrainer(LLMTrainer):
    def __init__(
        self,
        config: CosmosConfig,
        parallel_dims: ParallelDims,
        train_stream: torch.cuda.Stream,
        data_packer: Optional[BaseDataPacker] = None,
        val_data_packer: Optional[BaseDataPacker] = None,
        **kwargs,
    ):
        if parallel_dims.pp_enabled:
            raise NotImplementedError(
                "DPOTrainer does not support pipeline parallelism (pp_size > 1). "
                "DPO requires dual forward (chosen + rejected) which is incompatible with the current PP design."
            )

        super(DPOTrainer, self).__init__(
            config,
            parallel_dims,
            train_stream=train_stream,
            data_packer=data_packer,
            val_data_packer=val_data_packer,
            **kwargs,
        )

        custom = getattr(config, "custom", None) or {}
        self.beta = getattr(
            config.train.train_policy,
            "dpo_beta",
            custom.get("dpo_beta", 0.1)
            if isinstance(custom, dict)
            else getattr(custom, "dpo_beta", 0.1),
        )
        # MPO: loss_type=["sigmoid", "bco_pair", "sft"], loss_weights=[0.8, 0.2, 1.0]
        self.loss_types = (
            custom.get("dpo_loss_type", ["sigmoid"])
            if isinstance(custom, dict)
            else getattr(custom, "dpo_loss_type", ["sigmoid"])
        )
        self.loss_weights = (
            custom.get("dpo_loss_weights", [1.0])
            if isinstance(custom, dict)
            else getattr(custom, "dpo_loss_weights", [1.0])
        )
        if len(self.loss_weights) != len(self.loss_types):
            self.loss_weights = [1.0] * len(self.loss_types)
        self.enable_dp_load_balancing = False

    def load_model(self):
        """Load model weights from checkpoint if available. Required by SFTPolicyWorker.build_runner."""
        ckpt_total_steps = 0
        train_step = 0
        ckpt_extra_vars = {}
        if (
            not self.parallel_dims.dp_replicate_enabled
        ) or self.parallel_dims.dp_replicate_coord[0] == 0:
            if self.config.train.resume:
                try:
                    ckpt_extra_vars = self.model_resume_from_checkpoint()
                    ckpt_total_steps = ckpt_extra_vars.get("total_steps", 0)
                    train_step = ckpt_extra_vars.get("step", 0)
                except Exception as e:
                    logger.error(
                        f"Cannot resume due to error: {e}. Trying to load from HuggingFace..."
                    )
                    self.lr_schedulers = None
                    self.build_optimizers()
                    self.model.load_hf_weights(
                        self.config.policy.model_safetensor_path
                        or self.config.policy.model_name_or_path,
                        self.parallel_dims,
                        self.device,
                        revision=self.config.policy.model_revision,
                    )
            else:
                self.model_load_from_hf()

        if self.parallel_dims.dp_replicate_enabled:
            if self.config.train.resume:
                ckpt_total_steps = dist_util.broadcast_object_cpu(
                    ckpt_total_steps,
                    group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                    group_src=0,
                )
                train_step = dist_util.broadcast_object_cpu(
                    train_step,
                    group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                    group_src=0,
                )
                if (
                    self.parallel_dims.dp_replicate_coord[0] != 0
                    and ckpt_total_steps > 0
                ):
                    self.lr_schedulers = build_lr_schedulers(
                        self.optimizers, self.config, ckpt_total_steps
                    )
                if ckpt_total_steps > 0:
                    assert self.lr_schedulers is not None

            send_recv_hook = partial(
                dist.broadcast,
                group=self.parallel_dims.mesh["dp_replicate"].get_group(),
                group_src=0,
            )
            len_params = self.sync_all_states(
                is_send=self.parallel_dims.dp_replicate_coord[0] == 0,
                send_hook=send_recv_hook,
                recv_hook=send_recv_hook,
            )
            logger.info(
                f"Synchronized {len_params} parameters across data parallel replicas."
            )

        self.model.train()
        return ckpt_total_steps, train_step, ckpt_extra_vars

    def _compute_logprobs_and_sum(
        self,
        batch: Dict[str, Any],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token log probs and sum over response tokens per sequence."""
        input_ids = batch["input_ids"]
        logprob_masks = batch.get("logprob_masks")
        if logprob_masks is None:
            # Fallback: use label_ids != -100
            label_ids = batch.get("label_ids")
            if label_ids is not None:
                logprob_masks = label_ids != -100
            else:
                raise ValueError("DPO batch needs logprob_masks or label_ids")

        logps, cu_seqlens, _ = util.compute_logprobs(
            input_ids,
            logprob_masks,
            logits,
            is_full_logits=True,
        )

        # Sum log probs per sequence
        bsz = input_ids.shape[0]
        per_seq_logps = logps.new_zeros(bsz)
        for i in range(bsz):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            if end > start:
                per_seq_logps[i] = logps[start:end].sum()
        return per_seq_logps

    def _concat_dpo_batches(
        self,
        chosen_batch: Dict[str, Any],
        rejected_batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Concatenate chosen and rejected batches along batch dim for single forward."""
        combined = {}
        all_keys = set(chosen_batch.keys()) | set(rejected_batch.keys())
        for k in all_keys:
            c, r = chosen_batch.get(k), rejected_batch.get(k)
            if c is None and r is None:
                combined[k] = None
            elif c is None:
                combined[k] = r
            elif r is None:
                combined[k] = c
            elif isinstance(c, torch.Tensor) and isinstance(r, torch.Tensor):
                combined[k] = torch.cat([c, r], dim=0)
            elif isinstance(c, (list, tuple)) and isinstance(r, (list, tuple)):
                combined[k] = c + r
            else:
                combined[k] = c
        return combined

    def _compute_sft_loss(
        self, chosen_batch: Dict[str, Any], chosen_logits: torch.Tensor
    ) -> torch.Tensor:
        """Cross-entropy on chosen response tokens (TRL sft loss)."""
        shift_logits = (
            chosen_logits[..., :-1, :].contiguous().view(-1, chosen_logits.size(-1))
        )
        shift_labels = chosen_batch["input_ids"][..., 1:].contiguous().view(-1)
        # Mask: predict only on response positions (logprob_masks=1 at target position)
        logprob_masks = chosen_batch.get("logprob_masks")
        if logprob_masks is None:
            logprob_masks = (
                chosen_batch.get("label_ids", chosen_batch["input_ids"]) != -100
            ).long()
        shift_mask = logprob_masks[..., 1:].contiguous().view(-1).bool()
        if shift_mask.sum() == 0:
            return chosen_logits.new_zeros(1)
        loss = F.cross_entropy(
            shift_logits[shift_mask],
            shift_labels[shift_mask],
            reduction="mean",
        )
        return loss

    def _dpo_forward_and_loss(
        self,
        chosen_batch: Dict[str, Any],
        rejected_batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Single forward: concat chosen+rejected, run one forward, compute combined loss (MPO-style)."""
        self.model.train()
        chosen_batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in chosen_batch.items()
        }
        rejected_batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in rejected_batch.items()
        }

        # Concatenate chosen and rejected for single forward (avoids issues with dual forward)
        combined_batch = self._concat_dpo_batches(chosen_batch, rejected_batch)

        # Ensure position_ids for combined batch
        if "position_ids" not in combined_batch:
            position_ids, _, _ = self.model.get_position_ids(**combined_batch)
            combined_batch["position_ids"] = position_ids

        with self.act_offloading_ctx_manager:
            output = self.model(**combined_batch)

        bsz = chosen_batch["input_ids"].shape[0]
        # Split logits: first half = chosen, second half = rejected
        chosen_logits = output.logits[:bsz]
        rejected_logits = output.logits[bsz:]

        chosen_logps = self._compute_logprobs_and_sum(chosen_batch, chosen_logits)
        rejected_logps = self._compute_logprobs_and_sum(rejected_batch, rejected_logits)

        # Combined loss (MPO: sigmoid + bco_pair + sft)
        total_loss = chosen_logps.new_zeros(1)
        for loss_type, weight in zip(self.loss_types, self.loss_weights, strict=False):
            if loss_type == "sigmoid":
                total_loss = total_loss + weight * dpo_loss(
                    chosen_logps, rejected_logps, self.beta
                )
            elif loss_type == "bco_pair":
                # TRL bco_pair: -logsigmoid(β*chosen) - logsigmoid(-β*rejected) for quality
                chosen_rewards = self.beta * chosen_logps
                rejected_rewards = self.beta * rejected_logps
                bco = (
                    -F.logsigmoid(chosen_rewards) - F.logsigmoid(-rejected_rewards)
                ).mean()
                total_loss = total_loss + weight * bco
            elif loss_type == "sft":
                sft_loss = self._compute_sft_loss(chosen_batch, chosen_logits)
                total_loss = total_loss + weight * sft_loss
            else:
                raise ValueError(
                    f"Unknown dpo_loss_type: {loss_type}. Use sigmoid, bco_pair, sft."
                )

        return total_loss

    def step_training(
        self,
        global_batch: List[Dict[str, Any]],
        total_steps: int,
        train_step: int,
        save_freq: int,
        inter_policy_nccl: Optional[dist_util.HighAvailabilitylNccl] = None,
        data_arrival_event: Optional[torch.cuda.Event] = None,
    ):
        if self.lr_schedulers is None:
            assert train_step == 0
            self.lr_schedulers = build_lr_schedulers(
                self.optimizers, self.config, total_steps
            )

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        self.optimizers.zero_grad()

        # global_batch: list of {"chosen": {...}, "rejected": {...}}
        # Use data_packer.dpo_collate_fn if available, else assume already batched
        if hasattr(self.data_packer, "dpo_collate_fn"):
            chosen_batch, rejected_batch = self.data_packer.dpo_collate_fn(global_batch)
        else:
            # Fallback: stack chosen and rejected from list
            chosen_list = [x["chosen"] for x in global_batch]
            rejected_list = [x["rejected"] for x in global_batch]
            chosen_batch = self._stack_dpo_batch(chosen_list)
            rejected_batch = self._stack_dpo_batch(rejected_list)

        loss = self._dpo_forward_and_loss(chosen_batch, rejected_batch)
        loss.backward()

        all_params = [
            p
            for m in [model for model in self.model_parts if model is not None]
            for p in m.parameters()
        ]
        grad_norm = dist_util.gradient_norm_clipping(
            all_params,
            self.config.train.optm_grad_norm_clip,
            foreach=True,
            pp_mesh=self.parallel_dims.mesh["pp"]
            if self.parallel_dims.pp_enabled
            else None,
            return_norm_only=(self.config.train.optm_grad_norm_clip <= 0.0),
        )

        self.optimizers.step()
        self.lr_schedulers.step()

        step_hook_report_data = self.model.step_hook(train_step)
        report_data = step_hook_report_data if step_hook_report_data is not None else {}

        end_event.record()

        loss_cpu = loss.detach().cpu()
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
        ):
            torch.distributed.all_reduce(
                loss_cpu,
                op=torch.distributed.ReduceOp.AVG,
                group=self.parallel_dims.mesh["dp_cp"].get_group(),
            )

        if self.config.logging.logger and data_arrival_event is not None:
            assert end_event.query()
            fwd_bwd_time = start_event.elapsed_time(end_event) / 1000.0
            batch_arrival_time = data_arrival_event.elapsed_time(start_event) / 1000.0
            report_data["train/loss_avg"] = loss_cpu.item()
            report_data["train/iteration_time"] = fwd_bwd_time
            report_data["train/batch_arrival_time_mean"] = batch_arrival_time
            report_data["optimizer/grad_norm"] = (
                grad_norm if grad_norm is not None else -1.0
            )

        return report_data

    def _stack_dpo_batch(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Stack a list of per-sample dicts into a batched dict. Pads sequences to max len."""
        pad_id = self.data_packer.pad_token_id
        batched = {}

        # Handle input_ids, label_ids, logprob_masks with padding
        for key in ["input_ids", "label_ids", "logprob_masks"]:
            if key in items[0]:
                tensors = [x[key] for x in items]
                max_len = max(t.shape[-1] for t in tensors)
                padded = []
                for t in tensors:
                    if t.shape[-1] < max_len:
                        pad_val = (
                            pad_id
                            if key == "input_ids"
                            else (-100 if key == "label_ids" else 0)
                        )
                        t = F.pad(t, (0, max_len - t.shape[-1]), value=pad_val)
                    padded.append(t)
                batched[key] = torch.stack(padded)

        # Other tensors: pad or stack as appropriate
        for key in items[0].keys():
            if key in batched:
                continue
            vals = [x[key] for x in items]
            if isinstance(vals[0], torch.Tensor):
                if vals[0].dim() == 0:
                    batched[key] = torch.stack(vals)
                elif vals[0].shape[0] == 1 and all(
                    v.shape == vals[0].shape for v in vals
                ):
                    batched[key] = torch.cat(vals, dim=0)
                else:
                    batched[key] = torch.stack(vals)
            elif isinstance(vals[0], (list, tuple)):
                batched[key] = sum(vals, [])
            else:
                batched[key] = vals
        return batched

    def step_validation(
        self,
        global_batch,
        train_step: int,
        total_steps: int,
    ) -> float:
        """DPO validation: compute validation loss (no backward)."""
        with torch.no_grad():
            if hasattr(self.data_packer, "dpo_collate_fn"):
                chosen_batch, rejected_batch = self.data_packer.dpo_collate_fn(
                    global_batch
                )
            else:
                chosen_list = [x["chosen"] for x in global_batch]
                rejected_list = [x["rejected"] for x in global_batch]
                chosen_batch = self._stack_dpo_batch(chosen_list)
                rejected_batch = self._stack_dpo_batch(rejected_list)

            chosen_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in chosen_batch.items()
            }
            rejected_batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in rejected_batch.items()
            }
            loss = self._dpo_forward_and_loss(chosen_batch, rejected_batch)
        return loss.item()

    def build_lr_schedulers(self):
        """Stub for abstract method; LR schedulers built lazily in step_training."""
        pass

    @property
    def pp_loss_fn(self):
        """
        Pipeline parallelism loss function. DPO computes loss in step_training
        (chosen/rejected forward); this is used by parallelize_fn when pp_size > 1.
        Returns a pass-through that accepts (loss, target) and returns loss.mean().
        """

        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            return loss.mean()

        return fake_compute_loss
