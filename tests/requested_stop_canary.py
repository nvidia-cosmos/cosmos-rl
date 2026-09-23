# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application-side stop and final-save failure fixture.

Requires the RL-Gym companion modules and its normal role launcher. Select
trainer_type="stop_canary_trainer", checkpointing enabled, and a finite horizon.
Run from the companion project root so its relative assets resolve.

STOP_CANARY_AFTER=0 requests step-zero stop; set 2 for partial progress.
STOP_CANARY_FAIL_SAVE=1 injects final checkpoint failure (expect job failure).
STOP_CANARY_DURING_VALIDATION=1 waits for an active validation round with
multiple prompt requests; enable validation for this case. Check the terminal ACKs,
actual saved step, unchanged horizon, and all role exits, not just batch status.

The published fixture uses per-process output directories to prevent independent
trainers from racing on a shared temporary checkpoint file. These are test-only
checkpoints, not a distributed checkpoint format or a resume implementation.
"""

import os
import faulthandler
from pathlib import Path
import uuid
import torch

from cosmos_rl.utils.model_config import register_local_model_config
from cosmos_rl.utils.no_op_tokenizer import NoOpTokenizer
from cosmos_rl.utils.util import register_tokenizer_loader
from cosmos_rl.policy.trainer.base import TrainerRegistry
from rl_gym.simple_weight_mapper import SimpleRLConfig
from rl_gym.policy.trainer.simple_trainer import SimpleRLTrainer
from rl_gym.rollouts.modular_rollout_worker import ModularRolloutWorker  # noqa: F401
from rl_gym.reward_function import rl_gym_reward_fn
from rl_gym.simple_data_packer import SimpleRLDataPacker, TransportSimpleRLDataPacker

if os.environ.get("STOP_CANARY_STACK_TIMEOUT"):
    faulthandler.dump_traceback_later(
        float(os.environ["STOP_CANARY_STACK_TIMEOUT"]), repeat=True
    )

register_tokenizer_loader(
    predicate=lambda path: str(path).endswith(".toml"),
    loader=lambda path: NoOpTokenizer(),
)
register_local_model_config(
    predicate=lambda path: str(path).endswith(".toml"),
    factory=lambda path: SimpleRLConfig(),
)


@TrainerRegistry.register("stop_canary_trainer")
class StopCanaryTrainer(SimpleRLTrainer):
    def invalidate_checkpoint_completion(self, current_step):
        path = self.checkpoint_root() / "stop-final.complete"
        path.unlink(missing_ok=True)

    def checkpoint_root(self):
        if not hasattr(self, "_canary_checkpoint_root"):
            self._canary_checkpoint_root = (
                Path(self.config.train.output_dir) / f"stop-canary-{uuid.uuid4().hex}"
            )
        return self._canary_checkpoint_root

    def save_checkpoint(
        self, current_step, total_steps, remain_samples_num, is_final=False
    ):
        if is_final and os.environ.get("STOP_CANARY_FAIL_SAVE") == "1":
            print(
                f"[STOP-CANARY] injecting final save failure step={current_step}",
                flush=True,
            )
            raise RuntimeError("STOP-CANARY injected final checkpoint failure")
        root = self.checkpoint_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / "stop-final.pt"
        temporary = root / "stop-final.pt.tmp"
        torch.save(
            {
                "step": current_step,
                "total_steps": total_steps,
                "remain_samples_num": remain_samples_num,
                "model": self.model.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "is_final": is_final,
            },
            temporary,
        )
        temporary.replace(target)
        (root / "stop-final.complete").touch()
        print(
            f"[STOP-CANARY] saved step={current_step} horizon={total_steps} final={is_final} path={target}",
            flush=True,
        )


role = os.environ["COSMOS_ROLE"].lower()
if role == "controller":
    from cosmos_rl.dispatcher import run_web_panel

    @run_web_panel.app.middleware("http")
    async def application_budget(request, call_next):
        controller = run_web_panel.controller
        manager = controller.policy_status_manager
        stop_after = int(os.environ.get("STOP_CANARY_AFTER", "0"))
        rollouts_ready = (
            len(controller.rollout_status_manager.get_all_atoms_arrived_replicas())
            >= controller.config.rollout.parallelism.n_init_replicas
        )
        validation_active = controller.data_fetcher.activated_val_iter is not None
        if request.query_params.get("validation_step") not in (None, ""):
            controller._canary_validation_requests = (
                getattr(controller, "_canary_validation_requests", 0) + 1
            )
        validation_reported = sum(
            len(group)
            for groups in manager.val_report_data.values()
            for group in groups
        )
        validation_ready = os.environ.get("STOP_CANARY_DURING_VALIDATION") != "1" or (
            validation_active
            and getattr(controller, "_canary_validation_requests", 0) >= 2
        )
        if (
            manager.policy_init_done
            and rollouts_ready
            and validation_ready
            and manager.current_step >= stop_after
            and manager.stop_reason is None
        ):
            accepted = await controller.request_stop("stop canary application budget")
            print(
                f"[STOP-CANARY] requested step={manager.current_step} accepted={accepted} validation_active={validation_active} validation_reported={validation_reported}",
                flush=True,
            )
        response = await call_next(request)
        if manager.stop_reason is not None:
            print(
                f"[STOP-CANARY] completion acks={len(manager.completion_acks)}/{len(manager.completion_recipients)} terminal={manager.terminal_complete}",
                flush=True,
            )
        return response

    run_web_panel.main()
elif role == "policy":
    from cosmos_rl.policy.policy_entry import policy_entry
    from cosmos_rl.dispatcher.api.client import APIClient

    metadata = APIClient(role="POLICY").get_controller_metadata()
    # Colocated payloads are already local; no transport prefetch is submitted.
    packer = (
        SimpleRLDataPacker()
        if metadata["config"]["mode"] == "colocated"
        else TransportSimpleRLDataPacker()
    )
    policy_entry(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
elif role == "rollout":
    from cosmos_rl.rollout.rollout_entry import run_rollout

    packer = SimpleRLDataPacker()
    run_rollout(
        data_packer=packer, val_data_packer=packer, reward_fns=[rl_gym_reward_fn]
    )
