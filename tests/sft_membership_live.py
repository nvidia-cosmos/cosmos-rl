# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Full-launcher multi-replica SFT membership containment, using a tiny Llama.

python tests/sft_membership_live.py --prepare /tmp/sft-membership-assets
SFT_MEMBERSHIP_CASE=healthy cosmos-rl --config /tmp/sft-membership-assets/sft.toml \
    --policy 2 --rollout 0 tests/sft_membership_live.py

Other cases: late-join, replacement, departure, rebuild. The last two deliberately
terminate the controller; require the INCOMPLETE marker, not merely nonzero exit.
No claim of peer recovery or scheduler-wide failure propagation. Use finite jobs.
"""

import os
from pathlib import Path
import sys


def prepare(root):
    import toml
    from requested_stop_sft_live import prepare_assets

    prepare_assets(root)
    config = toml.load(root / "sft.toml")
    config["train"]["train_policy"]["trainer_type"] = "membership_live_sft"
    config["train"]["max_num_steps"] = 2
    (root / "sft.toml").write_text(toml.dumps(config))


def run():
    from cosmos_rl.policy.trainer.base import TrainerRegistry
    from cosmos_rl.policy.trainer.llm_trainer.sft_trainer import SFTTrainer

    case = os.environ.get("SFT_MEMBERSHIP_CASE", "healthy")
    if case not in {"healthy", "late-join", "replacement", "departure", "rebuild"}:
        raise ValueError(f"Unknown membership case {case}")

    @TrainerRegistry.register("membership_live_sft")
    class MembershipSFT(SFTTrainer):
        def step_training(self, *args, **kwargs):
            result = super().step_training(*args, **kwargs)
            step = kwargs["train_step"] + 1
            # Observe real post-optimizer model state, without modifying the
            # objective, reduction or training schedule.
            import hashlib
            import torch

            digest = hashlib.sha256()
            for model in self.model_parts:
                for parameter in model.parameters():
                    value = parameter.detach()
                    if hasattr(value, "to_local"):
                        value = value.to_local()
                    digest.update(
                        value.cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    )
            print(
                f"SFT_MEMBERSHIP_UPDATE case={case} step={step} digest={digest.hexdigest()}",
                flush=True,
            )
            return result

    if os.environ["COSMOS_ROLE"].lower() != "controller":
        from cosmos_rl.policy.policy_entry import policy_entry

        policy_entry()
        return

    from copy import copy
    from starlette.responses import JSONResponse
    from cosmos_rl.dispatcher import run_web_panel as panel
    from cosmos_rl.utils.api_suffix import COSMOS_API_POLICY_TRAIN_ACK_SUFFIX

    injected = False

    @panel.app.middleware("http")
    async def observe(request, call_next):
        nonlocal injected
        status = panel.controller.policy_status_manager
        is_ack = request.url.path == COSMOS_API_POLICY_TRAIN_ACK_SUFFIX
        body = await request.json() if is_ack else None
        if (
            is_ack
            and "val/avg_loss" not in body.get("report_data", {})
            and not injected
        ):
            injected = True
            names = frozenset(status.sft_cohort)
            assert len(names) == 2
            before = status.remain_samples_num
            original = status.policy_replicas[body["replica_name"]]
            if case in {"late-join", "replacement"}:
                extra = copy(next(iter(original.atoms.values())))
                extra.report_session_id = "injected-replacement"
                if case == "late-join":
                    extra.replica_name = "injected-late-replica"
                try:
                    status.register(
                        extra, status.config, panel.controller.rollout_status_manager
                    )
                except (ValueError, RuntimeError) as error:
                    expected = (
                        "membership is sealed"
                        if case == "late-join"
                        else "changed identity"
                    )
                    assert expected in str(error), error
                else:
                    raise AssertionError(
                        "Membership injection was incorrectly accepted"
                    )
                assert frozenset(status.policy_replicas) == names
                assert status.remain_samples_num == before
                assert status.terminal_error is None
                print(
                    f"SFT_MEMBERSHIP_REJECTED case={case} cohort_unchanged=True",
                    flush=True,
                )
            elif case in {"departure", "rebuild"}:
                try:
                    if case == "departure":
                        status.unregister(
                            next(name for name in names if name != original.name)
                        )
                    else:
                        status.trigger_rebuild_mesh(
                            status.get_all_atoms_arrived_replicas()
                        )
                except RuntimeError as error:
                    assert "SFT completion is uncertain" in str(error), error
                else:
                    raise AssertionError("Uncertain execution was incorrectly accepted")
                assert status.terminal_error is not None
                assert not status.training_finished()
                assert status.remain_samples_num == before
                assert frozenset(status.sft_cohort) == names
                assert not any(
                    g.settled
                    for (val, _), g in status.sft_ack_groups.items()
                    if not val
                )
                print(
                    f"SFT_MEMBERSHIP_INCOMPLETE case={case} original_cohort=True accounting_unchanged=True",
                    flush=True,
                )
                return JSONResponse(
                    status_code=503, content={"error": str(status.terminal_error)}
                )
        response = await call_next(request)
        if is_ack and response.status_code == 200:
            phase = "val/avg_loss" in body.get("report_data", {})
            group = status.sft_ack_groups[(phase, body["weight_step"])]
            assert group.participants == frozenset(status.sft_cohort)
            print(
                f"SFT_MEMBERSHIP_ACK case={case} validation={phase} step={group.step} "
                f"receipts={len(group.report_digests)}/2 settled={group.settled} "
                f"finished={status.training_finished()}",
                flush=True,
            )
            if not phase and len(group.report_digests) == 1:
                assert not status.training_finished()
            if not phase and group.step == 2 and group.settled:
                print(
                    f"SFT_MEMBERSHIP_FINAL case={case} steps=2 original_cohort=True",
                    flush=True,
                )
        return response

    panel.main()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--prepare":
        prepare(Path(sys.argv[2]))
    else:
        run()
