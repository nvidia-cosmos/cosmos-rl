# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Two-process numerical qualification, launched by test_execution_distributed."""

import json
import os
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from test_execution_objectives import configuration, sample
from cosmos_rl.execution.token_objectives import TokenObjectiveBatch


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=20))
    rank = dist.get_rank()
    all_rows = [sample([0, 1, 2]), sample([1, 0]), sample([2, 1, 0, 2], rl=False)]
    dummy = sample([0, 1])
    dummy["label_ids"] = [-100, -100]
    rows = all_rows[:2] if rank == 0 else [all_rows[2], dummy]
    initial = torch.arange(9, dtype=torch.float32).reshape(3, 3) / 10
    model = DistributedDataParallel(
        torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.03
    )
    config = configuration(supervised=0.7, policy_gradient=0.3)
    window = TokenObjectiveBatch(rows, config, device="cpu")
    for index, row in enumerate(rows):
        ids = torch.tensor(row["input_ids"])[None]
        labels = torch.tensor(row["label_ids"])[None]
        window.loss(model(ids), labels, start=index).backward()
    optimizer.step()

    reference = torch.nn.Embedding.from_pretrained(initial.clone(), freeze=False)
    expected_optimizer = torch.optim.SGD(
        reference.parameters(), lr=0.1, momentum=0.9, weight_decay=0.03
    )
    ce, pg = [], []
    for row in all_rows:
        ids = torch.tensor(row["input_ids"])
        logits = reference(ids)[:-1]
        ce.append(F.cross_entropy(logits, ids[1:], reduction="none"))
        if any(row["_execution_record"].masks["policy_gradient"]):
            logp = logits.log_softmax(-1).gather(-1, ids[1:, None]).squeeze(-1)
            ratio = (logp + 1.4).exp()
            pg.append(-torch.minimum(ratio * 0.3, ratio.clamp(0.8, 1.2) * 0.3))
    (0.7 * torch.cat(ce).mean() + 0.3 * torch.cat(pg).mean()).backward()
    expected_optimizer.step()
    assert torch.allclose(model.module.weight, reference.weight, atol=1e-7)
    actual_momentum = optimizer.state_dict()["state"][0]["momentum_buffer"]
    expected_momentum = expected_optimizer.state_dict()["state"][0]["momentum_buffer"]
    assert torch.allclose(actual_momentum, expected_momentum, atol=1e-7)

    # A whole rank has no contributing tokens but still runs every backward.
    with torch.no_grad():
        model.module.weight.copy_(initial)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.03
    )
    optimizer.zero_grad()
    empty = sample([0, 1], supervised=False, rl=False)
    empty["label_ids"] = [-100, -100]
    empty_rank_rows = all_rows if rank == 0 else [empty] * len(all_rows)
    window = TokenObjectiveBatch(empty_rank_rows, config, device="cpu")
    for index, row in enumerate(empty_rank_rows):
        ids = torch.tensor(row["input_ids"])[None]
        labels = torch.tensor(row["label_ids"])[None]
        window.loss(model(ids), labels, start=index).backward()
    optimizer.step()
    assert torch.allclose(model.module.weight, reference.weight, atol=1e-7)
    assert torch.allclose(
        optimizer.state_dict()["state"][0]["momentum_buffer"],
        expected_momentum,
        atol=1e-7,
    )
    assert TokenObjectiveBatch([empty], config, device="cpu").window.empty

    bad = configuration(supervised=0.7 + rank, policy_gradient=0.3)
    try:
        TokenObjectiveBatch(rows, bad, device="cpu")
    except ValueError as error:
        assert "disagree" in str(error)
    else:
        raise AssertionError("Mismatched objectives entered training")

    broken = rows.copy()
    if rank == 1:
        broken[0] = dict(broken[0])
        broken[0]["_execution_record"] = broken[0]["_execution_record"].model_copy(
            deep=True
        )
        broken[0]["_execution_record"].masks["policy_gradient"] = [True] * (
            len(broken[0]["input_ids"]) - 1
        )
        broken[0]["_execution_record"].behavior_logprobs = None
    try:
        TokenObjectiveBatch(broken, config, device="cpu")
    except ValueError as error:
        assert "behavior" in str(error)
    else:
        raise AssertionError("A peer preparation error entered training")

    teacher_row = sample([0, 1], rl=False)
    teacher_row["_execution_record"].annotations.update(
        teacher={"role": "teacher", "name": "recorded-teacher"},
        teacher_logprobs=torch.zeros(1, 3)
        if rank == 1
        else torch.full((1, 3), -torch.log(torch.tensor(3.0))),
    )
    try:
        TokenObjectiveBatch(
            [teacher_row], configuration(distillation=1), device="cpu", vocab_size=3
        )
    except ValueError as error:
        assert "Teacher targets" in str(error)
    else:
        raise AssertionError("Malformed teacher targets entered training")

    # Source opening agrees both membership and failures before model setup.
    from cosmos_rl.execution.corpus import CorpusWriter
    from cosmos_rl.execution.records import SampleRecord
    from cosmos_rl.policy.config import Config
    from test_resume_data_index import _WorkerHarness

    with tempfile.TemporaryDirectory() as root:
        writer = CorpusWriter(root, run_id="source-agreement")
        writer.commit(
            SampleRecord(
                sample_id=str(rank), occurrence_id=str(rank), inputs={}, targets="data"
            )
        )
        writer.finalize()
        source_config = Config.model_validate(
            {
                "execution": {
                    "path": "train_from_data",
                    "source": {"kind": "corpus", "path": root},
                    "objectives": {"supervised": 1},
                },
                "train": {"train_policy": {"type": "sft"}},
                "policy": {"parallelism": {"tp_size": 1}},
            }
        )
        worker = _WorkerHarness(source_config)
        worker.parallel_dims.mesh = {"dp": SimpleNamespace(get_group=lambda: None)}
        try:
            worker.build_runner()
        except ValueError as error:
            assert "source revision" in str(error)
        else:
            raise AssertionError("Different source snapshots entered training")
        if rank == 1:
            from pathlib import Path

            (Path(root) / "complete.json").unlink()
        try:
            worker.build_runner()
        except ValueError as error:
            assert "not complete" in str(error)
        else:
            raise AssertionError("A rank-local source opening failure entered training")

    # An empty exception message still denotes failure on the whole cohort.
    from cosmos_rl.policy.worker.sft_worker import SFTPolicyWorker

    worker = object.__new__(SFTPolicyWorker)
    worker.parallel_dims = SimpleNamespace(
        mesh={"dp": SimpleNamespace(get_group=lambda: None)}
    )
    worker.config = SimpleNamespace(
        execution=SimpleNamespace(source=SimpleNamespace(kind="corpus"))
    )

    def broken_loader():
        if rank == 1:
            raise RuntimeError("")
        yield 1

    try:
        list(worker.get_batch_from_dataloader(broken_loader()))
    except ValueError as error:
        assert "Corpus preparation failed" in str(error)
    else:
        raise AssertionError("An empty-message peer failure entered training")

    class BrokenSample(dict):
        def __getitem__(self, key):
            if rank == 1:
                raise RuntimeError("")
            return super().__getitem__(key)

    try:
        TokenObjectiveBatch(
            [BrokenSample(label_ids=[0, 1])],
            configuration(supervised=1),
            device="cpu",
        )
    except ValueError as error:
        assert str(error) == ""
    else:
        raise AssertionError("An empty-message peer preparation failure was ignored")

    dist.barrier()
    if rank == 0:
        print(
            json.dumps(
                {
                    "gradient_equivalence": True,
                    "optimizer_equivalence": True,
                    "cohort_disagreement": True,
                    "peer_preparation_failure": True,
                    "empty_rank": True,
                    "all_empty": True,
                    "malformed_teacher_agreement": True,
                    "source_revision_agreement": True,
                    "source_opening_failure_agreement": True,
                }
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    main()
