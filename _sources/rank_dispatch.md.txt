# Optional rank-affine prompt dispatch

`train.train_policy.data_dispatch_as_rank_in_mesh` defaults to `false`. Normal
shared-queue completion dispatch does not need it. When enabled, dataset indices
select both the rollout replica and the destination policy replica by modulo
their respective mesh sizes in disaggregated mode. A rollout replica can reach
only policy replicas
in its modulo-`gcd(rollout_replicas, policy_replicas)` class.

## Reservation and replacement contract

In disaggregated non-DAPO training, prompt capacity is per requested weight
version **and destination policy replica**, not just a global prompt count.
The controller records destination affinity with the issued producer reservation.
A returned payload cannot change that destination. Its reported generation
version is not used to reconstruct either its original quota or its destination.

Terminal rejection reopens the affected destination's prompt capacity. Partial
losses accumulate per `(requested_version, policy_rank)` and are rounded up to
whole `n_generation` groups there. Losses on two different policy queues must not
share one replacement group: no prompt can replenish both queues. Report replay
and producer retirement use the same original coordinates and cannot reopen a
slot twice. Completed work releases its affinity metadata; trained versions
release their fetch/refill bookkeeping.
Retirement validates against the reserved mesh, not the number of still-live
trainers: the last trainer can leave before rollout-owned slots are released.

The lazy fetcher applies destination quotas before pulling the dataset and
retains the existing bounded scans and separate validation iterator. Global
version assignment must not relabel its already sealed rank-local reservation.

## Admission and bounded surplus

Only complete, balanced sets of policy queues advance the estimated training
version. Staleness admission instead estimates each completion's actual position
in **its own destination queue**, including already reserved lookahead. Surplus
on another queue must not make a needed completion appear one training step
older. The dispatch-time staleness check remains authoritative.

Whole prompt groups may overfill a queue. For batch size `B`, group size `G`
and `P` policy replicas, the ordinary group-rounded capacity is
`P * ceil(B / G) * G`. Replenishment rounding adds fewer than `G` completions
per destination per version. Rank-quota-controlled admission therefore adds
`P * (G - 1)` headroom to the existing soft/hard sample ceilings, so bounded
surplus cannot starve a short peer. Destination quotas still constrain fetching;
this does not disable admission limits or increase the permitted staleness.

Colocated preparation owns its local usable-batch budget and can fetch
replacement groups until ready or genuinely exhausted. Its rank-affine prompt
fetches therefore retain the modulo shard mapping and bounded scans, but do not
apply the disaggregated fixed prompt quota. A local rejection must not strand
preparation at that quota. Colocated workers forward their rollout mesh rank
just like disaggregated workers; paired policies consume that rollout shard
locally rather than rerouting completions through global policy queues.

## End of data

Foreign-rank spill rows do not suppress a source's end-of-data receipt. Already
buffered rows for another source remain available to that source. A known final
iterator does not advance epochs repeatedly while sources drain those rows.

After the final training iterator ends, the existing serialized controller
status tick checks each policy queue's maximum possible remaining capacity:
accepted rows plus sealed lookahead plus producer-owned slots plus
`n_generation * unissued prompts`. If even this optimistic bound cannot fill
one policy batch, future prompt input closes for every producer. Issued work and
reports still drain normally; ordinary end-of-data completion preserves final
validation and checkpointing at the last completed update. It does not invent
samples, optimizer updates, or successful recovery of a lost trainer cohort.

Default shared-queue behavior and DAPO, SFT and validation fetch budgets retain
their existing policies. This does not add elastic mesh recovery,
change advantages or losses, or qualify any particular payload transport.

## Regression validation

`tests/test_rank_dispatch_refill.py` covers rectangular meshes, rank-local
partial-loss rounding, unchanged HTTP retry semantics, both identified and
ordinary reports, reservation affinity, producer retirement and staleness
admission, source-local EOF and conservative partial-tail input closure.
`tests/rank_dispatch_canary.py` runs two real collective optimizer
participants on batches admitted after injected partial failures:

```bash
torchrun --standalone --nproc-per-node=2 tests/rank_dispatch_canary.py --backend gloo
torchrun --standalone --nproc-per-node=2 tests/rank_dispatch_canary.py --backend nccl
```

The canary uses synthetic generation and disables automatic command publication;
it validates controller accounting plus collective progress, not end-to-end
rollout payload transfer or production throughput.
