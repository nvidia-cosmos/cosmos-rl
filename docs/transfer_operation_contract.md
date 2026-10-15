# Transfer readiness and accepted-operation ownership

P2R readiness is a command-scoped CPU agreement before native initialization,
including warm communicators. Every source and receiver rank must arrive.
`COSMOS_P2R_READY_TIMEOUT_S` defaults to 600 seconds from command creation;
`COSMOS_ROLLOUT_MESH_BUILD_TIMEOUT_MS` independently bounds native initialization.
These are different phases, not a longer native timeout. Commands preserve their
deadline across serialization. Changed metadata/declarations fail the agreement;
expired operations cannot be revived. Deploy matching controller/worker versions.

NCCL payload requests have unique operation identities. Atomic acceptance and
cancellation prevent duplicate sends and lost-reply ambiguity. Accepted sends
retain the UID value and payload lease, not merely an expiring Redis key. The
original request budget includes queueing, communicator initialization, enqueue
and device completion. A hard watchdog runs independently of the native caller;
a per-rendezvous batched observer correlates peer failure even during blocked
initialization. Control-plane observation cannot delay the hard deadline.

Pre-accept missing/rejected data remains skippable. Once accepted, a failure
whose native completion is unknown must not masquerade as a missing sample.
Sync and prefetch callers preserve terminal errors; native operands and pins
remain owned through terminal exit. This contains failure; it does not recover
a transport or provide cross-node scheduler propagation. Use a finite job limit.
The producer still permits different pairs to run concurrently, while preserving
FIFO order within a pair. Single sends no longer open unnecessary NCCL groups.

Cache invalidation includes queued and in-progress builds. A late native result
from an invalidated generation is aborted, never published or returned. Owned
teardown closes the cache permanently, while an explicit abort permits a fresh
generation. A watchdog does not prove that an abort stopped accessing storage.

UCXX sync/prefetch preserve terminal errors and independent operation deadlines
without cancelling native waiters. Pinned storage is recycled only after proven
copy completion; uncertain request/endpoint ownership remains retained. See
[UCXX operation lifetime](ucxx_operation_lifetime.md) for accepted producer sends,
originating-loop retirement and last-owner context reset as well. These are
ownership/containment contracts, not a UCXX receive-memory budget.

P2R temporary receives now register their
copy-back reader stream before scheduling the update, keeping allocator ownership
through the final read even after Python closures/queue entries disappear. This
adds no host synchronization or global barrier. It does not make the temporary
queue a hard memory cap or provide backend phase isolation. Only recorded
copy-completion events enter that queue; unissued copies remain owned by their
round's completion closures. Later temporary allocations drain older submitted
copies at the configured count, including off-device destinations. One atomic
round may exceed that count: it cannot wait for its own unissued copies while
the NCCL group is open. This is backpressure, not a byte budget. Grouped P2R/R2R
callers separately use a managed local enqueue scope: start/body/end uncertainty
is terminal and a group-owned handle cannot be freed by concurrent abort. This
does not prove peer device completion, recover a partial group, or replace an
end-to-end operation budget.

Portable focused canaries (two GPUs, local `redis-server`):

```bash
python tests/transfer_contract_canary.py --case ready-delay
python tests/transfer_contract_canary.py --case ready-missing
python tests/transfer_contract_canary.py --case queue-uid --prefetch --bounded
python tests/transfer_contract_canary.py --case accepted-failure
python tests/transfer_contract_canary.py --case warm-dead-peer --prefetch
torchrun --standalone --nproc-per-node=2 tests/p2r_copyback_canary.py
```

The parent owns the HTTP/Redis fixtures even after injected worker exit. The
readiness fixture uses the production registry/client, not a full controller.
Fault cases require an explicit transport-fatal marker and prohibit successful
continuation; a timeout of the canary itself is a failure, not passing evidence.

For two-node validation, launch `tests/transfer_contract_distributed_canary.py`
with two torchrun ranks (one GPU per rank), using the same case/prefetch/bounded
arguments. Each supervisor owns a child worker. A separate Gloo control group
collects exit evidence and cleans up the peer after an injected failure; the
receiver must emit its own transport-fatal marker. This tests the transfer
contract across nodes, not a production launcher's failure propagation.

The copy-back canary uses real cached-UID P2P construction, native send/receive,
delayed worker copy-back and allocator churn for two successive updates. Gloo
stands in for its readiness exchange; it does not test colocated IPC routing.
