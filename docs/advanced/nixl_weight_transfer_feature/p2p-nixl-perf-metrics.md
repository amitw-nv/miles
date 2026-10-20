# P2P NIXL perf metrics (refactored p2p)

**Status: implemented** (not yet run on a multi-node job). Deviations from this plan: the collector
identifies a GPU by `gpu` only (no `pp_rank` / `gathered_dp_rank`, which the log never printed), and version 1 is
marked `(warm-up)` in the section header. Open points 2-4 below stay deferred.

This is the metrics spec for the refactored p2p path
(`amitw/miles-nixl-refactoringcheck`). It replaces the spec written for the
`upstream-exp` design. The goal is the same: on every NIXL P2P weight
transfer, collect the metrics below and append them to `p2p_nixl_perf.log` in
the trainer working directory. No flag; on whenever `--p2p-transport nixl`
(this replaces `--update-weight-transfer-backend nixl`). Rank 0 writes the
file after gathering from sender GPUs.

One `update_weights` is one full weight transfer (all parameters). Later
training steps call it again; each call is its own log section. We do **not**
add those together.

---

## What changed in the architecture, and what it does to the metrics

| # | Refactor change | Effect on measurement |
|---|---|---|
| 1 | `p2p_nixl_perf.py`, `p2p_transfer_utils.py`, `P2PTransferManager`, `wrap_weight_iter`, `_do_p2p_write_one_session`, `_do_nixl_write`, `_get_transfer_ready_params` are gone. `p2p.py` is split into `transports/{nixl,mooncake,remote_shard,rollout_engine_write_threads}.py` and `protocols/utils/*`. | Every hook site moves. The perf module has to be rewritten against new names (table in "Hook sites"). |
| 2 | Transport is chosen by `--p2p-transport`, not `--update-weight-transfer-backend`. | The on/off gate is `args.p2p_transport == "nixl"`. |
| 3 | The shared pinned CPU replica (`load_weights` copy, then RDMA from it) is replaced by `ModelReplica.load_into(buffer, ...)`: sglang's own loader writes straight into a pinned **transfer buffer** with the param storage bound to buffer slices. It also runs bind/unbind and state-compare asserts on every load, and on the first load of each param a zero/0xFF double load plus `count_nonzero`. | `cpu_load` now means `load_into`. The first load of every param is much slower than steady state, so `weight_version` 1 (and any reconnect that builds a new replica) is not representative. |
| 4 | **Two transfer buffers** (`_NUM_TRANSFER_BUFFERS = 2`). `acquire()` blocks only until the previous writes reading that buffer return. Writes are fire-and-forget futures; the main thread no longer joins them per replica. | RDMA now overlaps with the next load and the next all-gather. Old rule "replicas run one after another, so sum wire_time" no longer holds. We must measure how much RDMA is **not** hidden (new `buffer_wait`, `drain`). |
| 5 | Writes run on **one thread per rollout engine** (`RolloutEngineWriteThreads`, `max_workers=1`) instead of a shared 4-worker pool. | Shards on different engines run in parallel; writes to the same engine queue behind each other. Queue wait is real now and must not be folded into `cpu_prep`. |
| 6 | One NIXL WRITE per **(shard, chunk)** carrying all names as descriptors (`NixlTransport._write_shard`), instead of one per session per bucket. Poll loop sleeps 0.5 ms between checks. | `_write_shard` is the new `_do_nixl_write`: same body (descriptor build, `initialize_xfer`, `transfer`, poll, release), so it stays the `wire_time` clock. Poll quantization adds up to ~0.5 ms per write. |
| 7 | Loop nest is deeper: bucket (iterator) -> runner role (`target`, `draft`) -> `ModelParamStager.stage` -> `pack_into_buffers` chunk -> replica target -> `acquire` / `load_into` / `transport.write`. | There are now three loops, not two. Folds are defined below against these. A GPU may also write to the MTP draft runner, so its sums include both roles. |
| 8 | `loads_hf_tensor` drops HF tensors the engine's loader ignores. | Wire bytes are what is actually written, not everything the iterator produced. |
| 9 | First `begin_sync` runs one **extra full iterator pass** (`iter_buckets(materialize=True)`) to learn HF tensor specs, then `map_hf_names` probes and buffer creation. | Gather/convert timers fire during that pass. Counters must be reset when `begin_sync` returns, so the probe pass is not billed to the first transfer. |
| 10 | Iterator became a stream of per-param **units** (`_iter_hf_param_units`), bucketed by `pack_units_by_size` and `assemble_atomic_update_groups`; multimodal tower units are appended. | Time spent in unit assembly, bucketing and tower units lands in the `next(iterator)` residual, not in gather/convert. |
| 11 | Direct iterator, routed experts with **ETP == 1** (`_convert_and_gather_expert_batch`): convert/quantize **before** the gather, then `ExpertGather` moves the already-converted HF tensors (packed uint8 payload, one collective per group; first call also does an `all_gather_object` layout exchange). The ETP > 1 path (`_gather_megatron_expert_batch`) keeps gather-then-convert. | For this path gather and convert are no longer sequential, and there is no TP all-gather for experts. `gather` must be the sum of its parts, not a clock wrapped around the function (that would contain convert). |
| 12 | Non-sender ranks still run the ETP == 1 expert convert + gather (they own an EDP share of the experts). | Their time is not logged (non-senders send `None`), but it shows up on senders as wait inside `gather_ep`. |
| 13 | `WeightTransferProtocol` has `update_weight_metrics` / `pop_metrics`; the actor drains them into `log_perf_data`. | Optional second sink (wandb) in addition to the file. Not part of the first cut. |
| 14 | `updater.update_weights` has no iterator-wrap hook. | We need a small default-identity hook on the protocol base class to time each `next(iterator)` and start `trainer_active_time`. |

Consequence in one line: the old design was **serial** (load, RDMA, load, RDMA)
so all categories added up to the active time. The refactor is **pipelined**,
so the categories overlap and we need explicit numbers for the part of RDMA
that stays on the critical path.

---

## Loops and which loop each metric uses

- **Outer loop:** `for bucket in iter_hf_weights: send_bucket(bucket)`.
  All-gather, convert, quant for the bucket run inside `next(iterator)`.
- **Mid loop:** inside `send_bucket`, `for runner_sender in _runner_senders`:
  `stage`, then `pack_into_buffers` yields one or more **chunks** (each fits
  one transfer buffer).
- **Inner loop:** for each chunk, `for target in replica_targets`:
  `acquire` a buffer, `load_into`, `transport.write` (returns one future per
  remote shard, one write thread per rollout engine).

A **write group** is one `transport.write(...)` call: one (chunk, replica
target) pair fanned out to its shards. Shards of one group run in parallel on
different engine threads.

---

## max_num_wire_bytes_per_trainer

**What it measures:** bytes each trainer GPU sends on the wire in this full
transfer.

**How we measure it:** in `NixlTransport._write_shard`, `sum(tensor.nbytes)`
over `tensors_by_name` for that shard (this equals the published size, checked
by `assert_tensors_fit`). Sum every shard, every group, every runner role.

**Log:** GPU with the most bytes. Printed as decimal GB (10^9), so
`bytes / wire_time` compares directly with a NIC line rate.

---

## wire_time

**What it measures:** time the trainer had RDMA in flight, from entering
`_write_shard` until it returns (descriptor build, `initialize_xfer`,
`transfer`, poll to DONE). Same span as the old `_do_nixl_write`.

**How we measure it:** `_write_shard` records `(t_start, t_end, nbytes)` with
`time.monotonic()` and returns it on its future. No lock on the RDMA path.

Because buffers overlap, summing durations double counts. Two numbers:

- `wire_time` = **union** of all `[t_start, t_end]` intervals on this GPU
  (interval merge, wall clock during which at least one write was in
  flight). This is what `bytes / wire_time` should use.
- `wire_time_sum` = sum over write groups of the **longest** shard in the
  group. This is the old definition (parallel sessions counted once, groups
  summed) and is kept so old and new logs can be compared. It is `>=
  wire_time` when groups overlap.

**Log:** GPU with the largest `wire_time`.

---

## trainer_prep_time

The log has **three** numbers, each with the GPU that owns that max (can be
three different GPUs): max `gpu_prep`, max `cpu_prep`, and max
`gpu_prep + cpu_prep` on the **same** GPU. Never add two category maxes from
different GPUs.

### 1. gpu

**What it measures:** everything the main thread does on the GPU side and in
the CPU load, excluding the wait for buffers and the RDMA itself.

**How we measure it:** a sum of three additive timers per GPU over the full
transfer:

- time **inside** `next(iterator)` (all-gather, HF convert, quant, unit
  assembly, bucketing),
- every `ModelParamStager.stage` call (per runner role, plus the
  `loads_hf_tensor` filter) as `stage`,
- every `load_into` as `cpu_load`.

This equals the old "clock from `next(iterator)` to after staging, plus each
load" when there is one runner role.

**gpu_prep parts.** Gather, convert, stage and cpu_load are extra counters
inside gpu_prep. Rank 0 does **not** pick a max GPU per part: it prints the
numbers from the GPU that owns `argmax(gpu_prep + cpu_prep)`. They need not
add up to that GPU's `gpu_prep`; the rest (unit assembly, bucketing, tower
units, tqdm, checksums) is the residual.

- **gather** — sum of the gather sub-parts below (see item 11: on the ETP == 1
  expert path convert sits between them, so a single wrapping clock would
  include convert).
  - **gather_load** — `_load_or_allocate_params` (copy this rank's shards to
    the GPU or allocate receive buffers, then `cuda.synchronize`); on the
    ETP == 1 expert path, the `.to(device)` of owned experts.
  - **gather_pp** — `_broadcast_across_pp` when PP is gathered; on the ETP == 1
    expert path, the `ExpertGather` over the PP group. `0.00s` when
    `gather_pp` is false.
  - **gather_tp** — `all_gather_params_async` (dense TP, or ETP for routed
    experts on the ETP > 1 path). Not run for ETP == 1 experts.
    - **gather_tp_start**, **gather_tp_wait**, **gather_tp_concat** — the
      three sequential chunks inside it, as before.
  - **gather_ep** — ETP > 1 path: `all_gather_object` (names) plus each EP
    `all_gather` and the wait of those handles. ETP == 1 path: the
    `ExpertGather` call over the combined TP/DP/CP group (payload pack, the
    collective, view unpack, wait). Near zero when `ep.size == 1`.
  - Bridge: `export_hf_weights` is one `next()`, so the whole call is
    `gather` and the inner lines are `0.00s`. This includes the lazy
    `_process_conversion_tasks` `.cuda()` copies.
- **convert** — `convert_to_hf` per param in `_convert_to_hf_param_units`
  (raw, both expert paths), or `_postprocess_and_quantize` (bridge). Includes
  vocab unpad, HF layout, `quantize_params`. Not `load_into`.
- **stage** — `ModelParamStager.stage`, as above.
- **cpu_load** — `load_into`: sglang `load_weights` into the bound transfer
  buffer, plus bind/unbind and state-compare asserts. Sum every chunk, every
  replica target, every runner role.

### 2. cpu

**What it measures:** main-thread host work from the end of `load_into` until
the write is handed to the engine threads.

**How we measure it:** one `time.monotonic()` right after `load_into` returns;
stop when `transport.write(...)` returns (shape checks `assert_tensors_fit`
and executor submit). Sum every group.

Not part of this number, because they are not main-thread host prep any more:

- `write_queue` — time from `write(...)` returning until `_write_shard` starts
  on its engine thread (thread wake plus waiting behind an earlier write to
  the same engine). Per group take the longest shard, sum over groups. This
  is wire backlog, not prep. Logged on its own line.
- descriptor build and `initialize_xfer` — inside `_write_shard`, so inside
  `wire_time`, same as the old `_do_nixl_write`.

### 3. exposed wire (new)

`trainer_prep_time` no longer adds up to anything close to the transfer time
once buffers overlap, so we add the part of RDMA that the main thread actually
waited for:

- `buffer_wait` — time blocked in `TransferBuffers.acquire()` waiting for the
  writes that still read the next buffer (sum over the transfer).
- `drain` — `wait_for_all_writes()` in `after_base_weights` (tail of the last
  writes).
- `trainer_exposed_wire_time = buffer_wait + drain`. If this is small relative
  to `wire_time`, the double buffer hides RDMA; if it is close to `wire_time`,
  the transfer is wire-bound.

**Log:** GPU with the largest `buffer_wait + drain`.

---

## trainer_active_time

**What it measures:** wall clock of this full transfer: first all-gather until
every write is finished. Not a sum of anything (the loops overlap).

**How we measure it:** start at the first `next(iterator)` of this
`update_weights`; stop right after `TransferBuffers.wait_for_all_writes()` in
`after_base_weights`. The gloo barrier after it is not included.

**Log:** GPU with the largest active time.

---

## How to implement (plan, nothing written yet)

New file `miles/backends/training_utils/weight_update/protocols/utils/p2p_nixl_perf.py`
(collector, fold, formatting, log). Ported from
`upstream-exp:.../protocols/p2p_nixl_perf.py` (`TimeMonitor`,
`format_perf_section`, `gather_and_write_perf_log` carry over; the wire-group
bookkeeping is replaced by the interval fold below).

Layering: the iterators must not import the protocol module. Put the part
timers (`timed_gpu_prep_part`, part names, `bind_gpu_prep_parts`) in a small
neutral module `miles/backends/training_utils/weight_update/gpu_prep_timing.py`
that is a no-op when no collector is bound.

Rules:

- No lock on the RDMA path. `_write_shard` times itself and returns a small
  `WriteStats(nbytes, t_start, t_end)` (a NamedTuple in `remote_shard.py`).
  `TransferBuffers` only inspects exceptions, so returning a value is safe.
  The protocol keeps `(group_id, t_submit_return, futures)` for each group
  and folds `f.result()` on the **main thread** after `wait_for_all_writes()`.
- Only NIXL records: the collector is `None` for Mooncake, and every hook is
  a no-op for `None`.
- Identity: local GPU = gloo `global_rank` (`gpu=`); also store `pp_rank` and
  `gathered_dp_rank` on the payload.
- Reset at the **end** of `begin_sync` (after the probe pass), so probe-pass
  gather/convert time is not counted.
- Every sender rank measures itself; rank 0 gathers (gloo `gather_object`) and
  is the only writer of the file. Non-senders send `None`.

Aggregation (full transfer = one log section):

| Metric | Per event | Parallel shards of one group | Overlapping groups | Across chunks / roles / buckets | Log |
|---|---|---|---|---|---|
| `max_num_wire_bytes_per_trainer` | `sum(nbytes)` in `_write_shard` | sum (every shard) | n/a | sum | GPU with max bytes |
| `wire_time` | `[t_start, t_end]` of `_write_shard` | union | **union** | union | GPU with max |
| `wire_time_sum` | same | **max** (count once) | sum | sum | GPU with max |
| `write_queue` | `t_start - t_submit_return` | **max** | sum | sum | that GPU's value |
| `trainer_prep_time` gpu | inside `next(iterator)` + `stage` + `load_into` | n/a (main thread) | n/a | sum | GPU with max gpu_prep |
| gpu parts (`gather*`, `convert`, `stage`, `cpu_load`) | see sections above | n/a | n/a | sum | the max-total GPU's values |
| `trainer_prep_time` cpu | `load_into` return -> `transport.write` return | n/a (main thread) | n/a | sum | GPU with max cpu_prep |
| `buffer_wait` | `acquire()` | n/a | n/a | sum | that GPU's value |
| `drain` | `wait_for_all_writes()` | n/a | n/a | n/a (once) | that GPU's value |
| `trainer_exposed_wire_time` | `buffer_wait + drain` | n/a | n/a | n/a | GPU with max |
| `trainer_active_time` | first `next(iterator)` -> after drain | n/a | n/a | n/a (one clock) | GPU with max |

Per-sender payload carries: `gpu_prep`, `cpu_prep`, the gpu parts
(`gather`, `gather_load`, `gather_pp`, `gather_tp`, `gather_tp_start`,
`gather_tp_wait`, `gather_tp_concat`, `gather_ep`, `convert`, `stage`,
`cpu_load`), `wire_bytes`, `wire_time`, `wire_time_sum`, `write_queue`,
`buffer_wait`, `drain`, `active`.

Rank 0 prints three prep picks (`argmax(gpu_prep)`, `argmax(cpu_prep)`,
`argmax(gpu_prep + cpu_prep)` on one GPU) and then that last GPU's parts.
Do not define the third as `max(gpu_prep) + max(cpu_prep)` across GPUs.

### Hook sites

| Old site (`upstream-exp`) | New site (`refactoringcheck`) | Records |
|---|---|---|
| `p2p.py` `update_weights` wrap via `wrap_weight_iter` | new default-identity `WeightTransferProtocol.wrap_bucket_iter`, called from the `for bucket in iter_hf_weights(...)` loop in `updater.py` | start `active` on first `next()`, time inside each `next()` as gpu_prep |
| `p2p.py` `begin_sync` `reset_collector` | `UpdateWeightP2P.begin_sync`, last line before `return True` | reset counters, set `weight_version` |
| `hf_weight_iterator_direct.py` `_materialize_non_expert_batch` | same function (`_load_or_allocate_params`, `_broadcast_across_pp`, `all_gather_params_async`) | gather_load, gather_pp, gather_tp |
| `_gather_megatron_expert_batch` (ETP > 1) | same function, plus EP `all_gather_object`, `all_gather`, `handle.wait` | gather_load, pp, tp, ep |
| (did not exist) `_convert_and_gather_expert_batch` (ETP == 1) | the `.to(device)` generator, each `gather(units, ...)` in `batch.gathers` (index 0 is PP when `placement.gather_pp`, else EP) | gather_load, gather_pp, gather_ep |
| `convert_to_hf` call | `_convert_to_hf_param_units` | convert |
| `hf_weight_iterator_bridge.py` `export_hf_weights` next / `_postprocess_and_quantize` | same | gather (inner lines 0), convert |
| `_get_transfer_ready_params` | `UpdateWeightP2P._send_runner_bucket`: `model_param_stager.stage(...)` | stage |
| `model_replica.load_weights` | `_send_runner_bucket`: `target.model_replica.load_into(...)` | cpu_load; then start `cpu_prep` clock |
| `begin_wire_group` / `_do_p2p_write_one_session` | `_send_runner_bucket`: after `transport.write(...)` returns, stop `cpu_prep`; record `t_submit_return` and the futures as one group | cpu_prep, group bookkeeping |
| (none) | `TransferBuffers.acquire` call site in `_send_runner_bucket` | buffer_wait |
| `_do_nixl_write` | `NixlTransport._write_shard` | `WriteStats(nbytes, t_start, t_end)` |
| `after_base_weights` `wait_transfers` + `stop_active_time` + `gather_and_write_perf_log` | `UpdateWeightP2P.after_base_weights`: time `wait_for_all_writes()` as drain, stop active, fold futures, `gather_object`, rank 0 appends | drain, active, fold, log |

### Log format

Rank 0, append, one section per `weight_version`. New lines marked `(new)`.

```text
=== p2p nixl perf  weight_version=12 ===
max_num_wire_bytes_per_trainer: gpu=5 bytes=13.31GB
wire_time: gpu=5 busy=2.110s sum_groups=3.480s
write_queue: gpu=5 time=0.62s                                   (new)
trainer_prep_time_gpu: gpu=5 time=0.91s
trainer_prep_time_cpu: gpu=2 time=0.05s
trainer_prep_time_total: gpu=5 gpu_prep=0.91s cpu_prep=0.03s total=0.94s
trainer_prep_time_gpu_gather: gpu=5 time=0.50s
trainer_prep_time_gpu_gather_load: gpu=5 time=0.04s
trainer_prep_time_gpu_gather_pp: gpu=5 time=0.00s
trainer_prep_time_gpu_gather_tp: gpu=5 time=0.41s
trainer_prep_time_gpu_gather_tp_start: gpu=5 time=0.02s
trainer_prep_time_gpu_gather_tp_wait: gpu=5 time=0.35s
trainer_prep_time_gpu_gather_tp_concat: gpu=5 time=0.04s
trainer_prep_time_gpu_gather_ep: gpu=5 time=0.04s
trainer_prep_time_gpu_convert: gpu=5 time=0.30s
trainer_prep_time_gpu_stage: gpu=5 time=0.01s
trainer_prep_time_gpu_cpu_load: gpu=5 time=0.15s
trainer_exposed_wire_time: gpu=5 buffer_wait=0.40s drain=0.35s total=0.75s   (new)
trainer_active_time: gpu=5 time=2.600s
```

(Numbers are illustrative, not measured.)

---

## Clock start and stop

X-axis is pipeline order, not measured seconds. `max_num_wire_bytes_per_trainer`
is not a clock; it is sampled inside `_write_shard`.

One GPU, two chunks of one bucket, one replica target each, two buffers (A, B).
Main thread on top, one engine write thread below:

```text
main   : [next(iterator): gather+convert][stage][acq A][load A][submit]
                                         [acq B][load B][submit] [next(iterator) ...]
engine :                                          [ queue ][== _write_shard A ==]
                                                                   [queue][== _write_shard B ==]
clocks :
  active    |<------------------------------------------------------------ ... drain>|
  gpu_prep  gather+convert (inside next) + stage + each load_into
  cpu_prep  load_into end -> write() returns         (main thread, per group)
  queue     write() returns -> _write_shard starts   (engine thread, per group)
  wire_time _write_shard start -> end                (union over shards and groups)
  buffer_wait  [acq X] blocks only if X is still being read by a write
  drain     wait_for_all_writes() at the end of the transfer
```

`trainer_active_time` starts at the first `next(iterator)` of this
`update_weights` and stops after `wait_for_all_writes()` in
`after_base_weights`. Writes of the last chunks can still be in flight until
that wait, which is exactly what `drain` measures.

---

## Open points to settle before implementing

1. **Hook in the base protocol.** `wrap_bucket_iter` is a default-identity
   method on `WeightTransferProtocol` plus one call in `updater.py`. It is the
   only edit outside p2p/iterator files. Alternative: time gaps between
   `send_bucket` calls, which is unreliable at the first bucket (session
   pause, barriers).
2. **Warm-up flag.** The first load of each param runs the byte-coverage check
   (two extra loads). We flag `weight_version == 1` as warm-up, but a
   reconnect that builds a new replica repeats the check and cannot be seen
   from the protocol. Exposing a "this load ran the check" bit from
   `ModelReplica.load_into` would fix that.
3. **Mooncake.** The hooks live in shared code (`RolloutEngineWriteThreads`
   path), so Mooncake could record the same numbers cheaply. This spec keeps
   NIXL only, as before.
4. **wandb.** `pop_metrics` could also export `wire_time`, `exposed_wire_time`
   and `active_time` (rank-0 values). Deferred; the file stays the source of
   truth.
