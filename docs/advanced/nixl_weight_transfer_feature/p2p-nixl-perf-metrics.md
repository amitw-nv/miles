# P2P NIXL perf metrics

On every NIXL P2P weight transfer, collect these metrics and append them to
`p2p_nixl_perf.log` in the trainer working directory. No flag. Rank 0 writes
the file after gathering from sender GPUs.

One `update_weights` is one full weight transfer (all parameters). The run
calls it again on later training steps; each call is its own log section. We
do **not** add those together.

Inside one full transfer there are two loops:

- **Outer loop:** `for bucket in iter_hf_weights: send_bucket(bucket)`. One
  outer step is one `send_bucket`. All-gather + HF convert for that bucket run
  in `next(iterator)`, just before `send_bucket`.
- **Inner loop:** inside `send_bucket`, `for replica in _transfer_engine_meta_list`:
  `load_weights` then RDMA to each remote session. One GPU can have more than
  one CPU replica.

Which loop a metric uses:

- `trainer_prep_time` gpu — outer (`next(iterator)` until before `load_weights`).
  Sum every outer step of this full transfer.
- `trainer_prep_time` cpu, `max_num_wire_bytes_per_trainer`, `wire_time` —
  inner (replica / session). Then sum across inner **and** outer so the
  logged number is the full transfer.
- `trainer_active_time` — the whole `update_weights` send path, not a sum of
  either loop.

---

## max_num_wire_bytes_per_trainer

**What it measures:** How many bytes each trainer GPU sends on the wire in this
full transfer.

**How we measure it:** Inner: in `_do_p2p_write_one_session`, sum `source_lens`
(CPU replica shard, `numel * element_size`, once per remote session). That is
the RDMA payload. Sum every inner write on that GPU, over every outer
`send_bucket`.

**What you see in the log:** Which GPU sent the most, and how many bytes.

Bytes print as decimal **GB** (10^9), not GiB (2^30), so this line divided by
`wire_time` is directly comparable to a NIC line rate without a base change.
Note the surrounding `memory_utils` lines (`total_GB`, `free_GB`) are binary
values carrying a `GB` label, so the two are not the same unit despite the
matching suffix.

---

## wire_time

**What it measures:** Time the trainer spent in RDMA: from the CPU starting
the NIXL write until it returns.

**How we measure it:** Inner: timer around `_do_nixl_write` (start right before,
stop when it returns).

One replica can RDMA the same buffer to several targets in parallel (one
session per engine). Count that group **once**: take the longest session
time, not the sum. Then sum across CPU replicas on this GPU (they run one
after the other on the shared buffer) and across every outer `send_bucket`.

**What you see in the log:** The GPU with the largest work.

---

## trainer_prep_time

Two parts, still measured separately per GPU. The log has **three** numbers,
each with the GPU that owns that max (they can be three different GPUs):

- max `gpu_prep`
- max `cpu_prep`
- max `gpu_prep + cpu_prep` on the **same** GPU (do not add the two category
  maxes from different GPUs)

### 1. gpu

**What it measures:** All-gather + HF convert + quant, until the bucket is
ready to load to CPU.

**How we measure it:** Outer: start at `next(iterator)`, stop right before
`load_weights`. Sum every outer step of this full transfer, per GPU.

### 2. cpu

**What it measures:** Load to CPU until the moment before the RDMA write.

**How we measure it:** Inner: time `load_weights` plus pointer setup in
`_do_p2p_write_one_session`, stop right before `_do_nixl_write`. Replicas on
one GPU run one after the other: **sum** them. Pointer setup for several
sessions of the same replica can run in parallel: count that group **once**
(longest), like `wire_time`. Then sum every outer `send_bucket`.

**What you see in the log:** Three lines: GPU with max gpu-prep, GPU with max
cpu-prep, GPU with max `gpu_prep + cpu_prep` (same GPU).

---

## trainer_active_time

**What it measures:** Wall clock of this full transfer: first all-gather until
every write is finished. Not a sum of outer `send_bucket` times and not a sum
of inner replica times (those overlap).

**How we measure it:** Start at the first `next(iterator)` of this
`update_weights`. Stop after `wait_transfers()` in `after_base_weights`.

**What you see in the log:** The GPU with the largest active time.

---

## How to implement

New file `miles/backends/training_utils/weight_update/protocols/p2p_nixl_perf.py`.
On whenever `--update-weight-transfer-backend nixl`. No flag.

There is no lock on the RDMA path today, and we do not add one for metrics.
`P2PTransferManager` runs each session in a thread pool, so several
`_do_nixl_write` calls can finish at once. Time the write **inside that
call**, return `(bytes, duration)` on the future, and fold max/sum on the
**main thread** after `f.result()` / `wait_transfers()`. The clock is just
`time.monotonic()` around `_do_nixl_write`; the fold is not on the RDMA
critical path.

Identity: local GPU = gloo `global_rank` (log as `gpu=`). Also store `pp_rank`
and `gathered_dp_rank` on the payload. Reset counters at the start of each
`update_weights` send path.

Every sender rank measures itself. Rank 0 only **gathers and writes the
file** so one shared log is not appended by many processes at once.
Non-senders send `None`. Rank 0 prints the bottleneck GPU per metric (max
bytes, max wire_time, max gpu_prep, max cpu_prep, max gpu+cpu on one GPU,
max active). That is the trainer-side P2P path, not rollout
`post_load_weights`. Because all-gather is lockstep and `update_weights`
ends on a gloo barrier, the max `trainer_active_time` is the stall the
system waits on.

Aggregation (full transfer = one log section):

| Metric | Per event | Fold parallel sessions | Fold sequential replicas | Fold outer `send_bucket` | Log |
|---|---|---|---|---|---|
| `max_num_wire_bytes_per_trainer` | `sum(source_lens)` in `_do_p2p_write_one_session` | **sum** (every session) | sum | sum | GPU with max bytes |
| `wire_time` | timer around `_do_nixl_write` | **max** (count once) | sum | sum | GPU with max work |
| `trainer_prep_time` gpu | `next(iterator)` until before `load_weights` | n/a (outer, main thread) | n/a | sum | GPU with max gpu_prep |
| `trainer_prep_time` cpu | `load_weights` + setup until before `_do_nixl_write` | **max** on setup | **sum** (`load_weights`) | sum | GPU with max cpu_prep |
| `trainer_active_time` | first `next(iterator)` → `wait_transfers()` done | n/a | n/a | n/a (one wall clock) | GPU with max active |

Each sender payload carries that rank’s `gpu_prep` and `cpu_prep`. Rank 0
prints three picks:

- `argmax(gpu_prep)`
- `argmax(cpu_prep)`
- `argmax(gpu_prep + cpu_prep)` on **that same GPU**

Do not define the third number as `max(gpu_prep) + max(cpu_prep)` across
different GPUs.

Hook sites:

- `updater.py` `update_weights`: start `trainer_active_time`; time each
  `next(iterator)` as prep gpu; protocol exposes the collector (updater does
  not import NIXL).
- `p2p.py` `send_bucket`: time `load_weights` per replica as prep cpu.
- `p2p.py` `_do_p2p_write_one_session`: add bytes; cpu-prep continues until
  `_do_nixl_write`; group sessions of one replica so wire_time / cpu-setup
  use max, not sum.
- `p2p.py` `_do_nixl_write`: start/stop for wire_time.
- `p2p.py` `after_base_weights`: `wait_transfers()`, stop active time,
  `gather_object` on gloo, rank 0 appends to `p2p_nixl_perf.log`.

Log (rank 0, append, one section per `weight_version`):

```text
=== p2p nixl perf  weight_version=12 ===
max_num_wire_bytes_per_trainer: gpu=5 bytes=13.31GB
wire_time: gpu=5 work=2.110s
trainer_prep_time_gpu: gpu=5 time=0.91s
trainer_prep_time_cpu: gpu=2 time=0.55s
trainer_prep_time_total: gpu=5 gpu_prep=0.91s cpu_prep=0.40s total=1.31s
trainer_active_time: gpu=5 time=3.410s
```

---

## Clock start and stop

X-axis is pipeline order, not measured seconds. A bar is one
`time.monotonic()` interval. `max_num_wire_bytes_per_trainer` is not a
clock; it is sampled during pointer setup (`sum(source_lens)`).

### One `send_bucket`

Pipeline on this plot: all-gather, HF convert, quant, staging,
`load_weights`, pointer setup, `_do_nixl_write`. Staging is still gpu-prep
(`_get_transfer_ready_params` in `send_bucket`). Extra CPU replicas on this
GPU repeat load → setup → RDMA after the first replica; the plot shows one
replica after gpu-prep.

![Clock start and stop on one send_bucket](clock-send-bucket.svg)

`trainer_active_time` starts at the first `next(iterator)` of this
`update_weights` and does **not** stop at the end of this bucket.

`trainer_prep_time` gpu starts at that `next(iterator)` (all-gather + HF
convert + quant run inside the yield) and stops at `stop_gpu_prep()`, after
staging, right before `load_weights`.

`trainer_prep_time` cpu is **two** clocks, not one continuous interval:

- `load_weights`: copy ready HF tensors into the shared pinned CPU replica.
  That is the actual CPU loading.
- pointer setup in `_do_p2p_write_one_session`: walk names, collect CPU and
  remote pointers, `add_wire_bytes`. Not a tensor copy. Stops at
  `cpu_setup.stop()`, immediately before `_do_nixl_write`.

`wire_time` starts immediately before `_do_nixl_write` and stops when that
call returns (including the DONE poll).

### One full `update_weights`

Two outer `send_bucket` steps, then `wait_transfers()`. gpu-prep, cpu-prep,
and `wire_time` are disjoint bars that the log **sums**. `trainer_active_time`
is one wall clock around them, including the gaps.

![Clock start and stop on one full update_weights](clock-update-weights.svg)

On this plot, **load** is `load_weights` and **setup** is pointer setup.
Together they are `trainer_prep_time` cpu. RDMA is `wire_time`, not cpu-prep.

`trainer_active_time` starts at the first `next(iterator)` and stops in
`after_base_weights` after `wait_transfers()`. Last-replica writes can still
be in flight until that wait.

Parallel sessions of one replica each start and stop their own setup and
`wire_time` clocks; the replica keeps the longest, not the sum.

