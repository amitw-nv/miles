# Bucket-fill pipeline

`send_bucket` loads one GPU bucket into the pinned CPU replica and RDMA-writes
it. The next bucket is created and filled while that work is still running.
No extra copy. No second pinned model. No flag. Mooncake uses the same
`send_bucket`.

## What overlaps

The updater loop is unchanged: `next(iterator)` then `send_bucket`.

- **Fill** is `next(iterator)`: all-gather, HF convert, quant, then pack into a
  new list of GPU tensors. Those tensors are allocated per bucket and stay
  valid while the caller holds them. The next iteration does not reuse that
  storage.
- **Load + RDMA** copies those GPU tensors into the pinned CPU replica
  (`load_weights` → `param.data.copy_`) and RDMA-writes from those registered
  addresses.

The fill buffer and the RDMA source are already different memory, so a staging
buffer is not required. A staging copy (GPU bucket → staging → pinned params)
would add a copy and still would not overlap anything until the main thread
returned to the iterator.

```text
main:    fill0 | submit0 | fill1          | join0 | submit1 | fill2
worker:           load0 + RDMA0 ..........|         load1 + RDMA1 ...
```

`fill1` runs during `load0` and RDMA of bucket 0. One bucket is in flight.
Peak GPU memory grows by one bucket (`--update-weight-buffer-size`, default
512MB) while the previous tensors are still being copied.

The next `load_weights` starts only after that RDMA has finished, so the pinned
params are free to overwrite.

## Where it runs

`UpdateWeightP2P` in `miles/backends/training_utils/weight_update/protocols/p2p.py`.

- The main thread still stages (`_get_transfer_ready_params`) and stops
  gpu-prep. Staging does not touch the pinned params.
- When this bucket has tensors to load, the main thread joins the previous
  bucket, submits this one, and returns. The next `next(iterator)` then fills
  the following bucket.
- `_load_and_write_bucket` runs on a **one-thread** executor, not on the RDMA
  pool. That thread waits on pool futures; sharing the pool can deadlock.
- The replica loop is unchanged. Replicas share one `param.data`, so a
  non-last replica's RDMA finishes before the next replica loads. The last
  replica's sessions stay fire-and-forget on `P2PTransferManager`.
- Join is `future.result()` plus `transfer_manager.wait_transfers()`.
  `after_base_weights` joins the same way before it stops `trainer_active_time`.

Every rank, including non-senders, still enters the next all-gather together.
Non-senders already block in that collective. Senders arrive while their copy
and RDMA are still running.

## Perf log

Metric definitions are unchanged. `gpu_prep`, `cpu_prep`, and `wire_time` are
still summed per bucket. `load_weights` time is still recorded around that
call; the call now runs on the bucket thread, as the RDMA timing already does.

`gpu_prep` stops before the join, so time spent waiting out the previous RDMA
is not counted as gather or convert. `trainer_active_time` is still one wall
clock from the first `next(iterator)` until after the final join. That wall
clock now includes the overlap of bucket N+1's fill with bucket N's load and
RDMA, so it can be shorter than the sum of the per-bucket bars.
