---
title: Miles NIXL — Step 2 Mission: Weight Transfer
---

## What this step delivers

By the end of this step, Miles can transfer weights to SGLang via NIXL RDMA WRITE — fully replacing the Mooncake `TransferEngine` write path when `--mode nixl` is used. Correctness is validated with `--check-weight-update-equal`.

This step builds directly on Step 1 (handshake established, `nixl_agent` available on `self.nixl_agent`).

## Scope

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`
- Add `register_cpu_memory_nixl(params_dict, nixl_agent)`: registers CPU pinned tensors with the NIXL agent as `"DRAM"` memory regions using `register_memory([(addr, size, 0, "")], "DRAM")` 4-tuples (same format as SGLang's `register_memory_region_nixl` for VRAM). Stores the returned `descs` handle on the agent (`nixl_agent._cpu_weight_descs`) to keep registrations alive for the process lifetime.

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`
**Buffer registration** (called once at init, after CPU replica is built):
- Replace `register_cpu_memory(params_dict, transfer_engine)` with `register_cpu_memory_nixl(params_dict, self.nixl_agent)` on the NIXL path.

**Flush / write to target** (called every bucket, every step):
- Replace `transfer_engine.transfer(session_id, src, size, dst)` with the NIXL three-step sequence:

```python
# Step 1: descriptors. get_xfer_descs takes 3-tuples (addr, size, device_id).
src_descs = nixl_agent.get_xfer_descs([(src_addr, size, 0)], "DRAM")
dst_descs = nixl_agent.get_xfer_descs([(dst_addr, size, device_id)], "VRAM")

# Step 2: initialize transfer. peer_name = agent_name UUID from RemoteWeightInfo.
xfer_handle = nixl_agent.initialize_xfer("WRITE", src_descs, dst_descs, peer_name, b"")
if not xfer_handle:
    raise RuntimeError("NIXL failed to create weight transfer handle")

# Step 3: post and poll completion.
state = nixl_agent.transfer(xfer_handle)
if state == "ERR":
    raise RuntimeError("NIXL failed to post weight transfer")
while nixl_agent.check_xfer_state(xfer_handle) != "DONE":
    pass  # blocking poll — runs inside P2PTransferManager thread pool
```

`device_id` comes from the 4th element of `weights_info_dict[param_name]`. `peer_name` is `RemoteWeightInfo.agent_name`.

The `P2PTransferManager` thread pool wraps this sequence the same way it wraps the Mooncake write — fire-and-forget for the last engine, blocking wait for non-last engines.

## Key API notes
- `register_memory` → 4-tuples `(addr, size, device_id, "")`
- `get_xfer_descs` → 3-tuples `(addr, size, device_id)` (no trailing `""`)
- `transfer(xfer_handle)` returns a state string; `"ERR"` means failed to post
- Completion is polled via `check_xfer_state(xfer_handle)` returning `"DONE"` or `"ERR"`
- No `wait()` method exists on `nixl_agent`

All API patterns confirmed from `sglang/srt/disaggregation/nixl/conn.py`.

## Out of scope for this step
- Any changes to the bucketed all-gather or HF conversion pipeline — those are backend-agnostic and untouched.
- LoRA adapter transfer.
- Broadcast mode.
