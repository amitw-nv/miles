---
title: Miles NIXL — Step 2 Roadmap: Weight Transfer
---

## Prerequisites
Step 1 (handshake) is merged and passing. SGLang is running with `--remote-instance-weight-loader-start-seed-via-nixl`.

## Implementation order

### 1. `p2p_transfer_utils.py` — add `register_cpu_memory_nixl()`
Add the new function in isolation. No callers yet. Write a unit test or quick standalone script that creates a dummy `nixl_agent`, allocates a small pinned CPU tensor, calls `register_cpu_memory_nixl`, and verifies `descs` is non-empty and `_cpu_weight_descs` is set.

### 2. `p2p.py` — CPU registration at init
Wire in `register_cpu_memory_nixl` on the NIXL path, after the CPU model replica is built (same location as the existing `register_cpu_memory` call). Confirm the NIXL path reaches this point without error by running with `--mode nixl` and checking logs.

### 3. `p2p.py` — replace the write call
Swap out `transfer_engine.transfer(session_id, ...)` for the `get_xfer_descs` → `initialize_xfer` → `transfer` → `check_xfer_state` sequence on the NIXL path. The Mooncake path (`backend == "mooncake"`) is completely unchanged — the branch is a simple `if self.nixl_agent is not None`.

### 4. End-to-end validation
Run with `--mode nixl --check-weight-update-equal` on a small model (e.g. Qwen3-4B, single node). This validates that every transferred weight matches the training-side value exactly.

## Testing

| Test | Command | Pass criterion |
|---|---|---|
| Mooncake regression | `--mode p2p --check-weight-update-equal` | No change in behavior |
| NIXL correctness | `--mode nixl --check-weight-update-equal` | All weights equal |
| NIXL multi-node | `--mode nixl` on a 2-node model (e.g. Moonlight-16B-A3B) | No crash, generation resumes |

## Definition of done
- `--mode p2p` (Mooncake) passes `--check-weight-update-equal` unchanged.
- `--mode nixl` passes `--check-weight-update-equal` on at least one validated model.
