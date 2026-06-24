---
title: NIXL Weight Transfer Backend — Implementation
description: Per-file implementation plan for the NIXL P2P weight transfer backend in Miles.
---

Implements the design in `nixl-weight-transfer-design.md`. This is the Miles-side counterpart to SGLang's `rfork_nixl_p2p_impl.md`.

## Files to Add or Change

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`

**`create_nixl_agent()` (new function)**

Mirrors `create_transfer_engine()`. Constructs a NIXL agent for the Miles training rank using the same pattern as `sglang/srt/disaggregation/nixl/conn.py`:

```python
from nixl._api import nixl_agent, nixl_agent_config

def create_nixl_agent() -> nixl_agent:
    backend = os.environ.get("SGLANG_DISAGGREGATION_NIXL_BACKEND", "UCX")
    cfg = nixl_agent_config(backends=[backend])
    return nixl_agent(str(uuid.uuid4()), cfg)
```

**`register_cpu_memory_nixl(params_dict, nixl_agent)` (new function)**

Mirrors `register_cpu_memory()` for NIXL. Registers CPU pinned tensors as `"DRAM"` memory regions. The `register_memory` API takes 4-tuples `(addr, size, device_id, "")` — the same format used by SGLang's `register_memory_region_nixl` for VRAM, with `device_id=0` for CPU:

```python
def register_cpu_memory_nixl(params_dict: dict, nixl_agent) -> tuple[dict, Any]:
    regions = []
    weight_dict = {}
    for name, cpu_tensor in params_dict.items():
        addr = cpu_tensor.data_ptr()
        size = cpu_tensor.numel() * cpu_tensor.element_size()
        regions.append((addr, size, 0, ""))  # 4-tuple: (addr, size, device_id=0 for CPU, "")
        weight_dict[name] = (addr, cpu_tensor.numel(), cpu_tensor.element_size())
    descs = nixl_agent.register_memory(regions, "DRAM")
    nixl_agent._cpu_weight_descs = descs  # keep alive for process lifetime
    return weight_dict, descs
```

**`RemoteWeightInfo` (extend)**

Add fields for NIXL alongside existing Mooncake fields:

```python
@dataclasses.dataclass
class RemoteWeightInfo:
    session_id: str | None          # mooncake
    agent_name: str | None          # nixl
    weights_info: dict[str, tuple]  # 3-tuple (mooncake) or 4-tuple (nixl)
    backend: str = "mooncake"
```

**`query_remote_weight_infos()` (update)**

Parse the new tagged-dict response format. When `backend == "nixl"`:
1. Base64-decode `agent_metadata`.
2. Call `nixl_agent.add_remote_agent(decoded_bytes)` to register the SGLang peer.
3. Store `agent_name` and 4-tuple `weights_info_dict` in `RemoteWeightInfo`.

When `backend == "mooncake"` (or legacy untagged format): existing path unchanged.

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

**Initialization**

Branch on `args.update_weight_transfer_backend`:
- `"mooncake"` (default) → `create_transfer_engine()`, existing path.
- `"nixl"` → `create_nixl_agent()`, store as `self.nixl_agent`.

CPU model replica construction and `ParameterMapper` logic are **unchanged** — format conversion is backend-agnostic.

**Buffer registration**

Replace `register_cpu_memory(params_dict, transfer_engine)` with `register_cpu_memory_nixl(params_dict, nixl_agent)` when on the NIXL path. The returned `_weight_descs` handle is stored on the class to keep registrations alive.

**Flush / write to target**

Replace the `transfer_engine.transfer(session_id, src, size, dst)` call with the NIXL three-step sequence (mirroring `sglang/srt/disaggregation/nixl/conn.py`):

```python
# NIXL WRITE: src is CPU DRAM, dst is remote GPU VRAM
# Step 1: build descriptors. get_xfer_descs takes 3-tuples (addr, size, device_id).
src_descs = nixl_agent.get_xfer_descs([(src_addr, size, 0)], "DRAM")
dst_descs = nixl_agent.get_xfer_descs([(dst_addr, size, device_id)], "VRAM")

# Step 2: initialize transfer. peer_name is the agent_name UUID from RemoteWeightInfo.
xfer_handle = nixl_agent.initialize_xfer("WRITE", src_descs, dst_descs, peer_name, b"")
if not xfer_handle:
    raise RuntimeError("NIXL failed to create weight transfer handle")

# Step 3: post transfer and poll for completion.
state = nixl_agent.transfer(xfer_handle)
if state == "ERR":
    raise RuntimeError("NIXL failed to post weight transfer")
while nixl_agent.check_xfer_state(xfer_handle) != "DONE":
    pass  # blocking poll — runs in P2PTransferManager thread pool
```

`device_id` comes from the 4th element of the `weights_info_dict` entry. The `P2PTransferManager` thread pool wraps this blocking sequence identically to the Mooncake path.

### `miles/utils/arguments.py`

Add `--update-weight-transfer-backend {mooncake,nixl}` (default `"mooncake"`). This is orthogonal to `--update-weight-transfer-mode`; NIXL is only valid when mode is `p2p`. Validation: reject `backend=nixl` when mode is not `p2p`.

### `miles/backends/megatron_utils/actor.py`

Pass `transfer_backend` to `UpdateWeightP2P` constructor.

### `miles/backends/sglang_utils/sglang_engine.py`

Fix the pending endpoint URL TODO (line ~314). The SGLang HTTP server exposes `/remote_instance_transfer_engine_info` but Miles currently queries `/get_remote_instance_transfer_engine_info`:

```python
# Before (stale, has TODO comment):
response = requests.get(
    f"http://{self.server_host}:{self.server_port}/get_remote_instance_transfer_engine_info",
    ...
)
# After:
response = requests.get(
    f"http://{self.server_host}:{self.server_port}/remote_instance_transfer_engine_info",
    ...
)
```

This is a bug fix for both backends — the SGL impl doc confirms the public endpoint has always been `/remote_instance_transfer_engine_info`.

### `examples/p2p_weight_transfer/run.py`

Add `nixl` as a new choice for `--mode` alongside `p2p` and `broadcast`. **`p2p` mode is completely unchanged.**

| `--mode` | `--update-weight-transfer-mode` → train.py | `--update-weight-transfer-backend` → train.py | SGLang seed flag |
|---|---|---|---|
| `broadcast` | `broadcast` | _(not set)_ | _(not set)_ |
| `p2p` | `p2p` | _(not set, defaults to `mooncake`)_ | `--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine` |
| `nixl` | `p2p` | `nixl` | `--sglang-remote-instance-weight-loader-start-seed-via-nixl` |

The two `mode == "p2p"` blocks in `run.py` (lines ~1015 and ~1053) must each be extended to also match `mode == "nixl"`, since NIXL shares all P2P SGLang args and buffer-size logic. Only the SGLang seed flag and backend flag differ.

## Launch Command

```bash
# Mooncake (unchanged)
python examples/p2p_weight_transfer/run.py run GLM-Z1-9B-0414 --mode p2p

# NIXL (new)
python examples/p2p_weight_transfer/run.py run GLM-Z1-9B-0414 --mode nixl
```

SGLang does not need to be launched separately; run.py controls both the train.py invocation and the SGLang seed flag via `--sglang-*` args.
