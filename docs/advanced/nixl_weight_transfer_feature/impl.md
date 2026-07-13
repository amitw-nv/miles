# Implementation: Miles NIXL backend for P2P weight transfer

Implements the design in `design.md`.

## Implementation roadmap

Each step is independently testable. Complete them in order.

Each step below lists only its test names + what they check. The runnable commands and expected
results live in `test.md`.

---

### Step 1 — Backend flag, mode, and endpoint fix

**Files:** `miles/backends/megatron_utils/arguments.py`, `examples/p2p_weight_transfer/run.py`, `miles/backends/sglang_utils/sglang_engine.py`

- Add `--update-weight-transfer-backend` argument with choices `["mooncake", "nixl"]`, default `"mooncake"`.
- Add `--mode nixl` choice to the `run` sub-command of `run.py`; when selected, pass
  `--update-weight-transfer-backend nixl` to Miles and
  `--remote-instance-weight-loader-start-seed-via-nixl` to the SGLang seed.
- Fix the endpoint URL in `sglang_engine.py`: replace the stale
  `/get_remote_instance_transfer_engine_info` with `/remote_instance_transfer_engine_info`
  (matches SGLang's registered route).
- Forward `transfer_backend` from `arguments.py` through `actor.py` to `UpdateWeightP2P`.

**Tests** (details in `test.md`):
- **1a** — `--update-weight-transfer-backend` flag appears in `--help`.
- **1b** — `--mode nixl` in `run.py` emits both the Miles and SGLang NIXL flags; `mooncake` flags are absent.
- **1c** — `mooncake` is still the default when `--mode` is omitted.
- **1d** — `sglang_engine.py` source contains the correct URL and not the stale one.

---

### Step 2 — Schema consumer: `RemoteWeightInfo` + query function

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`

- Extend `RemoteWeightInfo` with two new optional fields: `agent_name: str = ""` and
  `backend: str = "mooncake"`.
- Update `query_remote_weight_infos()` to detect `backend == "nixl"` in the HTTP response and
  parse the 4-field dict format `{"addr", "numel", "element_size", "device_id"}`; keep backward
  compatibility with the existing Mooncake 3-field tuple format.
- Add `add_nixl_remote_agent(nixl_agent, agent_metadata_b64)`: base64-decode the string and call
  `nixl_agent.add_remote_agent(raw)`.

**Tests** (details in `test.md`):
- **2a** — `query_remote_weight_infos()` parses the NIXL dict format and populates `agent_name` and `backend`.
- **2b** — `query_remote_weight_infos()` still parses the Mooncake tuple format unchanged (regression).
- **2c** — `RemoteWeightInfo` has the two new optional fields with the correct defaults.

---

### Step 3 — NIXL agent init and peer connection

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`, `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Add `create_nixl_agent()` in `p2p_transfer_utils.py`: construct and return a `nixl.Agent` instance.
- In `p2p.py` init, branch on `transfer_backend`:
  - `"nixl"` → call `create_nixl_agent()` once, then loop over all targets returned by `plan_p2p()`:
    for each `(engine_ind, engine_rank)` target, query the SGLang metadata endpoint to get
    `agent_name`, `agent_metadata`, and `weights_info_dict`; call `add_nixl_remote_agent()` with the
    decoded blob; collect `ServerArgs` via `get_server_info` Ray call per engine (same as Mooncake).
    Build and store an `agent_name → weights_info_dict` map and an `agent_name → ServerArgs` map
    (parallel to Mooncake's `session_id → weights_info` and `session_id → ServerArgs`). The handshake
    loop runs once at init, not per weight-update iteration.
  - `"mooncake"` → existing Mooncake init path unchanged.

**Tests** (details in `test.md`):
- **3a** — `create_nixl_agent()` exists and returns a NIXL agent object (requires NIXL installed).
- **3b** — p2p init branches correctly: NIXL path calls `create_nixl_agent`; Mooncake path does not.
- **3c** — handshake integration: `add_remote_agent` completes without error against a live SGLang
  NIXL seed (requires GPU + NIXL + SGLang).

---

### Step 4 — CPU DRAM memory registration

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`, `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Add `register_cpu_memory_nixl(nixl_agent, tensors)` in `p2p_transfer_utils.py`: call
  `agent.register_memory([(addr, size, 0, "")], "DRAM")` for each pinned CPU tensor; return the
  registration handles. Assert tensors are pinned; `device_id = 0` for all DRAM regions.
- Add `deregister_cpu_memory_nixl(nixl_agent, handles)`: deregister all handles.
- In `p2p.py`, call `register_cpu_memory_nixl()` after format conversion and before any
  `get_xfer_descs` call; wrap in `try/finally` so `deregister_cpu_memory_nixl()` always runs.

**Tests** (details in `test.md`):
- **4a** — `register_cpu_memory_nixl` and `deregister_cpu_memory_nixl` exist (static).
- **4b** — registration succeeds for a pinned tensor; deregistration does not raise (requires NIXL).
- **4c** — non-pinned tensor raises `AssertionError` with a clear message (requires NIXL).

---

### Step 5 — NIXL WRITE transfer

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Add `_transfer_weights_nixl(cpu_tensors, remote_infos)`: for each parameter, build `src_descs`
  from the CPU tensor and `dst_descs` from `RemoteWeightInfo`, then run the NIXL 3-step sequence:
  `get_xfer_descs` → `initialize_xfer("WRITE", …)` → `transfer()` → poll `check_xfer_state()`
  until `"DONE"` or raise on `"ERR"`.
- Branch the existing write loop: if `backend == "nixl"`, call `_transfer_weights_nixl()`; otherwise
  keep the existing Mooncake write path unchanged.
- MoE models: preserve the existing expert / non-expert parameter split; issue separate
  `_transfer_weights_nixl` calls for each group using the same `nixl_agent` and `agent_name`.

**Tests** (details in `test.md`):
- **5a** — NIXL write path calls `get_xfer_descs`, `initialize_xfer`, `transfer`, `check_xfer_state`
  in the correct order (mock NIXL agent, no GPU).
- **5b** — `check_xfer_state` returning `"ERR"` raises `RuntimeError` (mock, no GPU).
- **5c** — `deregister_cpu_memory_nixl` is called even if `transfer` raises (mock, no GPU).
- **5d** — Mooncake write path is not modified and passes its existing tests (regression).
- **5e** — small-model E2E: `--check-weight-update-equal` passes with `--mode nixl` (requires GPU + NIXL + SGLang).
