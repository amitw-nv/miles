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
  parse the 4-field weight entries `{"addr", "numel", "element_size", "device_id"}`; continue
  parsing SGLang's tagged Mooncake dict with 3-field weight entries.
- Add `add_nixl_remote_agent(nixl_agent, agent_metadata_b64)`: base64-decode the string and call
  `nixl_agent.add_remote_agent(raw)`.

**Tests** (details in `test.md`):
- **2a** — `query_remote_weight_infos()` parses the NIXL dict format and populates `agent_name` and `backend`.
- **2b** — `query_remote_weight_infos()` still parses the tagged Mooncake dict with 3-field weight
  entries (regression).
- **2c** — `RemoteWeightInfo` has the two new optional fields with the correct defaults.
- **2d** — mandatory within the full NIXL E2E Test 5e: against the SGLang seed launched by Miles,
  parse the real metadata and construct the installed SGLang `ServerArgs` from server info.

---

### Step 3 — NIXL agent init and peer connection

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`, `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Add `create_nixl_agent()` in `p2p_transfer_utils.py`: construct and return a `nixl.Agent` instance.
- In `UpdateWeightP2P.connect_rollout_engines()`, after the shared `plan_p2p()` call, branch on
  `transfer_backend`:
  - `"nixl"` → call `create_nixl_agent()` once, then loop over all targets returned by `plan_p2p()`:
    for each `(engine_ind, engine_rank)` target, query the SGLang metadata endpoint to get
    `agent_name`, `agent_metadata`, and `weights_info_dict`; call `add_nixl_remote_agent()` with the
    decoded blob; collect `ServerArgs` via `get_server_info` Ray call per engine (same as Mooncake).
    Build and store an `agent_name → weights_info_dict` map and an `agent_name → ServerArgs` map
    (parallel to Mooncake's `session_id → weights_info` and `session_id → ServerArgs`). The handshake
    loop runs once during connection, not per weight-update iteration.
  - `"mooncake"` → call the existing `query_remote_weight_infos()` and
    `create_transfer_engine()` path unchanged.
- Keep `_create_cpu_replica()`, format conversion, and `_transfer_engine_meta_list` construction
  shared between both backends.

**Tests** (details in `test.md`):
- **3a** — `create_nixl_agent()` exists and returns a NIXL agent object (requires NIXL installed).
- **3b** — `connect_rollout_engines()` branches correctly: the NIXL path calls
  `create_nixl_agent()`, while the Mooncake path calls `create_transfer_engine()`.
- **3c** — handshake integration: `add_remote_agent` completes without error against a live SGLang
  NIXL seed (requires GPU + NIXL + SGLang).

---

### Step 4 — CPU DRAM memory registration

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`, `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Add `register_cpu_memory_nixl(nixl_agent, tensors)` in `p2p_transfer_utils.py`: call
  `agent.register_memory([(addr, size, 0, "")], "DRAM")` for each shared pinned CPU tensor and
  return the source metadata needed by the write path. Assert tensors are pinned; `device_id = 0`
  for all DRAM regions.
- In `UpdateWeightP2P._pause_and_prepare_engines()`, branch on `transfer_backend` when
  `_model_registered` is false: call `register_cpu_memory_nixl()` for NIXL or the existing
  `register_cpu_memory()` for Mooncake, then set `_model_registered = True`.
- Keep the registration alive with the shared CPU buffers across weight-update iterations. Do not
  register or deregister memory in the per-update write path.

**Tests** (details in `test.md`):
- **4a** — `register_cpu_memory_nixl` exists and produces source metadata for a pinned tensor
  (requires NIXL).
- **4b** — `_pause_and_prepare_engines()` registers NIXL memory only on its first call.
- **4c** — non-pinned tensor raises `AssertionError` with a clear message (requires NIXL).

---

### Step 5 — Branch the existing P2P WRITE

**Files:** `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`

- Branch `_do_p2p_write_one_session()` on the remote target's backend:
  - `"nixl"` → for each parameter, derive `(addr, size, 0)` source descriptors from the one-time
    CPU registration and `(addr, size, device_id)` destination descriptors from
    `RemoteWeightInfo`; run `get_xfer_descs` → `initialize_xfer("WRITE", …)` → `transfer()` and poll
    `check_xfer_state()` until `"DONE"` or raise on `"ERR"`.
  - `"mooncake"` → keep the existing `batch_transfer_sync_write()` call unchanged.
- Keep staging, `load_weights()`, synchronous protection of reusable replicas, background
  scheduling of the final replica, MoE handling, and `wait_transfers()` shared.

**Tests** (details in `test.md`):
- **5a** — NIXL write path calls `get_xfer_descs`, `initialize_xfer`, `transfer`, `check_xfer_state`
  in the correct order (mock NIXL agent, no GPU).
- **5b** — `check_xfer_state` returning `"ERR"` raises `RuntimeError` (mock, no GPU).
- **5c** — a transfer failure does not discard or repeat the one-time CPU memory registration
  (mock, no GPU).
- **5d** — Mooncake write path is not modified and passes its existing tests (regression).
- **5e** — small-model E2E: `--check-weight-update-equal` passes with `--mode nixl` (requires GPU + NIXL + SGLang).
