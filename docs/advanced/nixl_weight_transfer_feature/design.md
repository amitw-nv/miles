# Design: Miles NIXL backend for P2P weight transfer

## 1. Goal & framing

Miles should push updated training weights directly into SGLang's GPU VRAM via NIXL RDMA WRITE,
replacing the current Mooncake `TransferEngine` path.

Miles is the **active driver**. It does everything:

1. **Connect** to SGLang's NIXL peer by querying the metadata endpoint and calling `add_remote_agent`.
2. **Register** its CPU weight buffers (post format-conversion) with a local NIXL agent as DRAM regions.
3. **Write** each parameter into SGLang's GPU VRAM using NIXL `get_xfer_descs` / `initialize_xfer` /
   `transfer` / `check_xfer_state`.

SGLang is the **passive target**. It only registers its GPU buffers and publishes metadata over the
existing HTTP endpoint. No SGLang code changes are needed for the transfer itself.

This design **adds NIXL as a second branch inside the existing Mooncake weight-transfer machinery**
rather than a parallel subsystem. The existing flags, `RemoteWeightInfo`, query helpers, and p2p
update loop are all reused; we add a `nixl` branch at each decision point.

### How it is launched

`run.py` owns the entry point. Adding `--mode nixl` makes it:
- Pass `--update-weight-transfer-backend nixl` to Miles training.
- Pass `--remote-instance-weight-loader-start-seed-via-nixl` to the SGLang seed it launches.

The NIXL seed flag is an internal contract between Miles and SGLang; the user only sets `--mode nixl`.

### Backend branch map

The NIXL backend stays inside `UpdateWeightP2P`; only five points branch, while planning, conversion,
staging, scheduling, completion, and engine lifecycle remain shared:

```text
run.py  (--mode nixl → --update-weight-transfer-backend nixl)  ── BRANCH 0 (launch flags)
  │
  └─ UpdateWeightP2P                                   [SHARED class, mode == p2p]
       connect_rollout_engines
         plan_p2p()                                    [SHARED]
         query_remote_weight_infos()                   ── BRANCH 1 (metadata / handshake)
         create_transfer_engine / create_nixl_agent    ── BRANCH 2 (engine vs agent)
         _create_cpu_replica() + conversion + meta_list [SHARED]
       _pause_and_prepare_engines
         register_cpu_memory[_nixl]()                  ── BRANCH 3 (one-time registration)
       _update_weight_implementation
         staging + load_weights + scheduling           [SHARED]
         _do_p2p_write_one_session
           batch_transfer_sync_write / NIXL xfer seq   ── BRANCH 4 (the write)
       _gather_and_update_expert_weights → wait_transfers()   [SHARED]
       _finalize_and_resume_engines (post_load_weights=True)  [SHARED]
```

## 2. Assumptions

These are the assumptions this design is built on. If any is wrong the corresponding section changes.

- **A1 — Miles is the active NIXL driver; SGLang is the passive target.** Miles issues all
  `add_remote_agent`, `get_xfer_descs`, `initialize_xfer`, `transfer`, and `check_xfer_state` calls.
  SGLang only registers memory and exports metadata.
- **A2 — We reuse the existing Mooncake infrastructure as-is.** The `RemoteWeightInfo` dataclass, the
  HTTP discovery URL, `query_remote_weight_infos`, and the p2p update loop are all kept. NIXL branches
  are inserted rather than new parallel structures.
- **A3 — Miles retains the CPU model replica.** Format conversion (Megatron → HF layout, resharding)
  happens on the CPU exactly as today. NIXL writes the already-converted CPU tensors into SGLang's
  GPU VRAM; it does not change the conversion step.
- **A4 — Memory registered on the Miles side is CPU DRAM (pinned).** Tensors must be pinned for NIXL
  DRAM registration. `device_id = 0` is used for all CPU regions.
- **A5 — Same-shape / same-layout assumption holds.** SGLang's `weights_info_dict` keys match Miles'
  HF parameter names after conversion, and `numel` / `element_size` agree. This mirrors the existing
  Mooncake contract; no new alignment logic is needed.
- **A6 — NIXL requires an explicit metadata handshake per target.** Unlike Mooncake's implicit
  `P2PHANDSHAKE` + `session_id`, NIXL needs Miles to call `add_remote_agent(agent_metadata)` before
  any transfer. The handshake mirrors the Mooncake `query_remote_weight_infos` loop: for each
  `(engine_ind, engine_rank)` target returned by `plan_p2p()`, Miles queries the SGLang metadata
  endpoint (getting `agent_name`, `agent_metadata`, and `weights_info_dict`), then calls
  `add_remote_agent` with the decoded blob. The result is a mapping of
  `agent_name → weights_info_dict` (parallel to Mooncake's `session_id → weights_info`), which is
  used to drive the per-target transfer loop. This is the single unavoidable schema change on the
  consumer side.
- **A7 — One `nixl_agent` per Miles rank.** The agent is created once at process startup and reused
  across all weight-update iterations and all SGLang tp_ranks it serves. A single Miles rank may be
  mapped to more than one SGLang worker by `plan_p2p()` (e.g. 1:2). In that case the same
  `nixl_agent` calls `add_remote_agent` once per target and iterates targets sequentially during
  transfer — one agent, N handshakes, N transfer loops.
- **A8 — Transfer direction is always WRITE (Miles → SGLang).** No READ / pull path exists or is
  planned. Miles is always the source.
- **A9 — JSON transport for metadata.** The existing HTTP endpoint is JSON; `agent_metadata` arrives
  as a base64-encoded string and Miles decodes it before calling `add_remote_agent`.

## 3. Baseline: what the existing Mooncake path already does in Miles

We get all of this for free and reuse it:

- **Backend selection** — `--mode mooncake` in `run.py` (will add `nixl` alongside it).
- **Endpoint query** — `sglang_engine.py` fetches `/remote_instance_transfer_engine_info` and returns
  the raw JSON (URL was wrong; the fix is part of this feature).
- **`RemoteWeightInfo`** — dataclass holding `(addr, numel, element_size, device_id)` per parameter;
  `query_remote_weight_infos()` builds it from the JSON response.
- **P2P update loop** — `p2p.py` iterates over `RemoteWeightInfo` entries after format conversion and
  calls the transfer backend to write each parameter.
- **`UpdateWeightP2P` / `actor.py`** — orchestrates the loop; receives args from `run.py`.

The peer (SGLang) exposes the metadata; Miles reads it, converts weights, and writes. NIXL adds a
new agent object and a new write sequence but fits entirely within this existing structure.

### 3.1 Block-by-block: each Mooncake API and the NIXL parallel Miles must provide

| # | Pipeline block | Mooncake path | NIXL path | Who owns it |
|---|---|---|---|---|
| B0 | Launch selection | `--mode mooncake` selects the default backend | `--mode nixl` emits the Miles backend flag and SGLang NIXL seed flag | `run.py` |
| S1 | Transfer planning | `plan_p2p()` maps each source rank to target `(engine_ind, engine_rank)` pairs | Same; backend-independent | Miles rank |
| B1 | Metadata discovery / peer handshake | Query the endpoint for `{session_id, weights_info_dict}`; the `session_id` drives implicit `P2PHANDSHAKE` | Query the same endpoint for `{backend, agent_name, agent_metadata, weights_info_dict}`; call `add_remote_agent(decoded_agent_metadata)` once per target | Miles rank |
| B2 | Local transfer object | Construct one Mooncake `TransferEngine` for all targets | Construct one local `nixl.Agent()` for all targets | Miles rank |
| S2 | CPU replica, conversion, and target grouping | `_create_cpu_replica()` creates shared pinned buffers; conversion and `_transfer_engine_meta_list` group work by target rank | Same; only the remote identity in each metadata entry differs | Miles rank |
| B3 | Source memory registration | `register_cpu_memory()` registers the shared pinned buffers with the transfer engine | `register_cpu_memory_nixl()` registers `(addr, size, 0, "")` DRAM regions with the agent | Miles rank, once on first prepare |
| S3 | Staging and scheduling | Stage shards, call `load_weights()`, synchronously protect reusable replicas, and submit the final replica's writes in the background | Same | Miles rank |
| B4a | Per-tensor source descriptors | Registered `(addr, numel, element_size)` tuple | `(addr, size, 0)` tuple passed to `get_xfer_descs` | Miles rank |
| B4b | Per-tensor destination descriptors | Remote `(addr, numel, element_size)` from `RemoteWeightInfo` | Remote `(addr, size, device_id)` derived from the four-field weight metadata | Miles rank |
| B4c | Transfer call | `batch_transfer_sync_write(session_id, source_ptrs, target_ptrs, source_lens)` | `get_xfer_descs` → `initialize_xfer("WRITE", …)` → `transfer()` → poll `check_xfer_state()` | Miles rank |
| S4 | Completion barrier | `_gather_and_update_expert_weights()` calls `wait_transfers()` | Same | Miles rank |
| S5 | Engine finalization | Update the weight version and resume with `post_load_weights=True` | Same | Miles / SGLang lifecycle |

**Reading the map.** `B0`–`B4` correspond to the five branch points above; `S1`–`S5` make the shared
blocks explicit. All stages run inside Miles. SGLang registers GPU memory, serves metadata, and runs
the existing post-load lifecycle hook, but it does not issue transfer operations. The crux of the
difference is `B1`: Mooncake uses an implicit `P2PHANDSHAKE`; NIXL requires an explicit
`add_remote_agent` call before any transfer can proceed (assumption **A6**).

## 4. Minimal delta for NIXL

Per touch point, the smallest change that makes NIXL work. Nothing is duplicated that can be branched.

### 4.1 Backend flag + run.py mode (tiny)

- Add `--update-weight-transfer-backend {mooncake,nixl}` to `arguments.py`.
- Add `--mode nixl` choice to `run.py`; when selected, emit the Miles backend flag and the SGLang
  NIXL seed flag.
- Fix the endpoint URL in `sglang_engine.py` (`/remote_instance_transfer_engine_info`, not
  `/get_remote_instance_transfer_engine_info`).

### 4.2 Schema consumer: `RemoteWeightInfo` + query function (branch in one helper)

- Extend `RemoteWeightInfo` with two new optional fields: `agent_name` (str) and `backend` (str,
  default `"mooncake"`).
- Update `query_remote_weight_infos()` to detect `backend == "nixl"` in the response and parse the
  4-field dict format, while keeping backward compatibility with the Mooncake 3-field tuple format.
- Add `add_nixl_remote_agent(nixl_agent, agent_metadata_b64)`: base64-decode and call
  `nixl_agent.add_remote_agent(raw)`.

### 4.3 NIXL agent init (branch in one method)

- Add `create_nixl_agent()` helper in `p2p_transfer_utils.py`.
- In `p2p.py` init, branch on `transfer_backend`: if `"nixl"`, call `create_nixl_agent()` then
  `add_nixl_remote_agent()`; otherwise run the existing Mooncake init path unchanged.
- Store `nixl_agent` and `backend` on the P2P object for use in the transfer loop.

### 4.4 CPU DRAM registration (branch in one helper)

- Add `register_cpu_memory_nixl(nixl_agent, tensors)` in `p2p_transfer_utils.py`.
- Tensors must be pinned; `device_id = 0` for all DRAM regions.
- `_pause_and_prepare_engines()` registers the shared CPU buffers on its first call, mirroring
  Mooncake's one-time registration. The buffers and registration remain valid for subsequent
  weight-update iterations.

### 4.5 NIXL WRITE transfer loop (branch in the write path)

- Branch `_do_p2p_write_one_session()`: if `backend == "nixl"`, run the NIXL transfer sequence per
  parameter; otherwise keep the existing Mooncake `batch_transfer_sync_write()` call.
- No Mooncake code is modified; the two paths are fully independent.

## 5. The metadata schema Miles consumes

SGLang publishes per-tp_rank metadata at `/remote_instance_transfer_engine_info`. The NIXL response
is:

```json
{
  "backend": "nixl",
  "agent_name": "<uuid string identifying this SGLang worker>",
  "agent_metadata": "<base64 of agent.get_agent_metadata()>",
  "weights_info_dict": {
    "<param_name>": [addr, numel, element_size, device_id]
  }
}
```

Compare with the existing Mooncake response (unchanged):

```json
{
  "backend": "mooncake",
  "session_id": "<host:port>",
  "weights_info_dict": {
    "<param_name>": [addr, numel, element_size]
  }
}
```

The only schema changes Miles must handle as a consumer are:
- `backend` field — new, used to select branch.
- `agent_name` / `agent_metadata` — new NIXL-only fields; `agent_metadata` is base64 and must be
  decoded before `add_remote_agent`.
- 4-field `weights_info_dict` entries (adds `device_id`) — needed for NIXL `get_xfer_descs` dst
  descriptors.

The HTTP URL (`/remote_instance_transfer_engine_info`) is unchanged; Miles' query function just
reads a richer payload.
