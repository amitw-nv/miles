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

| # | Pipeline block | Mooncake API (exists) | NIXL parallel (to add) | Who owns it |
|---|---|---|---|---|
| 1 | Agent / engine object | `TransferEngine` handle from SGLang `session_id` | `nixl.Agent()` constructed locally by Miles | Miles rank |
| 2 | Peer connection | `session_id` host:port string, implicit P2PHANDSHAKE | `add_remote_agent(decoded_agent_metadata)` called once per target at init (not per iteration); results stored as `agent_name → weights_info_dict` map | Miles rank |
| 3 | Source memory reg | CPU tensors referenced by pointer only | `agent.register_memory([(addr, size, 0, "")], "DRAM")` with pinned tensors | Miles rank |
| 4 | Per-tensor source descriptors | `(addr, numel, element_size)` 3-tuple | `(addr, size, 0)` 3-tuple for `get_xfer_descs` | Miles rank |
| 5 | Per-tensor dest descriptors | remote `(addr, numel, element_size)` from `RemoteWeightInfo` | `(addr, size, device_id)` 3-tuple from extended `RemoteWeightInfo` | Miles rank |
| 6 | Transfer call | `transfer_engine.transfer(session_id, local, remote, lens)` | `get_xfer_descs` → `initialize_xfer("WRITE", …)` → `transfer()` → poll `check_xfer_state()` | Miles rank |
| 7 | Metadata discovery | `GET /remote_instance_transfer_engine_info` → `{session_id, weights_info_dict}` | same URL → `{backend, agent_name, agent_metadata(b64), weights_info_dict}` with 4-field entries | Miles (consumer) |

**Reading the map.** All stages run inside Miles. SGLang is a pure data source: it registers GPU
memory and serves metadata, but no SGLang code executes during the transfer. The crux of the
difference is stage 2: Mooncake uses an implicit `P2PHANDSHAKE`; NIXL requires an explicit
`add_remote_agent` call before any transfer can proceed (assumption **A6**). Everything else is a
1:1 substitution of the API call within an already-existing block.

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

- Add `register_cpu_memory_nixl(nixl_agent, tensors)` and `deregister_cpu_memory_nixl(nixl_agent,
  handles)` in `p2p_transfer_utils.py`.
- Tensors must be pinned; `device_id = 0` for all DRAM regions.
- Registration happens after format conversion, before `get_xfer_descs`; deregistration is in a
  `finally` block to avoid leaks on error.

### 4.5 NIXL WRITE transfer loop (branch in the write path)

- Add `_transfer_weights_nixl()` in `p2p.py` — the NIXL 3-step sequence per parameter.
- Branch the existing write loop: if `backend == "nixl"`, call `_transfer_weights_nixl()`; otherwise
  keep the existing Mooncake call.
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
