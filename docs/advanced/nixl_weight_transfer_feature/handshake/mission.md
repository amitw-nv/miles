---
title: Miles NIXL — Step 1 Mission: Flags & Handshake
---

## What this step delivers

By the end of this step, Miles can:
1. Accept `--mode nixl` at the run.py level and translate it to the correct train.py flags.
2. Create a local `nixl_agent` on each training rank.
3. Query SGLang's `/remote_instance_transfer_engine_info` endpoint and parse the new tagged-dict response.
4. Call `add_remote_agent(decoded_metadata)` so the Miles NIXL agent and the SGLang NIXL agent know about each other.

No weights are transferred yet. The step is complete when the NIXL peer connection is established end-to-end without errors.

## Scope

### `examples/p2p_weight_transfer/run.py`
- Add `nixl` as a valid `--mode` choice.
- When `mode == "nixl"`: pass `--update-weight-transfer-mode p2p` + `--update-weight-transfer-backend nixl` to train.py, and pass `--sglang-remote-instance-weight-loader-start-seed-via-nixl` to SGLang (instead of the `transfer-engine` variant).
- Extend the two `mode == "p2p"` blocks (~lines 1015 and 1053) to also match `mode == "nixl"` so all shared P2P args and buffer-size logic apply.

### `miles/utils/arguments.py`
- Add `--update-weight-transfer-backend {mooncake,nixl}` (default `"mooncake"`).
- Validation: reject `backend=nixl` when `--update-weight-transfer-mode` is not `p2p`.

### `miles/backends/sglang_utils/sglang_engine.py`
- Fix the endpoint URL: `/get_remote_instance_transfer_engine_info` → `/remote_instance_transfer_engine_info` (resolves existing TODO comment).

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`
- Add `create_nixl_agent()`: constructs a `nixl_agent` using `nixl_agent_config` with `SGLANG_DISAGGREGATION_NIXL_BACKEND` (default `"UCX"`), same pattern as `sglang/srt/disaggregation/nixl/conn.py`.
- Extend `RemoteWeightInfo` with `agent_name: str | None`, `backend: str = "mooncake"`.
- Update `query_remote_weight_infos()`: detect `backend == "nixl"` in the response dict, base64-decode `agent_metadata`, call `nixl_agent.add_remote_agent(decoded_bytes)`, store `agent_name` in `RemoteWeightInfo`.

### `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`
- Branch on `args.update_weight_transfer_backend` at init: `"nixl"` → call `create_nixl_agent()` and store as `self.nixl_agent`; `"mooncake"` → existing `create_transfer_engine()` path unchanged.

### `miles/backends/megatron_utils/actor.py`
- Pass `transfer_backend` from args to `UpdateWeightP2P` constructor.

## Out of scope for this step
- CPU memory registration with NIXL.
- Any NIXL WRITE transfer calls.
- Changes to the bucketed flush / write-to-target logic.
