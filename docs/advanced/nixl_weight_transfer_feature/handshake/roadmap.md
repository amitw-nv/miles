---
title: Miles NIXL — Step 1 Roadmap: Flags & Handshake
---

## Implementation order

### 1. `miles/utils/arguments.py`
Add `--update-weight-transfer-backend` — no logic, just the new arg. Touch nothing else. Validates cleanly and is safe to merge in isolation.

### 2. `miles/backends/megatron_utils/actor.py`
Forward `args.update_weight_transfer_backend` to `UpdateWeightP2P`. One-line change; confirms the arg flows through before any backend-specific code exists.

### 3. `examples/p2p_weight_transfer/run.py`
Add `nixl` choice to `--mode`. Wire up the flag table (see mission.md). Extend both `mode == "p2p"` blocks to also match `"nixl"`. Confirm `--mode p2p` is untouched by running the existing mooncake test.

### 4. `miles/backends/sglang_utils/sglang_engine.py`
Fix the endpoint URL TODO. This is a standalone bug fix affecting both backends — do it before adding any NIXL parsing so the base HTTP call is correct.

### 5. `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py`
- Add `create_nixl_agent()`.
- Extend `RemoteWeightInfo` with `agent_name` and `backend` fields.
- Update `query_remote_weight_infos()`: branch on `backend`, base64-decode `agent_metadata`, call `add_remote_agent`.

### 6. `miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py`
Branch on `args.update_weight_transfer_backend` at init to call `create_nixl_agent()`.

## Testing
- `--mode p2p` (Mooncake): full existing test must pass unchanged.
- `--mode nixl` smoke test: start SGLang with `--remote-instance-weight-loader-start-seed-via-nixl`, start Miles with `--mode nixl`. Confirm no crash at handshake, NIXL peers connected. Weight transfer will not succeed yet (no CPU registration / no WRITE logic) — that is expected.

## Definition of done
- `--mode p2p` unchanged and passing.
- Miles connects to SGLang-NIXL, `add_remote_agent` completes without error, process does not crash at handshake initialization.
