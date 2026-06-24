---
title: NIXL Weight Transfer Backend — Design
description: Design for replacing Mooncake's TransferEngine with NIXL for direct RDMA weight writes from Miles to SGLang.
---

This document describes the design for adding NIXL as a P2P weight transfer backend in Miles. It is the Miles-side counterpart to SGLang's `rfork_nixl_p2p_design.md`. The implementation plan is in `nixl-weight-transfer-impl.md`.

## Background and Motivation

The existing P2P weight transfer path uses **Mooncake's `TransferEngine`** (`from mooncake.engine import TransferEngine`). In this path:

- Miles registers CPU pinned weight buffers with Mooncake.
- SGLang registers its GPU weight buffers with Mooncake.
- Miles writes weights over RDMA from CPU (Miles) → GPU (SGLang) via Mooncake's session-ID–based protocol.

**NIXL** (NVIDIA's transfer library, already used for PD-disaggregation in SGLang) provides a unified RDMA transfer primitive that works independently of Mooncake. The goal is to support NIXL as a P2P backend.

The roles are unchanged: **SGLang is the passive target** (registers GPU memory, publishes metadata). **Miles is the active driver** (discovers metadata, calls `add_remote_agent`, issues WRITE transfers).

## Assumptions

- **Miles is the NIXL initiator.** Miles constructs a local `nixl_agent`, registers its CPU pinned source buffers, calls `add_remote_agent` with SGLang's opaque metadata blob, then issues NIXL WRITE xfers targeting SGLang's GPU addresses. SGLang never calls `add_remote_agent` back.
- **SGLang has already been updated** per `rfork_nixl_p2p_design.md`: it accepts `--remote-instance-weight-loader-start-seed-via-nixl`, registers GPU VRAM buffers with its own `nixl_agent`, and publishes a tagged metadata dict via the existing `/remote_instance_transfer_engine_info` endpoint.
- **CPU model replica is still required** on the Miles side for Megatron→HF format conversion and weight re-sharding. The CPU replica itself doesn't change; only how its buffers are registered and transferred changes.
- **NIXL transport backend** is chosen via the existing `SGLANG_DISAGGREGATION_NIXL_BACKEND` env var (default `"UCX"`), consistent with SGLang's NIXL disaggregation path.
- **One `nixl_agent` per Miles training rank.** The agent is initialized once and lives for the process lifetime, mirroring the Mooncake `TransferEngine`.
- **`weights_info_dict` entries are now 4-tuples** `(addr, numel, element_size, device_id)` from the SGLang endpoint. The Mooncake path used 3-tuples `(addr, numel, element_size)`.

## Metadata Handshake

SGLang's `/remote_instance_transfer_engine_info?rank=<tp_rank>` now returns a tagged dict:

```json
{
  "backend": "nixl",
  "agent_name": "<uuid>",
  "agent_metadata": "<base64-encoded bytes from nixl_agent.get_agent_metadata()>",
  "weights_info_dict": {
    "<param_name>": [addr, numel, element_size, device_id],
    ...
  }
}
```

Miles must:

1. Query the endpoint (already done by `query_remote_weight_infos`).
2. Detect `backend == "nixl"` in the response.
3. Base64-decode `agent_metadata`.
4. Call `miles_nixl_agent.add_remote_agent(decoded_bytes)` — this registers the SGLang rank as a known NIXL peer, enabling Miles to address that GPU's memory.
5. Store `(agent_name, weights_info_dict)` as the remote descriptor in place of `(session_id, weights_info_dict)`.

## Architecture Delta

The diagram below shows what changes versus the Mooncake P2P path.

```
┌─────────────────────────────────────────────────────────────┐
│                    Miles Training Rank                      │
│                                                             │
│  DistBucketedWeightUpdateMixin (UNCHANGED)                  │
│  ├── TP/EP all-gather                                       │
│  ├── HF format conversion                                   │
│  └── bucketed flush ──→ UpdateWeightP2P._flush_bucket()     │
│                              │                              │
│          ┌───────────────────┴──────────────────────┐       │
│          │  mooncake path        │  NIXL path (NEW)  │      │
│          │  transfer_engine      │  nixl_agent        │      │
│          │  .transfer(           │  .initialize_xfer( │      │
│          │    session_id,        │    src_descs,      │      │
│          │    src_addr, size,    │    dst_descs,      │      │
│          │    dst_addr)          │    peer_name)      │      │
│          └───────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            │ RDMA WRITE
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   SGLang Rollout Rank                       │
│                                                             │
│  GPU VRAM — weight tensors registered with nixl_agent       │
│  (addr, numel, element_size, device_id) per param           │
└─────────────────────────────────────────────────────────────┘
```

## Invariants and Constraints

| Invariant | Notes |
|---|---|
| `add_remote_agent` is called once per `(miles_rank, sgl_rank)` pair at init time | Re-querying the endpoint on weight reallocations re-calls `add_remote_agent` with updated metadata |
| `nixl_agent` and `_weight_descs` must outlive all transfers | Both stored at instance scope |
| NIXL WRITE is unidirectional | SGLang never writes to Miles; Miles never reads from SGLang via NIXL |
| CPU model replica is still needed | Format conversion (Megatron→HF) and re-sharding are not NIXL concerns |
| Expert params and non-expert params use separate NIXL descriptor sets | Mirrors the existing bucketed-all-gather split in `mixin.py` |

## Out of Scope

- Broadcast mode NIXL support — NIXL is WRITE-only P2P; collective broadcast stays NCCL.
- LoRA adapter transfer via NIXL.
- NIXL READ path (SGLang pulling from Miles) — not needed; Miles always pushes.
- CPU-side `"DRAM"`-to-`"DRAM"` NIXL transfers — all target addresses are SGLang GPU VRAM.

## Open Questions

- **Q1 — NIXL descriptor granularity. RESOLVED.** SGLang registers contiguous merged blocks with `register_memory`, but `weights_info_dict` still carries individual per-tensor `(addr, numel, element_size, device_id)`. Miles calls `get_xfer_descs([(tensor_addr, tensor_bytes, device_id)], "VRAM")` per tensor — this addresses individual bytes within a registered region, which NIXL supports. No offset mapping needed on the Miles side.
- **Q2 — Completion polling. RESOLVED.** The completion API is `check_xfer_state(xfer_handle)` returning `"DONE"` or `"ERR"` (confirmed from `sglang/srt/disaggregation/nixl/conn.py`). There is no `wait()` method. The blocking poll loop runs inside `P2PTransferManager`'s thread pool, same as the Mooncake path.
- **Q3 — Re-registration on SGLang model reload.** If SGLang reallocates weights (e.g., quant swap), `agent_metadata` changes and Miles must re-query and re-call `add_remote_agent`. The current Mooncake path already re-queries each round; the same guard applies here.
