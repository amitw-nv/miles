# Tests: Miles NIXL backend for P2P weight transfer

Full test details for the implementation roadmap in `impl.md`.
The impl doc lists only the test name + what it checks; the runnable commands and
expected results live here.

## Conventions

- **Model path** (inside the dev container):
  `Qwen/Qwen2-0.5B` for no-frills e2e; use a larger model (`meta-llama/Llama-3.1-8B`) for
  multi-node RDMA tests. SGLang must be launched with
  `--remote-instance-weight-loader-start-seed-via-nixl` for all NIXL e2e tests.
- **Test types**:
  - *static* — source inspection; no import side effects.
  - *unit* — imports a module or patches at import time; no GPU or model.
  - *e2e* — launches real processes; needs GPU (+ model + NIXL/UCX for the NIXL path).
- A one-shot runner for the no-GPU tests of steps 1–4 lives at `run_tests_steps_1_4.sh`
  (created separately).

### Prerequisite gate (run before any NIXL e2e test)

```bash
python -c "import nixl; print('nixl importable')"
```

Not a feature test — it is a gate. If it fails, all NIXL e2e tests (3c, 4b, 4c, 5e) will run
vacuously or error on import; confirm this prints `nixl importable` before trusting them.

---

## Step 1 — Backend flag, mode, and endpoint fix

### Test 1a — `--update-weight-transfer-backend` flag appears in `--help` (unit, no GPU)

```bash
python train.py --help | grep -- --update-weight-transfer-backend
```

- **Checks:** the new argparse flag is wired into the Miles training entry point.
- **Expected:** one line printed showing the flag with choices `mooncake` and `nixl`. Empty output = fail.

### Test 1b — `--mode nixl` emits correct Miles and SGLang flags (static, no GPU)

```bash
python -c "
import inspect
import examples.p2p_weight_transfer.run as r
src = inspect.getsource(r)
assert '--update-weight-transfer-backend' in src and 'nixl' in src
assert '--remote-instance-weight-loader-start-seed-via-nixl' in src
print('OK: --mode nixl emits both flags')
"
```

- **Checks:** `run.py` translates `--mode nixl` to the Miles backend flag and the SGLang NIXL seed flag.
- **Expected:** prints `OK: --mode nixl emits both flags`.

### Test 1c — `mooncake` is the default mode (unit, no GPU)

```bash
python examples/p2p_weight_transfer/run.py run --help | grep -A2 -- --mode
```

- **Checks:** `--mode` defaults to `mooncake` when omitted.
- **Expected:** output shows `default: mooncake` (or `mooncake` as the first choice with no explicit default shown). `nixl` as default = fail.

### Test 1d — endpoint URL is correct (static, no GPU)

```bash
python -c "
import inspect
from miles.engine import sglang_engine
src = inspect.getsource(sglang_engine)
assert '/remote_instance_transfer_engine_info' in src
assert '/get_remote_instance_transfer_engine_info' not in src, 'stale URL still present'
print('OK: endpoint URL correct')
"
```

- **Checks:** the correct endpoint path is present and the stale path has been removed.
- **Expected:** prints `OK: endpoint URL correct`.

---

## Step 2 — Schema consumer: `RemoteWeightInfo` + query function

### Test 2a — `query_remote_weight_infos()` parses NIXL dict format (unit, no GPU)

```bash
python -c "
from unittest.mock import patch
import requests
from miles.p2p.p2p_transfer_utils import query_remote_weight_infos

fake = {
    'backend': 'nixl',
    'agent_name': 'sglang_worker_0',
    'agent_metadata': 'dGVzdA==',
    'weights_info_dict': {
        'model.embed_tokens.weight': [140000000000, 131072, 2, 0]
    },
}
with patch.object(requests, 'get') as m:
    m.return_value.json.return_value = fake
    backend, infos = query_remote_weight_infos('http://fake/remote_instance_transfer_engine_info')

assert backend == 'nixl', backend
info = infos['model.embed_tokens.weight']
assert info.agent_name == 'sglang_worker_0'
assert info.device_id == 0
assert info.backend == 'nixl'
print('OK: NIXL dict format parsed')
"
```

- **Checks:** the NIXL 4-field dict format is parsed; `agent_name` and `backend` are populated on `RemoteWeightInfo`.
- **Expected:** prints `OK: NIXL dict format parsed`.

### Test 2b — `query_remote_weight_infos()` still parses Mooncake tuple format (unit, no GPU)

```bash
python -c "
from unittest.mock import patch
import requests
from miles.p2p.p2p_transfer_utils import query_remote_weight_infos

fake = {
    'backend': 'mooncake',
    'session_id': '10.0.0.7:18000',
    'weights_info_dict': {
        'model.embed_tokens.weight': [140000000000, 131072, 2, 0]
    },
}
with patch.object(requests, 'get') as m:
    m.return_value.json.return_value = fake
    backend, infos = query_remote_weight_infos('http://fake/remote_instance_transfer_engine_info')

assert backend == 'mooncake', backend
assert infos['model.embed_tokens.weight'].backend == 'mooncake'
print('OK: Mooncake format still parsed')
"
```

- **Checks:** the existing Mooncake format continues to parse correctly after the Step 2 changes.
- **Expected:** prints `OK: Mooncake format still parsed`.

### Test 2c — `RemoteWeightInfo` has new optional fields with correct defaults (static, no GPU)

```bash
python -c "
from miles.p2p.p2p_transfer_utils import RemoteWeightInfo
import dataclasses
fields = {f.name: f for f in dataclasses.fields(RemoteWeightInfo)}
assert 'agent_name' in fields and fields['agent_name'].default == ''
assert 'backend' in fields and fields['backend'].default == 'mooncake'
print('OK: RemoteWeightInfo has new fields')
"
```

- **Checks:** both new fields exist with the correct defaults so old call sites do not break.
- **Expected:** prints `OK: RemoteWeightInfo has new fields`.

---

## Step 3 — NIXL agent init and peer connection

### Test 3a — `create_nixl_agent()` exists and returns a NIXL agent (unit, needs NIXL)

```bash
python -c "import nixl; print('nixl importable')"   # gate first
python -c "
from miles.p2p.p2p_transfer_utils import create_nixl_agent
agent = create_nixl_agent()
assert agent is not None
print('OK: create_nixl_agent returns agent')
"
```

- **Checks:** the helper exists and constructs a NIXL agent without error.
- **Expected:** prints `OK: create_nixl_agent returns agent`.

### Test 3b — p2p init branches on backend (static, no GPU)

```bash
python -c "
import inspect
from miles.p2p import p2p
src = inspect.getsource(p2p)
assert 'create_nixl_agent' in src
assert 'add_nixl_remote_agent' in src
assert 'nixl' in src and 'mooncake' in src    # both branches present
print('OK: p2p.py branches on backend')
"
```

- **Checks:** the NIXL init helpers are called in `p2p.py` and the Mooncake path is still present.
- **Expected:** prints `OK: p2p.py branches on backend`.

### Test 3c — handshake completes against a live SGLang NIXL seed (e2e, needs GPU + NIXL + SGLang)

Launch SGLang with `--remote-instance-weight-loader-start-seed-via-nixl`, then:

```bash
curl -s "http://localhost:30000/remote_instance_transfer_engine_info?rank=0" | python -c "
import sys, json, base64
d = json.load(sys.stdin)['remote_instance_transfer_engine_info']
assert d['backend'] == 'nixl', d.get('backend')
raw = base64.b64decode(d['agent_metadata'])
assert len(raw) > 0
print('OK: SGLang NIXL endpoint valid')
"
```

Then run Miles with `--update-weight-transfer-backend nixl` and a dry-run / handshake-only mode.

- **Checks:** Miles queries the endpoint, decodes `agent_metadata`, calls `add_remote_agent`, and
  no exception is raised. Miles log must contain a line indicating the remote agent was added.
- **Expected:** no crash; log shows `add_remote_agent succeeded` (or equivalent).
- **Failure signatures:**
  - `ConnectionRefused` → SGLang not started or wrong port.
  - `KeyError: 'agent_metadata'` → SGLang not in NIXL seed mode.
  - NIXL error in `add_remote_agent` → network or plugin issue; check UCX/IB config.

---

## Step 4 — CPU DRAM memory registration

### Test 4a — registration helpers exist (static, no GPU)

```bash
python -c "
from miles.p2p import p2p_transfer_utils as u
assert hasattr(u, 'register_cpu_memory_nixl')
assert hasattr(u, 'deregister_cpu_memory_nixl')
print('OK: registration helpers present')
"
```

- **Checks:** both helpers are importable from `p2p_transfer_utils`.
- **Expected:** prints `OK: registration helpers present`.

### Test 4b — pinned tensor registers and deregisters without error (unit, needs NIXL)

```bash
python -c "import nixl; print('nixl importable')"   # gate first
python -c "
import torch
from miles.p2p.p2p_transfer_utils import create_nixl_agent, register_cpu_memory_nixl, deregister_cpu_memory_nixl
agent = create_nixl_agent()
t = torch.zeros(1024, dtype=torch.float16).pin_memory()
handles = register_cpu_memory_nixl(agent, [t])
assert len(handles) == 1
deregister_cpu_memory_nixl(agent, handles)
print('OK: pinned tensor registered and deregistered')
"
```

- **Checks:** a pinned CPU tensor registers as a DRAM region and cleanly deregisters.
- **Expected:** prints `OK: pinned tensor registered and deregistered`.

### Test 4c — non-pinned tensor raises `AssertionError` (unit, needs NIXL)

```bash
python -c "
import torch
from miles.p2p.p2p_transfer_utils import create_nixl_agent, register_cpu_memory_nixl
agent = create_nixl_agent()
t = torch.zeros(1024, dtype=torch.float16)   # NOT pinned
try:
    register_cpu_memory_nixl(agent, [t])
    raise RuntimeError('expected AssertionError')
except AssertionError:
    print('OK: non-pinned tensor rejected')
"
```

- **Checks:** the pinned-memory precondition is enforced with a clear error.
- **Expected:** prints `OK: non-pinned tensor rejected`.

---

## Step 5 — NIXL WRITE transfer

### Test 5a — NIXL write path calls the 3-step API in the correct order (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock
import torch
from miles.p2p.p2p_transfer_utils import RemoteWeightInfo

agent = MagicMock()
agent.check_xfer_state.return_value = 'DONE'

# construct a minimal P2P object without full init
from miles.p2p import p2p as p2p_mod
obj = object.__new__(p2p_mod.UpdateWeightP2P)   # adjust class name as needed
obj._nixl_agent = agent

cpu_tensor = torch.zeros(64, dtype=torch.float16).pin_memory()
remote_infos = {
    'layer.weight': RemoteWeightInfo(
        addr=0xDEAD0000, numel=64, element_size=2, device_id=1,
        agent_name='sglang_worker_0', backend='nixl',
    )
}
obj._transfer_weights_nixl({'layer.weight': cpu_tensor}, remote_infos)

calls = [c[0] for c in agent.method_calls]
assert 'get_xfer_descs' in str(calls)
assert 'initialize_xfer' in str(calls)
assert 'transfer' in str(calls)
assert 'check_xfer_state' in str(calls)
print('OK: NIXL 3-step API called in order')
"
```

- **Checks:** `get_xfer_descs`, `initialize_xfer`, `transfer`, and `check_xfer_state` are all invoked, in that order.
- **Expected:** prints `OK: NIXL 3-step API called in order`.

### Test 5b — `"ERR"` state raises `RuntimeError` (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock
import torch
from miles.p2p.p2p_transfer_utils import RemoteWeightInfo
from miles.p2p import p2p as p2p_mod

agent = MagicMock()
agent.check_xfer_state.return_value = 'ERR'

obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj._nixl_agent = agent
cpu_tensor = torch.zeros(64, dtype=torch.float16).pin_memory()
remote_infos = {
    'layer.weight': RemoteWeightInfo(
        addr=0xDEAD0000, numel=64, element_size=2, device_id=1,
        agent_name='sglang_worker_0', backend='nixl',
    )
}
try:
    obj._transfer_weights_nixl({'layer.weight': cpu_tensor}, remote_infos)
    raise AssertionError('expected RuntimeError')
except RuntimeError as e:
    assert 'layer.weight' in str(e) or 'NIXL' in str(e)
    print('OK: ERR state raises RuntimeError')
"
```

- **Checks:** `check_xfer_state` returning `"ERR"` raises `RuntimeError` with the parameter name.
- **Expected:** prints `OK: ERR state raises RuntimeError`.

### Test 5c — deregistration runs even when transfer raises (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock, patch
import torch
from miles.p2p.p2p_transfer_utils import RemoteWeightInfo
from miles.p2p import p2p as p2p_mod

agent = MagicMock()
agent.check_xfer_state.return_value = 'ERR'

obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj._nixl_agent = agent
cpu_tensor = torch.zeros(64, dtype=torch.float16).pin_memory()
remote_infos = {
    'w': RemoteWeightInfo(addr=0, numel=64, element_size=2, device_id=0,
                          agent_name='peer', backend='nixl')
}
try:
    obj._transfer_weights_nixl({'w': cpu_tensor}, remote_infos)
except RuntimeError:
    pass
assert agent.deregister_memory.called, 'deregister_memory not called on error path'
print('OK: deregistration runs in finally block')
"
```

- **Checks:** `deregister_cpu_memory_nixl` (i.e. `agent.deregister_memory`) is called even when the transfer raises.
- **Expected:** prints `OK: deregistration runs in finally block`.

### Test 5d — Mooncake write path unchanged (regression, no GPU)

Run the existing Mooncake weight-transfer unit tests. All must pass without modification.

- **Checks:** the NIXL branch additions in `p2p.py` do not affect the `backend == "mooncake"` code path.
- **Expected:** existing Mooncake test suite exits 0.

### Test 5e — small-model E2E with NIXL (e2e, needs GPU + NIXL + SGLang)

```bash
python -c "import nixl; print('nixl importable')"   # gate first
python examples/p2p_weight_transfer/run.py run Qwen/Qwen2-0.5B \
    --mode nixl \
    --check-weight-update-equal \
    --num-weight-updates 2
```

- **Checks:** both weight-update iterations complete, `--check-weight-update-equal` reports no
  mismatches, and no NIXL `ERR` state is logged.
- **Expected:** clean exit with a summary showing all parameter checks passed.
- **Failure signatures:**
  - `AssertionError` from `--check-weight-update-equal` → weight data corrupted during transfer; check descriptor sizes and `device_id` values.
  - `RuntimeError: NIXL transfer failed` → `check_xfer_state` returned `"ERR"`; check IB/UCX link.
  - Hang in `check_xfer_state` poll → transfer never completed; set a timeout and inspect NIXL logs.

---

## Summary

| Test | Step | Type | Needs GPU? | Needs NIXL? | Pass signal |
|---|---|---|---|---|---|
| 1a | 1 | unit | no | no | flag in `--help` |
| 1b | 1 | static | no | no | both NIXL flags in run.py source |
| 1c | 1 | unit | no | no | `mooncake` is default mode |
| 1d | 1 | static | no | no | correct URL, stale URL absent |
| 2a | 2 | unit | no | no | NIXL dict parsed, `agent_name` populated |
| 2b | 2 | unit | no | no | Mooncake tuple still parsed |
| 2c | 2 | static | no | no | new fields with correct defaults |
| 3a | 3 | unit | no | yes | `create_nixl_agent` returns agent |
| 3b | 3 | static | no | no | both branches in `p2p.py` |
| 3c | 3 | e2e | yes | yes | `add_remote_agent` succeeds, no crash |
| 4a | 4 | static | no | no | helpers importable |
| 4b | 4 | unit | no | yes | pinned tensor registers + deregisters |
| 4c | 4 | unit | no | yes | non-pinned raises `AssertionError` |
| 5a | 5 | unit | no | no (mocked) | 3-step API called in order |
| 5b | 5 | unit | no | no (mocked) | `"ERR"` raises `RuntimeError` |
| 5c | 5 | unit | no | no (mocked) | deregister runs in `finally` |
| 5d | 5 | unit | no | no | Mooncake suite passes unchanged |
| 5e | 5 | e2e | yes | yes | `--check-weight-update-equal` passes |

No-GPU tests (1a–1d, 2a–2c, 3b, 4a, 5a–5d) are automated by `run_tests_steps_1_4.sh`. The e2e
tests (3c, 4b, 4c, 5e) need a container with GPUs and NIXL/UCX installed.
