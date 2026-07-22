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
- The commands below are the source of truth; no aggregate runner is assumed.

### Prerequisite gate (run before any NIXL e2e test)

```bash
python -c "import nixl; print('nixl importable')"
```

Not a feature test — it is a gate. If it fails, the NIXL-dependent tests (3a, 3c, 4c, and 5e)
will error on import or cannot run; confirm this prints `nixl importable` before trusting them.

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
from miles.backends.sglang_utils import sglang_engine
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
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    query_remote_weight_infos,
)

response = {
    'backend': 'nixl',
    'agent_name': 'sglang_worker_0',
    'agent_metadata': 'dGVzdA==',
    'weights_info_dict': {
        'model.embed_tokens.weight': [140000000000, 131072, 2, 0]
    },
}
engine = MagicMock()
target = SimpleNamespace(engine_ind=0, engine_rank=0)

# query result, parallelism info, then server info
# ServerArgs requires model_path; the mock must include it.
with patch(
    'miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils.ray.get',
    side_effect=[response, {'tp_rank': 0}, {'model_path': 'dummy'}],
):
    remote_by_id, target_to_id, _ = query_remote_weight_infos([engine], [target])

remote_id = target_to_id[(0, 0)]
assert remote_id == 'sglang_worker_0'
weights_info = remote_by_id[remote_id][0]
assert weights_info['model.embed_tokens.weight'] == (140000000000, 131072, 2, 0)
print('OK: NIXL dict format parsed')
"
```

- **Checks:** the existing Ray-actor query contract accepts the NIXL response, uses `agent_name` as
  the remote identity, and preserves the four-field destination metadata.
- **Expected:** prints `OK: NIXL dict format parsed`.

### Test 2b — `query_remote_weight_infos()` parses Mooncake tagged dict format (unit, no GPU)

```bash
python -c "
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    query_remote_weight_infos,
)

response = {
    'backend': 'mooncake',
    'session_id': '10.0.0.7:18000',
    'weights_info_dict': {
        'model.embed_tokens.weight': [140000000000, 131072, 2]
    },
}
engine = MagicMock()
target = SimpleNamespace(engine_ind=0, engine_rank=0)

with patch(
    'miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils.ray.get',
    side_effect=[response, {'tp_rank': 0}, {'model_path': 'dummy'}],
):
    remote_by_id, target_to_id, _ = query_remote_weight_infos([engine], [target])

session_id = target_to_id[(0, 0)]
assert session_id == '10.0.0.7:18000'
assert remote_by_id[session_id][0]['model.embed_tokens.weight'] == (140000000000, 131072, 2)
print('OK: Mooncake format still parsed')
"
```

- **Checks:** the tagged Mooncake dict with 3-field weight entries continues to parse correctly.
- **Expected:** prints `OK: Mooncake format still parsed`.

### Test 2c — `RemoteWeightInfo` has new optional fields with correct defaults (static, no GPU)

```bash
python -c "
import dataclasses
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    RemoteWeightInfo,
)
fields = {f.name: f for f in dataclasses.fields(RemoteWeightInfo)}
assert 'agent_name' in fields and fields['agent_name'].default == ''
assert 'backend' in fields and fields['backend'].default == 'mooncake'
print('OK: RemoteWeightInfo has new fields')
"
```

- **Checks:** both new fields exist with the correct defaults so old call sites do not break.
- **Expected:** prints `OK: RemoteWeightInfo has new fields`.

### Test 2d — Miles-launched SGLang schema and real `ServerArgs` compatibility (mandatory E2E)

This check is mandatory in the full NIXL E2E Test 5e; it is not an optional standalone Step 2 test.
Step 2 alone cannot complete a NIXL Miles launch because agent connection, memory registration, and
WRITE are implemented in Steps 3–5.

When Test 5e launches Miles with `--mode nixl`, Miles launches the real SGLang seed and
`UpdateWeightP2P.connect_rollout_engines()` must successfully call `query_remote_weight_infos()`
against those real Ray actors. That call consumes the real transfer metadata, parallelism info, and
server info and constructs the installed SGLang `ServerArgs`.

- **Checks:** the Miles-launched SGLang returns tagged NIXL metadata with `agent_name`,
  `agent_metadata`, and four-field weight entries; Miles parses it and constructs real `ServerArgs`
  (including required `model_path`) before any transfer starts.
- **Pass condition:** the connection phase of mandatory Test 5e completes. A schema mismatch,
  missing `model_path`, or invalid `ServerArgs` aborts Test 5e and fails the feature E2E.

---

## Step 3 — NIXL agent init and peer connection

### Test 3a — `create_nixl_agent()` exists and returns a NIXL agent (unit, needs NIXL)

```bash
python -c "import nixl; print('nixl importable')"   # gate first
python -c "
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    create_nixl_agent,
)
agent = create_nixl_agent()
assert agent is not None
print('OK: create_nixl_agent returns agent')
"
```

- **Checks:** the helper exists and constructs a NIXL agent without error.
- **Expected:** prints `OK: create_nixl_agent returns agent`.

### Test 3b — `connect_rollout_engines()` branches on backend (static, no GPU)

```bash
python -c "
import inspect
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p import (
    UpdateWeightP2P,
)
src = inspect.getsource(UpdateWeightP2P.connect_rollout_engines)
assert 'create_nixl_agent' in src
assert 'add_nixl_remote_agent' in src
assert 'create_transfer_engine' in src
print('OK: connect_rollout_engines branches on backend')
"
```

- **Checks:** backend-specific metadata/handshake and transfer-object creation live in
  `connect_rollout_engines()` while both backends remain present.
- **Expected:** prints `OK: connect_rollout_engines branches on backend`.

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

### Test 4a — NIXL registration helper exists (static, no GPU)

```bash
python -c "
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p_transfer_utils as u
assert hasattr(u, 'register_cpu_memory_nixl')
print('OK: NIXL registration helper present')
"
```

- **Checks:** the one-time NIXL registration helper is importable from `p2p_transfer_utils`.
- **Expected:** prints `OK: NIXL registration helper present`.

### Test 4b — prepare registers NIXL memory only once (unit, no GPU)

```bash
python -c "
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as p2p_mod

obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj.transfer_plan = SimpleNamespace(_gathered_dp_rank=0, _rollout_num_gpus=1)
obj.args = SimpleNamespace(update_weight_transfer_backend='nixl')
obj._model_registered = False
obj._shared_params_dict = {'w': MagicMock()}
obj._nixl_agent = MagicMock()

with patch.object(
    p2p_mod.DistBucketedWeightUpdateMixin, '_pause_and_prepare_engines'
), patch.object(
    p2p_mod, 'register_cpu_memory_nixl', return_value={'w': (1, 2, 0)}
) as register:
    obj._pause_and_prepare_engines()
    obj._pause_and_prepare_engines()

assert register.call_count == 1
assert obj._model_registered
print('OK: NIXL memory registered once')
"
```

- **Checks:** `_pause_and_prepare_engines()` registers the persistent shared buffers only on its
  first call.
- **Expected:** prints `OK: NIXL memory registered once`.

### Test 4c — non-pinned tensor raises `AssertionError` (unit, needs NIXL)

```bash
python -c "
import torch
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    create_nixl_agent,
    register_cpu_memory_nixl,
)
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

## Step 5 — Branch the existing P2P WRITE

### Test 5a — NIXL write path calls the 3-step API in the correct order (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    RemoteWeightInfo,
)

agent = MagicMock()
agent.check_xfer_state.return_value = 'DONE'

# construct a minimal P2P object without full init
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as p2p_mod
obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj._nixl_agent = agent
obj._weight_memory_registry = {'layer.weight': (0xCAFE0000, 64, 2)}

remote = RemoteWeightInfo(
    session_id='sglang_worker_0',
    weights_info={'layer.weight': (0xDEAD0000, 64, 2, 1)},
    agent_name='sglang_worker_0',
    backend='nixl',
)
obj._do_p2p_write_one_session(remote, ['layer.weight'])

calls = [c[0] for c in agent.method_calls]
names = [str(call) for call in calls]
order = ['get_xfer_descs', 'initialize_xfer', 'transfer', 'check_xfer_state']
positions = [next(i for i, name in enumerate(names) if expected in name) for expected in order]
assert positions == sorted(positions)
print('OK: NIXL 3-step API called in order')
"
```

- **Checks:** `get_xfer_descs`, `initialize_xfer`, `transfer`, and `check_xfer_state` are all invoked, in that order.
- **Expected:** prints `OK: NIXL 3-step API called in order`.

### Test 5b — `"ERR"` state raises `RuntimeError` (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    RemoteWeightInfo,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as p2p_mod

agent = MagicMock()
agent.check_xfer_state.return_value = 'ERR'

obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj._nixl_agent = agent
obj._weight_memory_registry = {'layer.weight': (0xCAFE0000, 64, 2)}
remote = RemoteWeightInfo(
    session_id='sglang_worker_0',
    weights_info={'layer.weight': (0xDEAD0000, 64, 2, 1)},
    agent_name='sglang_worker_0',
    backend='nixl',
)
try:
    obj._do_p2p_write_one_session(remote, ['layer.weight'])
    raise AssertionError('expected RuntimeError')
except RuntimeError as e:
    assert 'layer.weight' in str(e) or 'NIXL' in str(e)
    print('OK: ERR state raises RuntimeError')
"
```

- **Checks:** `check_xfer_state` returning `"ERR"` raises `RuntimeError` with the parameter name.
- **Expected:** prints `OK: ERR state raises RuntimeError`.

### Test 5c — transfer failure does not repeat registration (unit, no GPU)

```bash
python -c "
from unittest.mock import MagicMock, patch
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    RemoteWeightInfo,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as p2p_mod

agent = MagicMock()
agent.check_xfer_state.return_value = 'ERR'

obj = object.__new__(p2p_mod.UpdateWeightP2P)
obj._nixl_agent = agent
obj._model_registered = True
obj._weight_memory_registry = {'w': (0xCAFE0000, 64, 2)}
remote = RemoteWeightInfo(
    session_id='peer',
    weights_info={'w': (0xDEAD0000, 64, 2, 0)},
    agent_name='peer',
    backend='nixl',
)

with patch.object(p2p_mod, 'register_cpu_memory_nixl') as register:
    try:
        obj._do_p2p_write_one_session(remote, ['w'])
    except RuntimeError:
        pass

assert obj._model_registered
register.assert_not_called()
print('OK: transfer failure leaves one-time registration intact')
"
```

- **Checks:** registration is not part of the write/error path and remains valid after a failed
  transfer.
- **Expected:** prints `OK: transfer failure leaves one-time registration intact`.

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

- **Checks:** Miles launches the real SGLang NIXL seed; its connection phase passes mandatory Test
  2d by parsing live tagged metadata and constructing real `ServerArgs`; both weight-update
  iterations complete, `--check-weight-update-equal` reports no mismatches, and no NIXL `ERR` state
  is logged.
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
| 2b | 2 | unit | no | no | Mooncake tagged dict still parsed |
| 2c | 2 | static | no | no | new fields with correct defaults |
| 2d | 2 | e2e (mandatory in 5e) | yes | yes | Miles-launched SGLang metadata parses with real `ServerArgs` |
| 3a | 3 | unit | no | yes | `create_nixl_agent` returns agent |
| 3b | 3 | static | no | no | both branches in `p2p.py` |
| 3c | 3 | e2e | yes | yes | `add_remote_agent` succeeds, no crash |
| 4a | 4 | static | no | no | registration helper importable |
| 4b | 4 | unit | no | no (mocked) | prepare registers only once |
| 4c | 4 | unit | no | yes | non-pinned raises `AssertionError` |
| 5a | 5 | unit | no | no (mocked) | 3-step API called in order |
| 5b | 5 | unit | no | no (mocked) | `"ERR"` raises `RuntimeError` |
| 5c | 5 | unit | no | no (mocked) | failure leaves registration intact |
| 5d | 5 | unit | no | no | Mooncake suite passes unchanged |
| 5e | 5 | e2e | yes | yes | `--check-weight-update-equal` passes |

No-GPU tests are listed as individual commands above. Tests 3a and 4c require NIXL but no GPU;
tests 2d, 3c, and 5e require a container with GPUs and NIXL/UCX installed.
