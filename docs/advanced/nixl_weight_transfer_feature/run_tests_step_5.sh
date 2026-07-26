#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 5 tests.
# See test.md § "Step 5 — Branch the existing P2P WRITE".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_5.sh
#
# Runs all Step 5 tests:
#   5a — NIXL write calls get_xfer_descs/initialize_xfer/transfer/check_xfer_state in order
#   5b — check_xfer_state returning "ERR" raises RuntimeError
#   5c — a failed transfer leaves the one-time CPU registration intact
#   5d — the Mooncake write path is unchanged (mock transfer engine, 3-field weight entries)
#   5e — small-model E2E with --mode nixl (opt-in: needs GPUs, a prepared checkpoint, NIXL)
#   5f — descriptor contents: DRAM/VRAM types, sizes, device_ids, one transfer per batch
#   5g — write-path branch structure (AST only)
#
# Configuration:
#   RUN_E2E=1                    # opt in to 5e; it is skipped by default
#   E2E_MODEL=GLM-Z1-9B-0414     # model name for 5e, as accepted by run.py list
#
# Inside the container every dependency is real and nothing is stubbed. Outside it, a heavy
# third-party dep (torch, ray, mooncake, sglang, megatron, ...) is stubbed only once its absence
# has actually broken the import of the module under test, so third-party probes for optional
# modules keep failing as their authors intended.
# Tests that need a real dependency are reported as SKIP, never as PASS.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
SKIP=0
declare -a RESULTS

RUN_E2E="${RUN_E2E:-0}"
E2E_MODEL="${E2E_MODEL:-GLM-Z1-9B-0414}"

green()  { printf '\033[32m%s\033[0m' "$1"; }
red()    { printf '\033[31m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }

# Shared preamble: expose
#   _p2p  — the p2p module under test, or None when its imports cannot be satisfied
#   _write_target() — a bare UpdateWeightP2P carrying only what the write path reads
IMPORT_PREAMBLE='
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from unittest.mock import MagicMock

# Roots that must never be stubbed: miles is the code under test, nixl-dependent tests must
# skip rather than run against a mock, and deep_ep is monkey-patched at import time by
# miles.backends.megatron_utils, which needs either the real package or its ImportError path.
_NEVER_STUB = {"miles", "nixl", "deep_ep"}

# Filled in only by _import_with_stubs, one root per import that actually failed. Blanket
# stubbing is not safe: third-party code probes for optional modules (_winapi, simplejson, ...)
# and a stub that answers those probes sends it down a wrong branch.
_stub_roots: set[str] = set()


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve an explicit set of missing third-party modules to MagicMock stand-ins."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _stub_roots:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        module = MagicMock(name=spec.name)
        module.__name__ = spec.name
        module.__spec__ = spec
        module.__loader__ = self
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


sys.meta_path.append(_StubFinder())


def _import_with_stubs(do_import, attempts=40):
    """Import, stubbing whichever missing third-party module blocks progress, and retrying."""
    while True:
        try:
            return do_import()
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if not root or root in _NEVER_STUB or root in _stub_roots or attempts <= 0:
                raise
            _stub_roots.add(root)
            attempts -= 1


def _import_p2p():
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p

    return p2p


_p2p = None
_P2P_IMPORT_ERROR = None

try:
    _p2p = _import_with_stubs(_import_p2p)
except Exception as _p2p_error:
    _P2P_IMPORT_ERROR = _p2p_error


def _write_target(**attrs):
    """An UpdateWeightP2P with only the attributes the write path reads, no __init__."""
    obj = object.__new__(_p2p.UpdateWeightP2P)
    for name, value in attrs.items():
        setattr(obj, name, value)
    return obj


def _nixl_remote(weights_info, agent_name="sglang_worker_0"):
    return _p2p.RemoteWeightInfo(
        session_id=agent_name,
        weights_info=weights_info,
        agent_name=agent_name,
        backend="nixl",
    )
'

record_skip() {
    local id="$1" desc="$2" reason="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    echo "  -> $(yellow SKIP): $reason"
    SKIP=$((SKIP + 1))
    RESULTS+=("SKIP  $id  $desc ($reason)")
}

record_result() {
    local id="$1" desc="$2" status="$3"
    if [ "${status}" -eq 0 ]; then
        echo "  -> $(green PASS)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $id  $desc")
    fi
}

# Run a snippet with the import preamble available.
run_py() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    python -c "${IMPORT_PREAMBLE}${code}"
    record_result "$id" "$desc" "$?"
}

# Run a snippet without the preamble (real dependencies only).
run_py_raw() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    python -c "$code"
    record_result "$id" "$desc" "$?"
}

run_shell() {
    local id="$1" desc="$2"
    shift 2
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    "$@"
    record_result "$id" "$desc" "$?"
}

has_nixl() {
    python -c "import nixl" >/dev/null 2>&1
}

has_gpu() {
    python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1
}

# Print nothing when p2p.py is importable, otherwise the blocking import error.
p2p_error() {
    python -c "${IMPORT_PREAMBLE}
print(\"\" if _p2p is not None else f\"{type(_P2P_IMPORT_ERROR).__name__}: {_P2P_IMPORT_ERROR}\")
" 2>/dev/null | tail -n 1
}

P2P_ERROR="$(p2p_error)"

# Run a mock-based write test, or skip the whole group when p2p.py cannot be imported.
run_write_py() {
    local id="$1" desc="$2" code="$3"
    if [ -n "${P2P_ERROR}" ]; then
        record_skip "$id" "$desc" "p2p.py cannot be imported here (${P2P_ERROR})"
    else
        run_py "$id" "$desc" "$code"
    fi
}

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 5"
echo "repo=${REPO_ROOT}"
echo "e2e=${RUN_E2E} (model=${E2E_MODEL})"
echo "================================================================"

# 5a — the NIXL transfer sequence runs in the documented order
run_write_py "5a" "NIXL write calls the transfer API in order" '
agent = MagicMock()
agent.check_xfer_state.return_value = "DONE"

obj = _write_target(
    _nixl_agent=agent,
    _weight_memory_registry={"layer.weight": (0xCAFE0000, 64, 2)},
)
obj._do_p2p_write_one_session(_nixl_remote({"layer.weight": (0xDEAD0000, 64, 2, 1)}), ["layer.weight"])

names = [str(call) for call in agent.method_calls]
order = ["get_xfer_descs", "initialize_xfer", "transfer", "check_xfer_state"]
positions = [next(i for i, name in enumerate(names) if expected in name) for expected in order]
assert positions == sorted(positions), names
print("OK: NIXL transfer API called in order")
'

# 5b — an ERR state is turned into a RuntimeError
run_write_py "5b" "check_xfer_state ERR raises RuntimeError" '
agent = MagicMock()
agent.check_xfer_state.return_value = "ERR"

obj = _write_target(
    _nixl_agent=agent,
    _weight_memory_registry={"layer.weight": (0xCAFE0000, 64, 2)},
)
try:
    obj._do_p2p_write_one_session(_nixl_remote({"layer.weight": (0xDEAD0000, 64, 2, 1)}), ["layer.weight"])
except RuntimeError as e:
    assert "NIXL" in str(e), str(e)
else:
    raise AssertionError("expected RuntimeError for an ERR transfer state")

# The handle is released even on the error path, so a failure cannot leak transfer handles.
agent.release_xfer_handle.assert_called_once_with(agent.initialize_xfer.return_value)
print("OK: ERR state raises RuntimeError and releases the handle")
'

# 5c — a failed transfer must not disturb the one-time registration
run_write_py "5c" "transfer failure leaves the one-time registration intact" '
from unittest.mock import patch

agent = MagicMock()
agent.check_xfer_state.return_value = "ERR"

obj = _write_target(
    _nixl_agent=agent,
    _model_registered=True,
    _weight_memory_registry={"w": (0xCAFE0000, 64, 2)},
)

with patch.object(_p2p, "register_cpu_memory_nixl") as register, patch.object(
    _p2p, "register_cpu_memory"
) as register_mooncake:
    try:
        obj._do_p2p_write_one_session(_nixl_remote({"w": (0xDEAD0000, 64, 2, 0)}), ["w"])
    except RuntimeError:
        pass

assert obj._model_registered
assert obj._weight_memory_registry == {"w": (0xCAFE0000, 64, 2)}
register.assert_not_called()
register_mooncake.assert_not_called()
agent.deregister_memory.assert_not_called()
print("OK: transfer failure leaves one-time registration intact")
'

# 5d — the Mooncake write path still behaves exactly as before
run_write_py "5d" "Mooncake write path unchanged" '
engine = MagicMock()
engine.batch_transfer_sync_write.return_value = 0
agent = MagicMock()

obj = _write_target(
    _transfer_engine=engine,
    _nixl_agent=agent,
    _weight_memory_registry={"layer.weight": (0xCAFE0000, 64, 2)},
)
# Mooncake weight entries have three fields and the dataclass defaults to backend="mooncake".
remote = _p2p.RemoteWeightInfo(
    session_id="10.0.0.7:18000",
    weights_info={"layer.weight": (0xDEAD0000, 64, 2)},
)
obj._do_p2p_write_one_session(remote, ["layer.weight"])

engine.batch_transfer_sync_write.assert_called_once_with(
    "10.0.0.7:18000", [0xCAFE0000], [0xDEAD0000], [128]
)
assert not agent.method_calls, agent.method_calls

engine.batch_transfer_sync_write.return_value = -1
try:
    obj._do_p2p_write_one_session(remote, ["layer.weight"])
except RuntimeError as e:
    assert "10.0.0.7:18000" in str(e), str(e)
else:
    raise AssertionError("expected RuntimeError for a negative Mooncake return code")
print("OK: Mooncake write path unchanged and NIXL agent untouched")
'

# 5e — small-model E2E through run.py (opt-in; needs GPUs + a prepared checkpoint + NIXL)
if [ "${RUN_E2E}" != "1" ]; then
    record_skip "5e" "small-model E2E with --mode nixl" \
        "opt-in only; run RUN_E2E=1 E2E_MODEL=${E2E_MODEL} $0"
elif ! has_nixl; then
    record_skip "5e" "small-model E2E with --mode nixl" "Python package nixl is not installed"
elif ! has_gpu; then
    record_skip "5e" "small-model E2E with --mode nixl" "no CUDA device available"
else
    # run.py already appends --check-weight-update-equal unless SKIP_VALIDATION=1.
    run_shell "5e" "small-model E2E with --mode nixl" \
        python examples/p2p_weight_transfer/run.py run "${E2E_MODEL}" --mode nixl
fi

# 5f — the descriptors handed to NIXL, and one transfer per batch
run_write_py "5f" "descriptors carry sizes, device_ids, and batch in one transfer" '
from unittest.mock import call

agent = MagicMock()
agent.check_xfer_state.return_value = "DONE"

obj = _write_target(
    _nixl_agent=agent,
    _weight_memory_registry={"a": (0x1000, 4, 4), "b": (0x2000, 8, 2)},
)
remote = _nixl_remote({"a": (0xA000, 4, 4, 3), "b": (0xB000, 8, 2, 3)})
obj._do_p2p_write_one_session(remote, ["a", "b"])

# size is numel * element_size; sources are DRAM on device 0, targets are VRAM on the
# device_id published by SGLang. Both parameters travel in a single transfer request.
assert agent.get_xfer_descs.call_args_list == [
    call([(0x1000, 16, 0), (0x2000, 16, 0)], "DRAM"),
    call([(0xA000, 16, 3), (0xB000, 16, 3)], "VRAM"),
], agent.get_xfer_descs.call_args_list

operation, source_descs, target_descs, remote_agent = agent.initialize_xfer.call_args.args
assert operation == "WRITE", operation
assert remote_agent == "sglang_worker_0", remote_agent
assert source_descs is agent.get_xfer_descs.return_value
assert target_descs is agent.get_xfer_descs.return_value
assert agent.initialize_xfer.call_count == 1
agent.transfer.assert_called_once_with(agent.initialize_xfer.return_value)
agent.release_xfer_handle.assert_called_once_with(agent.initialize_xfer.return_value)
print("OK: descriptors correct and the batch travels in one transfer")
'

# 5g — structural check: the write branches on the remote backend, both paths intact
run_py_raw "5g" "_do_p2p_write_one_session branches on the remote backend" '
import ast
from pathlib import Path

p2p_path = Path(
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py"
)
tree = ast.parse(p2p_path.read_text())
update_weight_p2p = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "UpdateWeightP2P"
)
write = next(
    node
    for node in update_weight_p2p.body
    if isinstance(node, ast.FunctionDef) and node.name == "_do_p2p_write_one_session"
)
write_dump = ast.dump(write)
assert "batch_transfer_sync_write" in write_dump, "Mooncake write call is gone"
assert "backend" in write_dump, "write path does not branch on the remote backend"

# The NIXL sequence lives in the class, and registration stays out of the write path.
class_calls = {
    node.func.attr
    for node in ast.walk(update_weight_p2p)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
}
assert {"get_xfer_descs", "initialize_xfer", "transfer", "check_xfer_state"} <= class_calls, class_calls
assert "register_memory" not in write_dump and "register_cpu_memory" not in write_dump, (
    "registration must stay out of the per-update write path"
)
print("OK: write path branches on backend, both backends intact")
'

echo "================================================================"
echo "Summary"
echo "================================================================"
for line in "${RESULTS[@]}"; do
    echo "  $line"
done
echo
echo "PASS=${PASS}  FAIL=${FAIL}  SKIP=${SKIP}"
if [ "${FAIL}" -ne 0 ]; then
    exit 1
fi
exit 0
