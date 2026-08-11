#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 4 tests.
# See test.md § "Step 4 — CPU DRAM memory registration".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_4.sh
#
# Runs all Step 4 tests:
#   4a — register_cpu_memory_nixl is importable from p2p_transfer_utils (needs p2p_transfer_utils)
#   4b — _pause_and_prepare_engines registers NIXL memory only once (needs importable p2p.py)
#   4c — real NIXL agent registration, pinned and non-pinned (needs NIXL + torch;
#        the pinned half additionally needs a CUDA driver)
#   4d — DRAM registration args and returned source metadata (fake agent + tensors)
#   4e — _pause_and_prepare_engines branch structure (always, AST only)
#
# Inside the container every dependency is real and nothing is stubbed. Outside it, a heavy
# third-party dep (torch, ray, mooncake, sglang, megatron, ...) is stubbed only once its absence
# has actually broken the import of the module under test, so third-party probes for optional
# modules keep failing as their authors intended.
# Tests that need a real dependency are reported as SKIP, never as PASS.
#
# Note: a full Miles `run.py --mode nixl` weight-update E2E still needs Step 5.
# This script only validates Step 4 (one-time CPU DRAM registration).

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
SKIP=0
declare -a RESULTS

green()  { printf '\033[32m%s\033[0m' "$1"; }
red()    { printf '\033[31m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }

# Shared preamble: stub missing third-party deps, then expose
#   _ptu  — the p2p_transfer_utils module under test
#   _p2p  — the p2p module under test, or None when its imports cannot be satisfied
IMPORT_PREAMBLE='
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from pathlib import Path
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

_UTILS_PATH = Path(
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py"
)
_P2P_PATH = Path(
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py"
)


def _stub_module(name):
    module = MagicMock(name=name)
    module.__name__ = name
    module.__path__ = []
    sys.modules[name] = module
    return module


def _load_standalone(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _import_with_stubs(do_import, attempts=40):
    """Import, stubbing whichever missing third-party module blocks progress, and retrying.

    Inside the container every dependency is installed, so the first attempt succeeds and no
    stub is ever created. Only modules whose absence propagates out of our own import chain get
    stubbed, which leaves third-party optional-dependency probes untouched.
    """
    while True:
        try:
            return do_import()
        except ModuleNotFoundError as e:
            root = (e.name or "").split(".")[0]
            if not root or root in _NEVER_STUB or root in _stub_roots or attempts <= 0:
                raise
            _stub_roots.add(root)
            attempts -= 1


def _import_utils():
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
        p2p_transfer_utils,
    )

    return p2p_transfer_utils


def _import_p2p():
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p

    return p2p


_ptu = None
_p2p = None
_UTILS_IMPORT_ERROR = None
_P2P_IMPORT_ERROR = None

try:
    _ptu = _import_with_stubs(_import_utils)
except Exception as _utils_error:
    _UTILS_IMPORT_ERROR = _utils_error
    try:
        # Bypass package __init__ side effects by loading the file as a standalone module.
        for _name in ("miles.backends.training_utils", "miles.backends.training_utils.parallel"):
            if _name not in sys.modules:
                _stub_module(_name)
        _ptu = _import_with_stubs(lambda: _load_standalone(_UTILS_PATH, "p2p_transfer_utils_under_test"))
        _UTILS_IMPORT_ERROR = None
    except Exception as _standalone_error:
        _UTILS_IMPORT_ERROR = _standalone_error

try:
    _p2p = _import_with_stubs(_import_p2p)
except Exception as _p2p_error:
    _P2P_IMPORT_ERROR = _p2p_error


class FakeTensor:
    """Minimal pinned-CPU tensor stand-in for registration tests without torch."""

    def __init__(self, addr, numel, element_size, pinned=True):
        self._addr = addr
        self._numel = numel
        self._element_size = element_size
        self._pinned = pinned

    def is_pinned(self):
        return self._pinned

    def data_ptr(self):
        return self._addr

    def numel(self):
        return self._numel

    def element_size(self):
        return self._element_size
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

# Run a snippet with the stub preamble available.
run_py() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    python -c "${IMPORT_PREAMBLE}${code}"
    record_result "$id" "$desc" "$?"
}

# Run a snippet without the stub preamble (real dependencies only).
run_py_raw() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    python -c "$code"
    record_result "$id" "$desc" "$?"
}

has_nixl() {
    python -c "import nixl" >/dev/null 2>&1
}

has_torch() {
    python -c "import torch" >/dev/null 2>&1
}

# Print nothing when the module under test is available, otherwise the blocking import error.
module_error() {
    python -c "${IMPORT_PREAMBLE}
module, error = (_ptu, _UTILS_IMPORT_ERROR) if \"$1\" == \"utils\" else (_p2p, _P2P_IMPORT_ERROR)
print(\"\" if module is not None else f\"{type(error).__name__}: {error}\")
" 2>/dev/null | tail -n 1
}

UTILS_ERROR="$(module_error utils)"
P2P_ERROR="$(module_error p2p)"

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 4"
echo "repo=${REPO_ROOT}"
echo "================================================================"

# 4a — the one-time NIXL registration helper is importable
if [ -n "${UTILS_ERROR}" ]; then
    record_skip "4a" "register_cpu_memory_nixl is importable" \
        "p2p_transfer_utils.py cannot be imported here (${UTILS_ERROR})"
else
    run_py "4a" "register_cpu_memory_nixl is importable" '
assert hasattr(_ptu, "register_cpu_memory_nixl"), "register_cpu_memory_nixl missing"
print("OK: NIXL registration helper present")
'
fi

# 4b — prepare registers the shared buffers exactly once
if [ -z "${P2P_ERROR}" ]; then
    run_py "4b" "_pause_and_prepare_engines registers NIXL memory once" '
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

obj = object.__new__(_p2p.UpdateWeightP2P)
obj.transfer_plan = SimpleNamespace(_gathered_dp_rank=0, _rollout_num_gpus=1)
obj.args = SimpleNamespace(update_weight_transfer_backend="nixl")
obj.transfer_backend = "nixl"
obj._model_registered = False
obj._shared_params_dict = {"w": MagicMock()}
obj._nixl_agent = MagicMock()

with patch.object(
    _p2p.DistBucketedWeightUpdateMixin, "_pause_and_prepare_engines"
), patch.object(
    _p2p, "register_cpu_memory_nixl", return_value={"w": (1, 2, 0)}
) as register, patch.object(
    _p2p, "register_cpu_memory"
) as register_mooncake:
    obj._pause_and_prepare_engines()
    obj._pause_and_prepare_engines()

assert register.call_count == 1, register.call_count
register_mooncake.assert_not_called()
assert obj._model_registered
assert obj._weight_memory_registry == {"w": (1, 2, 0)}
print("OK: NIXL memory registered once")
'
else
    record_skip "4b" "_pause_and_prepare_engines registers NIXL memory once" \
        "p2p.py cannot be imported here (${P2P_ERROR})"
fi

# 4c — registration against a real NIXL agent, both preconditions
if ! has_nixl; then
    record_skip "4c" "real NIXL agent registers pinned DRAM and rejects non-pinned" \
        "Python package nixl is not installed"
elif ! has_torch; then
    record_skip "4c" "real NIXL agent registers pinned DRAM and rejects non-pinned" \
        "Python package torch is not installed"
else
    run_py_raw "4c" "real NIXL agent registers pinned DRAM and rejects non-pinned" '
import torch

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    create_nixl_agent,
    register_cpu_memory_nixl,
)

agent = create_nixl_agent()

not_pinned = torch.zeros(1024, dtype=torch.float16)
try:
    register_cpu_memory_nixl(agent, {"layer.weight": not_pinned})
except AssertionError as e:
    assert "pinned" in str(e), str(e)
    print("OK: non-pinned tensor rejected")
else:
    raise RuntimeError("expected AssertionError for a non-pinned tensor")

# Pinning needs a CUDA driver; the real registration path is only exercised when one is present.
if torch.cuda.is_available():
    pinned = torch.zeros(1024, dtype=torch.float16).pin_memory()
    registry = register_cpu_memory_nixl(agent, {"layer.weight": pinned})
    assert registry == {"layer.weight": (pinned.data_ptr(), 1024, 2)}, registry
    print("OK: pinned tensor registered as a DRAM region by a real NIXL agent")
else:
    print("NOTE: no CUDA driver, skipped the pinned registration half of 4c")
'
fi

# 4d — DRAM registration arguments and returned source metadata (no torch/NIXL needed)
if [ -n "${UTILS_ERROR}" ]; then
    record_skip "4d" "register_cpu_memory_nixl registers DRAM regions and returns source metadata" \
        "p2p_transfer_utils.py cannot be imported here (${UTILS_ERROR})"
else
    run_py "4d" "register_cpu_memory_nixl registers DRAM regions and returns source metadata" '
from unittest.mock import MagicMock

agent = MagicMock()
registry = _ptu.register_cpu_memory_nixl(agent, {"layer.weight": FakeTensor(0xCAFE0000, 64, 2)})

agent.register_memory.assert_called_once_with([(0xCAFE0000, 128, 0, "")], "DRAM")
assert registry == {"layer.weight": (0xCAFE0000, 64, 2)}, registry

# Non-pinned memory is rejected before any registration call is issued.
agent = MagicMock()
try:
    _ptu.register_cpu_memory_nixl(agent, {"layer.weight": FakeTensor(0x2000, 8, 4, pinned=False)})
except AssertionError as e:
    assert "pinned" in str(e), str(e)
else:
    raise RuntimeError("expected AssertionError for a non-pinned tensor")
agent.register_memory.assert_not_called()
print("OK: DRAM regions registered with device_id 0 and pinned memory enforced")
'
fi

# 4e — structural check: one-time registration branches on the backend
run_py_raw "4e" "_pause_and_prepare_engines branches on transfer_backend" '
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
prepare = next(
    node
    for node in update_weight_p2p.body
    if isinstance(node, ast.FunctionDef) and node.name == "_pause_and_prepare_engines"
)
calls = {
    node.func.id
    for node in ast.walk(prepare)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
assert {"register_cpu_memory_nixl", "register_cpu_memory"} <= calls, calls
assert "_model_registered" in ast.dump(prepare), "one-time registration guard missing"
assert "transfer_backend" in ast.dump(prepare), "backend branch missing"

# The per-update write path must not register or deregister memory.
write = next(
    node
    for node in update_weight_p2p.body
    if isinstance(node, ast.FunctionDef) and node.name == "_do_p2p_write_one_session"
)
write_dump = ast.dump(write)
assert "register_memory" not in write_dump and "register_cpu_memory" not in write_dump, (
    "registration must stay out of the per-update write path"
)
print("OK: registration branches on backend and stays out of the write path")
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
