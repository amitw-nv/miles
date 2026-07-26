#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 4 tests.
# See test.md § "Step 4 — CPU DRAM memory registration".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_4.sh
#
# Runs all Step 4 tests:
#   4a — register_cpu_memory_nixl is importable from p2p_transfer_utils (always)
#   4b — _pause_and_prepare_engines registers NIXL memory only once (needs importable p2p.py)
#   4c — real NIXL agent registration, pinned and non-pinned (needs NIXL + torch;
#        the pinned half additionally needs a CUDA driver)
#   4d — DRAM registration args and returned source metadata (always, fake agent + tensors)
#   4e — _pause_and_prepare_engines branch structure (always, AST only)
#
# Heavy third-party deps (torch, ray, mooncake, sglang, megatron, ...) are stubbed when
# missing so the schema/registration logic can still be exercised outside the container.
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

# miles must always resolve for real; nixl-dependent tests must skip rather than run
# against a mock; deep_ep is monkey-patched at import time by miles.backends.megatron_utils,
# which only works on the real package or its ImportError fallback.
_NEVER_STUB = {"miles", "nixl", "deep_ep"}


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Resolve otherwise-missing third-party modules to MagicMock stand-ins.

    Appended to sys.meta_path, so it is consulted only for modules no real finder
    could locate: inside the container every dependency imports normally.
    """

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _NEVER_STUB:
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

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


try:
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
        p2p_transfer_utils as _ptu,
    )
except Exception:
    # Bypass package __init__ side effects by loading the file as a standalone module.
    for _name in ("miles.backends.training_utils", "miles.backends.training_utils.parallel"):
        if _name not in sys.modules:
            _stub_module(_name)
    _ptu = _load_standalone(_UTILS_PATH, "p2p_transfer_utils_under_test")

try:
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p as _p2p
except Exception as _p2p_error:
    _p2p = None
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

has_p2p_module() {
    python -c "${IMPORT_PREAMBLE}
import sys
sys.exit(0 if _p2p is not None else 1)
" >/dev/null 2>&1
}

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 4"
echo "repo=${REPO_ROOT}"
echo "================================================================"

# 4a — the one-time NIXL registration helper is importable
run_py "4a" "register_cpu_memory_nixl is importable" '
assert hasattr(_ptu, "register_cpu_memory_nixl"), "register_cpu_memory_nixl missing"
print("OK: NIXL registration helper present")
'

# 4b — prepare registers the shared buffers exactly once
if has_p2p_module; then
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
        "p2p.py imports cannot be satisfied in this environment"
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
