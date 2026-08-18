#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 1 tests.
# See test.md § "Step 1 — Backend flag, mode, and endpoint fix".
#
# Usage (from the Miles repo root):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_1.sh
#
# Notes vs test.md:
# - 1c expects default --mode p2p (Mooncake), not mooncake. User-facing modes are
#   {p2p,nixl,broadcast}; nixl maps internally to P2P + NIXL backend.
# - 1b checks the Miles-prefixed SGLang seed flag
#   --sglang-remote-instance-weight-loader-start-seed-via-nixl.
# - 1a falls back to inspecting arguments.py if train.py cannot import sglang.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
declare -a RESULTS

green() { printf '\033[32m%s\033[0m' "$1"; }
red()   { printf '\033[31m%s\033[0m' "$1"; }

run_py() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    if python -c "$code"; then
        echo "  -> $(green PASS)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $id  $desc")
    fi
}

run_shell() {
    local id="$1" desc="$2"
    shift 2
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    if "$@"; then
        echo "  -> $(green PASS)"
        PASS=$((PASS + 1))
        RESULTS+=("PASS  $id  $desc")
    else
        echo "  -> $(red FAIL)"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $id  $desc")
    fi
}

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 1"
echo "repo=${REPO_ROOT}"
echo "================================================================"

# 1a — --update-weight-transfer-backend appears in training --help
run_shell "1a" "--update-weight-transfer-backend appears in --help" \
    bash -c '
set -euo pipefail
if out=$(python train.py --help 2>/dev/null); then
  echo "$out" | grep -- --update-weight-transfer-backend | grep -E "mooncake|nixl"
else
  echo "train.py --help unavailable (likely missing sglang); checking arguments.py"
  python - <<'"'"'PY'"'"'
from pathlib import Path
src = Path("miles/utils/arguments.py").read_text()
assert "--update-weight-transfer-backend" in src
assert "mooncake" in src and "nixl" in src
print("OK: flag present in miles/utils/arguments.py")
PY
fi
'

# 1b — --mode nixl emits Miles backend + SGLang NIXL seed flags
run_py "1b" "--mode nixl emits Miles and SGLang NIXL flags" '
import inspect
import examples.p2p_weight_transfer.run as r

src = inspect.getsource(r)
assert "--update-weight-transfer-backend" in src and "nixl" in src
assert "--sglang-remote-instance-weight-loader-start-seed-via-nixl" in src
# Miles-prefixed form still contains the SGLang flag name as a substring after the prefix.
assert "remote-instance-weight-loader-start-seed-via-nixl" in src
assert "transfer_mode = \"p2p\" if mode == \"nixl\" else mode" in src
print("OK: --mode nixl emits both flags")
'

# 1c — p2p remains the default user-facing mode (maps to Mooncake)
run_shell "1c" "--mode defaults to p2p (Mooncake)" \
    bash -c '
set -euo pipefail
out=$(python examples/p2p_weight_transfer/run.py run --help)
echo "$out" | grep -A2 -- --mode
echo "$out" | grep -q -- "--mode {p2p,nixl,broadcast}"
echo "$out" | grep -q "default: p2p"
! echo "$out" | grep -qi "default: nixl"
! echo "$out" | grep -qi "default: mooncake"
'

# 1d — endpoint URL is the current SGLang route, stale URL removed
run_py "1d" "endpoint URL is /remote_instance_transfer_engine_info" '
try:
    import inspect
    from miles.backends.sglang_utils import sglang_engine

    src = inspect.getsource(sglang_engine)
except ModuleNotFoundError as e:
    print(f"import unavailable ({e}); reading sglang_engine.py source")
    from pathlib import Path

    src = Path("miles/backends/sglang_utils/sglang_engine.py").read_text()

assert "/remote_instance_transfer_engine_info" in src
assert "/get_remote_instance_transfer_engine_info" not in src, "stale URL still present"
print("OK: endpoint URL correct")
'

echo "================================================================"
echo "Summary"
echo "================================================================"
for line in "${RESULTS[@]}"; do
    echo "  $line"
done
echo
echo "PASS=${PASS}  FAIL=${FAIL}"
if [ "${FAIL}" -ne 0 ]; then
    exit 1
fi
exit 0
