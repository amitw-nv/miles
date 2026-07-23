#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 3 tests.
# See test.md § "Step 3 — NIXL agent init and peer connection".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_3.sh
#
# Optional live-test configuration:
#   SGLANG_URL=http://localhost:30000
#   SGLANG_RANK=0
#   REQUIRE_LIVE_NIXL=1  # fail instead of skip when NIXL or the seed is unavailable

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
SKIP=0
declare -a RESULTS

SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
SGLANG_RANK="${SGLANG_RANK:-0}"
REQUIRE_LIVE_NIXL="${REQUIRE_LIVE_NIXL:-0}"
export SGLANG_URL SGLANG_RANK

green()  { printf '\033[32m%s\033[0m' "$1"; }
red()    { printf '\033[31m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }

record_skip() {
    local id="$1" desc="$2" reason="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    echo "  -> $(yellow SKIP): $reason"
    SKIP=$((SKIP + 1))
    RESULTS+=("SKIP  $id  $desc ($reason)")
}

record_missing_prerequisite() {
    local id="$1" desc="$2" reason="$3"
    if [ "${REQUIRE_LIVE_NIXL}" = "1" ]; then
        echo "----------------------------------------------------------------"
        echo "[$id] $desc"
        echo "  -> $(red FAIL): $reason"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  $id  $desc ($reason)")
    else
        record_skip "$id" "$desc" "$reason"
    fi
}

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

has_nixl() {
    python -c "import nixl" >/dev/null 2>&1
}

has_live_seed() {
    python -c '
import os
import urllib.request

url = (
    os.environ["SGLANG_URL"].rstrip("/")
    + "/remote_instance_transfer_engine_info?rank="
    + os.environ["SGLANG_RANK"]
)
with urllib.request.urlopen(url, timeout=3) as response:
    assert response.status == 200
' >/dev/null 2>&1
}

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 3"
echo "repo=${REPO_ROOT}"
echo "SGLANG_URL=${SGLANG_URL}  SGLANG_RANK=${SGLANG_RANK}"
echo "================================================================"

# 3a — real NIXL agent construction
if has_nixl; then
    run_py "3a" "create_nixl_agent returns a NIXL agent" '
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    create_nixl_agent,
)

agent = create_nixl_agent()
assert agent is not None
print("OK: create_nixl_agent returns agent")
'
else
    record_missing_prerequisite "3a" "create_nixl_agent returns a NIXL agent" \
        "Python package nixl is not installed"
fi

# 3b — dependency-free structural check after the parallel-query-helper refactor
run_py "3b" "connect_rollout_engines branches with parallel query helpers" '
import ast
from pathlib import Path

p2p_path = Path(
    "miles/backends/megatron_utils/update_weight/"
    "update_weight_from_distributed/p2p.py"
)
utils_path = Path(
    "miles/backends/megatron_utils/update_weight/"
    "update_weight_from_distributed/p2p_transfer_utils.py"
)

p2p_src = p2p_path.read_text()
p2p_tree = ast.parse(p2p_src)
update_weight_p2p = next(
    node
    for node in p2p_tree.body
    if isinstance(node, ast.ClassDef) and node.name == "UpdateWeightP2P"
)
connect = next(
    node
    for node in update_weight_p2p.body
    if isinstance(node, ast.FunctionDef) and node.name == "connect_rollout_engines"
)
connect_calls = {
    node.func.id
    for node in ast.walk(connect)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
assert {
    "create_nixl_agent",
    "query_remote_weight_infos_nixl",
    "query_remote_weight_infos",
    "create_transfer_engine",
} <= connect_calls

utils_src = utils_path.read_text()
utils_tree = ast.parse(utils_src)
nixl_query = next(
    node
    for node in utils_tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "query_remote_weight_infos_nixl"
)
nixl_query_calls = {
    node.func.id
    for node in ast.walk(nixl_query)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
}
assert "add_nixl_remote_agent" in nixl_query_calls
assert any(isinstance(node, ast.For) for node in ast.walk(nixl_query))
print("OK: backend branches and parallel NIXL query helper present")
'

# 3c — real handshake against a running SGLang NIXL seed
if ! has_nixl; then
    record_missing_prerequisite "3c" "handshake with live SGLang NIXL seed" \
        "Python package nixl is not installed"
elif ! has_live_seed; then
    record_missing_prerequisite "3c" "handshake with live SGLang NIXL seed" \
        "no seed metadata endpoint at ${SGLANG_URL} for rank ${SGLANG_RANK}"
else
    run_py "3c" "handshake with live SGLang NIXL seed" '
import json
import os
import urllib.request

from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.p2p_transfer_utils import (
    add_nixl_remote_agent,
    create_nixl_agent,
)

url = (
    os.environ["SGLANG_URL"].rstrip("/")
    + "/remote_instance_transfer_engine_info?rank="
    + os.environ["SGLANG_RANK"]
)
with urllib.request.urlopen(url, timeout=5) as response:
    info = json.load(response)["remote_instance_transfer_engine_info"]

assert info["backend"] == "nixl", info.get("backend")
agent_name = info["agent_name"]
assert agent_name
assert info["agent_metadata"]

agent = create_nixl_agent()
add_nixl_remote_agent(agent, info["agent_metadata"])
print(f"OK: add_remote_agent succeeded for {agent_name}")
'
fi

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
