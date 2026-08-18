#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 3 tests.
# See test.md § "Step 3 — NIXL agent init and peer connection".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_3.sh
#
# Runs all Step 3 tests:
#   3a — create_nixl_agent() (needs NIXL)
#   3b — connect_rollout_engines branch structure (always)
#   3c — live handshake against a SGLang NIXL seed (needs NIXL + GPU + model)
#
# For 3c the script reuses an existing seed at SGLANG_URL if one is already up;
# otherwise it launches SGLang with --remote-instance-weight-loader-start-seed-via-nixl,
# runs the handshake, and tears the seed down.
#
# Configuration:
#   MODEL_PATH=/path/to/model   # optional; otherwise auto-discover /root/models or download
#   DEFAULT_MODEL_NAME=GLM-Z1-9B-0414  # used when auto-download is needed
#   TP=1
#   PORT=30000
#   SGLANG_URL=http://localhost:30000
#   SGLANG_RANK=0
#   READY_TIMEOUT=1200
#
# If no HF model is present, 3c downloads DEFAULT_MODEL_NAME with:
#   python examples/p2p_weight_transfer/run.py prepare <name> --download-only
# (same prepare path used by examples/p2p_weight_transfer/GLM-Z1-9B.sh).
#
# Note: a full Miles `run.py --mode nixl` weight-update E2E still needs Steps 4-5.
# This script only validates Step 3 (agent init + peer handshake).

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
MODEL_PATH="${MODEL_PATH:-}"
DEFAULT_MODEL_NAME="${DEFAULT_MODEL_NAME:-GLM-Z1-9B-0414}"
TP="${TP:-1}"
PORT="${PORT:-30000}"
READY_TIMEOUT="${READY_TIMEOUT:-1200}"
export SGLANG_URL SGLANG_RANK

LAUNCHED_SGLANG_PID=""
LAUNCHED_SGLANG_LOG=""

green()  { printf '\033[32m%s\033[0m' "$1"; }
red()    { printf '\033[31m%s\033[0m' "$1"; }
yellow() { printf '\033[33m%s\033[0m' "$1"; }

cleanup_launched_sglang() {
    if [ -n "${LAUNCHED_SGLANG_PID}" ]; then
        kill "${LAUNCHED_SGLANG_PID}" 2>/dev/null || true
        wait "${LAUNCHED_SGLANG_PID}" 2>/dev/null || true
        LAUNCHED_SGLANG_PID=""
    fi
}
trap cleanup_launched_sglang EXIT

record_skip() {
    local id="$1" desc="$2" reason="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    echo "  -> $(yellow SKIP): $reason"
    SKIP=$((SKIP + 1))
    RESULTS+=("SKIP  $id  $desc ($reason)")
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

resolve_model_path() {
    if [ -n "${MODEL_PATH}" ]; then
        if [ -d "${MODEL_PATH}" ] && [ -f "${MODEL_PATH}/config.json" ]; then
            return 0
        fi
        echo "  MODEL_PATH=${MODEL_PATH} is missing or has no config.json"
        return 1
    fi

    local candidate
    for candidate in \
        "/root/models/GLM-Z1-9B-0414" \
        "/root/models/Qwen3-4B" \
        "/root/models/Qwen/Qwen2.5-0.5B" \
        "/root/models/Qwen2.5-0.5B" \
        "/sgl-workspace/llm_models/DeepSeek-V3-Lite/bf16"
    do
        if [ -d "${candidate}" ] && [ -f "${candidate}/config.json" ]; then
            MODEL_PATH="${candidate}"
            return 0
        fi
    done

    # Fall back to the first Hugging Face-like directory under /root/models.
    if [ -d "/root/models" ]; then
        candidate="$(
            find /root/models -mindepth 1 -maxdepth 3 -type f -name config.json 2>/dev/null \
                | head -n 1 \
                | xargs -r dirname
        )"
        if [ -n "${candidate}" ] && [ -d "${candidate}" ]; then
            MODEL_PATH="${candidate}"
            return 0
        fi
    fi

    return 1
}

download_default_model() {
    local model_name="${DEFAULT_MODEL_NAME:-GLM-Z1-9B-0414}"
    local model_dir="/root/models/${model_name}"

    echo "  No local HF model found; downloading ${model_name} (--download-only)"
    mkdir -p /root/models
    if ! python examples/p2p_weight_transfer/run.py prepare "${model_name}" --download-only; then
        echo "  -> $(red FAIL): prepare ${model_name} --download-only failed"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  launch  prepare ${model_name} --download-only failed")
        return 1
    fi

    if [ ! -f "${model_dir}/config.json" ]; then
        echo "  -> $(red FAIL): expected ${model_dir}/config.json after prepare"
        FAIL=$((FAIL + 1))
        RESULTS+=("FAIL  launch  missing ${model_dir}/config.json after prepare")
        return 1
    fi

    MODEL_PATH="${model_dir}"
    echo "  downloaded MODEL_PATH=${MODEL_PATH}"
    return 0
}

ensure_model_path() {
    if resolve_model_path; then
        echo "  using MODEL_PATH=${MODEL_PATH}"
        return 0
    fi
    download_default_model
}

launch_nixl_seed() {
    if ! ensure_model_path; then
        return 1
    fi

    SGLANG_URL="http://localhost:${PORT}"
    export SGLANG_URL

    LAUNCHED_SGLANG_LOG="$(mktemp)"
    echo "  launching SGLang NIXL seed (model=${MODEL_PATH}, tp=${TP}, port=${PORT})"
    echo "  log: ${LAUNCHED_SGLANG_LOG}"
    python -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --tp "${TP}" \
        --trust-remote-code \
        --port "${PORT}" \
        --remote-instance-weight-loader-start-seed-via-nixl \
        >"${LAUNCHED_SGLANG_LOG}" 2>&1 &
    LAUNCHED_SGLANG_PID=$!

    local waited=0
    while [ "${waited}" -lt "${READY_TIMEOUT}" ]; do
        if ! kill -0 "${LAUNCHED_SGLANG_PID}" 2>/dev/null; then
            echo "  -> $(red FAIL): SGLang exited before becoming ready"
            tail -n 40 "${LAUNCHED_SGLANG_LOG}" || true
            FAIL=$((FAIL + 1))
            RESULTS+=("FAIL  launch  SGLang exited early")
            return 1
        fi
        if grep -q "ready to roll" "${LAUNCHED_SGLANG_LOG}"; then
            if has_live_seed; then
                echo "  -> $(green PASS): SGLang NIXL seed ready at ${SGLANG_URL}"
                return 0
            fi
        fi
        if grep -qE "OutOfMemoryError|Scheduler hit an exception|Received sigquit" "${LAUNCHED_SGLANG_LOG}"; then
            echo "  -> $(red FAIL): detected crash while launching SGLang"
            tail -n 40 "${LAUNCHED_SGLANG_LOG}" || true
            FAIL=$((FAIL + 1))
            RESULTS+=("FAIL  launch  SGLang crashed during startup")
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    echo "  -> $(red FAIL): timed out waiting for SGLang NIXL seed"
    tail -n 40 "${LAUNCHED_SGLANG_LOG}" || true
    FAIL=$((FAIL + 1))
    RESULTS+=("FAIL  launch  timed out waiting for seed")
    return 1
}

ensure_live_seed() {
    if has_live_seed; then
        echo "Using existing SGLang seed at ${SGLANG_URL} (rank ${SGLANG_RANK})"
        return 0
    fi
    echo "No live seed found; launching SGLang NIXL seed for 3c"
    launch_nixl_seed
}

echo "================================================================"
echo "Miles NIXL weight-transfer tests: Step 3"
echo "repo=${REPO_ROOT}"
echo "sglang_url=${SGLANG_URL}  rank=${SGLANG_RANK}"
echo "model_path=${MODEL_PATH:-<auto>}  tp=${TP}  port=${PORT}"
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
    record_skip "3a" "create_nixl_agent returns a NIXL agent" \
        "Python package nixl is not installed"
fi

# 3b — dependency-free structural check after the parallel-query-helper refactor
run_py "3b" "connect_rollout_engines branches with parallel query helpers" '
import ast
from pathlib import Path

p2p_path = Path(
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py"
)
utils_path = Path(
    "miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py"
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

# 3c — real handshake against a SGLang NIXL seed (launch one if needed)
if ! has_nixl; then
    record_skip "3c" "handshake with live SGLang NIXL seed" \
        "Python package nixl is not installed"
elif ensure_live_seed; then
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
else
    # ensure_live_seed already recorded a FAIL for the launch path
    :
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
