#!/usr/bin/env bash
# Runner for Miles NIXL weight-transfer Step 2 tests.
# See test.md § "Step 2 — Schema consumer: RemoteWeightInfo + query function".
#
# Usage (from the Miles repo root, or inside the cluster container at /root/miles):
#   ./docs/advanced/nixl_weight_transfer_feature/run_tests_step_2.sh
#
# If mooncake/sglang are not installed, the runner stubs those imports so the
# unit tests can still exercise the schema parser.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

PASS=0
FAIL=0
declare -a RESULTS

green() { printf '\033[32m%s\033[0m' "$1"; }
red()   { printf '\033[31m%s\033[0m' "$1"; }

# Shared preamble: stub heavy deps when missing, then import the module under test.
IMPORT_PREAMBLE='
import sys
from unittest.mock import MagicMock

def _stub_if_missing(modname):
    try:
        __import__(modname)
    except ModuleNotFoundError:
        sys.modules[modname] = MagicMock()

for _m in (
    "torch",
    "torch.distributed",
    "ray",
    "ray.actor",
    "mooncake",
    "mooncake.engine",
    "sglang",
    "sglang.srt",
    "sglang.srt.server_args",
    "miles.backends.training_utils",
    "miles.backends.training_utils.parallel",
):
    _stub_if_missing(_m)

if "sglang.srt.server_args" in sys.modules and isinstance(sys.modules["sglang.srt.server_args"], MagicMock):
    import dataclasses as _dc
    @_dc.dataclass
    class ServerArgs:
        pass
    sys.modules["sglang.srt.server_args"].ServerArgs = ServerArgs

# Bypass package __init__ side effects by loading the file as a standalone module
# when the normal package import cannot resolve heavy deps.
try:
    from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import p2p_transfer_utils as _ptu
except Exception:
    import importlib.util
    from pathlib import Path
    path = Path("miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p_transfer_utils.py")
    spec = importlib.util.spec_from_file_location("p2p_transfer_utils_under_test", path)
    _ptu = importlib.util.module_from_spec(spec)
    sys.modules["p2p_transfer_utils_under_test"] = _ptu
    spec.loader.exec_module(_ptu)
'

run_py() {
    local id="$1" desc="$2" code="$3"
    echo "----------------------------------------------------------------"
    echo "[$id] $desc"
    if python -c "${IMPORT_PREAMBLE}${code}"; then
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
echo "Miles NIXL weight-transfer tests: Step 2"
echo "repo=${REPO_ROOT}"
echo "================================================================"

# 2a — NIXL dict format
run_py "2a" "query_remote_weight_infos_nixl parses NIXL dict format" '
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

query_remote_weight_infos_nixl = _ptu.query_remote_weight_infos_nixl

response = {
    "backend": "nixl",
    "agent_name": "sglang_worker_0",
    "agent_metadata": "dGVzdA==",
    "weights_info_dict": {
        "model.embed_tokens.weight": [140000000000, 131072, 2, 0]
    },
}
engine = MagicMock()
target = SimpleNamespace(engine_ind=0, engine_rank=0)
nixl_agent = MagicMock()

with patch.object(
    _ptu.ray,
    "get",
    side_effect=[response, {"tp_rank": 0}, {"model_path": "dummy"}],
):
    remote_by_id, target_to_id, _ = query_remote_weight_infos_nixl([engine], [target], nixl_agent)

remote_id = target_to_id[(0, 0)]
assert remote_id == "sglang_worker_0"
weights_info = remote_by_id[remote_id][0]
assert weights_info["model.embed_tokens.weight"] == (140000000000, 131072, 2, 0)
assert nixl_agent.add_remote_agent.called
print("OK: NIXL dict format parsed")
'

# 2b — Mooncake tagged dict format (regression)
run_py "2b" "query_remote_weight_infos still parses Mooncake format" '
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

query_remote_weight_infos = _ptu.query_remote_weight_infos

response = {
    "backend": "mooncake",
    "session_id": "10.0.0.7:18000",
    "weights_info_dict": {
        "model.embed_tokens.weight": [140000000000, 131072, 2]
    },
}
engine = MagicMock()
target = SimpleNamespace(engine_ind=0, engine_rank=0)

with patch.object(
    _ptu.ray,
    "get",
    side_effect=[response, {"tp_rank": 0}, {"model_path": "dummy"}],
):
    remote_by_id, target_to_id, _ = query_remote_weight_infos([engine], [target])

session_id = target_to_id[(0, 0)]
assert session_id == "10.0.0.7:18000"
assert remote_by_id[session_id][0]["model.embed_tokens.weight"] == (140000000000, 131072, 2)
print("OK: Mooncake format still parsed")
'

# 2c — RemoteWeightInfo optional fields
run_py "2c" "RemoteWeightInfo has new optional fields with correct defaults" '
import dataclasses

RemoteWeightInfo = _ptu.RemoteWeightInfo
fields = {f.name: f for f in dataclasses.fields(RemoteWeightInfo)}
assert "agent_name" in fields and fields["agent_name"].default == ""
assert "backend" in fields and fields["backend"].default == "mooncake"
print("OK: RemoteWeightInfo has new fields")
'

# Extra: add_nixl_remote_agent helper required by impl Step 2
run_py "2d" "add_nixl_remote_agent base64-decodes and calls add_remote_agent" '
import base64
from unittest.mock import MagicMock

add_nixl_remote_agent = _ptu.add_nixl_remote_agent
agent = MagicMock()
payload = b"nixl-meta"
add_nixl_remote_agent(agent, base64.b64encode(payload).decode("ascii"))
agent.add_remote_agent.assert_called_once_with(payload)
print("OK: add_nixl_remote_agent works")
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
