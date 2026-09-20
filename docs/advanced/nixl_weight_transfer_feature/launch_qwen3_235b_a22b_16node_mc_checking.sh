#!/bin/bash
# =============================================================================
# launch_qwen3_235b_a22b_16node_mc_checking.sh
#   Sanity check: does mainline Miles's P2P (Mooncake/RDMA) weight transfer
#   work as-is, no NIXL patches?
#
#   Miles  : amitw/miles-main-MC-checking (byte-identical to radixark/miles main)
#   SGLang : sglang-miles (the stock image's default branch — nothing to switch)
#
# This intentionally follows only what the docs say to do — no fork/branch
# juggling inside the container, because the stock radixark/miles:latest image
# already clones origin=radixark/miles @ main and SGLang @ sglang-miles
# (docker/Dockerfile ARG MILES_COMMIT=main, ARG SGLANG_BRANCH=sglang-miles).
# So "get on the right code" is just the quick-start's `git pull`:
#   docs/getting-started/quick-start.md, Step 1:
#     cd /root/miles && git pull && pip install -e . --no-deps
#
# And running the model is exactly the p2p_weight_transfer README's multi-node
# recipe:
#   examples/infra_features/p2p_weight_transfer/README.md:
#     bash examples/infra_features/p2p_weight_transfer/Qwen3-235B-A22B.sh p2p 0 $HEAD_NODE_IP  # head
#     bash examples/infra_features/p2p_weight_transfer/Qwen3-235B-A22B.sh p2p 1 $HEAD_NODE_IP  # worker 1
#     ... workers 2..15
# That one wrapper script already does prepare-if-missing + run.py run, so this
# launcher does not call run.py directly. Internally it trains
# Qwen3-235B-A22B-Instruct-2507; the wrapper filename differs from the model
# name run.py actually uses.
#
# Everything below this point (sbatch, container mounts, head-IP discovery) is
# cluster plumbing the docs don't cover, since they assume manual per-node
# docker rather than Slurm.
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest+100926.sqsh
# If it doesn't, build it from a compute node (login node has no disk space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# USAGE
# -----
#   bash launch_qwen3_235b_a22b_16node_mc_checking.sh [MODE]
#     MODE : p2p (default) | broadcast   (mainline has no "nixl" mode)
#
# MONITORING
# ----------
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-qwen3-235b-a22b-mc-checking-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
# =============================================================================

set -e

MODE="${1:-p2p}"

PORTFOLIO=network_research_advdev
PARTITION=batch
NUM_NODES=16

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest+100926.sqsh
C_NAME=amitw-miles-mc-checking-16node-100926
LOG_DIR=$LUSTRE/logs

# Same container paths the docs use directly (/root/models, /root/datasets,
# /root/multinode) — no renaming layer, matching quick-start.md and
# p2p-weight-transfer.md verbatim.
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
HOST_DATASETS=$LUSTRE/datasets
HOST_CKPT=$LUSTRE/multinode

if [ ! -f "$SQSH" ]; then
    echo "ERROR: sqsh file not found at $SQSH"
    echo "Build it from a compute node: enroot import docker://radixark/miles:latest"
    exit 1
fi

# pyxis will not create mount sources.
mkdir -p "$LOG_DIR" "$HOST_MODELS" "$HOST_DATASETS" "$HOST_CKPT"

echo ""
echo "=========================================="
echo "  Miles cluster launch — Qwen3-235B-A22B 16-node (mainline MC-checking)"
echo "  Mode      : $MODE"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Logs      : $LOG_DIR"
echo "=========================================="
echo ""

JOB_SCRIPT=$LOG_DIR/job-qwen3-235b-a22b-mc-checking.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-qwen3-235b-a22b-mc-checking
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=01:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out

mkdir -p $LOG_DIR

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_DATASETS:/root/datasets,$HOST_CKPT:/root/multinode" \\
    bash -lc '
set -ex

# --- quick-start.md, Step 1: get on latest mainline Miles --------------------
echo "--- [Node \${SLURM_NODEID}] git pull (mainline main) ---"
cd /root/miles
git pull
pip install -e . --no-deps

# --- docker/Dockerfile "Install sglang-miles" step, applied again at runtime -
# so SGLang moves forward in lockstep with Miles instead of staying pinned to
# whatever commit was baked into the sqsh at build time. HEAD is detached
# (docker/Dockerfile checks out FETCH_HEAD directly), so `git pull` will not
# work here — mirror the Dockerfile'"'"'s fetch+checkout instead.
echo "--- [Node \${SLURM_NODEID}] Updating SGLang (sglang-miles) ---"
git -C /sgl-workspace/sglang fetch origin sglang-miles
git -C /sgl-workspace/sglang checkout -f FETCH_HEAD

echo "--- [Node \${SLURM_NODEID}] Active code ---"
echo "Miles  : \$(git -C /root/miles log --oneline -1)"
echo "SGLang : \$(git -C /sgl-workspace/sglang log --oneline -1)"

# --- Resolve head node IP from SLURM_NODELIST ---------------------------------
NODE_RANK=\${SLURM_NODEID}
HEAD_NODE_IP=\$(python3 -c "
import socket, re, os, sys
nl = os.environ.get(\"SLURM_NODELIST\", \"\")
m = re.match(r\"(.+)-\[([^\]]+)\]\", nl)
if m:
    prefix, spec = m.group(1), m.group(2)
    first = re.split(r\"[,\-]\", spec)[0]
    hostname = prefix + \"-\" + first
elif \",\" in nl:
    hostname = nl.split(\",\")[0]
else:
    hostname = nl
try:
    print(socket.gethostbyname(hostname))
except Exception as e:
    print(\"ERROR resolving \" + hostname + \": \" + str(e), file=sys.stderr)
    sys.exit(1)
")
echo "--- [Node \$NODE_RANK] Head IP: \$HEAD_NODE_IP ---"

# --- p2p_weight_transfer/README.md multi-node usage, verbatim ----------------
bash examples/infra_features/p2p_weight_transfer/Qwen3-235B-A22B.sh $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"
'
SLURM

JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
echo "Submitted job $JOB_ID"
echo "  tail -f $LOG_DIR/miles-qwen3-235b-a22b-mc-checking-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
