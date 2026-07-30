#!/bin/bash
# =============================================================================
# launch_qwen3_30b_a3b_4node.sh — 4-node (32-GPU) Qwen3-30B-A3B Miles job on the cluster
# =============================================================================
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#
# If it doesn't exist, create it from a compute node (login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# HOW THIS SCRIPT WORKS
# ---------------------
# Submits a 4-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit).  Inside the job:
#   1-4. Same env setup as launch_on_cluster_1node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run Qwen3-30B-A3B in parallel with the correct rank and IP.
#
# Output is streamed to:
#   /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-qwen3-30b-a3b-<JOBID>.out
#
# USAGE
# -----
#   bash launch_qwen3_30b_a3b_4node.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-qwen3-30b-a3b-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome-4node) persists on each node. If it
# gets into a broken state, delete the sqsh to force a clean rebuild:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

MODEL=Qwen3-30B-A3B
MODE="${1:-nixl}"

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

SQSH=/lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
C_NAME=amitw-miles-nohome-4node

LOG_DIR=/lustre/fsw/portfolios/network/users/amitw/miles/logs

# --- Validate sqsh exists ----------------------------------------------------
if [ ! -f "$SQSH" ]; then
    echo ""
    echo "ERROR: sqsh file not found at $SQSH"
    echo ""
    echo "To create it, run on a compute node (login node has no disk space):"
    echo "  srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash"
    echo "  cd /lustre/fsw/portfolios/network/users/amitw/miles/"
    echo "  enroot import docker://radixark/miles:latest"
    echo ""
    exit 1
fi

mkdir -p "$LOG_DIR"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — Qwen3-30B-A3B 4-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Logs      : $LOG_DIR"
echo "=========================================="
echo ""

# --- Write job script --------------------------------------------------------
#
# Written to a file in lustre so it can be inspected after submission.
# Variables from the outer shell (SQSH, MILES_BRANCH, etc.) are expanded here;
# SLURM variables (\$SLURM_NODEID etc.) are escaped and evaluated at runtime.
#
JOB_SCRIPT=$LOG_DIR/job-qwen3-30b-a3b.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-qwen3-30b-a3b
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out

mkdir -p $LOG_DIR

srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    bash -lc '
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork $MILES_FORK 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH fork/$MILES_BRANCH

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork $SGLANG_FORK 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork $SGLANG_BRANCH
git -C /sgl-workspace/sglang checkout -B $SGLANG_BRANCH fork/$SGLANG_BRANCH

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
echo "--- [Node \${SLURM_NODEID}] Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- [Node \${SLURM_NODEID}] Active code ---"
echo "Miles  : \$(git -C /root/miles log --oneline -1)"
echo "SGLang : \$(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : \$(python -c '"'"'import numpy; print(numpy.__version__)'"'"')"
echo ""

# ---- Step 5: Discover head node IP from SLURM_NODELIST ----------------------
# Each node independently resolves the head hostname from SLURM_NODELIST —
# no shared file or coordination needed.
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
    ip = socket.gethostbyname(hostname)
    print(ip)
except Exception as e:
    print(\"ERROR resolving \" + hostname + \": \" + str(e), file=sys.stderr)
    sys.exit(1)
")
echo "--- [Node \$NODE_RANK] Head IP: \$HEAD_NODE_IP ---"

# ---- Step 6: Run the model --------------------------------------------------
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
cd /root/miles
bash examples/p2p_weight_transfer/$MODEL.sh $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

'
SLURM

# --- Submit ------------------------------------------------------------------
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/miles-qwen3-30b-a3b-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
