#!/bin/bash
# =============================================================================
# launch_kimi_k2_64node.sh — 64-node (512-GPU) Kimi-K2 Miles job on the cluster
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
# This script TRAINS ONLY. The HF model, the converted Megatron checkpoint and
# the datasets must already be on Lustre under
# /lustre/fsw/portfolios/network/users/amitw/miles/{models,multinode,datasets}.
# Unlike the Qwen launcher, preparation cannot happen inside the job: Kimi-K2 is
# a trillion parameters, so per-node download and single-node conversion do not
# fit. See the artifact checks below for exactly what is expected.
#
# HOW THIS SCRIPT WORKS
# ---------------------
# Submits a 64-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit).  Before submitting, it verifies every artifact the job
# needs, so a missing checkpoint costs nothing rather than a place in the queue
# for 512 GPUs.  Inside the job:
#   1-4. Same env setup as launch_on_cluster_1node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run Kimi-K2-Instruct in parallel with the correct rank and IP.
#
# The Lustre directories are bind-mounted at the container paths run.py
# hardcodes, so all 64 nodes read one copy of the model and checkpoint. Mounting
# the subdirectories is safe; mounting over /root or home breaks the Megatron
# import, hence --no-container-mount-home on the srun.
#
# 64 nodes is what RUN_CONFIGS["Kimi-K2-Instruct"] in
# examples/p2p_weight_transfer/run.py declares: 256 train + 256 rollout GPUs,
# disaggregated. Do not lower it without changing that entry too.
#
# Output is streamed to:
#   /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-<JOBID>.out
#
# USAGE
# -----
#   bash launch_kimi_k2_64node.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome-64node) persists on each node. If it
# gets into a broken state, delete the sqsh to force a clean rebuild:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

MODEL=Kimi-K2-Instruct
# The per-model launcher is named after the model family, not the registry key.
MODEL_SCRIPT=Kimi-K2
MODE="${1:-nixl}"

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch
NUM_NODES=64

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-experiments

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=amitw-miles-nohome-64node

LOG_DIR=$LUSTRE/logs

# Bind-mounted at the container paths run.py hardcodes, so these three are not
# free to rename.
HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets

# --- Validate the sqsh and the prepared artifacts ----------------------------
# Lustre is visible from the login node, so all of this runs before the job is
# queued.
fail() {
    echo ""
    echo "ERROR: $1"
    echo ""
    shift
    for line in "$@"; do echo "  $line"; done
    echo ""
    exit 1
}

if [ ! -f "$SQSH" ]; then
    fail "sqsh file not found at $SQSH" \
         "To create it, run on a compute node (login node has no disk space):" \
         "srun -A $PORTFOLIO -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash" \
         "cd $LUSTRE/" \
         "enroot import docker://radixark/miles:latest"
fi

# run.py passes this directory to SGLang as --hf-checkpoint. hf download writes
# the shard index last, so its presence means the download finished rather than
# merely started.
if [ ! -f "$HOST_MODELS/$MODEL/model.safetensors.index.json" ]; then
    fail "no complete HF checkpoint at $HOST_MODELS/$MODEL" \
         "Download it to Lustre before submitting:" \
         "hf download moonshotai/$MODEL --local-dir $HOST_MODELS/$MODEL"
fi

# convert_hf_to_torch_dist.py writes 'release' into the tracker as its last
# action, so anything else means the conversion did not finish.
TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    fail "no completed Megatron checkpoint at $HOST_CKPT/${MODEL}_torch_dist" \
         "Convert the HF checkpoint to torch_dist on Lustre before submitting." \
         "It does not fit on one node: a trillion parameters need a multi-node" \
         "torchrun, and the source must be BF16 (see tools/fp8_cast_bf16.py)."
fi

# run.py passes both files by absolute path, so these names are fixed.
for ds in dapo-math-17k/dapo-math-17k.jsonl aime-2024/aime-2024.jsonl; do
    if [ ! -f "$HOST_DATASETS/$ds" ]; then
        fail "dataset file missing: $HOST_DATASETS/$ds" \
             "Download it to Lustre before submitting:" \
             "hf download --repo-type dataset zhuzilin/${ds%%/*} --local-dir $HOST_DATASETS/${ds%%/*}"
    fi
done

mkdir -p "$LOG_DIR"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — Kimi-K2 ${NUM_NODES}-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Nodes     : $NUM_NODES"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Model dir : $HOST_MODELS/$MODEL"
echo "  Ckpt dir  : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Logs      : $LOG_DIR"
echo "=========================================="
echo ""

# --- Write job script --------------------------------------------------------
#
# Written to a file in lustre so it can be inspected after submission.
# Variables from the outer shell (SQSH, MILES_BRANCH, etc.) are expanded here;
# SLURM variables (\$SLURM_NODEID etc.) are escaped and evaluated at runtime.
#
JOB_SCRIPT=$LOG_DIR/job-kimi-k2.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out

mkdir -p $LOG_DIR

# Per-job signal directory. run.py has rank 0 write job_done_<mode> here when
# training ends and the 63 workers poll for it, so it has to be a shared mount —
# run.py's default is container-local and the workers would never see it.
# Keying it on the job id stops a leftover signal from an earlier run from
# making every worker exit immediately. pyxis will not create mount sources.
HOST_SIGNALS=$LOG_DIR/signals-\$SLURM_JOB_ID
mkdir -p "\$HOST_SIGNALS"

srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$HOST_SIGNALS:/root/signals" \\
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
# run.py raises NotImplementedError for Kimi-K2-Instruct unless SKIP_VALIDATION
# is set: the model has no --check-weight-update-equal path.
#
# The checkpoint was verified on the login node, so $MODEL_SCRIPT.sh takes the
# already-prepared branch and goes straight to training.
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/signals
cd /root/miles
bash examples/p2p_weight_transfer/$MODEL_SCRIPT.sh $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

'
SLURM

# --- Submit ------------------------------------------------------------------
JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/miles-kimi-k2-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
