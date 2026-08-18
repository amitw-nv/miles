#!/bin/bash
# =============================================================================
# launch_qwen3_235b_a22b_16node.sh — end-to-end Qwen3-235B-A22B: prepare → train
# =============================================================================
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest+120826.sqsh
#
# If it doesn't exist, create it from a compute node (login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# CODE STAGING (no GitHub access from compute nodes)
# --------------------------------------------------
# Compute nodes cannot reach GitHub. This script fetches Miles and SGLang
# from GitHub once on the login node into bare mirrors on Lustre, then
# every compute node fetches from those mirrors instead. The login node
# needs GitHub access; the compute nodes need none.
#
# END-TO-END FLOW
# ---------------
# Run this script once from the login node. It checks what is already staged on
# Lustre and submits only the jobs that are still needed, chaining them with
# --dependency=afterok so they run in sequence automatically:
#
#   [prepare job]  — submitted if the HF checkpoint, datasets, or the Megatron
#       torch_dist checkpoint are missing.
#       1 node, 8 GPUs. Runs `run.py prepare Qwen3-235B-A22B-Instruct-2507`, which
#       downloads Qwen/Qwen3-235B-A22B-Instruct-2507 plus both datasets and converts
#       the checkpoint to torch_dist (single-node torchrun, 8 GPUs). Writes onto
#       Lustre via bind-mounts. Time limit: 4 h.
#
#   [main job]     — always submitted, waits for the prepare job if there is one.
#       16 nodes, 128 GPUs. The Qwen3-235B-A22B RLVR training run. Time limit: 4 h.
#
# If everything is already staged on Lustre, only the main job is submitted —
# nothing is re-downloaded on the training nodes.
#
# HOW THE MAIN JOB WORKS
# ----------------------
# Submits a 16-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit).  Inside the job:
#   1-4. Same env setup as launch_qwen3_30b_a3b_4node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run Qwen3-235B-A22B in parallel with the correct rank and IP.
#
# Mount safety (Megatron):
#   Bind-mount ONLY the leaf dirs that run.py hardcodes
#   (/root/models, /root/multinode, /root/datasets, /root/signals).
#   Do NOT mount over /root or home — that shadows /root/miles and
#   /root/Megatron-LM and breaks the Megatron import. Always keep
#   --no-container-mount-home on the srun.
#
# Output is streamed to:
#   /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-qwen3-235b-a22b-<JOBID>.out
#
# USAGE
# -----
#   bash launch_qwen3_235b_a22b_16node.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-qwen3-235b-a22b-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome-16node) persists on each node. If it
# gets into a broken state, delete the sqsh to force a clean rebuild:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest+120826.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

MODEL=Qwen3-235B-A22B-Instruct-2507
MODE="${1:-nixl}"

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch
NUM_NODES=16

MILES_FORK=git@github.com:amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-final-rebase

SGLANG_FORK=git@github.com:amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl-rebase
SGLANG_COMMIT=bb82b4c628c0f1aaee330392d44dda9225017d79

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest+120826.sqsh
C_NAME=amitw-miles-nohome-16node

LOG_DIR=$LUSTRE/logs

# Bind-mounted at the container paths run.py hardcodes, so these three are not
# free to rename. HOST_MODELS is shared across models (GLM, Qwen, Kimi) and
# sits outside $LUSTRE on purpose.
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets

# Bare mirrors fetched here on the login node; compute nodes read from Lustre.
MILES_SRC=$LUSTRE/src/miles.git
SGLANG_SRC=$LUSTRE/src/sglang.git

# --- Validate sqsh exists ----------------------------------------------------
if [ ! -f "$SQSH" ]; then
    echo ""
    echo "ERROR: sqsh file not found at $SQSH"
    echo ""
    echo "To create it, run on a compute node (login node has no disk space):"
    echo "  srun -A $PORTFOLIO -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash"
    echo "  cd $LUSTRE/"
    echo "  enroot import docker://radixark/miles:latest"
    echo ""
    exit 1
fi

# pyxis will not create mount sources, so every bind-mount source has to exist
# before the srun or container setup fails before any of our code runs.
mkdir -p "$LOG_DIR" "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$LUSTRE/src"

# --- Stage code on Lustre (login node fetches from GitHub once) --------------
stage_repo() {
    local bare=$1 url=$2 branch=$3
    [ -d "$bare" ] || git init --bare -q "$bare"
    if ! git -C "$bare" fetch --force --quiet "$url" "+$branch:$branch"; then
        echo "" >&2
        echo "ERROR: could not fetch $branch from $url" >&2
        echo "Make sure this login node has GitHub access." >&2
        echo "" >&2
        return 1
    fi
}

stage_repo "$MILES_SRC"  "$MILES_FORK"  "$MILES_BRANCH"
stage_repo "$SGLANG_SRC" "$SGLANG_FORK" "$SGLANG_BRANCH"

MILES_SHA=$(git -C "$MILES_SRC" rev-parse "$MILES_BRANCH")

# Pinned to a specific SGLang commit rather than the branch tip. The branch
# fetch above pulls full history, so this commit is present in the mirror
# as long as it is an ancestor of $SGLANG_BRANCH.
SGLANG_SHA=$(git -C "$SGLANG_SRC" rev-parse --verify "${SGLANG_COMMIT}^{commit}")

echo "Miles  : $MILES_BRANCH @ ${MILES_SHA:0:9}"
echo "SGLang : $SGLANG_BRANCH @ ${SGLANG_SHA:0:9}"

# --- Determine whether preparation is needed ---------------------------------
NEEDS_PREPARE=0

# hf download writes the shard index last; its presence means download finished.
if [ ! -f "$HOST_MODELS/$MODEL/model.safetensors.index.json" ]; then
    echo "HF checkpoint missing — will submit prepare job."
    NEEDS_PREPARE=1
fi

# convert_hf_to_torch_dist.py writes 'release' into the tracker as its last
# action, so anything else means the conversion did not finish.
TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    echo "Megatron checkpoint missing or incomplete — will submit prepare job."
    NEEDS_PREPARE=1
fi

# run.py passes both files by absolute path, so these names are fixed.
for ds in dapo-math-17k/dapo-math-17k.jsonl aime-2024/aime-2024.jsonl; do
    if [ ! -f "$HOST_DATASETS/$ds" ]; then
        echo "Dataset missing: $ds — will submit prepare job."
        NEEDS_PREPARE=1
    fi
done

PREV_JOB_ID=""

# --- Submit prepare job if needed --------------------------------------------
if [ "$NEEDS_PREPARE" -eq 1 ]; then
    PREP_JOB_SCRIPT=$LOG_DIR/job-qwen3-235b-a22b-prepare.slurm
    cat > "$PREP_JOB_SCRIPT" <<PREP_SLURM
#!/bin/bash
#SBATCH --job-name=miles-qwen3-235b-a22b-prepare
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=$LOG_DIR/miles-qwen3-235b-a22b-prepare-%j.out

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-prepare" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$MILES_SRC:/root/src/miles.git:ro" \\
    bash -lc '
set -ex

# Fetch from the Lustre mirror — login node already pulled from GitHub.
echo "--- [Prepare] Updating Miles ($MILES_BRANCH @ ${MILES_SHA:0:9}) ---"
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch /root/src/miles.git $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH $MILES_SHA
pip install -q "numpy<2" "scipy<1.15"

# Downloads the HF checkpoint and both datasets, then converts to torch_dist.
# Writes land on Lustre via the bind-mounts above. Each step is individually
# idempotent, so a re-run resumes rather than redoes.
echo "--- [Prepare] Downloading and converting $MODEL ---"
cd /root/miles
python3 examples/infra_features/p2p_weight_transfer/run.py prepare $MODEL

echo "--- [Prepare] Done ---"
'
PREP_SLURM

    PREP_JOB_ID=$(sbatch --parsable "$PREP_JOB_SCRIPT")
    echo "Submitted prepare job $PREP_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-qwen3-235b-a22b-prepare-${PREP_JOB_ID}.out"
    PREV_JOB_ID=$PREP_JOB_ID
fi

MAIN_DEP=""
[ -n "$PREV_JOB_ID" ] && MAIN_DEP="--dependency=afterok:$PREV_JOB_ID"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — Qwen3-235B-A22B 16-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Nodes     : $NUM_NODES"
echo "  Miles     : $MILES_BRANCH @ ${MILES_SHA:0:9}"
echo "  SGLang    : $SGLANG_BRANCH @ ${SGLANG_SHA:0:9}"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Model dir : $HOST_MODELS/$MODEL"
echo "  Ckpt dir  : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Logs      : $LOG_DIR"
[ -n "$PREV_JOB_ID" ] && echo "  Waiting on: prepare job $PREV_JOB_ID"
echo "=========================================="
echo ""

# --- Write job script --------------------------------------------------------
#
# Written to a file in lustre so it can be inspected after submission.
# Variables from the outer shell (SQSH, MILES_BRANCH, etc.) are expanded here;
# SLURM variables (\$SLURM_NODEID etc.) are escaped and evaluated at runtime.
#
JOB_SCRIPT=$LOG_DIR/job-qwen3-235b-a22b.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-qwen3-235b-a22b
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out

mkdir -p $LOG_DIR

# Per-job signal directory. run.py has rank 0 write job_done_<mode> here when
# training ends and the workers poll for it, so it has to be a shared mount —
# run.py's default is container-local and the workers would never see it.
# Keying it on the job id stops a leftover signal from an earlier run from
# making every worker exit immediately. pyxis will not create mount sources.
HOST_SIGNALS=$LOG_DIR/signals-\$SLURM_JOB_ID
mkdir -p "\$HOST_SIGNALS"

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$HOST_SIGNALS:/root/signals,$MILES_SRC:/root/src/miles.git:ro,$SGLANG_SRC:/root/src/sglang.git:ro" \\
    bash -lc '
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH @ ${MILES_SHA:0:9}) ---"
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch /root/src/miles.git $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH $MILES_SHA

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating SGLang ($SGLANG_BRANCH @ ${SGLANG_SHA:0:9}) ---"
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch /root/src/sglang.git $SGLANG_BRANCH
git -C /sgl-workspace/sglang checkout -B $SGLANG_BRANCH $SGLANG_SHA

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
# HF weights / torch_dist ckpt / datasets come from the Lustre bind-mounts.
# Qwen3-235B-A22B.sh still has a prepare guard, but it will no-op once the
# Megatron checkpoint exists on the shared /root/multinode mount.
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
export MILES_LOG_DIR=/root/signals
# Qwen3-235B-A22B-Instruct-2507 has rope_theta=5000000; the model script
# defaults to 1000000 (an older Qwen3 value). Must be exported before run.py
# so the env var is inherited when the model script is sourced inside run.py.
export MODEL_ARGS_ROTARY_BASE=5000000
cd /root/miles
bash examples/infra_features/p2p_weight_transfer/Qwen3-235B-A22B.sh $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

'
SLURM

# --- Submit ------------------------------------------------------------------
JOB_ID=$(sbatch --parsable $MAIN_DEP "$JOB_SCRIPT")

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/miles-qwen3-235b-a22b-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
