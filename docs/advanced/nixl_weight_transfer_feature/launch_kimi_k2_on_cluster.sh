#!/bin/bash
# =============================================================================
# launch_kimi_k2_on_cluster.sh — 64-node (512-GPU) Kimi-K2 job on the cluster
# =============================================================================
#
# Run prepare_kimi_k2_on_cluster.sh FIRST. This script assumes the model, the
# datasets and the converted Megatron checkpoint are already on Lustre, and
# refuses to queue for 64 nodes if any of them is missing.
#
# WHAT IT DOES
# ------------
#   0.   Allocates 64 nodes with sbatch and starts one container per node.
#   1-4. Updates Miles + SGLang to the fork branches, pins numpy/scipy, prints
#        the active commits.  All 64 nodes.
#   5.   Runs examples/p2p_weight_transfer/Kimi-K2.sh on every node with its own
#        rank, which brings up Ray and submits the training job from rank 0.
#
# 64 nodes is what RUN_CONFIGS["Kimi-K2-Instruct"] in
# examples/p2p_weight_transfer/run.py declares: 256 train + 256 rollout GPUs,
# disaggregated. Do not lower it without changing that entry too — Kimi-K2.sh
# says outright that fewer than 64 nodes will not work.
#
# sbatch rather than an interactive srun --pty, because 512 GPUs will not be
# handed to an interactive session — which is also why step 5 launches the job
# itself instead of dropping into a shell.
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/$USER/miles/radixark+miles+latest.sqsh
#
# If it doesn't exist, create it from a compute node (login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/$USER/miles/
#   enroot import docker://radixark/miles:latest
#
# USAGE
# -----
#   bash launch_kimi_k2_on_cluster.sh              # nixl (default)
#   bash launch_kimi_k2_on_cluster.sh p2p          # mooncake
#   bash launch_kimi_k2_on_cluster.sh broadcast
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-<jobid>.out
#
# RESETTING
# ---------
# To rebuild the model or the checkpoint, see prepare_kimi_k2_on_cluster.sh.
# To rebuild the container image, delete and re-import the sqsh:
#   rm -f <LUSTRE>/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest
# =============================================================================

#SBATCH --job-name=miles-kimi-k2
#SBATCH --account=network_research_advdev
#SBATCH --partition=batch
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.err

set -e

# --- Config ------------------------------------------------------------------
# The #SBATCH directives above are parsed by sbatch before any of this runs, so
# node count, account and partition have to be literals up there. Keep the two
# in sync by hand.
PORTFOLIO=network_research_advdev

export MODE="${1:-nixl}"           # nixl | p2p | broadcast

export MODEL=Kimi-K2-Instruct

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2

# These four container paths are the ones run.py hardcodes, so the mounts below
# are not free to change. A named container persists /root between sessions on
# one node, but the other 63 cannot see it, so everything shared has to come
# from Lustre. Mounting the subdirectories is safe; mounting over /root or home
# breaks the Megatron import, hence --no-container-mount-home on the srun.
HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets
HOST_LOGS=$LUSTRE/kimi-k2-logs

case "$MODE" in
    nixl|p2p|broadcast) ;;
    *)
        echo "ERROR: unknown mode '$MODE'. Use one of: nixl, p2p, broadcast."
        exit 1
        ;;
esac

# --- Validate the sqsh and the prepared artifacts ----------------------------
# Lustre is visible from the login node, so this runs before the job is queued.
# Checking here rather than inside the job is the point: a missing checkpoint
# should cost nothing, not a place in the queue for 512 GPUs.
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

if [ ! -f "$HOST_MODELS/.${MODEL}.download_complete" ]; then
    fail "$MODEL has not been downloaded (sentinel missing under $HOST_MODELS)" \
         "Run the prepare job first:" \
         "bash $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/prepare_kimi_k2_on_cluster.sh"
fi

TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    fail "no completed Megatron checkpoint at $HOST_CKPT/${MODEL}_torch_dist" \
         "convert_hf_to_torch_dist.py writes 'release' into the tracker as its" \
         "last action, so anything else means the conversion did not finish." \
         "Run the prepare job first:" \
         "bash $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/prepare_kimi_k2_on_cluster.sh"
fi

for ds in dapo-math-17k/dapo-math-17k.jsonl aime-2024/aime-2024.jsonl; do
    if [ ! -f "$HOST_DATASETS/$ds" ]; then
        fail "dataset file missing: $HOST_DATASETS/$ds" \
             "Run the prepare job first:" \
             "bash $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/prepare_kimi_k2_on_cluster.sh"
    fi
done

# --- Submit pass: re-exec under sbatch ---------------------------------------
# Run from the login node there is no allocation, and the #SBATCH lines above
# are just comments. Create the log directory Slurm needs and hand the file to
# sbatch, which re-reads those directives for real.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$HOST_LOGS"
    echo "All artifacts present. Submitting 64-node Kimi-K2 job (mode=$MODE)..."
    exec sbatch "$(readlink -f "${BASH_SOURCE[0]}")" "$MODE"
fi

# --- Per-job signal directory ------------------------------------------------
# run.py has rank 0 write job_done_<mode> into MILES_LOG_DIR when training ends,
# and the 63 workers poll for it. It has to be a shared mount or they would
# never see it — run.py's default, /data/ray/signals, is container-local. Keying
# it on the job id is what stops a leftover signal from an earlier run from
# making every worker exit immediately.
HOST_SIGNALS=$HOST_LOGS/job-${SLURM_JOB_ID}
mkdir -p "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$HOST_SIGNALS"

# --- Resolve the head node IP ------------------------------------------------
# Has to happen out here: the container has no scontrol. Ray needs an address to
# gather on, and in a batch job that is the first host of the allocation; the
# rank comes from the scheduler.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
if [ -z "$HEAD_NODE_IP" ]; then
    echo "ERROR: could not resolve an IP for head node $HEAD_NODE"
    exit 1
fi

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — Kimi-K2 64-node"
echo "=========================================="
echo "  Job       : $SLURM_JOB_ID"
echo "  Portfolio : $PORTFOLIO"
echo "  Mode      : $MODE"
echo "  Nodes     : $SLURM_JOB_NUM_NODES (256 train + 256 rollout GPUs)"
echo "  Head node : $HEAD_NODE ($HEAD_NODE_IP)"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Signals   : $HOST_SIGNALS"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
# The heredoc is quoted, so nothing in it is expanded at submit time. Everything
# it needs was exported above and reaches the container through the environment;
# SLURM_NODEID is set per task by Slurm and is the rank Kimi-K2.sh needs.
INNER_CMD=$(cat <<'EOF'
set -ex

RANK="${SLURM_NODEID:-0}"

# ---- Step 1: Update Miles ---------------------------------------------------
# The sqsh has Miles cloned at /root/miles from the main branch at build time.
# Add the fork remote, discard any local edits so checkout cannot abort with
# "local changes would be overwritten", then fetch and check out.
echo "--- [rank $RANK] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" fork/"$MILES_BRANCH"

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [rank $RANK] Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork "$SGLANG_FORK" 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork "$SGLANG_BRANCH"
git -C /sgl-workspace/sglang checkout -B "$SGLANG_BRANCH" fork/"$SGLANG_BRANCH"

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
# The sqsh base image ships numpy 2.x and scipy 1.18. Megatron asserts numpy<2
# at startup, and scipy>=1.15 requires numpy>=2, so both move together.
echo "--- [rank $RANK] Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
# So a log on its own identifies exactly what ran.
echo ""
echo "--- [rank $RANK] Active code ---"
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ---- Step 5: Launch ---------------------------------------------------------
# Every rank runs this with its own rank; Kimi-K2.sh forwards all three
# arguments to run.py, which starts the Ray head on rank 0, joins the rest to
# it, waits for all 512 GPUs to register, then submits the training job.
#
# SKIP_VALIDATION is required: run.py raises NotImplementedError for
# Kimi-K2-Instruct unless it is set, so weight validation is off
# (see docs/advanced/p2p-weight-transfer.md).
#
# Kimi-K2.sh also runs `run.py prepare` itself when the torch_dist directory is
# missing. The launch script checked for it before submitting, so it never is —
# which is the point of converting in the prepare job instead: prepare would use
# the single-node PREPARE_CONFIGS sizes, which OOM on a trillion parameters.
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/signals

cd /root/miles
echo "--- [rank $RANK] Launching $MODEL, mode=$MODE, head=$HEAD_NODE_IP ---"
bash examples/p2p_weight_transfer/Kimi-K2.sh "$MODE" "$RANK" "$HEAD_NODE_IP"
EOF
)

# --- Run ---------------------------------------------------------------------
# One container per node, and no --mpi=pmix: nothing here is an MPI program.
# Ray does the bootstrapping — rank 0 starts the head, the rest join over TCP.
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$HOST_SIGNALS:/root/signals" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
