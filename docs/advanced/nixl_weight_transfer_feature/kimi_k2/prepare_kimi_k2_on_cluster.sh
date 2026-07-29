#!/bin/bash
# =============================================================================
# prepare_kimi_k2_on_cluster.sh — step 2 of 3: BF16 -> Megatron torch_dist
# =============================================================================
#
# THE THREE STEPS
# ---------------
#   1. cast_kimi_k2_to_bf16_on_cluster.sh   1 node    datasets, model, BF16 cast
#   2. prepare_kimi_k2_on_cluster.sh        8 nodes   BF16 -> Megatron torch_dist
#   3. launch_kimi_k2_on_cluster.sh        64 nodes   training
#
# All three are submitted from the login node and each self-submits with sbatch.
#
# Run step 1 first. This script converts the BF16 copy it produced and refuses
# to queue for 8 nodes if that copy is not there.
#
# WHAT IT DOES
# ------------
#   1-4. Update Miles + SGLang to the fork branches, pin numpy/scipy, print the
#        active commits.  All 8 nodes.
#   5.   Convert the BF16 HF checkpoint to Megatron torch_dist across 8 nodes /
#        64 GPUs.
#
# It is idempotent: a completed checkpoint is detected and skipped, so a rerun
# after a failure costs only the container setup.
#
# WHY 8 NODES
# -----------
# A trillion parameters do not fit on one GPU, so the conversion is a torchrun
# job that shards the model and has each rank build and write only its own
# slice. 4 nodes was tried first and is too tight: with EP 4 the shard is
# ~70 GiB of a 79 GiB card, and once peer and NCCL buffers take their ~8 GiB
# there is 1.38 GiB left, which is not enough to stage incoming tensors. 8 nodes
# with EP 8 halves the shard to roughly 35 GiB.
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
#   bash prepare_kimi_k2_on_cluster.sh
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-prep-<jobid>.out
#
# To change the walltime without editing this file, sbatch reads it from the
# environment:
#   SBATCH_TIMELIMIT=02:00:00 bash prepare_kimi_k2_on_cluster.sh
#
# RESETTING
# ---------
# To force a fresh conversion, delete the checkpoint and resubmit:
#   rm -rf <LUSTRE>/multinode/Kimi-K2-Instruct_torch_dist
# =============================================================================

# 8 nodes = the 64 GPUs the parallel sizes in step 5 are derived for. sbatch
# rather than an interactive srun --pty, because 64 GPUs will not be handed to
# an interactive session.
#SBATCH --job-name=miles-kimi-k2-prep
#SBATCH --account=network_research_advdev
#SBATCH --partition=batch
#SBATCH --nodes=8
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

export MODEL=Kimi-K2-Instruct
export MODEL_BF16=Kimi-K2-Instruct-bf16
export MODEL_TYPE=kimi-k2          # selects scripts/models/kimi-k2.sh

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

# Must match #SBATCH --nodes above and the parallel sizes in step 5.
export CONVERT_NODES=8

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2-prep

HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_LOGS=$LUSTRE/kimi-k2-logs

# --- Validate the sqsh and the BF16 checkpoint -------------------------------
# Lustre is visible from the login node, so this runs before the job is queued.
# Checking here rather than inside the job is the point: a missing BF16 copy
# should cost nothing, not a place in the queue for 64 GPUs.
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

# fp8_cast_bf16.py writes model.safetensors.index.json last, so its presence
# means the cast finished rather than merely started.
if [ ! -f "$HOST_MODELS/$MODEL_BF16/model.safetensors.index.json" ]; then
    fail "no completed BF16 checkpoint at $HOST_MODELS/$MODEL_BF16" \
         "The conversion cannot read the FP8 download directly — mbridge's" \
         "de-quant path OOMs and cannot take the loader's host tensors." \
         "Run step 1 first:" \
         "bash $(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/cast_kimi_k2_to_bf16_on_cluster.sh"
fi

# --- Submit pass: re-exec under sbatch ---------------------------------------
# Run from the login node there is no allocation, and the #SBATCH lines above
# are just comments. Create the log directory Slurm needs and hand the file to
# sbatch, which re-reads those directives for real.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$HOST_LOGS"
    echo "BF16 checkpoint present. Submitting 8-node Kimi-K2 conversion job..."
    exec sbatch "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# --- Mount sources -----------------------------------------------------------
# The checkpoint has to land on Lustre: a named container persists /root between
# sessions on one node, but the other seven cannot see it, and neither can the
# 64 nodes of the training job. These container paths are the ones run.py
# hardcodes. Mounting the subdirectories is safe; mounting over /root or home
# breaks the Megatron import, hence --no-container-mount-home on the srun below.
#
# pyxis will not create mount sources, so they have to exist first.
mkdir -p "$HOST_MODELS" "$HOST_CKPT" "$HOST_LOGS"

# --- Resolve the head node IP ------------------------------------------------
# Has to happen out here: the container has no scontrol. torchrun needs a
# rendezvous address and in a batch job that is the first host of the
# allocation.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
if [ -z "$HEAD_NODE_IP" ]; then
    echo "ERROR: could not resolve an IP for head node $HEAD_NODE"
    exit 1
fi

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster — Kimi-K2 conversion"
echo "=========================================="
echo "  Job       : $SLURM_JOB_ID"
echo "  Portfolio : $PORTFOLIO"
echo "  Nodes     : $SLURM_JOB_NUM_NODES (64 GPUs)"
echo "  Head node : $HEAD_NODE ($HEAD_NODE_IP)"
echo "  Source    : $HOST_MODELS/$MODEL_BF16"
echo "  Target    : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
# The heredoc is quoted, so nothing in it is expanded at submit time. Everything
# it needs was exported above and reaches the container through the environment;
# SLURM_NODEID is set per task by Slurm and is the rank torchrun needs.
INNER_CMD=$(cat <<'EOF'
set -ex

RANK="${SLURM_NODEID:-0}"

BF16_DIR=/root/models/$MODEL_BF16
TRACKER=/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt

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
echo ""
echo "--- [rank $RANK] Active code ---"
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ---- Step 5: Convert the checkpoint to Megatron torch_dist ------------------
# One torchrun job spanning 8 nodes and 64 GPUs. The pipeline size splits the 61
# layers into stages and the expert-parallel size splits each layer's experts,
# so a rank only ever builds its own slice. Each rank writes its own shard into
# the shared /root/multinode mount, so the nodes cooperatively produce one
# checkpoint directory.
#
# The parallel sizes are NOT optional, and kimi-k2.md § 3.3 omits them. Three
# ways of getting this wrong have already been tried:
#   * No parallel sizes, as printed in § 3.3. Nothing is sharded, every rank
#     builds the whole model, and it dies in load_weights with CUDA OOM.
#   * The PREPARE_CONFIGS values from examples/p2p_weight_transfer/run.py on a
#     single node. Megatron rejects it before loading any weights: "world_size
#     (8) is not divisible by expert_tensor_model_pipeline_parallel size (64)".
#   * EP 4 on 4 nodes. Megatron accepts it and builds the model, but the shard
#     is ~70 GiB of a 79 GiB card and load_weights OOMs with 1.38 GiB free.
#
# Two constraints fix the values below, and changing one means re-deriving the
# rest:
#   * ETP * EP * PP must equal the GPU count, 8 * 8 nodes = 64. Here
#     1 * 8 * 8 = 64. Megatron rejects any other product outright.
#   * The pipeline split must cover all 61 layers (scripts/models/kimi-k2.sh
#     sets --num-layers 61). Seven stages of 8 plus a last stage of 5 is 61,
#     hence --decoder-last-pipeline-num-layers 5.
#
# The source is the BF16 copy, never the FP8 download — see
# cast_kimi_k2_to_bf16_on_cluster.sh for why.
if [ -f "$TRACKER" ] && [ "$(tr -d '[:space:]' < "$TRACKER")" = "release" ]; then
    echo "--- [rank $RANK] Checkpoint already converted, skipping ---"
else
    echo "--- [rank $RANK] Converting across $CONVERT_NODES nodes ---"
    cd /root/miles
    source scripts/models/$MODEL_TYPE.sh      # defines MODEL_ARGS
    PYTHONPATH=/root/Megatron-LM/ torchrun \
        --nproc-per-node 8 \
        --master-addr "$HEAD_NODE_IP" --master-port 12345 \
        --nnodes="$CONVERT_NODES" --node-rank "$RANK" \
        tools/convert_hf_to_torch_dist.py \
        "${MODEL_ARGS[@]}" \
        --tensor-model-parallel-size 1 \
        --pipeline-model-parallel-size 8 \
        --expert-tensor-parallel-size 1 \
        --expert-model-parallel-size 8 \
        --decoder-last-pipeline-num-layers 5 \
        --hf-checkpoint "$BF16_DIR" \
        --save /root/multinode/${MODEL}_torch_dist
fi

# convert_hf_to_torch_dist.py writes "release" into the tracker as its very last
# action, after a barrier, so this both proves the checkpoint is complete and is
# what the training job checks for before asking for 64 nodes.
echo "--- [rank $RANK] Checkpoint tracker: $(cat "$TRACKER") ---"
echo "--- [rank $RANK] Done. Next: launch_kimi_k2_on_cluster.sh ---"
EOF
)

# --- Run ---------------------------------------------------------------------
# One container per node, and no --mpi=pmix: torchrun does its own rendezvous
# over the master address resolved above.
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
