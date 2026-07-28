#!/bin/bash
# =============================================================================
# launch_kimi_k2_on_cluster.sh — 64-node (512-GPU) Kimi-K2 Miles job
# =============================================================================
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
# HOW THIS SCRIPT WORKS
# ---------------------
# Same as launch_glm_z1_9b_1node.sh — steps 1-7 run inside the container on
# every launch and are idempotent, skipped instantly once already done:
#   1-4. Update Miles + SGLang to the fork branches, pin numpy/scipy
#   5. Download datasets dapo-math-17k (prompts) and aime-2024 (eval)
#   6. Download model    Kimi-K2-Instruct
#   7. Convert checkpoint to Megatron format
#
# Three things differ, all because Kimi-K2 needs 64 nodes instead of one:
#
#   * Batch, not interactive. 512 GPUs will not be handed to a --pty session,
#     so step 8 launches the model itself instead of dropping you into a shell.
#     Pick the transfer mode as the first argument.
#
#   * Steps 5-7 run on rank 0 only, with the other 63 nodes waiting for the
#     converted checkpoint to appear. Unguarded, all 64 would download the same
#     1 TB model at once. On the first launch those nodes idle through the
#     download and conversion; on later launches everything is already on
#     Lustre and every rank skips straight to step 8.
#
#   * Lustre is mounted, which the 1-node script deliberately avoids. A named
#     container persists /root between sessions on one node, but 63 other nodes
#     cannot see it, so the artifacts have to live on shared storage. They are
#     mounted at the paths run.py hardcodes for this model (--hf-checkpoint
#     /root/models/..., --ref-load /root/multinode/...; CKPT_SAVE_DIR is
#     honoured by `prepare` but NOT by `run` for Kimi-K2). Mounting these
#     subdirectories is safe; mounting over /root or home breaks the Megatron
#     import, which is why --no-container-mount-home stays.
#
# USAGE
# -----
#   # once: Slurm needs the --output directory to exist before the job starts
#   mkdir -p /lustre/fsw/portfolios/network/users/$USER/miles/kimi-k2-logs
#
#   sbatch launch_kimi_k2_on_cluster.sh          # nixl (default)
#   sbatch launch_kimi_k2_on_cluster.sh p2p      # mooncake
#   sbatch launch_kimi_k2_on_cluster.sh broadcast
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-<jobid>.out
#
# The --time below has to cover the first-launch download and conversion as
# well as training. Once the checkpoint is on Lustre, lower it.
#
# If moonshotai/Kimi-K2-Instruct is gated, export HF_TOKEN before submitting;
# enroot passes the environment through to the container.
#
# RESETTING
# ---------
# To force a fresh conversion, delete the checkpoint and resubmit:
#   rm -rf <LUSTRE>/multinode/Kimi-K2-Instruct_torch_dist
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
#SBATCH --time=16:00:00
#SBATCH --output=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.err

set -e

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev

export MODE="${1:-nixl}"          # nixl | p2p | broadcast

export MODEL=Kimi-K2-Instruct
export HF_REPO=moonshotai/Kimi-K2-Instruct
export MODEL_TYPE=kimi-k2

# Nodes taking part in the HF -> torch_dist conversion, per
# docs/models/kimi/kimi-k2.md § 3.3. Ranks 0..CONVERT_NODES-1 run it as one
# torchrun job; the rest wait for the tracker file. No parallel-size arguments
# are passed — the doc's command is MODEL_ARGS only, and adding to it is what
# broke the earlier single-node attempt.
export CONVERT_NODES=4

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2

HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets
HOST_LOGS=$LUSTRE/kimi-k2-logs/job-${SLURM_JOB_ID}

# --- Must be submitted, not executed -----------------------------------------
# The #SBATCH directives above are ordinary comments when this runs as a plain
# script. Without an allocation there is no node list to read, so it would
# otherwise fail further down with a confusing "could not resolve head node".
if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo ""
    echo "ERROR: this is a batch script — submit it, don't run it:"
    echo "  sbatch $(readlink -f "${BASH_SOURCE[0]}") $MODE"
    echo ""
    echo "It needs 64 nodes, so there is nothing it can do on the login node."
    echo ""
    exit 1
fi

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

# pyxis will not create mount sources, so they have to exist first. The log
# directory is per job, so the head's done-signal to the workers is never a
# leftover from an earlier run.
mkdir -p "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$HOST_LOGS"

# --- Resolve the head node IP (containers have no scontrol) ------------------
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
echo "  Logs      : $HOST_LOGS"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
#
# Steps 1-4 are identical to launch_on_cluster.sh.
# Steps 5-7 are the model setup, rank-gated.
# Step 8 launches instead of opening a shell.
#
# The heredoc is quoted, so nothing here is expanded at submit time. The values
# it needs are exported above and reach the container through the environment;
# SLURM_NODEID is set per task by Slurm.
#
INNER_CMD=$(cat <<'EOF'
set -ex

RANK="${SLURM_NODEID:-0}"
TRACKER=/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
DOWNLOAD_DONE=/root/models/.${MODEL}.download_complete

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- [rank $RANK] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" "fork/$MILES_BRANCH"

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [rank $RANK] Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork "$SGLANG_FORK" 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork "$SGLANG_BRANCH"
git -C /sgl-workspace/sglang checkout -B "$SGLANG_BRANCH" "fork/$SGLANG_BRANCH"

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
echo "--- [rank $RANK] Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- [rank $RANK] Active code ---"
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

if [ "$RANK" -eq 0 ]; then
    # ---- Step 5: Download datasets (skip if already present) ----------------
    # run.py passes both by absolute path, so the .jsonl files have to be
    # exactly where it expects.
    PROMPT_FILE=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
    if [ -f "$PROMPT_FILE" ]; then
        echo "--- Prompt dataset already present, skipping download ---"
    else
        echo "--- Downloading dataset dapo-math-17k ---"
        mkdir -p /root/datasets
        hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
    fi
    ls -la "$PROMPT_FILE"

    EVAL_FILE=/root/datasets/aime-2024/aime-2024.jsonl
    if [ -f "$EVAL_FILE" ]; then
        echo "--- Eval dataset already present, skipping download ---"
    else
        echo "--- Downloading dataset aime-2024 ---"
        hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024
    fi
    ls -la "$EVAL_FILE"

    # ---- Step 6: Download model (skip if already present) -------------------
    # ~1 TB. Keyed on its own sentinel, not on the converted checkpoint, so a
    # failed conversion does not drag the download check back in. hf download
    # resumes, so a rerun only fetches what is missing.
    MODEL_DIR=/root/models/$MODEL
    if [ -f "$DOWNLOAD_DONE" ]; then
        echo "--- Model already downloaded, skipping ---"
    else
        echo "--- Downloading model $HF_REPO ---"
        mkdir -p /root/models
        hf download "$HF_REPO" --local-dir "$MODEL_DIR"
        du -sh "$MODEL_DIR"
        touch "$DOWNLOAD_DONE"
    fi
fi

# ---- Step 7: Convert checkpoint (skip if already done) ----------------------
# Exactly the command from docs/models/kimi/kimi-k2.md § 3.3: 4 nodes, and no
# parallel-size arguments beyond MODEL_ARGS. An earlier version of this script
# ran it on rank 0 alone with the extra --expert-model-parallel-size 8 and
# --decoder-last-pipeline-num-layers 5 taken from PREPARE_CONFIGS in
# examples/p2p_weight_transfer/run.py, and Megatron rejected it before loading
# any weights: "world_size (8) is not divisible by
# expert_tensor_model_pipeline_parallel size (64)".
if [ "$RANK" -lt "$CONVERT_NODES" ]; then
    # Ranks 1..3 join the conversion, so they wait for rank 0's download.
    if [ "$RANK" -ne 0 ]; then
        echo "--- [rank $RANK] Waiting for rank 0 to finish downloading ---"
        while [ ! -f "$DOWNLOAD_DONE" ]; do
            sleep 30
        done
    fi

    if [ -f "$TRACKER" ]; then
        echo "--- [rank $RANK] Checkpoint already converted ($(cat $TRACKER)), skipping ---"
    else
        echo "--- [rank $RANK] Converting HF checkpoint across $CONVERT_NODES nodes ---"
        mkdir -p /root/multinode
        cd /root/miles
        source scripts/models/$MODEL_TYPE.sh
        PYTHONPATH=/root/Megatron-LM/ torchrun \
            --nproc-per-node 8 \
            --master-addr "$HEAD_NODE_IP" --master-port 12345 \
            --nnodes="$CONVERT_NODES" --node-rank "$RANK" \
            tools/convert_hf_to_torch_dist.py \
            "${MODEL_ARGS[@]}" \
            --hf-checkpoint /root/models/$MODEL \
            --save /root/multinode/${MODEL}_torch_dist
    fi
fi

# ---- All ranks: wait until the checkpoint is ready --------------------------
# convert_hf_to_torch_dist.py writes "release" into the tracker as its very last
# step, after a barrier, so it doubles as the ready signal for the 60 ranks that
# took no part in the conversion.
echo "--- [rank $RANK] Waiting for the converted checkpoint ---"
while [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; do
    sleep 30
done
echo "--- [rank $RANK] Checkpoint ready: $(cat $TRACKER) ---"

# ---- Step 8: Launch ---------------------------------------------------------
# Kimi-K2-Instruct refuses to run with --check-weight-update-equal, so weight
# validation is off (see docs/en/advanced/p2p-weight-transfer.md).
# MILES_LOG_DIR is on the shared mount: run.py has the head write
# job_done_<mode> there when training ends, and the workers poll for it.
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/logs

cd /root/miles
echo "--- [rank $RANK] Launching $MODEL, mode=$MODE, head=$HEAD_NODE_IP ---"
bash examples/p2p_weight_transfer/Kimi-K2.sh "$MODE" "$RANK" "$HEAD_NODE_IP"
EOF
)

# --- Launch ------------------------------------------------------------------
# No --mpi=pmix: nothing here is an MPI program. Ray does the bootstrapping —
# rank 0 starts the head, the rest join over TCP — so srun only has to start one
# container per node.
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$HOST_LOGS:/root/logs" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
