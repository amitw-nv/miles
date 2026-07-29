#!/bin/bash
# =============================================================================
# prepare_kimi_k2_on_cluster.sh — one-off Kimi-K2 download + checkpoint convert
# =============================================================================
#
# Companion to launch_kimi_k2_on_cluster.sh. Run this one FIRST, once. It puts
# the model, the datasets and the converted Megatron checkpoint on Lustre, where
# the 64-node launch job then finds them.
#
# WHY THIS IS A SEPARATE SCRIPT
# -----------------------------
# The download is ~1 TB and the conversion takes tens of minutes. Doing either
# inside the 64-node job would hold 512 GPUs idle for hours. Conversion needs 32
# GPUs, so 4 nodes is all this asks for.
#
# WHAT IT DOES
# ------------
#   1-4. Update Miles + SGLang to the fork branches, pin numpy/scipy, print the
#        active commits.  All 4 nodes.
#   5-7. Download the datasets, download the ~1 TB model, patch its config.json
#        for mbridge.  Rank 0 only; ranks 1-3 wait.
#   8.   Convert the HF checkpoint to Megatron torch_dist across 4 nodes / 32
#        GPUs.
#
# Every step is idempotent and checks for its own result first, so a rerun after
# a failure or a walltime expiry only redoes what is missing. `hf download`
# resumes, so an interrupted transfer is not restarted from zero.
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
# moonshotai/Kimi-K2-Instruct may be gated. If so, export HF_TOKEN before
# launching; the container runtime passes the environment through.
#
# USAGE
# -----
#   bash prepare_kimi_k2_on_cluster.sh     # from the login node; self-submits
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-prep-<jobid>.out
#
# If the partition caps walltime below the --time below, just resubmit after it
# expires — the download resumes and the finished steps are skipped.
#
# RESETTING
# ---------
# To force a fresh conversion, delete the checkpoint and resubmit:
#   rm -rf <LUSTRE>/multinode/Kimi-K2-Instruct_torch_dist
# To force a fresh download, delete the sentinel and the model:
#   rm -rf <LUSTRE>/models/Kimi-K2-Instruct <LUSTRE>/models/.Kimi-K2-Instruct.download_complete
# =============================================================================

# 4 nodes = the 32 GPUs the parallel sizes in step 8 are derived for. sbatch
# rather than an interactive srun --pty, because the download outlives any
# reasonable interactive session.
#SBATCH --job-name=miles-kimi-k2-prep
#SBATCH --account=network_research_advdev
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.out
#SBATCH --error=/lustre/fsw/portfolios/network/users/%u/miles/kimi-k2-logs/%x-%j.err

set -e

# --- Config ------------------------------------------------------------------
# The #SBATCH directives above are parsed by sbatch before any of this runs, so
# node count, account and partition have to be literals up there. Keep the two
# in sync by hand.
PORTFOLIO=network_research_advdev

export MODEL=Kimi-K2-Instruct
export HF_REPO=moonshotai/Kimi-K2-Instruct
export MODEL_TYPE=kimi-k2          # selects scripts/models/kimi-k2.sh

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

# Ranks 0..CONVERT_NODES-1 convert the checkpoint as one torchrun job. 8 GPUs x
# 4 nodes = the 32 the parallel sizes in step 8 are derived for, so this cannot
# be changed without re-deriving them and the #SBATCH --nodes above.
export CONVERT_NODES=4

# How long ranks 1-3 wait for rank 0 to finish downloading, in seconds. Rank 0
# is a single point of failure here: without a bound, a rank 0 that died mid
# download would leave the other three polling until the walltime expired.
export PREP_WAIT_TIMEOUT="${PREP_WAIT_TIMEOUT:-21600}"

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2-prep

HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets
HOST_LOGS=$LUSTRE/kimi-k2-logs

# --- Validate sqsh exists ----------------------------------------------------
# Lustre is visible from the login node, so this runs before the job is queued
# on the submit pass and again inside the job.
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

# --- Submit pass: re-exec under sbatch ---------------------------------------
# Run from the login node there is no allocation, and the #SBATCH lines above
# are just comments. Create the log directory Slurm needs and hand the file to
# sbatch, which re-reads those directives for real.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "$HOST_LOGS"
    echo "Submitting 4-node Kimi-K2 prepare job..."
    exec sbatch "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# --- Mount sources -----------------------------------------------------------
# The model, the datasets and the converted checkpoint have to live on Lustre:
# a named container persists /root between sessions on one node, but the other
# three cannot see it, and neither can the 64 nodes of the launch job. These
# three container paths are the ones run.py hardcodes, so they are not free to
# change. Mounting the subdirectories is safe; mounting over /root or home
# breaks the Megatron import, hence --no-container-mount-home on the srun below.
#
# pyxis will not create mount sources, so they have to exist first.
mkdir -p "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$HOST_LOGS"

# The prep signal of step 7 is per job, so a stale one from an earlier run can
# never release the converting ranks early.
HOST_SIGNALS=$HOST_LOGS/prep-job-${SLURM_JOB_ID}
mkdir -p "$HOST_SIGNALS"

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
echo "  Miles cluster prepare — Kimi-K2"
echo "=========================================="
echo "  Job       : $SLURM_JOB_ID"
echo "  Portfolio : $PORTFOLIO"
echo "  Nodes     : $SLURM_JOB_NUM_NODES (32 GPUs for the conversion)"
echo "  Head node : $HEAD_NODE ($HEAD_NODE_IP)"
echo "  Model     : $HF_REPO"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Models    : $HOST_MODELS"
echo "  Checkpoint: $HOST_CKPT"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
# The heredoc is quoted, so nothing in it is expanded at submit time. Everything
# it needs was exported above and reaches the container through the environment;
# SLURM_NODEID is set per task by Slurm and is the rank the steps key off.
INNER_CMD=$(cat <<'EOF'
set -ex

RANK="${SLURM_NODEID:-0}"

MODEL_DIR=/root/models/$MODEL
export MODEL_DIR
DOWNLOAD_DONE=/root/models/.${MODEL}.download_complete
TRACKER=/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
PREP_DONE=/root/signals/prep_done

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

# Steps 5, 6 and 7 are rank 0's alone. The artifacts are on shared storage, so
# exactly one node may write them — unguarded, all four would download the same
# 1 TB model into the same directory at once.
if [ "$RANK" -eq 0 ]; then
    # ---- Step 5: Download datasets ------------------------------------------
    # run.py passes both files by absolute path, so these destinations are not
    # free to change. They match hf_download_dataset's layout.
    PROMPT_FILE=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
    if [ -f "$PROMPT_FILE" ]; then
        echo "--- Prompt dataset already present, skipping download ---"
    else
        echo "--- Downloading dataset dapo-math-17k ---"
        hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
    fi
    ls -la "$PROMPT_FILE"

    # The Kimi recipe evaluates on AIME-2024, so the eval set is needed too.
    EVAL_FILE=/root/datasets/aime-2024/aime-2024.jsonl
    if [ -f "$EVAL_FILE" ]; then
        echo "--- Eval dataset already present, skipping download ---"
    else
        echo "--- Downloading dataset aime-2024 ---"
        hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024
    fi
    ls -la "$EVAL_FILE"

    # ---- Step 6: Download model ---------------------------------------------
    # ~1 TB, gated on a sentinel written only after hf download returns
    # successfully. Skipping on a merely non-empty directory, as the GLM recipe
    # does, would treat an interrupted transfer of this size as finished. hf
    # download resumes, so a rerun after a failure only fetches what is missing.
    #
    # The sentinel is keyed on the download alone, not on the converted
    # checkpoint of step 8, so a failed conversion does not drag the 1 TB back
    # into scope.
    if [ -f "$DOWNLOAD_DONE" ]; then
        echo "--- Model already downloaded, skipping ---"
    else
        echo "--- Downloading model $HF_REPO ---"
        hf download "$HF_REPO" --local-dir "$MODEL_DIR"
        du -sh "$MODEL_DIR"
        touch "$DOWNLOAD_DONE"
    fi

    # ---- Step 7: Point the config at the DeepSeek-V3 loader -----------------
    # mbridge has no kimi_k2 entry and fails with "Unregistered model type:
    # kimi_k2". Kimi-K2 is DeepSeek-V3-shaped and its HF config already declares
    # DeepseekV3ForCausalLM as the architecture, so only model_type is in the
    # way — this is the `sed` that docs/models/kimi/kimi-k2.md § 1 alludes to.
    #
    # A JSON load and dump rather than a text substitution, idempotent by
    # returning early when the value is already deepseek_v3, and it keeps a copy
    # of the original the first time.
    python3 - <<'PY'
import json, os, shutil

cfg = os.path.join(os.environ["MODEL_DIR"], "config.json")
with open(cfg) as f:
    conf = json.load(f)

if conf.get("model_type") == "deepseek_v3":
    print("--- config.json already patched (model_type=deepseek_v3) ---")
else:
    print(f"--- Patching config.json: model_type {conf.get('model_type')!r} -> 'deepseek_v3' ---")
    shutil.copy2(cfg, cfg + ".orig")
    conf["model_type"] = "deepseek_v3"
    with open(cfg, "w") as f:
        json.dump(conf, f, indent=2, ensure_ascii=False)

print("architectures:", conf.get("architectures"))
PY

    # Only now may ranks 1-3 start. Signalling after the patch has returned is
    # what keeps a converting rank from reading a half-written config.json, or
    # from reading it before the rewrite and hitting the mbridge error above.
    touch "$PREP_DONE"
else
    # Ranks 1-3 take part in the conversion but not in the download, so they
    # wait for rank 0 to finish fetching the model and patching its config.
    # Polling with a bound, so a dead rank 0 surfaces as a failure on every rank
    # instead of a wait that quietly runs out the walltime.
    waited=0
    echo "--- [rank $RANK] Waiting for rank 0 to prepare the HF checkpoint, up to ${PREP_WAIT_TIMEOUT}s ---"
    until [ -f "$PREP_DONE" ]; do
        if [ "$waited" -ge "$PREP_WAIT_TIMEOUT" ]; then
            echo "ERROR: [rank $RANK] gave up after ${waited}s waiting for $PREP_DONE." >&2
            echo "       Rank 0 probably failed — check the head of this log." >&2
            echo "       If the download is merely slow, raise PREP_WAIT_TIMEOUT." >&2
            exit 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
fi

# ---- Step 8: Convert the checkpoint to Megatron torch_dist ------------------
# A torchrun job spanning 4 nodes and 32 GPUs, which is what makes it possible
# at all: no single GPU can hold a trillion parameters. The pipeline size splits
# the 61 layers into stages and the expert-parallel size splits each layer's
# experts, so a rank only ever builds its own slice. Each rank writes its own
# shard into the shared /root/multinode mount, so the nodes cooperatively
# produce one checkpoint directory.
#
# The five parallel sizes are NOT optional, and kimi-k2.md § 3.3 omits them.
# Both ways of getting this wrong have already been tried:
#   * No parallel sizes, as printed in § 3.3. Nothing is sharded, every rank
#     builds the whole model, and it dies in load_weights with CUDA OOM at
#     79 GiB.
#   * The PREPARE_CONFIGS values from examples/p2p_weight_transfer/run.py on a
#     single node. Megatron rejects it before loading any weights: "world_size
#     (8) is not divisible by expert_tensor_model_pipeline_parallel size (64)".
#
# Two constraints fix the values below, and changing one means re-deriving the
# rest:
#   * ETP * EP * PP must equal the GPU count, 8 * 4 nodes = 32. Here
#     1 * 4 * 8 = 32. Megatron rejects any other product outright.
#   * The pipeline split must cover all 61 layers (scripts/models/kimi-k2.sh
#     sets --num-layers 61). Seven stages of 8 plus a last stage of 5 is 61,
#     hence --decoder-last-pipeline-num-layers 5.
#
# They mirror docs/models/deepseek/deepseek.md § 3.3, which is what § 3.3 of the
# Kimi page points at when it says to mirror the DeepSeek-V3 procedure. Do not
# reach for the training-side table in kimi-k2.md § 5.1 instead: those sizes
# (TP 8, EP 32) are for the 256-GPU training run, not for this 32-GPU conversion.
if [ -f "$TRACKER" ] && [ "$(tr -d '[:space:]' < "$TRACKER")" = "release" ]; then
    echo "--- [rank $RANK] Checkpoint already converted, skipping ---"
else
    echo "--- [rank $RANK] Converting HF checkpoint across $CONVERT_NODES nodes ---"
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
        --expert-model-parallel-size 4 \
        --decoder-last-pipeline-num-layers 5 \
        --hf-checkpoint "$MODEL_DIR" \
        --save /root/multinode/${MODEL}_torch_dist
fi

# convert_hf_to_torch_dist.py writes "release" into the tracker as its very last
# action, after a barrier, so this both proves the checkpoint is complete and is
# what the launch job checks for before asking for 64 nodes.
echo "--- [rank $RANK] Checkpoint tracker: $(cat "$TRACKER") ---"
echo "--- [rank $RANK] Prepare done. Now submit launch_kimi_k2_on_cluster.sh ---"
EOF
)

# --- Run ---------------------------------------------------------------------
# One container per node, and no --mpi=pmix: torchrun does its own rendezvous
# over the master address resolved above.
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$HOST_SIGNALS:/root/signals" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
