#!/bin/bash
# doc-dev: docs/advanced/nixl_weight_transfer_feature/launch_kimi.md
# =============================================================================
# launch_kimi_k2_on_cluster.sh — 64-node (512-GPU) Kimi-K2 job on the cluster
# =============================================================================
#
# launch_kimi.md is the spec for this file. Change the doc first, then conform
# this script to it.
#
# WHAT IT DOES
# ------------
#   0. Allocates 64 nodes with sbatch and starts one container per node.
#   1-4. Updates Miles + SGLang to the fork branches, pins numpy/scipy, prints
#        the active commits.  All 64 nodes.
#   5-6. Downloads the datasets and the ~1 TB model.  Rank 0 only.
#   7. Converts the HF checkpoint to Megatron torch_dist across 4 nodes / 32
#      GPUs, then every rank waits for the result.
#   8. Launches the training job.
#
# Every step is idempotent: it checks for its own result first and skips
# instantly when it is already there. On the first launch the download and the
# conversion dominate; afterwards both are on Lustre and every rank goes almost
# straight to step 8.
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
# Slurm needs the --output directory to exist before the job starts, so once:
#   mkdir -p /lustre/fsw/portfolios/network/users/$USER/miles/kimi-k2-logs
#
# moonshotai/Kimi-K2-Instruct may be gated. If so, export HF_TOKEN before
# submitting; the container runtime passes the environment through.
#
# USAGE
# -----
#   sbatch launch_kimi_k2_on_cluster.sh            # nixl (default)
#   sbatch launch_kimi_k2_on_cluster.sh p2p        # mooncake
#   sbatch launch_kimi_k2_on_cluster.sh broadcast
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-<jobid>.out
#
# The --time below assumes the model and the converted checkpoint are already on
# Lustre. A genuine first launch has to cover the ~1 TB download and the
# conversion too, so raise it along with PREP_WAIT_TIMEOUT and
# CONVERT_WAIT_TIMEOUT.
#
# RESETTING
# ---------
# To force a fresh conversion, delete the checkpoint and resubmit:
#   rm -rf <LUSTRE>/multinode/Kimi-K2-Instruct_torch_dist
# To force a fresh download, delete the sentinel and the model:
#   rm -rf <LUSTRE>/models/Kimi-K2-Instruct <LUSTRE>/models/.Kimi-K2-Instruct.download_complete
# To rebuild the container image, delete and re-import the sqsh:
#   rm -f <LUSTRE>/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest
# =============================================================================

# 64 nodes is what RUN_CONFIGS["Kimi-K2-Instruct"] in
# examples/p2p_weight_transfer/run.py declares: 256 train + 256 rollout GPUs.
# sbatch rather than an interactive srun --pty, because 512 GPUs will not be
# handed to an interactive session — which is also why step 8 launches the job
# itself instead of dropping into a shell.
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
PORTFOLIO=network_research_advdev

export MODE="${1:-nixl}"          # nixl | p2p | broadcast

export MODEL=Kimi-K2-Instruct
export HF_REPO=moonshotai/Kimi-K2-Instruct
export MODEL_TYPE=kimi-k2         # selects scripts/models/kimi-k2.sh

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

# Ranks 0..CONVERT_NODES-1 convert the checkpoint as one torchrun job; every
# other rank waits for the result. 8 GPUs x 4 nodes = the 32 the parallel sizes
# in step 7 are derived for, so this cannot be changed on its own.
export CONVERT_NODES=4

# How long a rank that only waits is willing to wait, in seconds. Rank 0 is a
# single point of failure for both waits: if it dies mid-download, the other 63
# nodes would otherwise poll until the walltime expired, burning 512 GPUs on a
# job that has already failed. Two knobs because the waits scale differently —
# the first covers a 1 TB download, the second a 32-GPU conversion.
export PREP_WAIT_TIMEOUT="${PREP_WAIT_TIMEOUT:-7200}"
export CONVERT_WAIT_TIMEOUT="${CONVERT_WAIT_TIMEOUT:-7200}"

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2

# --- Must be submitted, not executed -----------------------------------------
# The #SBATCH directives above are ordinary comments when this file is executed
# directly. Without an allocation there is no node list to read, so it would
# otherwise fail much further down with a confusing error about the head node.
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

# --- Mount sources -----------------------------------------------------------
# The model, the datasets and the converted checkpoint have to live on Lustre:
# a named container persists /root between sessions on one node, but the other
# 63 cannot see it. These four container paths are the ones run.py hardcodes,
# so they are not free to change. Mounting the subdirectories is safe; mounting
# over /root or home breaks the Megatron import, hence --no-container-mount-home
# on the srun below.
#
# The log directory is per job. That is what makes the prep_done signal of step 7
# impossible to inherit from an earlier run, which would otherwise release the
# converting ranks immediately.
HOST_MODELS=$LUSTRE/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets
HOST_LOGS=$LUSTRE/kimi-k2-logs/job-${SLURM_JOB_ID}

# pyxis will not create mount sources, so they have to exist first.
mkdir -p "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$HOST_LOGS"

# --- Resolve the head node IP ------------------------------------------------
# Has to happen out here: the container has no scontrol. kimi-k2.md § 3.1 asks
# for MASTER_ADDR and NODE_RANK to be exported by hand; in a batch job the
# master address is the first host of the allocation and the rank comes from the
# scheduler, so the script resolves both itself.
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
# The heredoc is quoted, so nothing in it is expanded at submit time. Everything
# it needs was exported above and reaches the container through the environment;
# SLURM_NODEID is set per task by Slurm and is the RANK the steps use to decide
# what they do.
INNER_CMD=$(cat <<'EOF'
set -ex

RANK="${SLURM_NODEID:-0}"

MODEL_DIR=/root/models/$MODEL
export MODEL_DIR
DOWNLOAD_DONE=/root/models/.${MODEL}.download_complete
TRACKER=/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
PREP_DONE=/root/logs/prep_done

# wait_for <label> <path> <timeout-seconds> <knob-name> [expected-contents]
#
# Polls every 30s until the file exists and, when expected-contents is given,
# matches once whitespace is stripped. On expiry it names the knob to raise and
# exits non-zero, so a dead rank 0 surfaces as a failure on every rank instead
# of a wait that quietly runs out the walltime.
wait_for() {
    local label="$1" path="$2" timeout="$3" knob="$4" want="${5:-}"
    local waited=0
    echo "--- [rank $RANK] Waiting for $label, up to ${timeout}s ---"
    while true; do
        if [ -f "$path" ] &&
           { [ -z "$want" ] || [ "$(tr -d '[:space:]' < "$path")" = "$want" ]; }; then
            return 0
        fi
        if [ "$waited" -ge "$timeout" ]; then
            echo "ERROR: [rank $RANK] gave up after ${waited}s waiting for $label." >&2
            echo "       Missing or incomplete: $path" >&2
            echo "       Rank 0 most likely failed — look for its error earlier in this log." >&2
            echo "       If this was a genuine first launch, raise $knob and --time." >&2
            exit 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

# ---- Step 1: Update Miles ---------------------------------------------------
# The image ships Miles at /root/miles on whatever branch it was built from.
# Switch it to the fork branch: add the remote (only new the first time, so a
# failure is tolerated), discard local edits so the checkout cannot fail on a
# dirty tree, then force the local branch to the fetched head. `restore` falls
# back to `checkout` for older git.
echo "--- [rank $RANK] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" "fork/$MILES_BRANCH"

# ---- Step 2: Update SGLang --------------------------------------------------
# The same four commands against /sgl-workspace/sglang.
echo "--- [rank $RANK] Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork "$SGLANG_FORK" 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork "$SGLANG_BRANCH"
git -C /sgl-workspace/sglang checkout -B "$SGLANG_BRANCH" "fork/$SGLANG_BRANCH"

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
# The image ships versions Miles cannot use. Pin them before anything imports
# them. pip install is idempotent, so this is safe to re-run.
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

# Steps 5, 6 and 7a are rank 0's alone. The artifacts are on shared storage, so
# exactly one node may write them — unguarded, all 64 would download the same
# 1 TB model into the same directory at once.
if [ "$RANK" -eq 0 ]; then
    # ---- Step 5: Download datasets ------------------------------------------
    # run.py passes both files by absolute path, so these destinations are not
    # free to change. Skip on the .jsonl being there, and ls -la it afterwards
    # so the log shows the size.
    PROMPT_FILE=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
    if [ -f "$PROMPT_FILE" ]; then
        echo "--- Prompt dataset already present, skipping download ---"
    else
        echo "--- Downloading dataset dapo-math-17k ---"
        mkdir -p /root/datasets
        hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
    fi
    ls -la "$PROMPT_FILE"

    # The eval set is the one addition to the GLM recipe, which downloads only
    # the prompt set: the Kimi recipe evaluates on AIME-2024 (kimi-k2.md § 3.2).
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
    # checkpoint of step 7, so a failed conversion does not drag the 1 TB back
    # into scope.
    if [ -f "$DOWNLOAD_DONE" ]; then
        echo "--- Model already downloaded, skipping ---"
    else
        echo "--- Downloading model $HF_REPO ---"
        mkdir -p /root/models
        hf download "$HF_REPO" --local-dir "$MODEL_DIR"
        du -sh "$MODEL_DIR"
        touch "$DOWNLOAD_DONE"
    fi

    # ---- Step 7a: Point the config at the DeepSeek-V3 loader ----------------
    # mbridge has no kimi_k2 entry and fails with "Unregistered model type:
    # kimi_k2, now only support dict_keys([...])". Kimi-K2 is DeepSeek-V3-shaped
    # and its HF config already declares DeepseekV3ForCausalLM as the
    # architecture, so only model_type is in the way — this is the `sed` that
    # kimi-k2.md § 1 alludes to.
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

    # Only now may ranks 1..3 start. Signalling after the patch has returned is
    # what keeps a converting rank from reading a half-written config.json, or
    # from reading it before the rewrite and hitting the mbridge error above.
    touch "$PREP_DONE"
fi

# ---- Step 7b: Convert the checkpoint to Megatron torch_dist -----------------
# A torchrun job spanning CONVERT_NODES nodes and 32 GPUs, which is what makes
# it possible at all: no single GPU can hold a trillion parameters. The pipeline
# size splits the 61 layers into stages and the expert-parallel size splits each
# layer's experts, so a rank only ever builds its own slice. Each rank writes
# its own shard into the shared /root/multinode mount, so the nodes
# cooperatively produce one checkpoint directory.
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
if [ "$RANK" -lt "$CONVERT_NODES" ]; then
    # Ranks 1..3 take part in the conversion but not in the download, so they
    # wait for rank 0 to finish fetching the model and patching its config.
    if [ "$RANK" -ne 0 ]; then
        wait_for "rank 0 to prepare the HF checkpoint" \
                 "$PREP_DONE" "$PREP_WAIT_TIMEOUT" PREP_WAIT_TIMEOUT
    fi

    if [ -f "$TRACKER" ]; then
        echo "--- [rank $RANK] Checkpoint already converted ($(cat $TRACKER)), skipping ---"
    else
        echo "--- [rank $RANK] Converting HF checkpoint across $CONVERT_NODES nodes ---"
        mkdir -p /root/multinode
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
            --hf-checkpoint /root/models/$MODEL \
            --save /root/multinode/${MODEL}_torch_dist
    fi
fi

# ---- Step 7d: Every rank waits for the checkpoint ---------------------------
# convert_hf_to_torch_dist.py writes "release" into the tracker as its very last
# action, after a barrier, so the file means the checkpoint is complete and
# usable. That makes it both the skip condition above and the ready signal for
# the 60 ranks that took no part in the conversion. Match on the contents rather
# than on the file existing, so a tracker mid-write is never mistaken for a
# finished checkpoint.
wait_for "the converted checkpoint" \
         "$TRACKER" "$CONVERT_WAIT_TIMEOUT" CONVERT_WAIT_TIMEOUT release
echo "--- [rank $RANK] Checkpoint ready: $(cat $TRACKER) ---"

# ---- Step 8: Launch ---------------------------------------------------------
# Every rank runs this with its own RANK; Kimi-K2.sh forwards all three
# arguments to run.py, which starts the Ray head on rank 0 and joins the rest
# to it.
#
# SKIP_VALIDATION is required: run.py refuses to run Kimi-K2-Instruct with
# --check-weight-update-equal unless it is set, so weight validation is off
# (see docs/advanced/p2p-weight-transfer.md).
#
# MILES_LOG_DIR has to be the shared mount. run.py has the head write
# job_done_<mode> there when training ends and the workers poll for it; on a
# container-local path they would never see it. Its default, /data/ray/signals,
# is not mounted.
#
# Kimi-K2.sh also runs `run.py prepare` itself when the torch_dist directory is
# missing. After step 7 it never is, which is the point of converting here:
# prepare would use the PREPARE_CONFIGS sizes ruled out above.
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/logs

cd /root/miles
echo "--- [rank $RANK] Launching $MODEL, mode=$MODE, head=$HEAD_NODE_IP ---"
bash examples/p2p_weight_transfer/Kimi-K2.sh "$MODE" "$RANK" "$HEAD_NODE_IP"
EOF
)

# --- Launch ------------------------------------------------------------------
# One container per node, and no --mpi=pmix: nothing here is an MPI program.
# Ray does the bootstrapping — rank 0 starts the head, the rest join over TCP.
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$HOST_LOGS:/root/logs" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
