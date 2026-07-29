#!/bin/bash
# =============================================================================
# cast_kimi_k2_to_bf16_on_cluster.sh — step 1 of 3: fetch Kimi-K2 and de-quantise
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
# WHY A BF16 COPY IS NEEDED
# -------------------------
# moonshotai/Kimi-K2-Instruct ships as a block-scaled FP8 checkpoint. Pointing
# the Megatron conversion straight at it makes mbridge take its
# dequant_fp8_safetensor_io path, which de-quantises every tensor on the GPU
# through a Triton kernel while loading. That does not work here, in two ways at
# once:
#   * convert_hf_to_torch_dist.py loads with memory_efficient=True, which stages
#     tensors on the host, and the Triton kernel cannot take a host pointer:
#     "ValueError: Pointer argument (at 0) cannot be accessed from Triton
#     (cpu tensor?)".
#   * Where it does reach the GPU there is no room for it. The model shard alone
#     is ~70 GiB of a 79 GiB card, so the de-quant buffer OOMs.
#
# So the FP8 -> BF16 cast happens here instead, offline and once, exactly as
# every DeepSeek recipe in this repo does it (scripts/run_deepseek.py converts
# from "<model>-bf16", and docs/models/deepseek/deepseek.md documents the same
# two-stage flow that kimi-k2.md § 3.3 says to mirror).
#
# Only the *conversion* needs BF16. Training still reads the original FP8
# directory, which is why this script leaves it untouched.
#
# WHY ONE NODE
# ------------
# tools/fp8_cast_bf16.py is single-process and single-GPU: it walks the
# safetensors shards serially, loading each to one device, de-quantising, and
# writing it back out. Nothing here is distributed, so asking for the 8 nodes
# the conversion needs would leave 63 GPUs idle for hours.
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
# DISK: the BF16 copy is roughly 2 TB, on top of the ~1 TB FP8 original. Check
# there is room before submitting:
#   df -h /lustre/fsw/portfolios/network/users/$USER
#
# moonshotai/Kimi-K2-Instruct may be gated. If so, export HF_TOKEN before
# launching; the container runtime passes the environment through.
#
# USAGE
# -----
#   bash cast_kimi_k2_to_bf16_on_cluster.sh
#
#   squeue -u $USER
#   tail -f <LUSTRE>/kimi-k2-logs/miles-kimi-k2-cast-<jobid>.out
#
# To raise or lower the walltime without editing this file, sbatch reads it from
# the environment:
#   SBATCH_TIMELIMIT=04:00:00 bash cast_kimi_k2_to_bf16_on_cluster.sh
#
# The cast is NOT resumable — it has no per-shard sentinel, so a run killed by
# the walltime restarts from the first shard. If 4 hours turns out to be too
# tight, raise SBATCH_TIMELIMIT rather than resubmitting repeatedly.
#
# RESETTING
# ---------
# To redo the cast, delete the BF16 copy and resubmit:
#   rm -rf <LUSTRE>/models/Kimi-K2-Instruct-bf16
# To redo the download, delete the sentinel and the model:
#   rm -rf <LUSTRE>/models/Kimi-K2-Instruct <LUSTRE>/models/.Kimi-K2-Instruct.download_complete
# =============================================================================

# One node. Only one of its GPUs is used, but whole-node allocation avoids
# contending with another job for host memory and Lustre bandwidth during a
# multi-hour 3 TB of I/O.
#SBATCH --job-name=miles-kimi-k2-cast
#SBATCH --account=network_research_advdev
#SBATCH --partition=batch
#SBATCH --nodes=1
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
export HF_REPO=moonshotai/Kimi-K2-Instruct

export MILES_FORK=https://github.com/amitw-nv/miles.git
export MILES_BRANCH=amitw/miles-nixl

export SGLANG_FORK=https://github.com/amitw-nv/sglang.git
export SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/$USER/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=${USER}-miles-kimi-k2-cast

HOST_MODELS=$LUSTRE/models
HOST_DATASETS=$LUSTRE/datasets
HOST_LOGS=$LUSTRE/kimi-k2-logs

# --- Validate sqsh exists ----------------------------------------------------
# Lustre is visible from the login node, so this runs before the job is queued
# and again inside the job.
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
    echo "Submitting 1-node Kimi-K2 download + BF16 cast job..."
    exec sbatch "$(readlink -f "${BASH_SOURCE[0]}")" "$@"
fi

# --- Mount sources -----------------------------------------------------------
# Everything produced here has to land on Lustre so the 8-node conversion and
# the 64-node training job can see it. These container paths are the ones run.py
# hardcodes, so they are not free to change. Mounting the subdirectories is
# safe; mounting over /root or home breaks the Megatron import, hence
# --no-container-mount-home on the srun below.
#
# pyxis will not create mount sources, so they have to exist first.
mkdir -p "$HOST_MODELS" "$HOST_DATASETS" "$HOST_LOGS"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster — Kimi-K2 download + cast"
echo "=========================================="
echo "  Job       : $SLURM_JOB_ID"
echo "  Portfolio : $PORTFOLIO"
echo "  Model     : $HF_REPO"
echo "  BF16 copy : $HOST_MODELS/$MODEL_BF16"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
# The heredoc is quoted, so nothing in it is expanded at submit time. Everything
# it needs was exported above and reaches the container through the environment.
INNER_CMD=$(cat <<'EOF'
set -ex

MODEL_DIR=/root/models/$MODEL
BF16_DIR=/root/models/$MODEL_BF16
export BF16_DIR
DOWNLOAD_DONE=/root/models/.${MODEL}.download_complete

# ---- Step 1: Update Miles ---------------------------------------------------
# The sqsh has Miles cloned at /root/miles from the main branch at build time.
# Add the fork remote, discard any local edits so checkout cannot abort with
# "local changes would be overwritten", then fetch and check out.
echo "--- Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" fork/"$MILES_BRANCH"

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork "$SGLANG_FORK" 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork "$SGLANG_BRANCH"
git -C /sgl-workspace/sglang checkout -B "$SGLANG_BRANCH" fork/"$SGLANG_BRANCH"

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
# The sqsh base image ships numpy 2.x and scipy 1.18. Megatron asserts numpy<2
# at startup, and scipy>=1.15 requires numpy>=2, so both move together.
echo "--- Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- Active code ---"
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ---- Step 5: Download datasets ----------------------------------------------
# run.py passes both files by absolute path, so these destinations are not free
# to change. They match hf_download_dataset's layout.
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

# ---- Step 6: Download the FP8 model -----------------------------------------
# ~1 TB, gated on a sentinel written only after hf download returns
# successfully. Skipping on a merely non-empty directory would treat an
# interrupted transfer of this size as finished. hf download resumes, so a rerun
# after a failure only fetches what is missing.
#
# This directory stays FP8 and is left exactly as downloaded: it is what
# run.py hands SGLang as --hf-checkpoint at training time.
if [ -f "$DOWNLOAD_DONE" ]; then
    echo "--- Model already downloaded, skipping ---"
else
    echo "--- Downloading model $HF_REPO ---"
    hf download "$HF_REPO" --local-dir "$MODEL_DIR"
    du -sh "$MODEL_DIR"
    touch "$DOWNLOAD_DONE"
fi

# ---- Step 7: Cast FP8 -> BF16 -----------------------------------------------
# The reason this script exists; see the header for why the conversion cannot
# read FP8 directly. Going through command_utils rather than calling
# tools/fp8_cast_bf16.py by hand, because the helper already skips the work when
# the destination's model.safetensors.index.json exists — and the tool writes
# that file last, so it is a real completion marker rather than a guess.
#
# Expect one to three hours and about 2 TB written. The tool keeps only two
# safetensors files resident at a time, so host memory is not the constraint;
# Lustre bandwidth is.
echo "--- Casting FP8 -> BF16 (this is the long step) ---"
cd /root/miles
python3 -c "
from miles.utils.external_utils.command_utils import fp8_cast_bf16
fp8_cast_bf16(path_src='${MODEL_DIR}', path_dst='${BF16_DIR}')
"
du -sh "$BF16_DIR"

# ---- Step 8: Make the BF16 config honest ------------------------------------
# fp8_cast_bf16.py copies config.json across verbatim, which leaves two things
# wrong for the copy it just wrote:
#
#   * quantization_config still advertises block-scaled FP8, but there are no
#     FP8 tensors and no _scale_inv entries left in the index. Leaving it would
#     invite mbridge back into the de-quant path this whole script exists to
#     avoid.
#   * model_type is kimi_k2, which mbridge has no entry for — it fails with
#     "Unregistered model type: kimi_k2". Kimi-K2 is DeepSeek-V3-shaped and the
#     config already declares DeepseekV3ForCausalLM as its architecture, so
#     model_type is the only thing in the way. This is the `sed` that
#     docs/models/kimi/kimi-k2.md § 1 alludes to.
#
# Only the BF16 copy is touched. The FP8 original keeps its real model_type and
# quantization_config, because that is what SGLang reads during training.
python3 - <<'PY'
import json, os

cfg = os.path.join(os.environ["BF16_DIR"], "config.json")
with open(cfg) as f:
    conf = json.load(f)

changed = False
if conf.pop("quantization_config", None) is not None:
    print("--- Removed stale quantization_config from the BF16 config ---")
    changed = True
if conf.get("model_type") != "deepseek_v3":
    print(f"--- model_type {conf.get('model_type')!r} -> 'deepseek_v3' ---")
    conf["model_type"] = "deepseek_v3"
    changed = True

if changed:
    with open(cfg, "w") as f:
        json.dump(conf, f, indent=2, ensure_ascii=False)
else:
    print("--- BF16 config already correct ---")

print("model_type   :", conf.get("model_type"))
print("architectures:", conf.get("architectures"))
print("quantization :", conf.get("quantization_config", "(none)"))
PY

echo "--- Cast done. Next: prepare_kimi_k2_on_cluster.sh ---"
EOF
)

# --- Run ---------------------------------------------------------------------
srun \
    --ntasks-per-node=1 \
    --container-image="$SQSH" \
    --container-name="$C_NAME" \
    --container-mounts="$HOST_MODELS:/root/models,$HOST_DATASETS:/root/datasets" \
    --container-workdir=/root/miles \
    --no-container-mount-home \
    bash -c "$INNER_CMD"
