#!/bin/bash
# =============================================================================
# download_kimi_k2.sh — fetch Kimi-K2-Instruct onto Lustre, into a fresh dir
# =============================================================================
#
# Companion to launch_kimi_k2_64node.sh, which deliberately never downloads the
# model ("This script does not download the model."). This one does that single
# job and nothing else.
#
# WHERE IT WRITES
# ---------------
#   /lustre/fsw/portfolios/network/users/amitw/models/new-download-kimi-k2
#
# That directory is the only thing this script creates or writes. The existing
# model directories under $HOST_MODELS — Kimi-K2-Instruct, the GLM and Qwen
# weights, any *-bf16 casts — are never read, moved, or deleted.
#
# SIZE AND TIME
# -------------
# Kimi-K2-Instruct is FP8, roughly 1 TB across ~60 safetensors shards. Make sure
# the Lustre quota has that much headroom before starting; note that the later
# convert step in launch_kimi_k2_64node.sh adds a ~2 TB BF16 cast on top.
#
# The default partition is cpu_datamover because it has no time limit. The 4 h
# cap on interactive/batch is a genuine risk at this size: finishing inside it
# needs a sustained ~70 MB/s with zero margin.
#
# Interrupted transfers resume. `hf download` skips shards that are already
# complete, so re-running this script after a timeout or a dead node continues
# where it stopped instead of starting over.
#
# PREREQUISITES
# -------------
# The same sqsh the launcher uses:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
# If it is missing, build it from a compute node (the login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive \
#        --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# For a gated or rate-limited repo, `export HF_TOKEN=...` before running. sbatch
# forwards the submitting environment, so the token reaches the job without ever
# being written into the generated .slurm file.
#
# USAGE
# -----
#   bash download_kimi_k2.sh [PARTITION]
#     PARTITION : cpu_datamover (default) | batch | interactive
#
# MONITORING
# ----------
#   squeue -u amitw
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-download-<JOBID>.out
#   scancel <JOBID>
#
# WIRING IT INTO THE LAUNCHER
# ---------------------------
# launch_kimi_k2_64node.sh resolves $HOST_MODELS/Kimi-K2-Instruct, so a download
# under a different name is invisible to it by design — that is what keeps this
# safe to run alongside the weights already in place. Once you have verified the
# new copy and want to switch over:
#   cd /lustre/fsw/portfolios/network/users/amitw/models
#   mv Kimi-K2-Instruct Kimi-K2-Instruct.old      # or rm, if it is a stale symlink
#   ln -sfn new-download-kimi-k2 Kimi-K2-Instruct
# =============================================================================

set -euo pipefail

# --- Config ------------------------------------------------------------------
REPO_ID=moonshotai/Kimi-K2-Instruct
REVISION=main

# Deliberately not "Kimi-K2-Instruct": a fresh name cannot collide with the
# weights the launcher already points at.
DEST_NAME=new-download-kimi-k2

PORTFOLIO=network_research_advdev
PARTITION="${1:-cpu_datamover}"
TIME_LIMIT=4:00:00

# cpu_datamover has no GPUs. Set to 1 if you retarget this at batch/interactive.
GPUS_PER_NODE=0

# Parallel shard downloads. 16 saturates the datamover link without thrashing
# Lustre; the hf default of 8 leaves throughput on the table for a 1 TB repo.
MAX_WORKERS=16

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
LOG_DIR=$LUSTRE/logs
C_NAME=amitw-miles-nohome-download

# Shared across models (GLM, Qwen, Kimi), which is why it sits outside $LUSTRE.
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
DEST=$HOST_MODELS/$DEST_NAME

# Mounted as one tree at the container path, matching the launcher's convention.
C_MODELS=/root/models
C_DEST=$C_MODELS/$DEST_NAME

# --- Preflight ---------------------------------------------------------------
fail() {
    echo ""
    echo "ERROR: $1"
    echo ""
    shift
    for line in "$@"; do echo "  $line"; done
    echo ""
    exit 1
}

[ -f "$SQSH" ] || fail "sqsh file not found at $SQSH" \
    "To create it, run on a compute node (login node has no disk space):" \
    "srun -A $PORTFOLIO -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash" \
    "cd $LUSTRE/" \
    "enroot import docker://radixark/miles:latest"

[ -d "$HOST_MODELS" ] || fail "models directory not found at $HOST_MODELS" \
    "Set HOST_MODELS to the Lustre directory that holds the weights."

# Guard the promise made in the header: never write over a sibling model.
case "$DEST_NAME" in
    ""|.|..|*/*) fail "DEST_NAME must be a plain directory name, got '$DEST_NAME'" ;;
esac
if [ -e "$DEST" ] && [ ! -d "$DEST" ]; then
    fail "$DEST exists and is not a directory" \
         "Move it aside, or pick a different DEST_NAME."
fi

# hf writes the shard index last, so its presence means the transfer finished.
if [ -f "$DEST/model.safetensors.index.json" ]; then
    echo "Kimi-K2 already present at $DEST — nothing to do."
    echo "Delete the directory to force a re-download."
    exit 0
fi

if [ -d "$DEST" ]; then
    echo "Found a partial download at $DEST — the job will resume it."
fi

mkdir -p "$LOG_DIR"

# --- Submit ------------------------------------------------------------------
JOB_SCRIPT=$LOG_DIR/job-kimi-k2-download.slurm
GPU_LINE=""
if [ "$GPUS_PER_NODE" -gt 0 ]; then
    GPU_LINE="#SBATCH --gpus-per-node=$GPUS_PER_NODE"
fi

cat > "$JOB_SCRIPT" <<DL_SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2-download
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=$TIME_LIMIT
#SBATCH --output=$LOG_DIR/miles-kimi-k2-download-%j.out
$GPU_LINE

srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}" \\
    --container-mounts="$HOST_MODELS:$C_MODELS" \\
    bash -lc '
set -euo pipefail

# Xet is the fast path; hf_transfer is retired in this huggingface_hub.
export HF_XET_HIGH_PERFORMANCE=1
# Belt and braces. --local-dir streams straight to the destination and does not
# touch the hub cache, so this should never fill up; pointing it inside the
# destination just guarantees that nothing can spill into a sibling model dir.
# Note this is the env var, not --cache-dir, which the CLI refuses alongside
# --local-dir.
export HF_HUB_CACHE=$C_DEST/.hf-cache

echo "--- [Download] node=\$(hostname) repo=$REPO_ID revision=$REVISION ---"
echo "--- [Download] dest=$C_DEST ---"

hf download $REPO_ID \\
    --revision $REVISION \\
    --local-dir $C_DEST \\
    --max-workers $MAX_WORKERS

echo "--- [Verify] Checking every shard named by the index is on disk ---"
python3 - <<PY
import json, os, sys

d = "$C_DEST"
index = os.path.join(d, "model.safetensors.index.json")
if not os.path.exists(index):
    sys.exit("FAIL: no model.safetensors.index.json — download is incomplete")

with open(index) as fh:
    idx = json.load(fh)

shards = sorted(set(idx["weight_map"].values()))
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
if missing:
    sys.exit("FAIL: %d/%d shards missing, first=%s" % (len(missing), len(shards), missing[0]))

on_disk = sum(os.path.getsize(os.path.join(d, s)) for s in shards)
expected = idx.get("metadata", {}).get("total_size")
print("shards=%d bytes_on_disk=%d expected=%s" % (len(shards), on_disk, expected))
if expected is not None and on_disk < expected:
    sys.exit("FAIL: short by %d bytes — download is truncated" % (expected - on_disk))
print("verify_ok")
PY

echo "--- [Smoke] Offline config/tokenizer load (advisory) ---"
# Kimi-K2 ships a custom TikTokenTokenizer through auto_map, so both loads need
# trust_remote_code. Advisory only: it depends on tiktoken/blobfile being in the
# image, and a missing dev dependency must not condemn a 1 TB transfer.
python3 - <<PY || echo "WARN: smoke test failed — weights verified above, check deps"
import os
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoConfig, AutoTokenizer

d = "$C_DEST"
cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
tok = AutoTokenizer.from_pretrained(d, trust_remote_code=True)
print("offline_resolve_ok arch=%s vocab=%d" % (cfg.architectures, len(tok)))
PY

echo "--- [Download] Done: $DEST ---"
'
DL_SLURM

JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

echo ""
echo "=========================================="
echo "  Kimi-K2 download"
echo "=========================================="
echo "  Repo      : $REPO_ID ($REVISION)"
echo "  Dest      : $DEST"
echo "  Partition : $PARTITION"
echo "  Time limit: $TIME_LIMIT"
echo "  Job ID    : $JOB_ID"
echo "=========================================="
echo ""
echo "  tail -f $LOG_DIR/miles-kimi-k2-download-${JOB_ID}.out"
echo "  scancel $JOB_ID"
echo ""
echo "Re-run this script after a timeout to resume; it exits early once done."
