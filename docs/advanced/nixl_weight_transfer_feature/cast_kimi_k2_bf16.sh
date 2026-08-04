#!/bin/bash
# =============================================================================
# cast_kimi_k2_bf16.sh — cast Kimi-K2 FP8 weights to BF16, one shard at a time
# =============================================================================
#
# Submits a 1-node, 1-GPU job that runs tools/cast_kimi_k2_bf16.py, which
# processes each of the 61 safetensor shards sequentially and frees GPU memory
# between shards. Existing output shards are skipped, so re-running is safe.
#
# BEFORE RUNNING: delete any incomplete BF16 directory from a prior failed run.
#   rm -rf /lustre/fsw/portfolios/network/users/amitw/models/Kimi-K2-Instruct-bf16
#
# HOW TO RUN:
#   sbatch cast_kimi_k2_bf16.sh
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-fp8cast-<JOBID>.out
#
#SBATCH --job-name=miles-kimi-k2-fp8cast
#SBATCH --partition=batch
#SBATCH --account=network_research_advdev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-fp8cast-%j.out

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-experiments

if [ ! -f "$SQSH" ]; then
    echo "ERROR: sqsh not found at $SQSH"
    exit 1
fi

mkdir -p "$LUSTRE/logs"

srun \
    --container-image="$SQSH" \
    --no-container-mount-home \
    --container-name=amitw-miles-fp8cast \
    --container-mounts="$HOST_MODELS:/root/models" \
    bash -lc '
set -ex

MILES_FORK='"$MILES_FORK"'
MILES_BRANCH='"$MILES_BRANCH"'

echo "--- Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" fork/"$MILES_BRANCH"
pip install -q "numpy<2" "scipy<1.15"

echo "--- Starting FP8 -> BF16 cast (one shard at a time) ---"
cd /root/miles/tools
python3 cast_kimi_k2_bf16.py \
    /root/models/Kimi-K2-Instruct \
    /root/models/Kimi-K2-Instruct-bf16

echo "--- Cast complete ---"
'
