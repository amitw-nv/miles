#!/bin/bash
# Launch a Miles P2P weight transfer job on the HPC cluster.
# Clones/updates Miles and SGLang from their forks to Lustre, then mounts
# them into the container so you can run updated code without rebuilding the image.
#
# Usage: bash launch_on_cluster.sh <portfolio> <model> [mode]
#   portfolio : network | trtllm | triton
#   model     : GLM-Z1-9B-0414 | Qwen3-4B | GLM-4.7-Flash | ...
#   mode      : p2p (default) | broadcast

set -e

# --- Args -------------------------------------------------------------------
PORTFOLIO_ARG="${1:-}"
MODEL="${2:-}"
MODE="${3:-p2p}"

case "$PORTFOLIO_ARG" in
    network) PORTFOLIO=network_research_advdev ;;
    trtllm)  PORTFOLIO=coreai_comparch_trtllm ;;
    triton)  PORTFOLIO=coreai_tritoninference_triton3 ;;
    *)
        echo "Error: first arg must be 'network', 'trtllm', or 'triton'"
        echo "Usage: bash launch_on_cluster.sh <portfolio> <model> [mode]"
        exit 1
        ;;
esac

if [ -z "$MODEL" ]; then
    echo "Error: model name required (e.g. GLM-Z1-9B-0414, Qwen3-4B, GLM-4.7-Flash)"
    echo "Usage: bash launch_on_cluster.sh <portfolio> <model> [mode]"
    exit 1
fi

# --- Repos & branches -------------------------------------------------------
MILES_REPO=git@github.com:amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl
SGLANG_REPO=git@github.com:amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

# --- Paths ------------------------------------------------------------------
BASEDIR=/lustre/fsw/portfolios/network/users/$USER
MILES_SRC=$BASEDIR/miles
SGLANG_SRC=$BASEDIR/sglang
LLM_MODELS=/lustre/fsw/portfolios/network/users/bbiber/llm_models

C_IMAGE=docker://radixark/miles:latest
C_SAVED=$BASEDIR/miles/radixark+miles+latest.sqsh
C_NAME=miles-dev-$USER

PARTITION=interactive
TIME=02:00:00
JOB_NAME=miles-p2p.$MODEL

# --- Sync repos to Lustre ---------------------------------------------------
sync_repo() {
    local repo=$1 branch=$2 dest=$3
    if [ -d "$dest/.git" ]; then
        echo "Updating $(basename $dest) ($branch)..."
        git -C "$dest" fetch origin
        git -C "$dest" checkout "$branch"
        git -C "$dest" pull origin "$branch"
    else
        echo "Cloning $(basename $dest) ($branch)..."
        git clone --branch "$branch" "$repo" "$dest"
    fi
}

sync_repo "$MILES_REPO"  "$MILES_BRANCH"  "$MILES_SRC"
sync_repo "$SGLANG_REPO" "$SGLANG_BRANCH" "$SGLANG_SRC"

# --- Mounts -----------------------------------------------------------------
SLURM_PATHS=/lib/x86_64-linux-gnu/libmunge.so.2,/run/munge,/etc/slurm,/cm/shared/apps/slurm/current:/opt/slurm

MOUNTS=$SLURM_PATHS
MOUNTS+=,$BASEDIR:/workspace/lustre
MOUNTS+=,$LLM_MODELS:/workspace/llm_models
MOUNTS+=,$MILES_SRC:/root/miles
MOUNTS+=,$SGLANG_SRC:/root/sglang
MOUNTS+=,/lustre:/lustre

# --- Command to run inside the container ------------------------------------
INNER_CMD=$(cat <<EOF
set -ex
pip install -e /root/miles --no-deps -q
pip install -e /root/sglang/python --no-deps -q
sed -i 's/model_loader_module\.post_load_weights/model_loader_module._post_load_weights/g' \
  /root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py
cd /root/miles
python examples/p2p_weight_transfer/run.py run $MODEL --mode $MODE
EOF
)

# --- Launch -----------------------------------------------------------------
echo "Launching: model=$MODEL mode=$MODE portfolio=$PORTFOLIO"
echo "Miles src : $MILES_SRC ($MILES_BRANCH)"
echo "SGLang src: $SGLANG_SRC ($SGLANG_BRANCH)"
echo ""

srun \
    -A "$PORTFOLIO" \
    -N 1 \
    --gpus-per-node=8 \
    -p "$PARTITION" \
    --time="$TIME" \
    -J "$JOB_NAME" \
    --container-image="$C_IMAGE" \
    --container-save="$C_SAVED" \
    --container-name="$C_NAME" \
    --no-container-mount-home \
    --container-mounts="$MOUNTS" \
    bash -c "$INNER_CMD"
