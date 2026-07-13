#!/bin/bash
# Launch a Miles P2P weight transfer job on the HPC cluster.
# - Miles: synced from fork to Lustre, mounted into container
# - SGLang: uses the container's built-in repo; checks out the fork branch inside
#
# Usage: bash launch_on_cluster.sh <portfolio> <model> [mode]
#   portfolio : network | trtllm | triton
#   model     : GLM-Z1-9B-0414 | Qwen3-4B | GLM-4.7-Flash | ...
#   mode      : p2p (default) | broadcast | nixl

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
SGLANG_FORK=git@github.com:amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

# --- Paths ------------------------------------------------------------------
BASEDIR=/lustre/fsw/portfolios/network/users/$USER
MILES_SRC=$BASEDIR/miles
LLM_MODELS=/lustre/fsw/portfolios/network/users/bbiber/llm_models

C_IMAGE=docker://radixark/miles:latest
C_SAVED=$BASEDIR/miles/radixark+miles+latest.sqsh
C_NAME=miles-dev-$USER

PARTITION=interactive
TIME=02:00:00
JOB_NAME=miles-p2p.$MODEL

# --- Sync Miles to Lustre ---------------------------------------------------
sync_repo() {
    local repo=$1 branch=$2 dest=$3
    if [ -d "$dest/.git" ]; then
        echo "Updating $(basename $dest) ($branch)..."
        git -C "$dest" fetch origin
        git -C "$dest" reset --hard origin/"$branch"
    else
        echo "Cloning $(basename $dest) ($branch)..."
        git clone --branch "$branch" "$repo" "$dest"
    fi
}

sync_repo "$MILES_REPO" "$MILES_BRANCH" "$MILES_SRC"

# --- Mounts -----------------------------------------------------------------
SLURM_PATHS=/lib/x86_64-linux-gnu/libmunge.so.2,/run/munge,/etc/slurm,/cm/shared/apps/slurm/current:/opt/slurm

MOUNTS=$SLURM_PATHS
MOUNTS+=,$BASEDIR:/workspace/lustre
MOUNTS+=,$LLM_MODELS:/workspace/llm_models
MOUNTS+=,$MILES_SRC:/root/miles
MOUNTS+=,/lustre:/lustre

# --- Command to run inside the container ------------------------------------
INNER_CMD=$(cat <<EOF
set -ex

# Install Miles from mounted Lustre source
pip install -e /root/miles --no-deps -q

# Checkout SGLang fork branch inside the container's built-in SGLang repo
SGLANG_GIT=\$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
git -C "\$SGLANG_GIT" remote add fork $SGLANG_FORK 2>/dev/null || true
git -C "\$SGLANG_GIT" fetch fork $SGLANG_BRANCH
git -C "\$SGLANG_GIT" checkout $SGLANG_BRANCH
pip install -e "\$SGLANG_GIT" --no-deps -q

# Apply known naming fix
sed -i 's/model_loader_module\.post_load_weights/model_loader_module._post_load_weights/g' \
  /root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py

cd /root/miles
python examples/p2p_weight_transfer/run.py run $MODEL --mode $MODE
EOF
)

# --- Launch -----------------------------------------------------------------
echo "Launching: model=$MODEL mode=$MODE portfolio=$PORTFOLIO"
echo "Miles src : $MILES_SRC ($MILES_BRANCH)"
echo "SGLang    : container repo, branch $SGLANG_BRANCH from fork"
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
