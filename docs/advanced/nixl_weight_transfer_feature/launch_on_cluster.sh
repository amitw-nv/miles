#!/bin/bash
# Launch a Miles P2P weight transfer job on the HPC cluster.
# - Miles: synced from fork to Lustre, mounted into container
# - SGLang: uses the container's built-in repo; checks out the fork branch inside
#
# Usage: bash launch_on_cluster.sh <portfolio> <model> [mode]
#   portfolio : network | trtllm | triton
#   model     : GLM-Z1-9B-0414 | Qwen3-4B | GLM-4.7-Flash | ...
#   mode      : p2p (default) | broadcast | nixl
#
# To reset a corrupted container:
#   rm -f /lustre/fsw/portfolios/network/users/$USER/miles/radixark+miles+latest.sqsh
#   The next run starts fresh from the Docker base image.

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
SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

# --- Paths ------------------------------------------------------------------
BASEDIR=/lustre/fsw/portfolios/network/users/$USER
MILES_SRC=$BASEDIR/miles
LLM_MODELS=/lustre/fsw/portfolios/network/users/amitw/models

# Container: use saved sqsh when it exists; fall back to Docker base so the
# sqsh can be deleted to force a clean rebuild.
C_BASE=radixark/miles:latest
C_SAVED=$BASEDIR/miles/radixark+miles+latest.sqsh
if [ -f "$C_SAVED" ]; then
    C_IMAGE=$C_SAVED
else
    C_IMAGE=$C_BASE
fi
C_NAME=amitw-miles-nohome

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
MOUNTS+=,$LLM_MODELS:/root/models
MOUNTS+=,$MILES_SRC:/root/miles
MOUNTS+=,/lustre:/lustre

# --- Command to run inside the container ------------------------------------
INNER_CMD=$(cat <<EOF
set -ex

# Apply naming fix before any Miles code runs; the container has miles as an
# editable install pointing to /root/miles, so the mounted branch code is
# active immediately.
sed -i 's/model_loader_module\.post_load_weights/model_loader_module._post_load_weights/g' \
  /root/miles/miles/backends/megatron_utils/update_weight/update_weight_from_distributed/p2p.py

# Checkout SGLang fork branch inside the container's built-in SGLang repo
SGLANG_PKG=\$(python -c "import sglang, pathlib; print(pathlib.Path(sglang.__file__).parent.parent)")
SGLANG_GIT=\$(git -C "\$SGLANG_PKG" rev-parse --show-toplevel)
git -C "\$SGLANG_GIT" remote add fork $SGLANG_FORK 2>/dev/null || true
git -C "\$SGLANG_GIT" fetch fork $SGLANG_BRANCH
git -C "\$SGLANG_GIT" checkout -B $SGLANG_BRANCH FETCH_HEAD
pip install -e "\$SGLANG_GIT/python" --no-deps -q

cd /root/miles

# Prepare: check tracker file so a partial conversion does not get skipped
CKPT_TRACKER=/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ "\$(cat "\$CKPT_TRACKER" 2>/dev/null)" != "release" ]; then
    python examples/p2p_weight_transfer/run.py prepare $MODEL
fi

python examples/p2p_weight_transfer/run.py run $MODEL --mode $MODE
EOF
)

# --- Launch -----------------------------------------------------------------
echo "Launching : model=$MODEL  mode=$MODE  portfolio=$PORTFOLIO"
echo "Miles src : $MILES_SRC ($MILES_BRANCH)"
echo "SGLang    : container repo, branch $SGLANG_BRANCH from fork"
echo "Container : $C_IMAGE  ->  saved to $C_SAVED"
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
