#!/bin/bash
# =============================================================================
# launch_glm_z1_9b_1node.sh — 1-node (8-GPU) GLM-Z1-9B Miles job on the cluster
# =============================================================================
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#
# If it doesn't exist, create it from a compute node (login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# HOW THIS SCRIPT WORKS
# ---------------------
# Same as launch_on_cluster.sh, plus steps 5-7 which are run automatically
# inside the container on every launch (idempotent — skipped if already done):
#   5. Download dataset  dapo-math-17k
#   6. Download model    GLM-Z1-9B-0414
#   7. Convert checkpoint to Megatron format
#
# On first launch all three run. On subsequent launches on the same node they
# are skipped instantly because the named container persists /root/ state.
#
# USAGE
# -----
#   bash launch_glm_z1_9b_1node.sh
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome) persists on each node. If it gets
# into a broken state, delete the sqsh to force a clean rebuild on next run:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-rebase

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl-rebase

SQSH=/lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
C_NAME=amitw-miles-nohome

# --- Validate sqsh exists ----------------------------------------------------
if [ ! -f "$SQSH" ]; then
    echo ""
    echo "ERROR: sqsh file not found at $SQSH"
    echo ""
    echo "To create it, run on a compute node (login node has no disk space):"
    echo "  srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash"
    echo "  cd /lustre/fsw/portfolios/network/users/amitw/miles/"
    echo "  enroot import docker://radixark/miles:latest"
    echo ""
    exit 1
fi

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — GLM-Z1-9B 1-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
#
# Steps 1-4 are identical to launch_on_cluster.sh.
# Steps 5-7 are the model-specific setup added here.
#
INNER_CMD=$(cat <<'EOF'
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- Updating Miles (amitw/miles-nixl-rebase) ---"
git -C /root/miles remote add fork https://github.com/amitw-nv/miles.git 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork amitw/miles-nixl-rebase
git -C /root/miles checkout -B amitw/miles-nixl-rebase fork/amitw/miles-nixl-rebase

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- Updating SGLang (amitw/sgl-miles-nixl-rebase) ---"
git -C /sgl-workspace/sglang remote add fork https://github.com/amitw-nv/sglang.git 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork amitw/sgl-miles-nixl-rebase
git -C /sgl-workspace/sglang checkout -B amitw/sgl-miles-nixl-rebase fork/amitw/sgl-miles-nixl-rebase

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
echo "--- Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- Active code ---"
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ---- Step 5: Download dataset (skip if already present) ---------------------
DATASET_FILE=/root/datasets/dapo-math-17k/dapo-math-17k.jsonl
if [ -f "$DATASET_FILE" ]; then
    echo "--- Dataset already present, skipping download ---"
else
    echo "--- Downloading dataset dapo-math-17k ---"
    mkdir -p /root/datasets
    hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
fi
ls -la "$DATASET_FILE"

# ---- Step 6: Download model (skip if already present) -----------------------
MODEL_DIR=/root/models/GLM-Z1-9B-0414
if [ -d "$MODEL_DIR" ] && [ -n "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "--- Model already present, skipping download ---"
else
    echo "--- Downloading model GLM-Z1-9B-0414 ---"
    mkdir -p /root/models
    hf download THUDM/GLM-Z1-9B-0414 --local-dir "$MODEL_DIR"
fi

# ---- Step 7: Convert checkpoint (skip if already done) ----------------------
TRACKER=/root/multinode/GLM-Z1-9B-0414_torch_dist/latest_checkpointed_iteration.txt
if [ -f "$TRACKER" ]; then
    echo "--- Checkpoint already converted ($(cat $TRACKER)), skipping ---"
else
    echo "--- Converting HF checkpoint to Megatron format ---"
    mkdir -p /root/multinode
    python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint(
    model_name='GLM-Z1-9B-0414',
    megatron_model_type='glm4-9B',
    num_gpus_per_node=8,
    multinode=False,
    extra_args='',
    dir_dst='/root/multinode',
)
"
fi
echo "--- Checkpoint tracker: $(cat $TRACKER) ---"

# ---- Step 8: Ready — drop into interactive shell ----------------------------
echo ""
echo "--- Setup complete. You are now inside the container. ---"
echo "--- Run your model, e.g.: bash examples/p2p_weight_transfer/GLM-Z1-9B.sh p2p ---"
cd /root/miles
exec /bin/bash
EOF
)

# --- Launch ------------------------------------------------------------------
srun \
    -A "$PORTFOLIO" \
    -N 1 \
    --gpus-per-node=8 \
    -p interactive \
    --time=02:00:00 \
    -J "miles-glm-z1-9b-1node" \
    --container-image="$SQSH" \
    --no-container-mount-home \
    --container-name="$C_NAME" \
    --pty bash -c "$INNER_CMD"
