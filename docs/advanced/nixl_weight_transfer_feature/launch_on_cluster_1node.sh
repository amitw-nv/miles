#!/bin/bash
# =============================================================================
# launch_on_cluster_1node.sh — Run a 1-node Miles P2P weight transfer job on the cluster
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
# 1. Runs srun to allocate a compute node and start the container from the sqsh.
#    - The sqsh is built from radixark/miles:latest (Docker Hub) and contains
#      the base Miles + SGLang environment.
#    - No --container-mounts: mounting causes Megatron import issues, and the
#      sqsh already has Miles + SGLang installed inside.
#    - --container-name persists the container across sessions on the same node.
#      The sqsh is only used on first run; subsequent runs reuse the saved state.
#
# 2. Inside the container, updates Miles and SGLang to your fork branches:
#    - Miles:  github.com/amitw-nv/miles  branch amitw/miles-nixl
#    - SGLang: github.com/amitw-nv/sglang branch amitw/sgl-miles-nixl
#    Both are installed as editable (pip install -e .), so git checkout alone
#    is enough to activate code changes — no reinstall needed.
#
# 3. Fixes package version mismatches caused by the sqsh base image:
#    - The sqsh's sglang base ships numpy 2.x, but Megatron requires numpy 1.x.
#    - Downgrading numpy requires also downgrading scipy (scipy>=1.15 needs numpy>=2).
#    These are installed once and persist in the named container.
#
# 4. Drops you into an interactive shell inside the container, ready to run
#    whatever model or command you need.
#
# USAGE
# -----
#   bash launch_on_cluster.sh
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
MILES_BRANCH=amitw/miles-nixl

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

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
echo "  Miles cluster launch"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "=========================================="
echo ""

# --- Command to run inside the container -------------------------------------
#
# This runs after the container starts. Steps:
#  1. Update Miles from fork (discard any local edits first to avoid merge conflicts)
#  2. Update SGLang from fork (same)
#  3. Fix numpy/scipy versions (Megatron needs numpy<2; sqsh ships numpy 2.x)
#  4. Print the active commits so you can verify the right code is running
#  5. Drop into interactive shell
#
INNER_CMD=$(cat <<EOF
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
# The sqsh has Miles cloned at /root/miles from the main branch at build time.
# We switch it to our fork branch. Steps:
#  - Add the fork remote (safe to re-run, || true ignores "already exists")
#  - Discard any local edits (e.g. sed patches from a previous session) so
#    checkout doesn't abort with "local changes would be overwritten"
#  - Fetch the branch from the fork and check it out
echo "--- Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork $MILES_FORK 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH fork/$MILES_BRANCH

# ---- Step 2: Update SGLang --------------------------------------------------
# SGLang lives at /sgl-workspace/sglang (not mounted, baked into the sqsh).
# Same approach: add fork remote, discard local changes, fetch and checkout.
echo "--- Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork $SGLANG_FORK 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork $SGLANG_BRANCH
git -C /sgl-workspace/sglang checkout -B $SGLANG_BRANCH fork/$SGLANG_BRANCH

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
# The sqsh base image (lmsysorg/sglang:v0.5.13) ships numpy 2.x and scipy 1.18.
# Megatron asserts numpy<2 at startup. Downgrading numpy also requires
# downgrading scipy because scipy>=1.15 requires numpy>=2.
# pip install is idempotent — safe to re-run on subsequent sessions.
echo "--- Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- Active code ---"
echo "Miles  : \$(git -C /root/miles log --oneline -1)"
echo "SGLang : \$(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : \$(python -c 'import numpy; print(numpy.__version__)')"
echo ""

# ---- Step 5: Ready — drop into interactive shell ----------------------------
# Environment is set up. Run your model manually, e.g.:
#   bash examples/p2p_weight_transfer/GLM-Z1-9B.sh p2p
#   bash examples/p2p_weight_transfer/GLM-Z1-9B.sh broadcast
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
    -J "miles-setup" \
    --container-image="$SQSH" \
    --no-container-mount-home \
    --container-name="$C_NAME" \
    --pty bash -c "$INNER_CMD"
