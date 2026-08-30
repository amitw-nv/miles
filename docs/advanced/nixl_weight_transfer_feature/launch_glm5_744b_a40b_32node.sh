#!/bin/bash
# =============================================================================
# launch_glm5_744b_a40b_32node.sh — end-to-end GLM-5 744B-A40B: prepare → train
# =============================================================================
#
# PREREQUISITES
# -------------
# The sqsh file must exist at:
#   /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest+300826.sqsh
#
# If it doesn't exist, create it from a compute node (login node has no space):
#   srun -A network_research_advdev -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash
#   cd /lustre/fsw/portfolios/network/users/amitw/miles/
#   enroot import docker://radixark/miles:latest
#
# CODE STAGING (no GitHub access from compute nodes)
# --------------------------------------------------
# Compute nodes cannot reach GitHub. This script fetches Miles and SGLang
# from GitHub once on the login node into bare mirrors on Lustre, then
# every compute node fetches from those mirrors instead. The login node
# needs GitHub access; the compute nodes need none.
#
# END-TO-END FLOW
# ---------------
# Run this script once from the login node. It checks what is already staged on
# Lustre and submits only the jobs that are still needed, chaining them with
# --dependency=afterok so they run in sequence automatically:
#
#   [download job] — submitted if the HF checkpoint or dapo-math-17k is missing.
#       1 node on cpu_datamover, 0 GPUs. Rank 0 runs
#       `run.py prepare GLM-5 --download-only`.
#       Must not allocate GPUs: cw-dfw's Occupied Idle Job Reaper cancels
#       batch jobs whose DCGM SM_ACTIVE stays <= 0.01 for 30 min (this is
#       what killed 16434818 — 112/128 GPUs idle while rank 0 downloaded).
#       cpu_datamover has no GPUs and no reaper. Time limit: 4 h.
#
#   [convert job]  — submitted if the Megatron torch_dist tracker is not
#       `release`. 16 nodes, 128 GPUs (PP=4 x EP=32). Waits for the download
#       job when there is one. Rank 0 runs `run.py prepare GLM-5` (HF/dataset
#       steps no-op if already on Lustre); the other ranks join Ray so convert
#       can fan out. Time limit: 4 h.
#
#   [main job]     — always submitted, waits for the last prepare job if any.
#       32 nodes, 256 GPUs. The GLM-5 RLVR training run. Time limit: 2 h.
#       Sets OccupiedIdleGPUsJobReaper exemptIdleTimeMins=60 reason=model_loading
#       (weight load + TileLang JIT; jobs 16502810/16502817 were reaped at 31 min).
#
# If everything is already staged on Lustre, only the main job is submitted —
# nothing is re-downloaded on the training nodes.
#
# HOW THE MAIN JOB WORKS
# ----------------------
# Submits a 32-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit).  Inside the job:
#   1-4. Same env setup as launch_qwen3_235b_a22b_16node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run GLM-5 in parallel with the correct rank and IP.
#
# Mount safety (Megatron):
#   Bind-mount ONLY the leaf dirs that run.py hardcodes
#   (/root/models, /root/multinode, /root/datasets, /root/signals).
#   Do NOT mount over /root or home — that shadows /root/miles and
#   /root/Megatron-LM and breaks the Megatron import. Always keep
#   --no-container-mount-home on the srun.
#
# Output is streamed to:
#   /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-glm5-744b-a40b-<JOBID>.out
#
# USAGE
# -----
#   bash launch_glm5_744b_a40b_32node.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   tail -f .../logs/miles-glm5-744b-a40b-download-<JOBID>.out
#   tail -f .../logs/miles-glm5-744b-a40b-convert-<JOBID>.out
#   tail -f .../logs/miles-glm5-744b-a40b-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome-32node) persists on each node. If it
# gets into a broken state, delete the sqsh to force a clean rebuild:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest+300826.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

MODEL=GLM-5
MODE="${1:-nixl}"

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch
DOWNLOAD_PARTITION=cpu_datamover
NUM_NODES=32
NUM_DOWNLOAD_NODES=1
NUM_PREPARE_NODES=16

MILES_FORK=git@github.com:amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-upstream-exp

SGLANG_FORK=git@github.com:amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl-upstream-exp

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest+300826.sqsh
C_NAME=amitw-miles-nohome-32node

LOG_DIR=$LUSTRE/logs

# Bind-mounted at the container paths run.py hardcodes, so these three are not
# free to rename. HOST_MODELS is shared across models (GLM, Qwen, Kimi) and
# sits outside $LUSTRE on purpose.
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets

# Bare mirrors fetched here on the login node; compute nodes read from Lustre.
MILES_SRC=$LUSTRE/src/miles.git
SGLANG_SRC=$LUSTRE/src/sglang.git

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

# pyxis will not create mount sources, so every bind-mount source has to exist
# before the srun or container setup fails before any of our code runs.
mkdir -p "$LOG_DIR" "$HOST_MODELS" "$HOST_CKPT" "$HOST_DATASETS" "$LUSTRE/src"

# --- Stage code on Lustre (login node fetches from GitHub once) --------------
stage_repo() {
    local bare=$1 url=$2 branch=$3
    [ -d "$bare" ] || git init --bare -q "$bare"
    if ! git -C "$bare" fetch --force --quiet "$url" "+$branch:$branch"; then
        echo "" >&2
        echo "ERROR: could not fetch $branch from $url" >&2
        echo "Make sure this login node has GitHub access." >&2
        echo "" >&2
        return 1
    fi
}

stage_repo "$MILES_SRC"  "$MILES_FORK"  "$MILES_BRANCH"
stage_repo "$SGLANG_SRC" "$SGLANG_FORK" "$SGLANG_BRANCH"

MILES_SHA=$(git -C "$MILES_SRC" rev-parse "$MILES_BRANCH")
SGLANG_SHA=$(git -C "$SGLANG_SRC" rev-parse "$SGLANG_BRANCH")

echo "Miles  : $MILES_BRANCH @ ${MILES_SHA:0:9}"
echo "SGLang : $SGLANG_BRANCH @ ${SGLANG_SHA:0:9}"

# --- Determine whether download / convert are needed -------------------------
NEEDS_DOWNLOAD=0
NEEDS_CONVERT=0

# hf download writes the shard index last; its presence means download finished.
if [ ! -f "$HOST_MODELS/$MODEL/model.safetensors.index.json" ]; then
    echo "HF checkpoint missing — will submit 1-node download job."
    NEEDS_DOWNLOAD=1
fi

# run.py passes this file by absolute path, so the name is fixed.
if [ ! -f "$HOST_DATASETS/dapo-math-17k/dapo-math-17k.jsonl" ]; then
    echo "Dataset missing: dapo-math-17k/dapo-math-17k.jsonl — will submit 1-node download job."
    NEEDS_DOWNLOAD=1
fi

# convert_hf_to_torch_dist.py writes 'release' into the tracker as its last
# action, so anything else means the conversion did not finish.
TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    echo "Megatron checkpoint missing or incomplete — will submit 16-node convert job."
    NEEDS_CONVERT=1
fi

PREV_JOB_ID=""

# --- Submit 1-node download job if needed ------------------------------------
if [ "$NEEDS_DOWNLOAD" -eq 1 ]; then
    DL_JOB_SCRIPT=$LOG_DIR/job-glm5-744b-a40b-download.slurm
    cat > "$DL_JOB_SCRIPT" <<DL_SLURM
#!/bin/bash
#SBATCH --job-name=miles-glm5-744b-a40b-download
#SBATCH --partition=$DOWNLOAD_PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_DOWNLOAD_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/miles-glm5-744b-a40b-download-%j.out

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-download" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$MILES_SRC:/root/src/miles.git:ro" \\
    bash -lc '
set -ex

# Fetch from the Lustre mirror — login node already pulled from GitHub.
echo "--- [Download] Updating Miles ($MILES_BRANCH @ ${MILES_SHA:0:9}) ---"
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch /root/src/miles.git $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH $MILES_SHA
pip install -q "numpy<2" "scipy<1.15"

# snapshot_download + dapo-math-17k. Convert is skipped. Writes land on
# Lustre via the bind-mounts. Re-runs resume rather than redo.
echo "--- [Download] $MODEL --download-only ---"
cd /root/miles
python3 examples/infra_features/p2p_weight_transfer/run.py prepare $MODEL --download-only

echo "--- [Download] Done ---"
'
DL_SLURM

    DL_JOB_ID=$(sbatch --parsable "$DL_JOB_SCRIPT")
    echo "Submitted download job $DL_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-glm5-744b-a40b-download-${DL_JOB_ID}.out"
    PREV_JOB_ID=$DL_JOB_ID
fi

# --- Submit 16-node convert job if needed ------------------------------------
if [ "$NEEDS_CONVERT" -eq 1 ]; then
    CONV_DEP=""
    [ -n "$PREV_JOB_ID" ] && CONV_DEP="--dependency=afterok:$PREV_JOB_ID"

    CONV_JOB_SCRIPT=$LOG_DIR/job-glm5-744b-a40b-convert.slurm
    cat > "$CONV_JOB_SCRIPT" <<CONV_SLURM
#!/bin/bash
#SBATCH --job-name=miles-glm5-744b-a40b-convert
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_PREPARE_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/miles-glm5-744b-a40b-convert-%j.out

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-convert" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,$MILES_SRC:/root/src/miles.git:ro" \\
    bash -lc '
set -ex

# Fetch from the Lustre mirror — login node already pulled from GitHub.
echo "--- [Convert Node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH @ ${MILES_SHA:0:9}) ---"
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch /root/src/miles.git $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH $MILES_SHA
pip install -q "numpy<2" "scipy<1.15"

NODE_RANK=\${SLURM_NODEID}
HEAD_NODE_IP=\$(python3 -c "
import socket, re, os, sys
nl = os.environ.get(\"SLURM_NODELIST\", \"\")
m = re.match(r\"(.+)-\[([^\]]+)\]\", nl)
if m:
    prefix, spec = m.group(1), m.group(2)
    first = re.split(r\"[,\-]\", spec)[0]
    hostname = prefix + \"-\" + first
elif \",\" in nl:
    hostname = nl.split(\",\")[0]
else:
    hostname = nl
try:
    print(socket.gethostbyname(hostname))
except Exception as e:
    print(\"ERROR resolving \" + hostname + \": \" + str(e), file=sys.stderr)
    sys.exit(1)
")
echo "--- [Convert Node \$NODE_RANK] Head IP: \$HEAD_NODE_IP ---"

# convert_checkpoint uses Ray to run torchrun on every node (PP=4 x EP=32).
# HF / dataset steps in run.py prepare no-op if already on the bind-mounts.
ray stop --force || true
if [ "\$NODE_RANK" -eq 0 ]; then
    RAY_memory_monitor_refresh_ms=0 ray start --head --node-ip-address "\$HEAD_NODE_IP" --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    expected_gpus=$((NUM_PREPARE_NODES * 8))
    echo "--- [Convert] Waiting for \$expected_gpus GPUs in Ray cluster ---"
    while true; do
        available=\$(python3 -c "import ray; ray.init(address=\"auto\", ignore_reinit_error=True); print(int(ray.cluster_resources().get(\"GPU\", 0))); ray.shutdown()" 2>/dev/null || echo 0)
        echo "  ... \$available / \$expected_gpus GPUs"
        [ "\$available" -ge "\$expected_gpus" ] && break
        sleep 5
    done
    echo "--- [Convert] Converting $MODEL ---"
    cd /root/miles
    python3 examples/infra_features/p2p_weight_transfer/run.py prepare $MODEL
    touch /root/multinode/.glm5-prepare-done-\$SLURM_JOB_ID
    echo "--- [Convert] Done ---"
else
    sleep 20
    RAY_memory_monitor_refresh_ms=0 ray start --address="\$HEAD_NODE_IP:6379" --num-gpus 8 --disable-usage-stats
    while [ ! -f /root/multinode/.glm5-prepare-done-\$SLURM_JOB_ID ]; do
        sleep 15
    done
    echo "--- [Convert Node \$NODE_RANK] Rank 0 finished ---"
fi
'
CONV_SLURM

    CONV_JOB_ID=$(sbatch --parsable $CONV_DEP "$CONV_JOB_SCRIPT")
    echo "Submitted convert job $CONV_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-glm5-744b-a40b-convert-${CONV_JOB_ID}.out"
    PREV_JOB_ID=$CONV_JOB_ID
fi

MAIN_DEP=""
[ -n "$PREV_JOB_ID" ] && MAIN_DEP="--dependency=afterok:$PREV_JOB_ID"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — GLM-5 744B-A40B 32-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION (train/convert), $DOWNLOAD_PARTITION (download)"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Nodes     : $NUM_NODES train / $NUM_PREPARE_NODES convert / $NUM_DOWNLOAD_NODES download"
echo "  Miles     : $MILES_BRANCH @ ${MILES_SHA:0:9}"
echo "  SGLang    : $SGLANG_BRANCH @ ${SGLANG_SHA:0:9}"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Model dir : $HOST_MODELS/$MODEL"
echo "  Ckpt dir  : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Logs      : $LOG_DIR"
[ -n "${DL_JOB_ID:-}" ] && echo "  Download  : job $DL_JOB_ID"
[ -n "${CONV_JOB_ID:-}" ] && echo "  Convert   : job $CONV_JOB_ID"
[ -n "$PREV_JOB_ID" ] && echo "  Train waits on: job $PREV_JOB_ID"
echo "  Idle exemption: 60 min model_loading (check scontrol Comment)"
echo "=========================================="
echo ""

# --- Write job script --------------------------------------------------------
#
# Written to a file in lustre so it can be inspected after submission.
# Variables from the outer shell (SQSH, MILES_BRANCH, etc.) are expanded here;
# SLURM variables (\$SLURM_NODEID etc.) are escaped and evaluated at runtime.
#
JOB_SCRIPT=$LOG_DIR/job-glm5-744b-a40b.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-glm5-744b-a40b
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out
#SBATCH --comment='{"OccupiedIdleGPUsJobReaper":{"exemptIdleTimeMins":"60","reason":"model_loading","description":"GLM-5 32-node weight load and TileLang JIT before training"}}'

mkdir -p $LOG_DIR

# Per-job signal directory. run.py has rank 0 write job_done_<mode> here when
# training ends and the workers poll for it, so it has to be a shared mount —
# run.py's default is container-local and the workers would never see it.
# Keying it on the job id stops a leftover signal from an earlier run from
# making every worker exit immediately. pyxis will not create mount sources.
HOST_SIGNALS=$LOG_DIR/signals-\$SLURM_JOB_ID
mkdir -p "\$HOST_SIGNALS"

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$HOST_SIGNALS:/root/signals,$MILES_SRC:/root/src/miles.git:ro,$SGLANG_SRC:/root/src/sglang.git:ro" \\
    bash -lc '
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH @ ${MILES_SHA:0:9}) ---"
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch /root/src/miles.git $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH $MILES_SHA

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating SGLang ($SGLANG_BRANCH @ ${SGLANG_SHA:0:9}) ---"
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch /root/src/sglang.git $SGLANG_BRANCH
git -C /sgl-workspace/sglang checkout -B $SGLANG_BRANCH $SGLANG_SHA

# ---- Step 3: Fix numpy and scipy versions -----------------------------------
echo "--- [Node \${SLURM_NODEID}] Fixing numpy and scipy versions ---"
pip install -q "numpy<2" "scipy<1.15"

# ---- Step 4: Print active commits -------------------------------------------
echo ""
echo "--- [Node \${SLURM_NODEID}] Active code ---"
echo "Miles  : \$(git -C /root/miles log --oneline -1)"
echo "SGLang : \$(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : \$(python -c '"'"'import numpy; print(numpy.__version__)'"'"')"
echo ""

# ---- Step 5: Discover head node IP from SLURM_NODELIST ----------------------
# Each node independently resolves the head hostname from SLURM_NODELIST —
# no shared file or coordination needed.
NODE_RANK=\${SLURM_NODEID}
HEAD_NODE_IP=\$(python3 -c "
import socket, re, os, sys
nl = os.environ.get(\"SLURM_NODELIST\", \"\")
m = re.match(r\"(.+)-\[([^\]]+)\]\", nl)
if m:
    prefix, spec = m.group(1), m.group(2)
    first = re.split(r\"[,\-]\", spec)[0]
    hostname = prefix + \"-\" + first
elif \",\" in nl:
    hostname = nl.split(\",\")[0]
else:
    hostname = nl
try:
    ip = socket.gethostbyname(hostname)
    print(ip)
except Exception as e:
    print(\"ERROR resolving \" + hostname + \": \" + str(e), file=sys.stderr)
    sys.exit(1)
")
echo "--- [Node \$NODE_RANK] Head IP: \$HEAD_NODE_IP ---"

# ---- Step 6: Run the model --------------------------------------------------
# HF weights / torch_dist ckpt / datasets come from the Lustre bind-mounts.
# GLM-5.sh still has a prepare guard, but it will no-op once the
# Megatron checkpoint exists on the shared /root/multinode mount.
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
export MILES_LOG_DIR=/root/signals
# run.py --ref-load is CKPT_SAVE_DIR/GLM-5_torch_dist; default is /root
# which is container-local. Convert wrote the tracker on this bind-mount.
export CKPT_SAVE_DIR=/root/multinode
cd /root/miles
bash examples/infra_features/p2p_weight_transfer/GLM-5.sh GLM-5 $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

'
SLURM

# --- Submit ------------------------------------------------------------------
JOB_ID=$(sbatch --parsable $MAIN_DEP "$JOB_SCRIPT")

echo "Submitted train job $JOB_ID"
echo ""
echo "Monitor:"
[ -n "${DL_JOB_ID:-}" ] && echo "  tail -f $LOG_DIR/miles-glm5-744b-a40b-download-${DL_JOB_ID}.out"
[ -n "${CONV_JOB_ID:-}" ] && echo "  tail -f $LOG_DIR/miles-glm5-744b-a40b-convert-${CONV_JOB_ID}.out"
echo "  tail -f $LOG_DIR/miles-glm5-744b-a40b-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
