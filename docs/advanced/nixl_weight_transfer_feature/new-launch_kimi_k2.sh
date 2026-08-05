#!/bin/bash
# =============================================================================
# new-launch_kimi_k2.sh — end-to-end Kimi-K2-Instruct: prepare → train
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
# The HF weights are already staged at:
#   /lustre/fsw/portfolios/network/users/amitw/models/new-download-kimi-k2
# That directory is bind-mounted onto /root/models/Kimi-K2-Instruct because
# run.py derives the checkpoint path from the registry key and cannot be
# pointed at another name. With the mount in place `hf download` sees a
# complete local dir and no-ops instead of pulling ~1 TB again.
#
# END-TO-END FLOW
# ---------------
# Run this script once from the login node. It checks what is already staged on
# Lustre and submits only the jobs that are still needed, chaining them with
# --dependency=afterok so they run in sequence automatically:
#
#   [prepare job]  — submitted if the HF checkpoint, datasets, or the Megatron
#       torch_dist checkpoint are missing.
#       8 nodes, 64 GPUs. Brings up a Ray cluster, then rank 0 runs
#       `run.py prepare Kimi-K2-Instruct`, which resolves the already-staged HF
#       weights, downloads both datasets and converts the checkpoint to
#       torch_dist. Writes onto Lustre via bind-mounts. Time limit: 4 h.
#
#       The convert has no partial resume: convert_checkpoint only skips when
#       the tracker reads 'release', so a timeout means starting over.
#
#       Why 8 nodes: Kimi-K2 is ~1T params, so ~2 TB in bf16, and
#       convert_hf_to_torch_dist.py builds the Megatron model on GPU. One node
#       tops out at 640 GB of HBM. docs/models/kimi/kimi-k2.md uses 4 nodes,
#       but 32 GPUs is a floor of 64 GB each and lands at 68 GB of an 80 GB
#       H100, leaving almost nothing for NCCL buffers and the checkpoint save
#       path. 64 GPUs give PP 8 x DP 8 with EP 8, or 34 GB per GPU.
#
#   [main job]     — always submitted, waits for the prepare job if there is one.
#       64 nodes, 512 GPUs. The Kimi-K2-Instruct RLVR training run. Time limit: 4 h.
#
# If everything is already staged on Lustre, only the main job is submitted —
# nothing is re-downloaded on the training nodes.
#
# HOW THE MAIN JOB WORKS
# ----------------------
# Submits a 64-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit).  Inside the job:
#   1-4. Same env setup as launch_on_cluster_1node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run Kimi-K2-Instruct in parallel with the correct rank and IP.
#
# run.py raises NotImplementedError for Kimi-K2-Instruct unless SKIP_VALIDATION=1,
# because the model does not support --check-weight-update-equal. Step 6 exports
# it; that also makes run.py pass --sglang-load-format dummy, so SGLang starts
# with random weights and only the transferred weights are real.
#
# Mount safety (Megatron):
#   Bind-mount ONLY the leaf dirs that run.py hardcodes
#   (/root/models/Kimi-K2-Instruct, /root/multinode, /root/datasets, /root/signals).
#   Do NOT mount over /root or home — that shadows /root/miles and
#   /root/Megatron-LM and breaks the Megatron import. Always keep
#   --no-container-mount-home on the srun.
#
# Output is streamed to:
#   /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-<JOBID>.out
#
# USAGE
# -----
#   bash new-launch_kimi_k2.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-<JOBID>.out
#   squeue -u amitw
#   scancel <JOBID>
#
# RESETTING THE CONTAINER
# -----------------------
# The named container (amitw-miles-nohome-64node) persists on each node. If it
# gets into a broken state, delete the sqsh to force a clean rebuild:
#   rm -f /lustre/fsw/portfolios/network/users/amitw/miles/radixark+miles+latest.sqsh
#   enroot import docker://radixark/miles:latest  # recreate it first
# =============================================================================

set -e

MODEL=Kimi-K2-Instruct
# The wrapper is named Kimi-K2.sh, not Kimi-K2-Instruct.sh, so it cannot be
# derived from $MODEL the way the Qwen script does.
RUN_SCRIPT=Kimi-K2.sh
MODE="${1:-nixl}"

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch
NUM_NODES=64

# PREP_NODES must match convert_num_nodes in PREPARE_CONFIGS["Kimi-K2-Instruct"]:
# run.py fans the convert out over exactly that many Ray nodes and asserts if
# fewer have joined. The wall clock has to cover reading ~1 TB of fp8 shards off
# Lustre and writing ~2 TB back, both spread over the 64 ranks.
PREP_NODES=8
PREP_TIME=04:00:00

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-experiments

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=amitw-miles-nohome-64node

LOG_DIR=$LUSTRE/logs

# Bind-mounted at the container paths run.py hardcodes, so these are not free to
# rename. HOST_MODELS is shared across models (GLM, Qwen, Kimi) and sits outside
# $LUSTRE on purpose. HOST_MODEL_DIR is the pre-staged Kimi download; it is
# mounted onto /root/models/$MODEL so run.py's hardcoded path resolves to it.
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
HOST_MODEL_DIR=$HOST_MODELS/new-download-kimi-k2
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets

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
mkdir -p "$LOG_DIR" "$HOST_MODEL_DIR" "$HOST_CKPT" "$HOST_DATASETS"

# --- Determine whether preparation is needed ---------------------------------
NEEDS_PREPARE=0

# hf download writes the shard index last; its presence means download finished.
if [ ! -f "$HOST_MODEL_DIR/model.safetensors.index.json" ]; then
    echo "HF checkpoint missing — will submit prepare job."
    NEEDS_PREPARE=1
fi

# convert_hf_to_torch_dist.py writes 'release' into the tracker as its last
# action, so anything else means the conversion did not finish.
TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    echo "Megatron checkpoint missing or incomplete — will submit prepare job."
    NEEDS_PREPARE=1
fi

# run.py passes both files by absolute path, so these names are fixed.
for ds in dapo-math-17k/dapo-math-17k.jsonl aime-2024/aime-2024.jsonl; do
    if [ ! -f "$HOST_DATASETS/$ds" ]; then
        echo "Dataset missing: $ds — will submit prepare job."
        NEEDS_PREPARE=1
    fi
done

PREV_JOB_ID=""

# --- Submit prepare job if needed --------------------------------------------
if [ "$NEEDS_PREPARE" -eq 1 ]; then
    PREP_JOB_SCRIPT=$LOG_DIR/job-kimi-k2-prepare.slurm
    cat > "$PREP_JOB_SCRIPT" <<PREP_SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2-prepare
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$PREP_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=$PREP_TIME
#SBATCH --output=$LOG_DIR/miles-kimi-k2-prepare-%j.out

# run.py drives the convert through exec_command_all_ray_node, which calls
# ray.init(address="auto"), so a Ray cluster has to be up across all
# $PREP_NODES nodes before rank 0 invokes it. Rank 0 releases the workers
# through this directory whether it succeeds or fails.
PREP_SIGNALS=$LOG_DIR/prep-signals-\$SLURM_JOB_ID
mkdir -p "\$PREP_SIGNALS"

# Mount ONLY leaf dirs — never /root or home (breaks Megatron imports).
srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-prepare" \\
    --container-mounts="$HOST_MODEL_DIR:/root/models/$MODEL,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$PREP_SIGNALS:/root/signals" \\
    bash -lc '
set -ex

echo "--- [Prepare node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork $MILES_FORK 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH fork/$MILES_BRANCH
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
    ip = socket.gethostbyname(hostname)
    print(ip)
except Exception as e:
    print(\"ERROR resolving \" + hostname + \": \" + str(e), file=sys.stderr)
    sys.exit(1)
")
echo "--- [Prepare node \$NODE_RANK] Head IP: \$HEAD_NODE_IP ---"

DONE_FILE=/root/signals/prepare_done

if [ "\$NODE_RANK" -eq 0 ]; then
    RAY_memory_monitor_refresh_ms=0 ray start --head --node-ip-address \$HEAD_NODE_IP \\
        --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    # exec_command_all_ray_node fans out one torchrun per alive node, so every
    # worker has to have joined before the convert starts or it runs short.
    python3 -c "
import ray, time
ray.init(address=\"auto\", ignore_reinit_error=True)
expected = $((PREP_NODES * 8))
while True:
    n = int(ray.cluster_resources().get(\"GPU\", 0))
    print(\"  ... detected %d/%d GPUs\" % (n, expected), flush=True)
    if n >= expected:
        break
    time.sleep(5)
ray.shutdown()
"

    # Resolves the pre-staged HF checkpoint through the bind-mount, downloads
    # both datasets, then converts to torch_dist across all $PREP_NODES nodes.
    # Writes land on Lustre via the bind-mounts above. Each step is individually
    # idempotent, so a re-run resumes rather than redoes.
    echo "--- [Prepare node \$NODE_RANK] Converting $MODEL ---"
    cd /root/miles
    set +e
    python3 examples/p2p_weight_transfer/run.py prepare $MODEL
    PREP_RC=\$?
    set -e

    # Always release the workers, otherwise a failure here leaves them polling
    # until the job hits its time limit. The exit code still propagates, so
    # --dependency=afterok keeps the training job from starting.
    echo "\$PREP_RC" > "\$DONE_FILE"
    echo "--- [Prepare node \$NODE_RANK] Done (rc=\$PREP_RC) ---"
    exit \$PREP_RC
else
    sleep 20
    RAY_memory_monitor_refresh_ms=0 ray start --address=\$HEAD_NODE_IP:6379 \\
        --num-gpus 8 --disable-usage-stats

    echo "--- [Prepare node \$NODE_RANK] Joined Ray, waiting for head ---"
    while [ ! -f "\$DONE_FILE" ]; do sleep 10; done
    echo "--- [Prepare node \$NODE_RANK] Head finished (rc=\$(cat \$DONE_FILE)), exiting ---"
fi
'
PREP_SLURM

    PREP_JOB_ID=$(sbatch --parsable "$PREP_JOB_SCRIPT")
    echo "Submitted prepare job $PREP_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-kimi-k2-prepare-${PREP_JOB_ID}.out"
    PREV_JOB_ID=$PREP_JOB_ID
fi

MAIN_DEP=""
[ -n "$PREV_JOB_ID" ] && MAIN_DEP="--dependency=afterok:$PREV_JOB_ID"

# --- Print launch info -------------------------------------------------------
echo ""
echo "=========================================="
echo "  Miles cluster launch — Kimi-K2-Instruct 64-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Nodes     : $NUM_NODES"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Model dir : $HOST_MODEL_DIR"
echo "  Ckpt dir  : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Logs      : $LOG_DIR"
[ -n "$PREV_JOB_ID" ] && echo "  Waiting on: prepare job $PREV_JOB_ID"
echo "=========================================="
echo ""

# --- Write job script --------------------------------------------------------
#
# Written to a file in lustre so it can be inspected after submission.
# Variables from the outer shell (SQSH, MILES_BRANCH, etc.) are expanded here;
# SLURM variables (\$SLURM_NODEID etc.) are escaped and evaluated at runtime.
#
JOB_SCRIPT=$LOG_DIR/job-kimi-k2.slurm
cat > "$JOB_SCRIPT" <<SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$NUM_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/%x-%j.out

mkdir -p $LOG_DIR

# Per-job signal directory. run.py has rank 0 write job_done_<mode> here when
# training ends and the 63 workers poll for it, so it has to be a shared mount —
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
    --container-mounts="$HOST_MODEL_DIR:/root/models/$MODEL,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$HOST_SIGNALS:/root/signals" \\
    bash -lc '
set -ex

# ---- Step 1: Update Miles ---------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork $MILES_FORK 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH fork/$MILES_BRANCH

# ---- Step 2: Update SGLang --------------------------------------------------
echo "--- [Node \${SLURM_NODEID}] Updating SGLang ($SGLANG_BRANCH) ---"
git -C /sgl-workspace/sglang remote add fork $SGLANG_FORK 2>/dev/null || true
git -C /sgl-workspace/sglang restore . 2>/dev/null || git -C /sgl-workspace/sglang checkout . 2>/dev/null || true
git -C /sgl-workspace/sglang fetch fork $SGLANG_BRANCH
git -C /sgl-workspace/sglang checkout -B $SGLANG_BRANCH fork/$SGLANG_BRANCH

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
# $RUN_SCRIPT still has a prepare guard, but it will no-op once the Megatron
# checkpoint exists on the shared /root/multinode mount.
# SKIP_VALIDATION is mandatory: run.py refuses to run Kimi-K2-Instruct without
# it, since the model does not support --check-weight-update-equal.
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/signals
cd /root/miles
bash examples/p2p_weight_transfer/$RUN_SCRIPT $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

'
SLURM

# --- Submit ------------------------------------------------------------------
JOB_ID=$(sbatch --parsable $MAIN_DEP "$JOB_SCRIPT")

echo "Submitted job $JOB_ID"
echo ""
echo "Monitor:"
echo "  tail -f $LOG_DIR/miles-kimi-k2-${JOB_ID}.out"
echo "  squeue -u amitw"
echo "  scancel $JOB_ID"
