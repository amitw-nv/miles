#!/bin/bash
# =============================================================================
# launch_kimi_k2_64node.sh — end-to-end Kimi-K2: convert → train
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
# The Kimi-K2 weights must already be on Lustre under $HOST_MODELS below. This
# script never downloads the model; it fails fast if the weights are absent.
# Both on-disk layouts are accepted:
#   $HOST_MODELS/Kimi-K2-Instruct                      (hf download --local-dir)
#   $HOST_MODELS/models--moonshotai--Kimi-K2-Instruct  (HF hub cache)
#
# END-TO-END FLOW
# ---------------
# Run this script once from the login node. It checks which steps are already
# done and submits only the jobs that are still needed, chaining each with
# --dependency=afterok so they run in sequence automatically:
#
#   [dataset job]   — submitted if either dataset is missing
#       1 node, 1 GPU. Downloads dapo-math-17k and aime-2024 to Lustre. The
#       model itself is never downloaded — see PREREQUISITES.
#       Time limit: 4 h.
#
#   [convert job]   — submitted if the Megatron torch_dist checkpoint is missing
#       8 nodes, 64 GPUs (PP=8, EP=8 → 32 GB of BF16 weights per GPU).
#       Phase a — node 0 only: detects FP8 weights and casts to BF16 if needed
#         (output: $HOST_MODELS/Kimi-K2-Instruct-bf16, next to the original
#         weights — that directory must be writable).
#         All other nodes wait via a Lustre flag file.
#       Phase b — all 8 nodes: multi-node torchrun converts the BF16 checkpoint
#         to Megatron torch_dist format
#         (output: $LUSTRE/multinode/Kimi-K2-Instruct_torch_dist).
#       Time limit: 4 h.
#
#   [main job]      — always submitted, waits for any prep jobs above
#       64 nodes, 512 GPUs. The Kimi-K2 RLVR training run.
#       Time limit: 4 h.
#
# If everything is already prepared on Lustre, only the main job is submitted.
#
# HOW THE MAIN JOB WORKS
# ----------------------
# Submits a 64-node batch job via sbatch (bypasses the interactive partition's
# per-user node limit). Inside the job:
#   1-4. Same env setup as launch_on_cluster_1node.sh
#   5.   Each node independently resolves the head node IP from SLURM_NODELIST
#        using Python's socket — no shared filesystem needed.
#   6.   All nodes run Kimi-K2-Instruct in parallel with the correct rank and IP.
#
# The Lustre directories are bind-mounted at the container paths run.py
# hardcodes, so all 64 nodes read one copy of the model and checkpoint. Mounting
# the subdirectories is safe; mounting over /root or home breaks the Megatron
# import, hence --no-container-mount-home on the srun.
#
# 64 nodes is what RUN_CONFIGS["Kimi-K2-Instruct"] in
# examples/p2p_weight_transfer/run.py declares: 256 train + 256 rollout GPUs,
# disaggregated. Do not lower it without changing that entry too.
#
# USAGE
# -----
#   bash launch_kimi_k2_64node.sh [MODE]
#     MODE : nixl (default) | p2p | broadcast
#
# MONITORING
# ----------
#   squeue -u amitw
#   tail -f /lustre/fsw/portfolios/network/users/amitw/miles/logs/miles-kimi-k2-<JOBID>.out
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
# The per-model launcher is named after the model family, not the registry key.
MODEL_SCRIPT=Kimi-K2
MODE="${1:-nixl}"

# Number of nodes for the multi-node torch_dist conversion.
# PP=8, EP=8 → world_size=64 → ~32 GB BF16 weights per H100, well within budget.
CONV_NODES=8

# --- Config ------------------------------------------------------------------
PORTFOLIO=network_research_advdev
PARTITION=batch
NUM_NODES=64

MILES_FORK=https://github.com/amitw-nv/miles.git
MILES_BRANCH=amitw/miles-nixl-experiments

SGLANG_FORK=https://github.com/amitw-nv/sglang.git
SGLANG_BRANCH=amitw/sgl-miles-nixl

LUSTRE=/lustre/fsw/portfolios/network/users/amitw/miles
SQSH=$LUSTRE/radixark+miles+latest.sqsh
C_NAME=amitw-miles-nohome-64node

LOG_DIR=$LUSTRE/logs

# Bind-mounted at the container paths run.py hardcodes, so these three are not
# free to rename. HOST_MODELS is the pre-populated weights directory — it sits
# outside $LUSTRE because it is shared across models (GLM, Qwen, Kimi).
HOST_MODELS=/lustre/fsw/portfolios/network/users/amitw/models
HOST_CKPT=$LUSTRE/multinode
HOST_DATASETS=$LUSTRE/datasets

# --- sqsh check --------------------------------------------------------------
fail() {
    echo ""
    echo "ERROR: $1"
    echo ""
    shift
    for line in "$@"; do echo "  $line"; done
    echo ""
    exit 1
}

if [ ! -f "$SQSH" ]; then
    fail "sqsh file not found at $SQSH" \
         "To create it, run on a compute node (login node has no disk space):" \
         "srun -A $PORTFOLIO -N 1 --gpus-per-node=8 -p interactive --time=01:00:00 --pty /bin/bash" \
         "cd $LUSTRE/" \
         "enroot import docker://radixark/miles:latest"
fi

# --- Locate the pre-downloaded HF checkpoint ---------------------------------
#
# run.py hardcodes /root/models/$MODEL, and $MODEL has to stay the registry key
# "Kimi-K2-Instruct" because that is what RUN_CONFIGS is keyed on. When the
# weights are stored in HF hub cache layout the two names differ, so bridge
# them with a symlink. The whole models directory is mounted as one tree, so
# the snapshot's relative ../../blobs/<sha> symlinks still resolve.
[ -d "$HOST_MODELS" ] || fail "models directory not found at $HOST_MODELS" \
    "Set HOST_MODELS to the directory holding the Kimi-K2 weights."

HF_DIR=$HOST_MODELS/$MODEL

if [ ! -f "$HF_DIR/config.json" ]; then
    if [ -d "$HF_DIR" ] && [ ! -L "$HF_DIR" ]; then
        fail "$HF_DIR exists but has no config.json" \
             "Remove the incomplete directory or point HOST_MODELS elsewhere."
    fi

    CACHE_NAME=models--moonshotai--$MODEL
    CACHE_DIR=$HOST_MODELS/$CACHE_NAME
    [ -d "$CACHE_DIR" ] || fail "Kimi-K2 weights not found under $HOST_MODELS" \
        "Expected one of:" \
        "  $HOST_MODELS/$MODEL" \
        "  $CACHE_DIR" \
        "This script does not download the model."

    # refs/main names the current revision; fall back to scanning snapshots for
    # caches created without a ref (e.g. a revision-pinned download).
    SNAPSHOT=""
    if [ -f "$CACHE_DIR/refs/main" ]; then
        REV=$(tr -d '[:space:]' < "$CACHE_DIR/refs/main")
        if [ -f "$CACHE_DIR/snapshots/$REV/config.json" ]; then
            SNAPSHOT=snapshots/$REV
        fi
    fi
    if [ -z "$SNAPSHOT" ]; then
        for d in "$CACHE_DIR"/snapshots/*; do
            if [ -f "$d/config.json" ]; then
                SNAPSHOT=snapshots/$(basename "$d")
                break
            fi
        done
    fi
    [ -n "$SNAPSHOT" ] || fail "no usable snapshot in $CACHE_DIR" \
        "None of the snapshots/* directories contains a config.json."

    # Relative on purpose: the link is dereferenced inside the container, where
    # this directory lives at /root/models and the host path does not exist.
    echo "Linking $HF_DIR -> $CACHE_NAME/$SNAPSHOT"
    ln -sfn "$CACHE_NAME/$SNAPSHOT" "$HF_DIR"
fi

# hf writes the shard index last; its absence means the download never finished.
[ -f "$HF_DIR/model.safetensors.index.json" ] || \
    fail "$HF_DIR has no model.safetensors.index.json" \
         "The checkpoint looks incomplete — re-download it before launching."

echo "Using HF checkpoint: $(readlink -f "$HF_DIR")"

# --- Determine which preparation steps are needed ----------------------------
NEEDS_DATASETS=0
NEEDS_CONVERT=0

# convert_hf_to_torch_dist.py writes 'release' into the tracker as its last
# action, so anything else means the conversion did not finish.
TRACKER=$HOST_CKPT/${MODEL}_torch_dist/latest_checkpointed_iteration.txt
if [ ! -f "$TRACKER" ] || [ "$(tr -d '[:space:]' < "$TRACKER")" != "release" ]; then
    echo "Megatron checkpoint missing or incomplete — will submit convert job."
    NEEDS_CONVERT=1
fi

# run.py passes both files by absolute path, so these names are fixed.
for ds in dapo-math-17k/dapo-math-17k.jsonl aime-2024/aime-2024.jsonl; do
    if [ ! -f "$HOST_DATASETS/$ds" ]; then
        echo "Dataset missing: $ds — will submit dataset job."
        NEEDS_DATASETS=1
    fi
done

mkdir -p "$LOG_DIR"
PREV_JOB_ID=""

# --- Submit dataset job if needed --------------------------------------------
if [ "$NEEDS_DATASETS" -eq 1 ]; then
    DS_JOB_SCRIPT=$LOG_DIR/job-kimi-k2-datasets.slurm
    cat > "$DS_JOB_SCRIPT" <<DS_SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2-datasets
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/miles-kimi-k2-datasets-%j.out

srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-datasets" \\
    --container-mounts="$HOST_DATASETS:/root/datasets" \\
    bash -lc '
set -ex

echo "--- [Datasets] Downloading dataset: dapo-math-17k ---"
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k

echo "--- [Datasets] Downloading dataset: aime-2024 ---"
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024

echo "--- [Datasets] Done ---"
'
DS_SLURM

    DS_JOB_ID=$(sbatch --parsable "$DS_JOB_SCRIPT")
    echo "Submitted dataset job $DS_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-kimi-k2-datasets-${DS_JOB_ID}.out"
    PREV_JOB_ID=$DS_JOB_ID
fi

# --- Submit convert job if needed --------------------------------------------
if [ "$NEEDS_CONVERT" -eq 1 ]; then
    CONV_DEP=""
    if [ -n "$PREV_JOB_ID" ]; then
        CONV_DEP="--dependency=afterok:$PREV_JOB_ID"
    fi

    CONV_JOB_SCRIPT=$LOG_DIR/job-kimi-k2-convert.slurm
    cat > "$CONV_JOB_SCRIPT" <<CONV_SLURM
#!/bin/bash
#SBATCH --job-name=miles-kimi-k2-convert
#SBATCH --partition=$PARTITION
#SBATCH --account=$PORTFOLIO
#SBATCH --nodes=$CONV_NODES
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=04:00:00
#SBATCH --output=$LOG_DIR/miles-kimi-k2-convert-%j.out

srun \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="${C_NAME}-convert" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode" \\
    bash -lc '
set -ex

echo "--- [Convert Node \$SLURM_NODEID] Updating Miles ($MILES_BRANCH) ---"
git -C /root/miles remote add fork $MILES_FORK 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork $MILES_BRANCH
git -C /root/miles checkout -B $MILES_BRANCH fork/$MILES_BRANCH
pip install -q "numpy<2" "scipy<1.15"

MASTER_ADDR=\$(python3 -c "
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
    print(\"ERROR: \" + str(e), file=sys.stderr)
    sys.exit(1)
")

FLAG=/root/multinode/.fp8_done_\$SLURM_JOB_ID

if [ \$SLURM_NODEID -eq 0 ]; then
    echo "--- [Convert] Checking for FP8 weights ---"
    if grep -ql fp8 /root/models/$MODEL/config.json 2>/dev/null; then
        echo "--- [Convert] FP8 model detected, casting to BF16 ---"
        if [ ! -f /root/models/${MODEL}-bf16/model.safetensors.index.json ]; then
            cd /root/miles/tools && python3 fp8_cast_bf16.py --input-fp8-hf-path /root/models/$MODEL --output-bf16-hf-path /root/models/${MODEL}-bf16
        else
            echo "--- [Convert] BF16 copy already exists, skipping cast ---"
        fi
        echo /root/models/${MODEL}-bf16 > \$FLAG
    else
        echo /root/models/$MODEL > \$FLAG
    fi
fi

echo "--- [Convert Node \$SLURM_NODEID] Waiting for node 0 FP8 cast ---"
while [ ! -f "\$FLAG" ]; do sleep 30; done
HF_CKPT=\$(cat \$FLAG)

echo "--- [Convert Node \$SLURM_NODEID] Converting \$HF_CKPT to torch_dist ---"
cd /root/miles
source scripts/models/kimi-k2.sh
torchrun --nnodes=$CONV_NODES --nproc_per_node=8 --node-rank=\$SLURM_NODEID --rdzv_id=\$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=\$MASTER_ADDR:29500 tools/convert_hf_to_torch_dist.py "\${MODEL_ARGS[@]}" --hf-checkpoint \$HF_CKPT --save /root/multinode/${MODEL}_torch_dist --num-layers 61 --pipeline-model-parallel-size 8 --expert-model-parallel-size 8 --decoder-last-pipeline-num-layers 5

if [ \$SLURM_NODEID -eq 0 ]; then rm -f \$FLAG; fi
echo "--- [Convert Node \$SLURM_NODEID] Done ---"
'
CONV_SLURM

    CONV_JOB_ID=$(sbatch --parsable $CONV_DEP "$CONV_JOB_SCRIPT")
    echo "Submitted conversion job $CONV_JOB_ID"
    echo "  tail -f $LOG_DIR/miles-kimi-k2-convert-${CONV_JOB_ID}.out"
    PREV_JOB_ID=$CONV_JOB_ID
fi

# --- Print launch info -------------------------------------------------------
MAIN_DEP=""
if [ -n "$PREV_JOB_ID" ]; then
    MAIN_DEP="--dependency=afterok:$PREV_JOB_ID"
fi

echo ""
echo "=========================================="
echo "  Miles cluster launch — Kimi-K2 ${NUM_NODES}-node"
echo "=========================================="
echo "  Portfolio : $PORTFOLIO"
echo "  Partition : $PARTITION"
echo "  Model     : $MODEL"
echo "  Mode      : $MODE"
echo "  Nodes     : $NUM_NODES"
echo "  Miles     : $MILES_BRANCH"
echo "  SGLang    : $SGLANG_BRANCH"
echo "  Container : $C_NAME (from $SQSH)"
echo "  Model dir : $(readlink -f "$HF_DIR")"
echo "  Ckpt dir  : $HOST_CKPT/${MODEL}_torch_dist"
echo "  Logs      : $LOG_DIR"
if [ -n "$PREV_JOB_ID" ]; then
    echo "  Waiting on: prep job $PREV_JOB_ID"
fi
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

srun --mpi=pmix \\
    --container-image="$SQSH" \\
    --no-container-mount-home \\
    --container-name="$C_NAME" \\
    --container-mounts="$HOST_MODELS:/root/models,$HOST_CKPT:/root/multinode,$HOST_DATASETS:/root/datasets,\$HOST_SIGNALS:/root/signals" \\
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
# run.py raises NotImplementedError for Kimi-K2-Instruct unless SKIP_VALIDATION
# is set: the model has no --check-weight-update-equal path.
echo "--- [Node \$NODE_RANK] Starting $MODEL (mode=$MODE) ---"
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/signals
cd /root/miles
bash examples/p2p_weight_transfer/$MODEL_SCRIPT.sh $MODE "\$NODE_RANK" "\$HEAD_NODE_IP"

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
