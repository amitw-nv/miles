# Launch spec: Kimi-K2 on the cluster

This document is the input an agent reads to build or rebuild
[launch_kimi_k2_on_cluster.sh](launch_kimi_k2_on_cluster.sh); the script is the
output.

Steps 1-6 are the same steps, in the same order, as steps 1-6 of
[launch_glm_z1_9b_1node.sh](launch_glm_z1_9b_1node.sh) — only the model and dataset
values change. Step 7 is Kimi-specific and comes from
[docs/models/kimi/kimi-k2.md](../../models/kimi/kimi-k2.md). Step 0 is the batch
wrapper around all of it, and step 8 launches the training job.

All seven in-container steps run on every launch and must be idempotent:
each one checks for its own result first and skips instantly if it is already
there. Use `set -ex` so every command is echoed and the first failure stops the
run.

Not every step runs on every node. The model, datasets and converted checkpoint
live on shared storage (see [step 0](#step-0-the-wrapper)), so exactly one node
may write them:

- Steps 1-4 run on all 64 nodes. They touch only container-local paths.
- Steps 5, 6 and 7a run on rank 0 only, then rank 0 signals `prep_done`.
  Unguarded, all 64 nodes would download the same 1 TB model into the same
  directory at once.
- Step 7b runs on the first `CONVERT_NODES` (4) ranks as one `torchrun` job.
  Ranks 1-3 wait for `prep_done` before joining.
- Every rank, including the 60 that take no part in the conversion, then waits
  for the step 7d tracker before going on to step 8.

## Values used below

| Name | Value |
|---|---|
| `MILES_FORK` | `https://github.com/amitw-nv/miles.git` |
| `MILES_BRANCH` | `amitw/miles-nixl` |
| `SGLANG_FORK` | `https://github.com/amitw-nv/sglang.git` |
| `SGLANG_BRANCH` | `amitw/sgl-miles-nixl` |
| `MODEL` | `Kimi-K2-Instruct` |
| `HF_REPO` | `moonshotai/Kimi-K2-Instruct` |
| `MODEL_TYPE` | `kimi-k2` (selects `scripts/models/kimi-k2.sh`) |
| `PORTFOLIO` | `network_research_advdev` |
| `LUSTRE` | `/lustre/fsw/portfolios/network/users/$USER/miles` |
| `SQSH` | `$LUSTRE/radixark+miles+latest.sqsh` |
| `C_NAME` | `${USER}-miles-kimi-k2` |
| `MODE` | first argument, `nixl` (default) \| `p2p` \| `broadcast` |
| `PREP_WAIT_TIMEOUT` | `7200` seconds, how long ranks 1-3 wait for `prep_done` (see 7e) |
| `CONVERT_WAIT_TIMEOUT` | `7200` seconds, how long every rank waits for the tracker (see 7e) |

Miles lives at `/root/miles` in the image and SGLang at `/sgl-workspace/sglang`.

## Step 0: The wrapper

Submit with `sbatch`, not `srun --pty` like the GLM script: 512 GPUs will not be
handed to an interactive session. So the script is a batch script, and step 8
launches the job itself instead of dropping into a shell.

The `#SBATCH` block asks for 64 nodes, which is what
`RUN_CONFIGS["Kimi-K2-Instruct"]` in
[examples/p2p_weight_transfer/run.py](../../../examples/p2p_weight_transfer/run.py)
declares (`nnodes=64`, 256 train + 256 rollout GPUs): job name `miles-kimi-k2`,
account `$PORTFOLIO`, partition `batch`, `--ntasks-per-node=1`,
`--gpus-per-node=8`, `--exclusive`, and a `--time` long enough to cover the
first-launch download and conversion as well as training. Send `--output` and
`--error` under `$LUSTRE/kimi-k2-logs/`; Slurm needs that directory to exist
before the job starts, so the usage notes have to say `mkdir -p` it once.

Refuse to run without `SLURM_JOB_ID` and print the `sbatch` line instead. The
`#SBATCH` directives are ordinary comments when the file is executed directly, so
without an allocation there is no node list to read and the script would
otherwise fail much further down with a confusing error about the head node.
Validate `$SQSH` the same way the GLM script does.

Resolve the head node on the host, before `srun`, and export the result:

```bash
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{print $1}')
```

The container has no `scontrol`, which is why this cannot move inside. Error out
if the lookup comes back empty rather than letting `torchrun` fail on an empty
`--master-addr`.

### Mounts

The GLM script deliberately mounts nothing — a named container persists `/root`
between sessions on one node — but 63 other nodes cannot see that, so every
artifact has to live on Lustre. Mount it at the paths `run.py` hardcodes:

| Host | Container | Why this exact path |
|---|---|---|
| `$LUSTRE/models` | `/root/models` | `--hf-checkpoint /root/models/{model_name}` |
| `$LUSTRE/multinode` | `/root/multinode` | `ckpt_dir`; `CKPT_SAVE_DIR` is honoured by `prepare` but not by `run` for Kimi-K2 |
| `$LUSTRE/datasets` | `/root/datasets` | `/root/datasets/<name>/<name>.jsonl` |
| `$LUSTRE/kimi-k2-logs/job-$SLURM_JOB_ID` | `/root/logs` | `MILES_LOG_DIR`, and the `prep_done` signal of 7a |

Mounting these four subdirectories is safe; mounting over `/root` or home breaks
the Megatron import, so `--no-container-mount-home` stays. pyxis will not create
mount sources, so `mkdir -p` all four on the host first. Keep the log directory
per job — that is what makes the `prep_done` signal of 7a impossible to inherit
from an earlier run.

Pass the in-container command as a single quoted heredoc so nothing is expanded at
submit time, and hand it the values it needs by exporting them; `SLURM_NODEID` is
set per task by Slurm and is the `RANK` the steps below refer to. `srun` takes
`--ntasks-per-node=1`, `--container-image=$SQSH`, `--container-name=$C_NAME` and
`--container-workdir=/root/miles`, and needs no `--mpi=pmix`: nothing here is an
MPI program, Ray bootstraps itself over TCP from the head, so `srun` only has to
start one container per node.

## Step 1: Update Miles

Check out the fork branch over whatever the image shipped with. Add the remote if
it is missing, discard any local modifications first so the checkout cannot fail
on a dirty tree, then force the local branch to the fetched head:

```bash
git -C /root/miles remote add fork "$MILES_FORK" 2>/dev/null || true
git -C /root/miles restore . 2>/dev/null || git -C /root/miles checkout . 2>/dev/null || true
git -C /root/miles fetch fork "$MILES_BRANCH"
git -C /root/miles checkout -B "$MILES_BRANCH" "fork/$MILES_BRANCH"
```

The `remote add` is tolerated failing because it is only new the first time. The
`restore` falls back to `checkout` for older git versions.

## Step 2: Update SGLang

The same four commands against `/sgl-workspace/sglang` with `SGLANG_FORK` and
`SGLANG_BRANCH`.

## Step 3: Fix numpy and scipy versions

The image ships versions Miles cannot use. Pin them before anything imports them:

```bash
pip install -q "numpy<2" "scipy<1.15"
```

## Step 4: Print active commits

So a log on its own identifies exactly what ran, print the head commit of both
repos and the resolved numpy version:

```bash
echo "Miles  : $(git -C /root/miles log --oneline -1)"
echo "SGLang : $(git -C /sgl-workspace/sglang log --oneline -1)"
echo "numpy  : $(python -c 'import numpy; print(numpy.__version__)')"
```

## Step 5: Download datasets

Rank 0 only. `run.py` passes both files by absolute path, so they have to land
exactly where it expects — the destination paths are not free to change. Skip on
the `.jsonl` being present, and `ls -la` it afterwards so the log shows the size.

Prompt set, `/root/datasets/dapo-math-17k/dapo-math-17k.jsonl`:

```bash
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k
```

Eval set, `/root/datasets/aime-2024/aime-2024.jsonl`:

```bash
hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024
```

The eval set is the one addition to the GLM step: GLM downloads only the prompt
set, while the Kimi recipe evaluates on AIME-2024
([kimi-k2.md § 3.2](../../models/kimi/kimi-k2.md)).

## Step 6: Download model

Rank 0 only.

```bash
mkdir -p /root/models
hf download "$HF_REPO" --local-dir /root/models/$MODEL
du -sh /root/models/$MODEL
```

Gate it on a completion sentinel, `/root/models/.${MODEL}.download_complete`,
written only after `hf download` returns successfully. This is the one place where
the GLM step cannot be copied as-is: GLM skips when the model directory is merely
non-empty, which for a ~1 TB download would treat an interrupted transfer as
finished. `hf download` resumes, so a rerun after a failure only fetches what is
missing.

Key the sentinel on the download alone, not on the converted checkpoint of step 7,
so a failed conversion does not drag the 1 TB download back into scope.

If `moonshotai/Kimi-K2-Instruct` is gated, `HF_TOKEN` has to be exported before
launch; the container runtime passes the environment through.

## Step 7: Convert the checkpoint to Megatron `torch_dist`

From [kimi-k2.md § 3.3](../../models/kimi/kimi-k2.md), which converts across 4
nodes mirroring the DeepSeek-V3 procedure. Note this differs in kind from GLM:
GLM calls the `convert_checkpoint()` Python helper on one node, whereas Kimi-K2 is
a `torchrun` job spanning 4 nodes and 32 GPUs.

Only the first 4 ranks convert. Any node outside that set waits for the tracker
file of 7d before going on, and 7a is rank 0's alone.

### 7a. Prerequisite: point the config at the DeepSeek-V3 loader

mbridge has no `kimi_k2` entry and fails with `Unregistered model type: kimi_k2,
now only support dict_keys([...])`. Kimi-K2 is DeepSeek-V3-shaped and its HF
config already declares `DeepseekV3ForCausalLM` as the architecture, so only
`model_type` is in the way — this is the `sed` that
[kimi-k2.md § 1](../../models/kimi/kimi-k2.md) alludes to.

Rewrite `model_type` in `/root/models/$MODEL/config.json` from `kimi_k2` to
`deepseek_v3`. Do it with a JSON load and dump rather than a text substitution,
make it idempotent by returning early when the value is already `deepseek_v3`, and
copy the original to `config.json.orig` the first time.

Rank 0 owns this, because it owns the download the config belongs to. Only once
the patch has returned does it `touch` the `prep_done` signal that releases ranks
1-3 into 7b — otherwise a converting rank could read a half-written
`config.json`, or read it before the rewrite and hit the mbridge error above. Put
the signal in the per-job log directory of step 0, never on a path that outlives
the job, so it cannot be a leftover from an earlier run that would release the
ranks immediately.

### 7b. The conversion command

```bash
cd /root/miles
source scripts/models/$MODEL_TYPE.sh      # defines MODEL_ARGS
PYTHONPATH=/root/Megatron-LM/ torchrun \
    --nproc-per-node 8 \
    --master-addr "$HEAD_NODE_IP" --master-port 12345 \
    --nnodes=4 --node-rank "$RANK" \
    tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 8 \
    --expert-tensor-parallel-size 1 \
    --expert-model-parallel-size 4 \
    --decoder-last-pipeline-num-layers 5 \
    --hf-checkpoint /root/models/$MODEL \
    --save /root/multinode/${MODEL}_torch_dist
```

`kimi-k2.md § 3.1` asks for `MASTER_ADDR` and `NODE_RANK` to be exported by hand.
In a batch job the script resolves them itself: the master address is the first
host of the allocation, and the rank comes from the scheduler.

### 7c. The five parallel-size flags § 3.3 omits

The doc's snippet passes no parallel sizes, and it must not be copied literally.
Both ways of getting this wrong have already been tried here:

- No parallel sizes at all, as printed in § 3.3. Nothing is sharded, every rank
  builds the whole trillion-parameter model, and it dies in `load_weights` with
  CUDA OOM at 79 GiB.
- The `PREPARE_CONFIGS` values from `examples/p2p_weight_transfer/run.py` on a
  single node. Megatron rejects it before loading any weights: `world_size (8) is
  not divisible by expert_tensor_model_pipeline_parallel size (64)`.

Two constraints fix the values above, and changing one means re-deriving the rest:

- `ETP * EP * PP` must equal the GPU count, `8 * 4 nodes = 32`. Here
  `1 * 4 * 8 = 32`. Megatron rejects any other product outright.
- The pipeline split must cover all 61 layers (`scripts/models/kimi-k2.sh` sets
  `--num-layers 61`). Seven stages of 8 plus a last stage of 5 is 61, hence
  `--decoder-last-pipeline-num-layers 5`.

The values come from
[deepseek.md § 3.3](../../models/deepseek/deepseek.md), which § 3.3 of the Kimi
page points at when it says to mirror the DeepSeek-V3 procedure. The four parallel
sizes are identical there. Only the layer split differs: DeepSeek-R1 covers its 61
layers as `7 + 6 x 8 + 6` using both a first- and last-stage override, while the
command above uses `7 x 8 + 5` and so needs the last-stage override only.

Do not reach for the training-side table in
[kimi-k2.md § 5.1](../../models/kimi/kimi-k2.md) instead. Those sizes (TP 8, EP 32)
are for the 256-GPU training run, not for this 32-GPU conversion.

### 7d. Skip condition

The tracker file
`/root/multinode/${MODEL}_torch_dist/latest_checkpointed_iteration.txt`.
`convert_hf_to_torch_dist.py` writes `release` into it as its very last action,
after a barrier, so its presence means the checkpoint is complete and usable — skip
the conversion when it is already there, and report the value it contains.

Because it is written last and only on success, the same file doubles as the ready
signal for the 60 ranks that took no part in the conversion. Every rank polls it
before step 8, and matches on the contents being `release`, not merely on the file
existing.

### 7e. Both waits are bounded

Neither poll may spin forever. Rank 0 is a single point of failure for both of
them: if it dies mid-download, ranks 1-3 wait on a `prep_done` that will never
appear and the other 60 wait on a tracker that will never be written, so all 63
poll until the walltime expires — hours of 512 idle GPUs on a job that has
already failed, and a log whose last line is a wait message rather than the error.

Both waits therefore share one helper, so they behave identically:

```bash
wait_for <label> <path> <timeout-seconds> <knob-name> [expected-contents]
```

It polls every 30 seconds and returns as soon as the file exists and — when
`expected-contents` is given — matches after whitespace is stripped. On expiry it
prints the label, the path, that rank 0 is the likely cause, and `knob-name` so
the log says which variable to raise, then exits non-zero so the failure surfaces
on every rank rather than only on the one that died.

The two timeouts are separate because the waits scale differently: the first
covers a 1 TB download, the second a 32-GPU conversion. Both default to two hours
and both read from the environment, so they can be raised at submit time. A
genuine first launch has to raise them together with `--time`, which bounds
everything anyway.

## Step 8: Launch

```bash
export SKIP_VALIDATION=1
export MILES_LOG_DIR=/root/logs
cd /root/miles
bash examples/p2p_weight_transfer/Kimi-K2.sh "$MODE" "$RANK" "$HEAD_NODE_IP"
```

Every rank runs this, with its own `RANK`; `Kimi-K2.sh` forwards all three to
`run.py`, which starts the Ray head on rank 0 and joins the rest to it.

`SKIP_VALIDATION=1` is required: `run.py` refuses to run Kimi-K2-Instruct with
`--check-weight-update-equal` unless it is set, so weight validation is off (see
[p2p-weight-transfer.md](../p2p-weight-transfer.md)).

`MILES_LOG_DIR` has to be the shared mount. `run.py` has the head write
`job_done_<mode>` there when training ends and the workers poll for it; on a
container-local path the workers would never see it. It defaults to
`/data/ray/signals`, which is not mounted, so set it explicitly.

`Kimi-K2.sh` also runs `run.py prepare` itself when
`/root/multinode/Kimi-K2-Instruct_torch_dist` is missing. After step 7 it never
is, which is the point of doing the conversion here: `prepare` would run it with
the `PREPARE_CONFIGS` sizes that 7c rules out.
