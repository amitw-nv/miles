"""Cast Kimi-K2 FP8 safetensor shards to BF16, one shard at a time.

Processes shards sequentially in a single process (so Triton compiles once)
and frees GPU memory between each shard to avoid OOM on 80 GiB GPUs.

Usage:
    python3 cast_kimi_k2_bf16.py <fp8_dir> <bf16_dir>

fp8_dir must be the full model directory (all shards present) so that
cross-shard scale_inv lookups can still load from the original files.
Existing output shards are skipped — safe to re-run after a failure.
"""
import json
import os
import struct
import sys
from glob import glob

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_cast_bf16 import weight_dequant
from param_name_remap import get_param_name_remap


def cast_all_shards(fp8_path: str, bf16_path: str) -> None:
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    for pattern in ["config.json", "*.py", "tokenizer*", "generation_config.json", "*.model"]:
        os.system(f"cp -f {fp8_path}/{pattern} {bf16_path}/ 2>/dev/null")

    with open(os.path.join(fp8_path, "model.safetensors.index.json")) as f:
        weight_map_raw = json.load(f)["weight_map"]

    remap = get_param_name_remap(os.path.join(fp8_path, "config.json"), weight_map_raw)
    raw_name_by_renamed = {remap(raw): raw for raw in weight_map_raw}

    shards = sorted(glob(os.path.join(fp8_path, "model-*-of-*.safetensors")))
    print(f"Found {len(shards)} shards")

    weight_map_bf16: dict[str, str] = {}

    for i, shard_file in enumerate(shards):
        shard_name = os.path.basename(shard_file)
        out_file = os.path.join(bf16_path, shard_name)

        if os.path.isfile(out_file):
            print(f"[{i+1}/{len(shards)}] Skipping {shard_name} (already done)")
            with open(out_file, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            weight_map_bf16.update({k: shard_name for k in header if k != "__metadata__"})
            continue

        print(f"[{i+1}/{len(shards)}] Casting {shard_name} ...")

        loaded_files: dict = {}

        def get_tensor(renamed_name: str):
            raw = raw_name_by_renamed[renamed_name]
            fname = weight_map_raw[raw]
            if fname not in loaded_files:
                loaded_files[fname] = load_file(os.path.join(fp8_path, fname), device="cuda")
            return loaded_files[fname][raw]

        state = load_file(shard_file, device="cuda")
        loaded_files[shard_name] = state

        out: dict = {}
        for raw_name, weight in state.items():
            name = remap(raw_name)
            if name.endswith("_scale_inv"):
                continue
            if weight.element_size() == 1:  # FP8
                try:
                    out[name] = weight_dequant(weight, get_tensor(f"{name}_scale_inv"))
                except KeyError:
                    print(f"  Warning: no scale_inv for {name}, keeping as-is")
                    out[name] = weight
            else:
                out[name] = weight

        save_file(out, out_file)
        weight_map_bf16.update({k: shard_name for k in out})

        del out, state, loaded_files
        torch.cuda.empty_cache()
        print(f"  -> {out_file}")

    index_path = os.path.join(bf16_path, "model.safetensors.index.json")
    with open(index_path, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map_bf16}, f, indent=2)
    print(f"Done. {len(weight_map_bf16)} tensors across {len(shards)} shards -> {index_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    cast_all_shards(sys.argv[1], sys.argv[2])
