#!/usr/bin/env python3
"""
Convert Qwen3-30B-A3B expert weights to binary format for fast pread.
Processes one shard at a time, saves per-layer .bin files.
Also saves pinned.safetensors + config.json.

After conversion, the HF cache can be deleted to free ~17 GB.
"""
import os, sys, json, time, gc, glob, shutil
import numpy as np
import mlx.core as mx

sys.stdout.reconfigure(line_buffering=True)

OUTPUT_DIR = os.path.expanduser("~/models/qwen3-30b")
PAGE_SIZE = 16384

TENSOR_NAMES = [
    "gate_proj.weight", "gate_proj.scales", "gate_proj.biases",
    "up_proj.weight", "up_proj.scales", "up_proj.biases",
    "down_proj.weight", "down_proj.scales", "down_proj.biases",
]


def convert_layer_to_bin(layer_data, layer_idx, num_experts):
    """Convert one layer's expert tensors to binary format."""
    # Calculate per-expert block size
    tensor_info = {}
    expert_block_size = 0
    for name in TENSOR_NAMES:
        key = f"{name}"
        t = layer_data[key]
        per_expert_shape = list(t.shape[1:])
        if t.dtype == mx.uint32:
            elem_size = 4
        elif t.dtype in (mx.bfloat16, mx.float16):
            elem_size = 2
        else:
            elem_size = 4
        nbytes = 1
        for s in per_expert_shape:
            nbytes *= s
        nbytes *= elem_size
        tensor_info[name] = {
            "shape_per_expert": per_expert_shape,
            "dtype": str(t.dtype).replace("mlx.core.", ""),
            "nbytes": nbytes,
            "inner_offset": expert_block_size,
        }
        expert_block_size += nbytes

    header = {
        "layer_idx": layer_idx,
        "num_experts": num_experts,
        "layout": {
            "expert_block_size": expert_block_size,
            "data_start": PAGE_SIZE,
            "tensors": tensor_info,
        }
    }
    header_bytes = json.dumps(header, indent=2).encode()
    assert len(header_bytes) < PAGE_SIZE
    header_bytes += b"\x00" * (PAGE_SIZE - len(header_bytes))

    out_path = f"{OUTPUT_DIR}/bin/moe_layer_{layer_idx:02d}.bin"
    with open(out_path, "wb") as f:
        f.write(header_bytes)
        for expert_id in range(num_experts):
            for name in TENSOR_NAMES:
                t = layer_data[name][expert_id]
                if t.dtype == mx.bfloat16:
                    raw = np.array(t.astype(mx.float16)).astype(np.float16).tobytes()
                elif t.dtype == mx.uint32:
                    raw = np.array(t).astype(np.uint32).tobytes()
                else:
                    raw = np.array(t).tobytes()
                f.write(raw)

    return os.path.getsize(out_path), expert_block_size


def main():
    cache_dir = os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3-30B-A3B-4bit"
    )
    snap = glob.glob(f"{cache_dir}/snapshots/*/")[0]

    print("=" * 60)
    print("  Convert Qwen3-30B-A3B to binary sniper format")
    print("=" * 60)

    os.makedirs(f"{OUTPUT_DIR}/bin", exist_ok=True)
    shutil.copy(f"{snap}/config.json", f"{OUTPUT_DIR}/config.json")

    with open(f"{snap}/model.safetensors.index.json") as f:
        idx = json.load(f)

    shards = sorted(set(idx["weight_map"].values()))
    print(f"  {len(shards)} shards")

    pinned = {}
    layers_done = set()

    for si, shard_name in enumerate(shards):
        print(f"\n[{si+1}/{len(shards)}] {shard_name}...")
        t0 = time.time()
        data = mx.load(f"{snap}/{shard_name}")
        print(f"  Loaded {len(data)} tensors")

        # Classify
        layer_experts = {}
        for key, tensor in data.items():
            if "switch_mlp" in key:
                layer = int(key.split(".layers.")[1].split(".")[0])
                short = key.split(".switch_mlp.")[1]
                layer_experts.setdefault(layer, {})[short] = tensor
            else:
                pinned[key] = tensor

        # Convert expert layers
        for layer_idx, tensors in layer_experts.items():
            if len(tensors) < 9:
                # Partial layer — accumulate
                existing_path = f"{OUTPUT_DIR}/bin/moe_layer_{layer_idx:02d}.partial.json"
                if os.path.exists(existing_path):
                    # Load partial and merge
                    # For now, save as safetensors and merge later
                    pass
                continue

            if layer_idx in layers_done:
                continue

            num_experts = tensors[list(tensors.keys())[0]].shape[0]
            sz, ebs = convert_layer_to_bin(tensors, layer_idx, num_experts)
            layers_done.add(layer_idx)
            print(f"  Layer {layer_idx}: {sz/1e6:.0f} MB ({ebs/1e6:.2f} MB/expert)")

        del data, layer_experts
        gc.collect()
        mx.clear_cache()
        print(f"  Done in {time.time()-t0:.1f}s")

    # Handle partial layers (span multiple shards)
    # Re-scan for incomplete layers
    incomplete = set(range(48)) - layers_done
    if incomplete:
        print(f"\nIncomplete layers: {sorted(incomplete)}")
        print("Re-scanning shards for partial layers...")
        for layer_idx in sorted(incomplete):
            tensors = {}
            for shard_name in shards:
                data = mx.load(f"{snap}/{shard_name}")
                for key, tensor in data.items():
                    if f".layers.{layer_idx}." in key and "switch_mlp" in key:
                        short = key.split(".switch_mlp.")[1]
                        tensors[short] = tensor
                del data
            if len(tensors) >= 9:
                num_experts = tensors[list(tensors.keys())[0]].shape[0]
                sz, ebs = convert_layer_to_bin(tensors, layer_idx, num_experts)
                layers_done.add(layer_idx)
                print(f"  Layer {layer_idx}: {sz/1e6:.0f} MB (merged)")

    # Save pinned
    print(f"\nSaving pinned ({len(pinned)} keys)...")
    pinned_path = f"{OUTPUT_DIR}/pinned.safetensors"
    mx.save_safetensors(pinned_path, pinned)
    psz = os.path.getsize(pinned_path) / 1e9
    print(f"  Pinned: {psz:.2f} GB")
    del pinned

    # Verify
    bin_files = sorted(glob.glob(f"{OUTPUT_DIR}/bin/moe_layer_*.bin"))
    total = sum(os.path.getsize(f) for f in bin_files)
    print(f"\n  Expert layers: {len(bin_files)}/48")
    print(f"  Expert total: {total/1e9:.2f} GB")
    print(f"  Pinned: {psz:.2f} GB")
    print(f"  DONE!")


if __name__ == "__main__":
    main()
