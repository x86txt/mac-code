"""
Fast expert reader for Nemotron — pread only active experts from binary files.

6 active experts × 5.61 MB = 33.7 MB per layer (vs 718 MB for full layer).
23 MoE layers × 33.7 MB = 775 MB per token.
At 5 GB/s NVMe: 155ms → ~6 tok/s theoretical.
"""

import os
import json
import fcntl
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx

F_NOCACHE = 48
PAGE_SIZE = 16384

MLX_DTYPES = {
    "uint32": mx.uint32,
    "float16": mx.float16,
    "bfloat16": mx.float16,  # We converted bf16→f16 in the bin files
    "float32": mx.float32,
}


class NemotronExpertReader:
    """Reads specific experts from binary layer files via pread."""

    def __init__(self, bin_dir, num_workers=4):
        self.bin_dir = bin_dir
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

        # Parse headers for all MoE layer files
        self.headers = {}
        self.fds = {}
        for f in os.listdir(bin_dir):
            if f.startswith("moe_layer_") and f.endswith(".bin"):
                layer_idx = int(f.split("_")[2].split(".")[0])
                path = os.path.join(bin_dir, f)
                with open(path, "rb") as fh:
                    raw = fh.read(PAGE_SIZE)
                self.headers[layer_idx] = json.loads(raw.rstrip(b"\x00"))

        if not self.headers:
            raise FileNotFoundError(f"No bin files in {bin_dir}")

        # Layout from first layer (all layers have same layout)
        first = next(iter(self.headers.values()))["layout"]
        self.expert_block_size = first["expert_block_size"]
        self.data_start = first["data_start"]
        self.tensor_layout = first["tensors"]

        # Stats
        self.read_time = 0.0
        self.reads = 0
        self.bytes_read = 0

    def _get_fd(self, layer_idx):
        if layer_idx not in self.fds:
            path = os.path.join(self.bin_dir, f"moe_layer_{layer_idx:02d}.bin")
            fd = os.open(path, os.O_RDONLY)
            fcntl.fcntl(fd, F_NOCACHE, 1)
            self.fds[layer_idx] = fd
        return self.fds[layer_idx]

    def _read_expert(self, layer_idx, expert_id):
        """Read one expert's raw bytes via pread."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size
        return os.pread(fd, self.expert_block_size, offset)

    def _parse_expert(self, raw_bytes):
        """Parse raw bytes into dict of MLX arrays."""
        result = {}
        for name, info in self.tensor_layout.items():
            off = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype = MLX_DTYPES.get(info["dtype"], mx.float16)

            arr_bytes = raw_bytes[off:off + nbytes]
            if dtype == mx.uint32:
                np_arr = np.frombuffer(arr_bytes, dtype=np.uint32).reshape(shape)
            elif dtype == mx.float16:
                np_arr = np.frombuffer(arr_bytes, dtype=np.float16).reshape(shape)
            elif dtype == mx.float32:
                np_arr = np.frombuffer(arr_bytes, dtype=np.float32).reshape(shape)
            else:
                np_arr = np.frombuffer(arr_bytes, dtype=np.float16).reshape(shape)

            result[name] = mx.array(np_arr)
        return result

    def get_experts(self, layer_idx, expert_ids):
        """Read and parse active experts for a layer.

        Args:
            layer_idx: MoE layer index
            expert_ids: list of active expert IDs (e.g., [3, 17, 42, 88, 100, 120])

        Returns:
            dict[expert_id] -> dict[tensor_name -> mx.array]
        """
        t0 = time.time()

        # Parallel pread for all active experts
        futures = {
            eid: self.executor.submit(self._read_expert, layer_idx, eid)
            for eid in expert_ids
        }

        experts = {}
        for eid, future in futures.items():
            raw = future.result()
            experts[eid] = self._parse_expert(raw)
            self.bytes_read += len(raw)

        self.read_time += time.time() - t0
        self.reads += len(expert_ids)
        return experts

    def stack_experts(self, expert_data, expert_ids, tensor_name):
        """Stack individual expert tensors into (K, ...) format for gather_qmm."""
        return mx.stack([expert_data[eid][tensor_name] for eid in expert_ids])

    def stats(self):
        if self.reads == 0:
            return "No reads"
        avg_ms = self.read_time / self.reads * 1000
        throughput = self.bytes_read / max(self.read_time, 0.001) / 1e9
        return (f"reads={self.reads}, avg={avg_ms:.1f}ms/expert, "
                f"throughput={throughput:.1f} GB/s, "
                f"total={self.bytes_read/1e9:.2f} GB")

    def close(self):
        for fd in self.fds.values():
            os.close(fd)
        self.executor.shutdown(wait=False)
