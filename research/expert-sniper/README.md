# Expert Sniper: Running a 17 GB Model with 0.87 GB of RAM

A custom MoE inference engine that runs models larger than your RAM on Apple Silicon by loading only the active experts from SSD.

## The Problem

Qwen3-30B-A3B is a 17.2 GB model at 4-bit quantization. On a 16 GB Mac, every standard tool fails:

| Tool | Result |
|------|--------|
| mlx_lm (standard MLX loader) | **OOM — crashes** |
| llama.cpp with Metal GPU | **OOM — crashes** |
| llama.cpp with mmap | **15+ min at 98% CPU, no output** |

## The Solution

MoE models have 128 experts per layer but only use 8 per token. Expert Sniper loads only those 8 from SSD — 21 MB per layer instead of 340 MB.

**Measured results on Apple M4, 16 GB RAM:**

| Metric | Value |
|--------|-------|
| Model on disk | 17.2 GB |
| RAM used (pinned) | 0.87 GB |
| Sustained speed (200 tokens) | **4.33 tok/s** |
| p50 latency | **201 ms (4.97 tok/s)** |
| p99 latency | 537 ms (1.86 tok/s) |
| Cache hit rate | 88.5% |
| Output quality | Coherent — code, math, translations, essays |

## How It Works

1. **Pinned weights (0.87 GB)** — attention, router, norms, embeddings. Always in RAM.
2. **Expert weights (16.3 GB)** — stored in per-layer binary files on SSD. Indexed for random access via `pread`.
3. **Per-expert LRU cache (2,000 experts)** — consecutive tokens use ~88% of the same experts. Cache catches most requests before they hit SSD.
4. **gather_qmm compute** — MLX's fused quantized matmul selects and computes only the active experts.

## Architecture

```
                    ┌─────────────────┐
                    │   Router (pinned)│
                    │   128 → top-8    │
                    └────────┬────────┘
                             │ expert IDs
                    ┌────────▼────────┐
                    │  Per-Expert LRU  │
                    │  Cache (5.3 GB)  │──── 88.5% hit
                    └────────┬────────┘
                      miss   │
                    ┌────────▼────────┐
                    │  SSD pread      │
                    │  21 MB/layer    │──── F_NOCACHE + 4 workers
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  gather_qmm     │
                    │  (0.1% of time) │──── SwiGLU fusion
                    └─────────────────┘
```

## Profiling

We instrumented every operation. This is where time goes per token (pre-cache baseline):

| Operation | Time | % |
|-----------|------|---|
| I/O from SSD | 298 ms | 44% |
| mx.eval GPU sync | 140 ms | 52% |
| gather_qmm (actual matmul) | **0.9 ms** | **0.1%** |
| Router | 17 ms | 3.6% |

The GPU does almost no work. The bottleneck is I/O and framework sync — which caching solves.

With per-expert LRU cache (88.5% hit rate), I/O drops from 298 ms to ~34 ms per token.

## Optimization Progression

Each step was a single code change with a measured result:

| Step | tok/s | Change |
|------|-------|--------|
| Baseline (96 evals/token) | 1.05 | — |
| Combined attention+router eval | 1.43 | +36% |
| Eval only router logits | 1.93 | +84% |
| Per-expert LRU cache (2,000 experts) | **4.33** | **+312%** |

## Expert Activation Analysis

Across 200 tokens of sustained generation:

- **79,863 total activations** (48 layers x ~8 experts x ~200 tokens)
- **4,319 unique (layer, expert) pairs** out of 6,144 possible (70.3%)
- **Top 20% of experts handle 61.4%** of all activations
- Cache hit rate reaches **88.5%** at steady state with 2,000-expert LRU

## Quick Start

### Prerequisites

- Apple Silicon Mac with 16+ GB unified memory
- Python 3.13, MLX, mlx-lm, transformers
- ~17 GB disk for model files

### Setup

```bash
# Install dependencies
pip install mlx mlx-lm transformers rich ddgs

# Download and preprocess the model
cd research/expert-sniper
python3 convert_qwen3_30b.py
# This downloads Qwen3-30B-A3B-4bit from HuggingFace,
# extracts expert layers into binary files (~16 GB),
# and saves pinned weights (~0.87 GB)
```

### Run the Agent

```bash
python3 qwen3_agent.py
```

Features: web search, shell commands, Python execution, file operations, 64K context window.

### Run the OpenAI-Compatible Server

```bash
python3 sniper_server.py --port 8899
```

Then use with any OpenAI client:
```bash
curl http://localhost:8899/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}'
```

### Run the Profiler

```bash
# Deep profiling — shows where every millisecond goes
python3 profile_deep.py

# Sustained generation with expert activation tracking
python3 measure_sustained.py

# Per-expert cache optimization
python3 profile_step3.py
```

## Files

| File | Purpose |
|------|---------|
| `expert_io.py` | Binary expert reader — `pread` with `F_NOCACHE`, multi-threaded, LRU cache |
| `convert_qwen3_30b.py` | Downloads model, converts to per-layer binary expert format |
| `profile_deep.py` | Deep profiler — breaks down every operation per layer |
| `profile_step3.py` | Per-expert LRU cache — the 4.33 tok/s version |
| `measure_sustained.py` | 200-token sustained run with expert activation tracking |
| `benchmark.py` | Head-to-head comparison against llama.cpp |
| `qwen3_agent.py` | Interactive agent with search, shell, Python, files |
| `sniper_server.py` | OpenAI-compatible HTTP API wrapping the sniper |

## Model Details

| Parameter | Value |
|-----------|-------|
| Model | Qwen3-30B-A3B |
| Architecture | Pure transformer MoE (qwen3_moe) |
| Parameters | 30.5B total, 3.3B active per token |
| Experts | 128 per layer, top-8 routing |
| Layers | 48 |
| Hidden size | 2048 |
| MoE intermediate | 768 |
| Quantization | 4-bit, group_size=64, affine mode |
| Attention | GQA (32 query heads, 4 KV heads) |
| Context | 40,960 tokens (model), ~64K practical with sniper |

## Comparison

| System | Model | Speed | Quality | RAM Used |
|--------|-------|-------|---------|----------|
| mlx_lm standard | Qwen3-30B Q4 | **OOM** | — | — |
| llama.cpp (GPU) | Qwen3-30B Q4_K_M | **OOM** | — | — |
| llama.cpp (mmap) | Qwen3-30B Q4_K_M | **>15 min, no output** | — | — |
| llama.cpp (in-RAM) | Qwen3.5-27B IQ2 | 7.5 tok/s | Degraded (2-bit) | 8.6 GB |
| **Expert Sniper** | **Qwen3-30B Q4** | **4.33 tok/s** | **Full Q4** | **0.87 GB** |

The dense 27B at 2-bit fits in RAM and is faster, but produces degraded output (wrong example outputs in code generation). Expert Sniper runs a larger model at full 4-bit quality.

## How This Differs from Standard Approaches

**Standard mmap:** The OS pages in expert weights reactively when touched. It has no concept of which experts matter — it pages in whatever the access pattern touches, including the 94% of experts that are idle. Result: thrashing.

**Expert Sniper:** Proactive, targeted I/O. The router tells us exactly which 8 of 128 experts to load. We read only those, cache them, and skip the rest. The 88.5% cache hit rate proves the access pattern is learnable.

## Citation

If you use this work, please cite:

```
@software{expert_sniper_2026,
  title={Expert Sniper: MoE Expert Streaming for Memory-Constrained Inference},
  author={walter-grace},
  url={https://github.com/walter-grace/mac-code},
  year={2026}
}
```

## License

Apache 2.0
