# 🍎 mac code

**22 GB model on a 16 GB Mac. 1.42 GB RAM. Coherent 35B reasoning. $0/month.**

---

## What This Does

Runs models that **don't fit in RAM** on Apple Silicon Macs — no quality compromise, no mmap thrashing.

**1. MoE Expert Sniper (the breakthrough):** Qwen3.5-35B-A3B at full Q4_K_M quality (22 GB) on a 16 GB Mac with only **1.42 GB of RAM**. The router predicts which 8 of 256 experts activate per token — we snipe only those from SSD via `F_NOCACHE` direct I/O with 8-thread parallel `pread`. The rest stays on disk. Result: coherent chain-of-thought reasoning at 1.54 tok/s.

```
> what is 2+2?

Thinking Process:
1. Analyze the Request: basic addition (2 + 2).
2. Determine the Answer: 2 + 2 = 4.
3. Self-Correction: The prompt asks for concise. A direct answer is best.

2 + 2 equals 4.

1.54 tok/s · 170 tokens · 1.42 GB RAM
```

**2. MoE Agent (fast, daily driver):** Same Qwen3.5-35B-A3B at IQ2_M (10.6 GB) via llama.cpp. Fits in RAM natively. 30 tok/s with web search, shell commands, and file tools.

**3. Dense Flash Streaming (research proof):** Qwen3-32B at 4-bit (18.4 GB) streamed from SSD. 4.5 GB pinned, 14.2 GB FFN per token via `F_NOCACHE`. 0.15 tok/s — 9x faster than mmap thrashing. Produced 387 tokens of coherent chain-of-thought.

**4. Tool calling at 2.6 bits per weight.** LLM classifies its own intent as plain text — "search" / "shell" / "chat" — and routes itself. 8/8 correct.

**5. 64K context via KV cache quantization.** Two llama.cpp flags shrink KV cache from 1024 MB to 288 MB. The 9B goes from 32K to 64K context.

| | **35B Expert Sniper** | **35B MoE (fast)** | **9B (long context)** | **32B Dense Stream** |
|---|---|---|---|---|
| **Backend** | Custom MLX + SSD | llama.cpp | llama.cpp or MLX | Custom MLX + SSD |
| **Speed** | 1.54 tok/s | 30 tok/s | 16-20 tok/s | 0.15 tok/s |
| **Model size** | 22 GB (Q4_K_M) | 10.6 GB (IQ2_M) | 5.3 GB (Q4_K_M) | 18.4 GB (4-bit) |
| **RAM used** | **1.42 GB** | 10.6 GB | Fits in RAM | 4.5 GB |
| **Quality** | Full Q4 + CoT | Good (2.6 bpw) | Full | Full 4-bit |
| **Fits in 16 GB?** | NO (22 GB) | YES | YES | NO (18.4 GB) |
| **Best for** | High-quality reasoning | Fast daily agent | Long context | Dense model proof |

---

## Quick Start

### Option A: llama.cpp + 35B MoE (default — 30 tok/s)

```bash
brew install llama.cpp
pip3 install rich ddgs huggingface-hub --break-system-packages

# Download 35B model (10.6 GB)
mkdir -p ~/models
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"

# Start server
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 12288 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -np 1 -t 4

# Run agent
python3 agent.py
```

### Option B: MLX + 9B (64K context, persistent KV cache)

```bash
pip3 install mlx-lm rich ddgs --break-system-packages

# Start MLX engine (downloads 9B model on first run)
python3 mlx/mlx_engine.py

# Run agent
python3 agent.py
```

### Option C: llama.cpp + 9B (if you want 64K context without MLX)

```bash
# Download 9B model
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"

# Start server (64K context with quantized KV cache)
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 65536 \
    --cache-type-k q4_0 --cache-type-v q4_0 \
    --n-gpu-layers 99 --reasoning off -t 4

# Run agent
python3 agent.py
```

---

## Flash Streaming — Out-of-Core Inference

This is the research component. Running a model that **genuinely does not fit in RAM** at full quality.

### The Problem

Qwen3-32B at MLX 4-bit (group_size=64) is 18.4 GB. On a 16 GB Mac:
- **llama.cpp mmap**: OS pages the entire model reactively. Result: 0.017 tok/s (system thrashes).
- **2-bit quantization (IQ2_M)**: Compresses to ~9 GB. Result: 6 tok/s, but quality degrades after ~60 tokens.

### What We Built

Split the model by access pattern:
- **Pinned in RAM (4.28 GB)**: Attention weights, embeddings, norms — always needed, random access
- **Streamed from SSD (14.16 GB)**: FFN layers (~221 MB each) — sequential, loaded per-layer, discarded after use

Key technique: `fcntl(F_NOCACHE)` + `os.pread()` bypasses macOS Unified Buffer Cache, preventing page cache pollution. 16KB-aligned custom binary format for DART IOMMU compatibility.

### Measured Results

**Dense model (Qwen3-32B, 18.4 GB):**

| Approach | Speed | Quality | RAM Used |
|----------|-------|---------|----------|
| llama.cpp mmap (18.4 GB on 16 GB) | 0.017 tok/s | Full 4-bit | Swap thrashing |
| **Flash Stream v2 (F_NOCACHE)** | **0.152 tok/s** | **Full 4-bit** | **4.5 GB stable** |
| 2-bit in-RAM (9.2 GB) | ~6 tok/s | Degraded (~60 tok) | 9.2 GB |

**MoE model (Qwen3.5-35B-A3B Q4_K_M, 22 GB) — the breakthrough:**

| Metric | Value |
|--------|-------|
| Model size | 22 GB (256 experts, 8 active per token) |
| RAM pinned | **1.42 GB** |
| Decode speed | **1.75 tok/s** |
| Prefill | 3.5s (3.4 tok/s) |
| SSD throughput | 1.3 GB/s (8-thread F_NOCACHE pread) |
| Memory stability | 1.42 GB flat (never grows) |

The MoE Expert Sniper reads only the 8 active experts per layer from SSD (~14 MB/layer vs 221 MB for dense). The router acts as a built-in prefetch predictor — telling us exactly which weights to load before we need them.

### Sample Output — 35B MoE (coherent chain-of-thought)

Prompt: "Explain how mixture of experts models work in one paragraph."

> `<think>` Here's a thinking process that leads to the suggested explanation of Mixture of Experts (MoE): 1. **Analyze the Request:** * **Topic:** Mixture of Experts (MoE)...

This is the model reasoning about its own architecture — chain-of-thought at 35B quality, streaming from SSD with only 1.42 GB in RAM.

**Dense 32B sample** (387 tokens, coherent):

> "The Flash Attention paper introduces an optimized attention mechanism in transformers, focusing on improving computational and memory efficiency during the self-attention operation... FlashAttention addresses this by reorganizing computation via tiling..."

### What We Discovered

- **MLX 4-bit overhead**: group_size=64 scales+biases add 15% overhead. A 32B model at "4-bit" is 18.4 GB, not 16 GB. This pushed it past the RAM limit.
- **nn.Module memory leak**: Injecting FFN weights via `setattr` keeps references alive — 3.6 GB growth per 16 layers. Fixed by using `mx.quantized_matmul` directly on loaded arrays.
- **Batching doesn't help**: Tested 8-layer batches (16 evals vs 128) — zero speedup. Bottleneck is I/O latency, not GPU sync overhead.
- **F_NOCACHE works**: 27% speedup over mmap by bypassing the Unified Buffer Cache.
- **GGUF is column-major**: The correct reshape from dequantized flat data is `flat.reshape(ne[1], ne[0])`, not `flat.reshape(ne[0], ne[1]).T`.

See `research/flash-streaming/` for full code, audit, and documentation.

---

## Benchmarks

### Agent Tasks (Mac mini M4, 35B default)

| Task | Time | Backend |
|---|---|---|
| Shell command | 7.6s | llama.cpp |
| Web search + answer | 8.1s | llama.cpp |
| Math reasoning | 9.8s | MLX (9B) |
| Code generation | 9.7s | MLX (9B) |

### llama.cpp vs MLX (same prompts, same Mac mini M4)

| Task | llama.cpp | MLX | Winner |
|---|---|---|---|
| Shell command | 7.9s | **7.6s** | MLX |
| Math | 12.4s | **9.8s** | **MLX (21%)** |
| Code gen | 12.3s | **9.7s** | **MLX (21%)** |
| Reasoning | 12.3s | **10.0s** | **MLX (19%)** |
| Web search | **45.7s** | 48.3s | llama.cpp |

### Context Persistence (MLX, 9B)

| Operation | Time |
|---|---|
| Reprocess 141 tokens | 1.01s |
| **SSD load** | **0.0003s (6,677x faster)** |
| R2 download + load | 1.5s |
| KV cache compression | 26.6 → 6.7 MB (4x, 0.993 cosine similarity) |

KV cache quantization uses llama.cpp's `--cache-type-k q4_0 --cache-type-v q4_0` flags, applying per-group 4-bit quantization to key/value states. Inspired by [Google's TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) research on extreme KV compression.

---

## How It Works

The LLM classifies its own intent:

```
"find me videos on my desktop"  → LLM says "shell"  → generates find command → executes
"who do the lakers play next"   → LLM says "search" → rewrites query → DuckDuckGo → answers
"explain quantum computing"     → LLM says "chat"   → streams directly
```

Three paths. No hardcoded rules. Upgrading the model upgrades every capability.

---

## MLX Backend — Persistent Context

The MLX backend adds features llama.cpp can't do:

```bash
# Save context after analyzing a codebase
curl -X POST localhost:8000/v1/context/save \
    -d '{"name":"my-project","prompt":"your codebase here"}'

# Next day — resume instantly (0.0003s vs minutes reprocessing)
curl -X POST localhost:8000/v1/context/load \
    -d '{"name":"my-project"}'

# Different Mac — download from R2 (1.5s)
curl -X POST localhost:8000/v1/context/download \
    -d '{"name":"my-project"}'
```

See `mlx/PROJECT.md` for the full research roadmap.

---

## Commands

Type `/` to see all commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode (default) |
| `/raw` | Direct streaming, no tools |
| `/model 9b` | Switch to 9B (64K ctx) |
| `/model 35b` | Switch to 35B MoE (llama.cpp only) |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/stats` | Session statistics |
| `/cost` | Cost savings vs cloud |
| `/good` / `/bad` | Grade response (self-improvement logging) |
| `/improve` | View grading stats |
| `/clear` | Reset conversation |
| `/quit` | Exit |

---

## Architecture

```
┌──────────────────────────────────────────────────┐
│  agent.py — LLM-as-Router                        │
│  search / shell / chat                           │
├──────────┬───────────────────────────────────────┤
│ llama.cpp│  MLX backend                          │
│ backend  │  + KV cache save/load                 │
│          │  + KV cache compression (4-bit)       │
│          │  + Cloudflare R2 sync                 │
│          │  + Flash Streaming (out-of-core)      │
├──────────┴───────────────────────────────────────┤
│  Apple Silicon — Unified Memory + NVMe SSD       │
└──────────────────────────────────────────────────┘
```

---

## Files

| File | What |
|---|---|
| `agent.py` | CLI agent — works with either backend |
| `chat.py` | Streaming chat |
| `dashboard.py` | Server monitor |
| `web/` | Retro Mac web UI |
| `mlx/mlx_engine.py` | MLX inference server with context API |
| `mlx/kv_cache.py` | KV cache save/load/compress |
| `mlx/r2_store.py` | Cloudflare R2 integration |
| `mlx/turboquant.py` | KV cache 4-bit compression |
| `mlx/paged_inference.py` | Process docs beyond context limit |
| `mlx/PROJECT.md` | MLX research roadmap |
| `research/flash-streaming/` | Out-of-core inference engine |

---

## Scaling

| Mac | RAM | What you can run |
|---|---|---|
| Any Mac (8GB) | 8 GB | 9B, 4K context |
| **Mac mini M4** | **16 GB** | **9B (64K) + 35B MoE (12K) + 32B Flash Stream** |
| Mac mini M4 Pro | 48 GB | 35B at Q4 + speculative decoding |
| Mac Studio Ultra | 192 GB | 397B frontier model |

Same `agent.py` at every level. Just swap the model.

---

## Research

This project builds on:
- **[Apple "LLM in a Flash"](https://machinelearning.apple.com/research/efficient-large-language)** — Inspiration for out-of-core weight streaming. Our Flash Streaming engine implements the split-model architecture (attention pinned, FFN streamed) with F_NOCACHE direct I/O.
- **[Google TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** — Inspiration for KV cache compression. We use llama.cpp's per-group 4-bit KV quantization.
- **[MLX](https://github.com/ml-explore/mlx)** — Apple's native ML framework

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models
- **[llama.cpp](https://github.com/ggerganov/llama.cpp)** — inference engine
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Cloudflare R2](https://developers.cloudflare.com/r2/)** — free object storage
- **[Rich](https://github.com/Textualize/rich)** — terminal UI

## License

MIT
