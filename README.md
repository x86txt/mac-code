# 🍎 mac code

**Open-source AI agent that runs powerful models on your Mac — leveraging Apple Silicon like it was meant to be used.**

There are 100 million Macs with Apple Silicon in the world. Every one of them has a GPU, unified memory, and a fast SSD. That's the largest untapped AI compute network on Earth — and it's sitting on people's desks doing nothing.

mac code is an open-source project to change that. We run frontier-class open-weight models locally on Macs, with no cloud dependency, no API keys, and no cost per token. The goal is to make every Mac an AI workstation.

---

## The Mission

**Put open-source AI models on every Mac.**

Apple Silicon is uniquely suited for local LLM inference. Unlike NVIDIA GPUs, Apple's unified memory architecture lets the CPU, GPU, and SSD share the same address space. This means:

- Models that don't fit in RAM can **page from the SSD** and the GPU still processes everything
- No PCIe bottleneck, no CPU fallback, no data copies between devices
- A $600 Mac mini can run a 35-billion parameter model at 30 tokens/second

We proved this works. Now we're open-sourcing the entire stack so anyone with a Mac can do it too.

## How it works

mac code uses a three-tier routing system that picks the fastest path for every query:

| Query type | What happens | Speed |
|---|---|---|
| **Web search** (news, scores, weather, prices) | LLM rewrites query → DuckDuckGo → page fetch → LLM answers | **~8-20s** |
| **File/exec** (read, write, run commands) | Routes through PicoClaw agent | ~4-10s |
| **Everything else** (reasoning, coding, math) | Streams directly from LLM | **~2-3s** |

The web search pipeline:
1. LLM rewrites your question into optimal search terms (~1s)
2. DuckDuckGo search (~0.2s)
3. Fetches the top result page for detailed content (~1-2s)
4. LLM answers using search snippets + page data (~3-5s)

### Test results (10/10 passed)

| Category | Query | Time | Accurate? |
|---|---|---|---|
| Sports live | NBA scores last night | 30.9s | Yes — real scores |
| Sports schedule | Next UFC fight card | 24.6s | Yes — date + fighters |
| Finance | Bitcoin price | 24.6s | Yes — current price |
| Tech news | Latest OpenAI news | 30.5s | Yes — recent events |
| Weather | NYC weather today | 22.5s | Yes — temperature |
| Science | Latest SpaceX launch | 18.5s | Yes — mission details |
| Awards | Best Picture 2026 | 20.4s | Yes — winner named |
| Coding | Latest Python version | 33.7s | Yes — 3.14.3 |
| Markets | Stock market today | 39.8s | Yes — S&P data |
| Local | SF restaurants | 39.8s | Yes — real restaurants |

## What makes this different

**The 35B model doesn't fit in RAM.** That's the whole point.

Qwen3.5-35B-A3B is a 10.6 GB model. A Mac mini M4 has 16 GB of RAM. After macOS takes its share, there's not enough room. The overflow pages from the SSD.

On any other hardware, this kills performance:

| Setup | Speed | Cost/hr | What happens |
|---|---|---|---|
| **Mac mini M4 + SSD paging** | **29.8 tok/s** | **$0.00** | **GPU processes everything via unified memory** |
| NVIDIA GPU + NVMe paging | 1.6 tok/s | $0.44 | CPU bottleneck — GPU can't access paged data |
| NVIDIA GPU + FUSE paging | 0.075 tok/s | $0.44 | Network storage — barely functional |
| NVIDIA GPU in-VRAM (no paging) | 42.5 tok/s | $0.34 | Fast, but costs money and needs big GPU |
| Claude Code (API) | ~80 tok/s | ~$0.50+ | Fastest, but every token costs money |

**Apple Silicon is 18.6x faster than NVIDIA when the model doesn't fit in memory.**

---

## Quick Start

### What you need

- Mac with Apple Silicon (M1 or later, 16GB+ RAM)
- [Homebrew](https://brew.sh)

### One-command setup

```bash
git clone https://github.com/walter-grace/mac-code.git
cd mac-code
chmod +x setup.sh && ./setup.sh
```

### Or step by step

**1 — Install dependencies**

```bash
brew install llama.cpp
pip3 install huggingface-hub rich ddgs --break-system-packages
```

**2 — Download the model**

```bash
mkdir -p ~/models

# 9B model (recommended — tools + web search work reliably)
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-9B-GGUF',
    'Qwen3.5-9B-Q4_K_M.gguf', local_dir='$HOME/models/')
"
```

Optional — 35B MoE for faster reasoning (manual swap with `/model 35b`):
```bash
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
    'Qwen3.5-35B-A3B-UD-IQ2_M.gguf', local_dir='$HOME/models/')
"
```

**3 — Start the server**

```bash
llama-server \
    --model ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --host 127.0.0.1 \
    --flash-attn on --ctx-size 32768 \
    --n-gpu-layers 99 --reasoning off -t 4
```

**4 — Run the agent**

```bash
python3 agent.py
```

**5 — (Optional) Run the web UI**

```bash
# Build PicoClaw for file operations
git clone https://github.com/sipeed/picoclaw.git
cd picoclaw && make deps && make build && cd ..
mkdir -p ~/.picoclaw/workspace
cp config.example.json ~/.picoclaw/config.json

# Start the web backend
python3 web/server.py

# Open the retro Mac UI
open http://localhost:8080
```

---

## What it looks like

```
  🍎 mac code
  claude code, but it runs on your Mac for free

  model  Qwen3.5-9B  8.95B dense · Q4_K_M · 32K ctx
  tools  search · fetch · exec · files
  cost   $0.00/hr  Apple M4 Metal · localhost:8000

  type / to see all commands

  auto 9b > who do the lakers play next?

  ⠋ rewriting query  1s
  ⠙ searching the web  2s
  ⠹ reading results  4s
  ⠸ generating answer  6s

  The Lakers play the Detroit Pistons on Monday, March 24, 2026
  at Little Caesars Arena in Detroit.

  ▸ search  8.8s  ·  16.3 tok/s

  auto 9b > what is 2*4^6?

  2 × 4^6 = 2 × 4096 = 8,192

  16.8 tok/s  ·  45 tokens  ·  2.7s
```

## Commands

Type `/` to see all commands:

| Command | Action |
|---|---|
| `/agent` | Agent mode — tools + web search (default) |
| `/raw` | Raw mode — direct streaming, no tools |
| `/model 9b` | Switch to 9B (32K ctx, tool calling) |
| `/model 35b` | Switch to 35B MoE (8K ctx, faster reasoning) |
| `/auto` | Toggle smart auto-routing |
| `/btw <q>` | Side question without context |
| `/search <q>` | Quick web search |
| `/bench` | Speed benchmark |
| `/loop 5m <p>` | Run prompt on recurring interval |
| `/stop` | Stop a running loop |
| `/branch` | Save conversation checkpoint |
| `/restore` | Restore to checkpoint |
| `/add-dir <path>` | Set working directory |
| `/save <file>` | Export conversation to JSON |
| `/clear` | Reset conversation |
| `/stats` | Session statistics |
| `/tools` | List available tools |
| `/system <msg>` | Set system prompt |
| `/compact` | Toggle markdown rendering |
| `/cost` | Show cost savings vs cloud |
| `/quit` | Exit |

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  mac code                  Python + Rich             │
│  Three-tier routing:                                 │
│    search → DuckDuckGo + LLM direct (~8s)           │
│    files  → PicoClaw agent (when needed)            │
│    chat   → stream from LLM (~2s)                   │
├──────────────────────────────────────────────────────┤
│  llama.cpp                 C++ + Metal GPU           │
│  OpenAI-compatible API @ localhost:8000              │
├──────────────────────────────────────────────────────┤
│  Qwen3.5-9B (Q4_K_M)      Qwen3.5-35B-A3B (IQ2_M)  │
│  32K ctx, 16 tok/s         8K ctx, 57 tok/s         │
├──────────────────────────────────────────────────────┤
│  Apple Silicon             Unified Memory + SSD      │
│  Metal GPU + flash paging                            │
└──────────────────────────────────────────────────────┘
```

### Web search pipeline

```
User: "who do the lakers play next?"
  │
  ├─ Step 1: LLM rewrites → "Lakers schedule March 24, 2026" (1s)
  ├─ Step 2: DuckDuckGo search (0.2s)
  ├─ Step 3: Fetch top result page for details (1-2s)
  └─ Step 4: LLM answers with search context (3-5s)
  │
  Total: ~8s with real, accurate data
```

### File operations

For file read/write/exec, mac code routes through [PicoClaw](https://github.com/sipeed/picoclaw) (Go agent framework). PicoClaw is only used for file operations — web search bypasses it for speed.

---

## Files

| File | What |
|---|---|
| `agent.py` | CLI agent — search, chat, slash commands, animated loading |
| `chat.py` | Lightweight streaming chat (no tools) |
| `dashboard.py` | Real-time server monitor with tok/s sparklines |
| `web/index.html` | Retro Macintosh web UI with live LLM streaming |
| `web/server.py` | Web backend — bridges browser to search + LLM |
| `config.example.json` | PicoClaw config for file operations |
| `setup.sh` | One-command install |
| `CLAUDE.md` | Setup instructions for Claude Code users |

---

## Scaling

| Mac | RAM | What you can run | Est. Speed |
|---|---|---|---|
| Any Mac (8GB) | 8 GB | 9B only, 4K context | ~15 tok/s |
| **Mac mini M4** | **16 GB** | **9B (32K ctx) + 35B MoE (8K ctx, SSD paging)** | **16-57 tok/s** |
| Mac mini M4 Pro | 48 GB | 35B MoE at Q4_K_M, 32K context | ~40+ tok/s |
| Mac Studio M4 Max | 128 GB | Qwen3.5-397B-A17B (frontier MoE) | ~10-20 tok/s |
| Mac Studio M4 Ultra | 192 GB | Qwen3.5-397B-A17B at Q4_K_M | ~15-30 tok/s |
| Mac Pro M4 Ultra | 512 GB | **Kimi K2.5 (1T MoE, 32B active)** | ~5-15 tok/s |

---

## Benchmarks

### Math (212 problems, SymPy verified)

| Category | Score |
|---|---|
| Linear Algebra | **100%** (22/22) |
| Number Theory | **100%** (22/22) |
| Logic | **100%** (20/20) |
| Differential Equations | 95% |
| Geometry | 91% |
| Algebra | 86% |
| **Overall** | **86.3%** (183/212) |

### Speed comparison (3 models on same hardware)

| Model | Speed | Accuracy |
|---|---|---|
| Qwen3.5-9B (Q4_K_M) | 18.3 tok/s | 3/4 |
| Qwen3.5-27B (IQ2_M) | 7.7 tok/s | 3/4 |
| **Qwen3.5-35B-A3B MoE (IQ2_M)** | **30.9 tok/s** | **3/4** |

The 27B dense model is the worst option — slower than both alternatives with the same accuracy. The MoE architecture wins because only 3B of 35B activates per token.

---

## Common Issues

- **GPU OOM after long sessions**: Reboot your Mac to clear Metal GPU memory
- **"Connection refused"**: The llama-server crashed or wasn't started. Restart it.
- **Slow web search (>30s)**: Page fetch timeout on slow sites. The search still works, just slower.
- **35B tool calling fails**: Expected at 2.6 bits/weight. Use 9B for agent tasks, 35B for reasoning only.

---

## Speed Experiments

We tested three techniques to push past 30 tok/s. All need more RAM than 16 GB:

| Experiment | Result | Needs |
|---|---|---|
| Speculative decoding (9B drafts, 35B verifies) | GPU OOM | 48 GB+ |
| Multi-Token Prediction | No gain in llama.cpp | vLLM/SGLang |
| mmap vs no-mmap | ~3% improvement | Already optimal |

**If you have 48 GB+ RAM**, try speculative decoding:

```bash
llama-server \
    --model ~/models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf \
    --model-draft ~/models/Qwen3.5-9B-Q4_K_M.gguf \
    --port 8000 --flash-attn on --ctx-size 8192 \
    --n-gpu-layers 99 -t 4 --draft-max 8 --draft-min 2
```

Expected: **60-90 tok/s**. Report results in an issue.

---

## License

MIT

## Credits

- **[Qwen3.5](https://huggingface.co/Qwen)** — the models (Alibaba)
- **[llama.cpp](https://github.com/ggergov/llama.cpp)** — inference engine (Georgi Gerganov)
- **[PicoClaw](https://github.com/sipeed/picoclaw)** — agent framework for file ops (Sipeed)
- **[Unsloth](https://huggingface.co/unsloth)** — GGUF quantizations
- **[Rich](https://github.com/Textualize/rich)** — terminal UI (Will McGugan)
- **[DuckDuckGo](https://duckduckgo.com)** — web search (no API key needed)
