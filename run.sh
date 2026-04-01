#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- 1. SRE Idempotent Environment Setup (M5 Optimized) ---
# We use Python 3.14.3 to leverage the improved JIT on M5 silicon
if [[ ! -d ".venv" ]] || [[ "$(cat .python-version 2>/dev/null)" != "3.14-m5" ]]; then
    echo -e "\033[0;33m📦 Initializing M5 High-Performance Environment... \033[0m"
    rm -rf .venv
    uv venv .venv --python 3.14
    echo "3.14-m5" > .python-version
    
    source .venv/bin/activate
    
    # Define dependencies including dev tools
    echo "llama-cpp-python" > requirements.in
    echo "ruff" >> requirements.in
    echo "black" >> requirements.in
    
    echo "🔐 Compiling M5-specific dependency lockfile..."
    uv pip compile requirements.in -o requirements.txt
    
    # Force recompile to link against M5 Fusion Architecture / Metal 4 headers
    echo "🛠️  Building llama-cpp with M5 Neural Accelerator support..."
    CMAKE_ARGS="-DGGML_METAL=on" uv pip install -r requirements.txt --no-cache
else
    echo -e "\033[0;32m⚡️ 🐍 Swapping to M5 Python Environment... \033[0m"
    source .venv/bin/activate && echo "✨ 🐍 .venv activated (Using $(python --version))"
fi

mkdir -p ./models

# --- 2. Hardware Guard ---
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string)
if [[ "$CHIP_NAME" != *"M5 Max"* ]]; then
    echo -e "\033[0;31m⚠️  ARCH MISMATCH: M5 optimizations active but detected $CHIP_NAME\033[0m"
fi

# --- 3. Model Logic (70B Genius Model) ---
TARGET_MODEL="DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
HF_REPO="unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"

if [[ ! -f "./models/$TARGET_MODEL" ]]; then
    echo -e "\033[0;33m📥 Model missing. Pulling 70B DeepSeek-R1 (Q4_K_M)... \033[0m"
    hf download "$HF_REPO" "$TARGET_MODEL" --local-dir ./models
fi

# --- 4. Launch Optimized for M5 Max (64GB) ---
# Threads 12: Matches the 12 Performance Cores
# Batch 2048: Saturates the 614GB/s bandwidth
# Reasoning ON: DeepSeek-R1 is built for Chain-of-Thought
echo -e "\033[0;34m🚀 Launching llama-server (M5 Fusion Mode)... \033[0m"
exec llama-server \
    --model "./models/$TARGET_MODEL" \
    --port 8000 \
    --host 127.0.0.1 \
    --ctx-size 32768 \
    --batch-size 2048 \
    --ubatch-size 1024 \
    --threads 12 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --reasoning on \
    --mlock
