#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- 1. SRE Idempotent Environment Setup ---
# Check for .venv and version match
if [[ ! -d ".venv" ]] || [[ "$(cat .python-version 2>/dev/null)" != "3.14" ]]; then
    echo -e "\033[0;33m📦 Initializing Environment (Python 3.14.3)... \033[0m"
    rm -rf .venv
    uv venv .venv --python 3.14
    echo "3.14" > .python-version
    
    source .venv/bin/activate
    
    # Define core dependencies for compilation
    echo "llama-cpp-python" > requirements.in
    
    # Generate locked requirements.txt if it doesn't exist
    echo "🔐 Compiling dependency lockfile..."
    uv pip compile requirements.in -o requirements.txt
    
    # Install with Metal support forced for M1 Max
    echo "🛠️  Building llama-cpp with Metal kernels..."
    CMAKE_ARGS="-DGGML_METAL=on" uv pip install -r requirements.txt --no-cache
else
    echo -e "\033[0;32m⚡️ 🐍 Swapping to Python Environment... \033[0m"
    source .venv/bin/activate && echo "✨ 🐍 .venv activated (Using $(python --version))"
fi

mkdir -p ./models

# --- 2. Hardware Guard ---
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string)
if [[ "$CHIP_NAME" != *"M1 Max"* ]]; then
    echo -e "\033[0;31m⚠️  ARCH MISMATCH: Detected $CHIP_NAME. M1 Max optimizations may underperform.\033[0m"
fi

# --- 3. Model Logic ---
TARGET_MODEL="Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
HF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"

if [[ ! -f "./models/$TARGET_MODEL" ]]; then
    echo -e "\033[0;33m📥 Model missing. Pulling Qwen3.5-35B (IQ2_M)... \033[0m"
    hf download "$HF_REPO" "$TARGET_MODEL" --local-dir ./models
fi

# --- 4. Launch Optimized for M1 Max (32GB) ---
# Note: We use 4 threads to keep the efficiency cores free for OS tasks.
echo -e "\033[0;34m🚀 Launching llama-server... \033[0m"
exec llama-server \
    --model "./models/$TARGET_MODEL" \
    --port 8000 \
    --host 127.0.0.1 \
    --ctx-size 24576 \
    --batch-size 1024 \
    --ubatch-size 1024 \
    --threads 4 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --reasoning off \
    --mlock
