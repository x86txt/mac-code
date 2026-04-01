#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- 1. Idempotent Python Environment Setup ---
if [[ ! -d ".venv" ]]; then
    echo -e "\033[0;33m📦 .venv not found. Creating with Python 3.14... \033[0m"
    # uv will download 3.14 automatically if it's not on your system
    uv venv .venv --python 3.14
    
    echo -e "\033[0;32m⚡️ 🐍 Swapping to Python Environment... \033[0m"
    source .venv/bin/activate
    
    # Ensure dependencies are installed in the new venv
    echo "dependencies: installing llama-cpp-python with Metal support..."
    CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache
else
    echo -e "\033[0;32m⚡️ 🐍 Swapping to Python Environment... \033[0m"
    source .venv/bin/activate && echo "✨ 🐍 .venv activated (Using $(python --version))"
fi

mkdir -p ./models

# --- 2. Hardware Verification (M1 Max Safeguard) ---
CHIP_NAME=$(sysctl -n machdep.cpu.brand_string)
if [[ "$CHIP_NAME" != *"M1 Max"* ]]; then
    echo -e "\033[0;31m⚠️ Warning: You are on the M1 Max branch but hardware is $CHIP_NAME\033[0m"
fi

# --- 3. Configuration (Optimized for 32GB RAM) ---
TARGET_MODEL="Qwen3.5-35B-A3B-UD-IQ2_M.gguf"
HF_REPO="unsloth/Qwen3.5-35B-A3B-GGUF"

# --- 4. Idempotent Model Check ---
if [[ ! -f "./models/$TARGET_MODEL" ]]; then
    echo -e "\033[0;33m📥 Model missing. Downloading Qwen3.5 35B MoE... \033[0m"
    if ! command -v hf &> /dev/null; then
        echo "❌ Error: 'hf' cli not found. Run: uv tool install huggingface_hub[cli]"
        exit 1
    fi
    hf download "$HF_REPO" "$TARGET_MODEL" --local-dir ./models
fi

# --- 5. Execution ---
# Flags: 24k Context, 4 Threads, Reasoning OFF for speed
echo -e "\033[0;34m🚀 Launching llama-server for M1 Max... \033[0m"
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
