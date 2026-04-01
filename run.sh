#!/bin/bash
set -e
cd "$(dirname "$0")"

# --- 1. Python Environment ---
echo -e "\033[0;32m⚡️ 🐍 Swapping to M5-Optimized Python Environment... \033[0m"
# On M5, we force a fresh uv sync to ensure Neural Accelerator support
if [[ ! -d ".venv" ]]; then
    uv venv
    CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache
fi
source .venv/bin/activate

# --- 2. M5 Max Logic ---
TARGET_MODEL="DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf"
HF_REPO="unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF"

# --- 3. Auto-Download ---
if [[ ! -f "./models/$TARGET_MODEL" ]]; then
    echo "📥 Downloading 70B Thinking Model for M5 Max..."
    hf download "$HF_REPO" "$TARGET_MODEL" --local-dir ./models
fi

# --- 4. Launch with M5 Fusion Architecture Tuning ---
# M5 Max has 12 Performance Cores; we use all of them for the KV-cache management.
# 614 GB/s bandwidth allows for a massive 2048 batch size.
exec llama-server \
    --model "./models/$TARGET_MODEL" \
    --port 8000 --host 127.0.0.1 \
    --ctx-size 32768 \
    --batch-size 2048 \
    --ubatch-size 1024 \
    --threads 12 \
    --n-gpu-layers 99 \
    --flash-attn on \
    --reasoning on \
    --mlock
