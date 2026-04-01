import time
from llama_cpp import Llama

# Config - Adjust path if needed
model_path = "./models/Qwen3.5-35B-A3B-UD-IQ2_M.gguf"

print("🧠 Loading model onto M1 Max GPU...")
# n_gpu_layers=-1 ensures all 40 layers offload to Metal
llm = Llama(
    model_path=model_path,
    n_gpu_layers=-1, 
    n_ctx=2048,      
    verbose=False
)

prompt = "<|im_start|>user\nWrite a high-performance concurrent TCP port scanner in Go.<|im_end|>\n<|im_start|>assistant\n"

# 1. Warmup Run (Optional but recommended for accurate SRE metrics)
# This primes the Metal kernels and the KV cache.
print("🔥 Priming GPU...")
llm(prompt, max_tokens=5, stop=["<|im_end|>"])

# 2. Actual Benchmark
print("🚀 Benchmarking generation speed...")
start_time = time.time()

output = llm(
    prompt, 
    max_tokens=128, # Generate enough tokens to get a solid average
    stop=["<|im_end|>", "<|im_start|>", "<|endoftext|>"], 
    echo=False
)

end_time = time.time()

# Extraction
tokens_generated = output["usage"]["completion_tokens"]
total_duration = end_time - start_time
tokens_per_sec = tokens_generated / total_duration

print("\n" + "="*30)
print(f"📊 BENCHMARK RESULTS")
print("="*30)
print(f"Model:           Qwen3.5-35B-A3B (IQ2_M)")
print(f"Tokens Gen:      {tokens_generated}")
print(f"Total Time:      {total_duration:.2f} seconds")
print(f"Speed:           {tokens_per_sec:.2f} tokens/sec")
print("="*30)

# Print a snippet to verify it's not gibberish
print(f"\nSample Output:\n{output['choices'][0]['text'][:100]}...")
