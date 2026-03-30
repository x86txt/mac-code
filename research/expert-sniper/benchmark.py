#!/usr/bin/env python3
"""
Head-to-head benchmark: MoE 30B Q4 (sniper) vs Dense 27B IQ2 (llama-server)

Tests:
  1. Factual accuracy (5 questions with known answers)
  2. Code generation (2 coding tasks)
  3. Reasoning (2 logic/math problems)
  4. Creative writing (1 essay prompt)
  5. Speed: tok/s, time-to-first-token, total time

Outputs a formatted comparison table + saves JSON.
"""
import sys, os, time, json, gc
sys.stdout.reconfigure(line_buffering=True)
import requests
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import NemotronExpertReader

BITS = 4; GROUP_SIZE = 64
BIN_DIR = os.path.expanduser("~/models/qwen3-30b/bin")
LLAMA_API = "http://localhost:8080/v1/chat/completions"

# ─── BENCHMARK PROMPTS ──────────────────────────────────────────────

BENCHMARKS = [
    # (category, prompt, expected_keyword_or_None)
    ("factual", "What is the capital of France? Answer in one sentence.", "Paris"),
    ("factual", "What year did World War 2 end? Answer with just the year.", "1945"),
    ("factual", "What is the chemical formula for water?", "H2O"),
    ("factual", "Who wrote Romeo and Juliet?", "Shakespeare"),
    ("factual", "What is the speed of light in km/s? Round to nearest thousand.", "300"),
    ("code", "Write a Python function called `is_prime(n)` that returns True if n is prime.", "def is_prime"),
    ("code", "Write a Python function called `fibonacci(n)` that returns the nth Fibonacci number.", "def fibonacci"),
    ("reasoning", "If a train travels 120 km in 2 hours, what is its speed in km/h? Answer with just the number.", "60"),
    ("reasoning", "I have 3 apples. I buy 5 more and eat 2. How many do I have? Answer with just the number.", "6"),
    ("creative", "Write a haiku about the ocean.", None),
]


# ─── PER-EXPERT CACHE ───────────────────────────────────────────────

class PerExpertCache:
    def __init__(self, max_experts=2000):
        self.cache = OrderedDict()
        self.max_experts = max_experts
        self.hits = 0; self.misses = 0
    def get(self, l, e):
        k = (l, e)
        if k in self.cache: self.hits += 1; self.cache.move_to_end(k); return self.cache[k]
        self.misses += 1; return None
    def put(self, l, e, d):
        self.cache[(l, e)] = d; self.cache.move_to_end((l, e))
        while len(self.cache) > self.max_experts: self.cache.popitem(last=False)
    def stats(self):
        t = self.hits + self.misses
        return f"{self.hits}/{t} ({self.hits/max(t,1)*100:.0f}%)"


# ─── DENSE 27B VIA LLAMA-SERVER ─────────────────────────────────────

def run_dense(prompt, max_tokens=200):
    """Non-streaming call to llama-server. Returns (text, ttft, total_time, tok_count)."""
    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": prompt},
    ]
    t0 = time.time()
    try:
        resp = requests.post(LLAMA_API, json={
            "messages": messages, "max_tokens": max_tokens, "temperature": 0.0,
        }, timeout=120)
        total = time.time() - t0
        r = resp.json()
        text = r["choices"][0]["message"]["content"]
        # Extract timing from response
        timings = r.get("timings", {})
        prompt_ms = timings.get("prompt_ms", 0)
        gen_tps = timings.get("predicted_per_second", 0)
        gen_tokens = timings.get("predicted_n", len(text.split()))
        ttft = prompt_ms / 1000.0
        return text, ttft, total, gen_tokens, gen_tps
    except requests.exceptions.ConnectionError:
        return "[llama-server not running]", 0, 0, 0, 0
    except Exception as e:
        return f"[ERROR: {e}]", 0, 0, 0, 0


# ─── MOE 30B SNIPER ─────────────────────────────────────────────────

def load_sniper():
    """Load the MoE sniper engine. Returns (model, tokenizer, reader, expert_cache)."""
    from mlx_lm.models.qwen3_moe import Model, ModelArgs

    with open(os.path.expanduser("~/models/qwen3-30b/config.json")) as f:
        config = json.load(f)
    model = Model(ModelArgs.from_dict(config))
    q = config["quantization"]
    def cp(p, m):
        if p in config.get("quantization", {}): return config["quantization"][p]
        if not hasattr(m, "to_quantized"): return False
        return True
    nn.quantize(model, group_size=q["group_size"], bits=q["bits"],
                mode=q.get("mode", "affine"), class_predicate=cp)
    pinned = mx.load(os.path.expanduser("~/models/qwen3-30b/pinned.safetensors"))
    if hasattr(model, "sanitize"): pinned = model.sanitize(pinned)
    model.load_weights(list(pinned.items()), strict=False)
    params = [p for name, p in tree_flatten(model.parameters()) if "switch_mlp" not in name]
    mx.eval(*params); del pinned; gc.collect()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    reader = NemotronExpertReader(BIN_DIR, num_workers=4)
    expert_cache = PerExpertCache(max_experts=2000)
    mx.set_memory_limit(12 * 1024**3)
    return model, tok, reader, expert_cache, config


def run_sniper(prompt, model, tok, reader, ec, max_tokens=200):
    """Run one prompt through the sniper. Returns (text, ttft, total_time, tok_count)."""
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.models.base import create_attention_mask

    messages = [
        {"role": "system", "content": "You are a helpful, concise assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tok.apply_chat_template(messages, tokenize=False,
                                    add_generation_prompt=True, enable_thinking=False)
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])
    cache = make_prompt_cache(model)

    t0 = time.time()
    generated = []
    ttft = None

    for tidx in range(max_tokens + 1):  # +1 for prefill
        h = model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, cache[0])

        for i, layer in enumerate(model.model.layers):
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=cache[i])
            h = h + attn_out; mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active = sorted(set(int(e) for e in np.array(inds).flatten()))
            ce = {}; miss = []
            for eid in active:
                c = ec.get(i, eid)
                if c: ce[eid] = c
                else: miss.append(eid)
            if miss:
                ld = reader.get_experts(i, miss)
                for eid in miss: ec.put(i, eid, ld[eid]); ce[eid] = ld[eid]
                del ld

            itl = {eid: j for j, eid in enumerate(active)}
            li = mx.array(np.vectorize(lambda x: itl.get(int(x), 0))(np.array(inds)))
            gw = mx.stack([ce[e]["gate_proj.weight"] for e in active])
            gs = mx.stack([ce[e]["gate_proj.scales"] for e in active])
            gb = mx.stack([ce[e]["gate_proj.biases"] for e in active])
            uw = mx.stack([ce[e]["up_proj.weight"] for e in active])
            us = mx.stack([ce[e]["up_proj.scales"] for e in active])
            ub = mx.stack([ce[e]["up_proj.biases"] for e in active])
            dw = mx.stack([ce[e]["down_proj.weight"] for e in active])
            ds = mx.stack([ce[e]["down_proj.scales"] for e in active])
            db = mx.stack([ce[e]["down_proj.biases"] for e in active])
            del ce

            x_exp = mx.expand_dims(normed, (-2, -3))
            go = mx.gather_qmm(x_exp, gw, scales=gs, biases=gb, rhs_indices=li,
                                transpose=True, group_size=GROUP_SIZE, bits=BITS)
            uo = mx.gather_qmm(x_exp, uw, scales=us, biases=ub, rhs_indices=li,
                                transpose=True, group_size=GROUP_SIZE, bits=BITS)
            hid = nn.silu(go) * uo
            do = mx.gather_qmm(hid, dw, scales=ds, biases=db, rhs_indices=li,
                                transpose=True, group_size=GROUP_SIZE, bits=BITS)
            while do.ndim > 4: do = do.squeeze(-2)
            eo = (do * scores[..., None]).sum(axis=-2)
            h = h + eo
            del gw, gs, gb, uw, us, ub, dw, ds, db

        h = model.model.norm(h)
        logits = model.lm_head(h); mx.eval(logits)
        nt = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        if ttft is None:
            ttft = time.time() - t0

        if nt in {151643, 151645}: break
        w = tok.decode([nt])
        if "<|im_end|>" in w or "<|endoftext|>" in w: break
        if tidx > 0:
            generated.append(w)
        input_ids = mx.array([[nt]])

    total = time.time() - t0
    out = "".join(generated)
    tps = len(generated) / total if total > 0 else 0
    return out, ttft, total, len(generated), tps


# ─── SCORING ─────────────────────────────────────────────────────────

def score_response(text, expected_keyword):
    """Simple keyword-based scoring. Returns 1 if keyword found, 0 otherwise."""
    if expected_keyword is None:
        return 1 if len(text.strip()) > 10 else 0  # Creative: just check non-empty
    return 1 if expected_keyword.lower() in text.lower() else 0


# ─── MAIN ────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  BENCHMARK: MoE 30B Q4 (JITEM sniper) vs Dense 27B IQ2 (llama.cpp)")
    print("  Same hardware: M4 Mac Mini 16GB")
    print("=" * 72)

    # Check llama-server
    try:
        requests.get("http://localhost:8080/health", timeout=2)
        dense_available = True
        print("  ✓ llama-server running (dense 27B)")
    except:
        dense_available = False
        print("  ✗ llama-server not running — dense results will be skipped")

    # Load sniper
    print("  Loading MoE sniper...", flush=True)
    model, tok, reader, ec, config = load_sniper()
    print("  ✓ MoE sniper loaded\n")

    results = []

    for bi, (category, prompt, expected) in enumerate(BENCHMARKS):
        print(f"  [{bi+1}/{len(BENCHMARKS)}] ({category}) {prompt[:60]}...")

        # Run sniper
        s_text, s_ttft, s_total, s_tokens, s_tps = run_sniper(
            prompt, model, tok, reader, ec, max_tokens=150)
        s_score = score_response(s_text, expected)

        # Run dense
        if dense_available:
            d_text, d_ttft, d_total, d_tokens, d_tps = run_dense(prompt, max_tokens=150)
            d_score = score_response(d_text, expected)
        else:
            d_text, d_ttft, d_total, d_tokens, d_tps, d_score = "", 0, 0, 0, 0, 0

        print(f"    MoE:   {s_tps:5.2f} tok/s | {'✓' if s_score else '✗'} | {s_text[:80]}")
        print(f"    Dense: {d_tps:5.1f} tok/s | {'✓' if d_score else '✗'} | {d_text[:80]}")

        results.append({
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "sniper": {
                "text": s_text, "ttft": s_ttft, "total": s_total,
                "tokens": s_tokens, "tps": s_tps, "score": s_score,
            },
            "dense": {
                "text": d_text, "ttft": d_ttft, "total": d_total,
                "tokens": d_tokens, "tps": d_tps, "score": d_score,
            },
        })

    # ─── RESULTS TABLE ───────────────────────────────────────────────

    print(f"\n{'='*72}")
    print(f"  RESULTS")
    print(f"{'='*72}\n")

    # Accuracy by category
    categories = ["factual", "code", "reasoning", "creative"]
    print(f"  {'Category':<12s} {'MoE Correct':>12s} {'Dense Correct':>14s}")
    print(f"  {'─'*12} {'─'*12} {'─'*14}")
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        s_correct = sum(r["sniper"]["score"] for r in cat_results)
        d_correct = sum(r["dense"]["score"] for r in cat_results)
        total = len(cat_results)
        print(f"  {cat:<12s} {s_correct:>5d}/{total:<6d} {d_correct:>5d}/{total}")

    s_total_score = sum(r["sniper"]["score"] for r in results)
    d_total_score = sum(r["dense"]["score"] for r in results)
    print(f"  {'TOTAL':<12s} {s_total_score:>5d}/{len(results):<6d} {d_total_score:>5d}/{len(results)}")

    # Speed comparison
    s_speeds = [r["sniper"]["tps"] for r in results if r["sniper"]["tps"] > 0]
    d_speeds = [r["dense"]["tps"] for r in results if r["dense"]["tps"] > 0]

    print(f"\n  {'Speed':<20s} {'MoE 30B Q4':>12s} {'Dense 27B IQ2':>14s}")
    print(f"  {'─'*20} {'─'*12} {'─'*14}")
    if s_speeds:
        print(f"  {'Avg tok/s':<20s} {sum(s_speeds)/len(s_speeds):>10.2f}   {sum(d_speeds)/len(d_speeds) if d_speeds else 0:>10.1f}")
        print(f"  {'Max tok/s':<20s} {max(s_speeds):>10.2f}   {max(d_speeds) if d_speeds else 0:>10.1f}")
    s_ttfts = [r["sniper"]["ttft"] for r in results if r["sniper"]["ttft"] > 0]
    d_ttfts = [r["dense"]["ttft"] for r in results if r["dense"]["ttft"] > 0]
    if s_ttfts:
        print(f"  {'Avg TTFT':<20s} {sum(s_ttfts)/len(s_ttfts):>9.2f}s   {sum(d_ttfts)/len(d_ttfts) if d_ttfts else 0:>9.1f}s")

    print(f"\n  Model details:")
    print(f"    MoE:   Qwen3-30B-A3B Q4 (17.2 GB on disk, 0.87 GB pinned in RAM)")
    print(f"    Dense: Qwen3.5-27B IQ2_XXS (8.6 GB, fully in RAM)")
    print(f"    mlx_lm standard: OOM on MoE model ❌")
    print(f"    Expert cache: {ec.stats()}")
    print(f"{'='*72}")

    # Save
    with open("benchmark_full_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to benchmark_full_results.json")


if __name__ == "__main__":
    main()
