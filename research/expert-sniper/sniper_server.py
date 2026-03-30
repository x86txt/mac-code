#!/usr/bin/env python3
"""
OpenAI-compatible HTTP server wrapping the MoE Expert Sniper.

Makes our 30B sniper available to ANY tool that speaks the OpenAI API:
  - Hermes Agent
  - Open Interpreter
  - Continue.dev
  - Cursor
  - Any MCP client
  - curl

Endpoints:
  GET  /v1/models              — list available models
  POST /v1/chat/completions    — chat (streaming + non-streaming)
  GET  /health                 — health check

Usage:
  python3 sniper_server.py                    # start on port 8899
  python3 sniper_server.py --port 9000        # custom port

Then point any OpenAI client at http://localhost:8899/v1
"""

import sys, os, time, json, gc, argparse, uuid
from collections import OrderedDict
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import NemotronExpertReader

BITS = 4; GROUP_SIZE = 64
BIN_DIR = os.path.expanduser("~/models/qwen3-30b/bin")
PINNED_PATH = os.path.expanduser("~/models/qwen3-30b/pinned.safetensors")
CONFIG_PATH = os.path.expanduser("~/models/qwen3-30b/config.json")
MODEL_NAME = "qwen3-30b-a3b-sniper"


# ─── Per-Expert Cache ────────────────────────────────────────────────

class PerExpertCache:
    def __init__(self, max_experts=2000):
        self.cache = OrderedDict()
        self.max_experts = max_experts
        self.hits = 0; self.misses = 0
    def get(self, l, e):
        k = (l, e)
        if k in self.cache:
            self.hits += 1; self.cache.move_to_end(k); return self.cache[k]
        self.misses += 1; return None
    def put(self, l, e, d):
        self.cache[(l, e)] = d; self.cache.move_to_end((l, e))
        while len(self.cache) > self.max_experts: self.cache.popitem(last=False)


# ─── Sniper Engine ───────────────────────────────────────────────────

class SniperEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.reader = None
        self.expert_cache = None

    def load(self):
        from mlx_lm.models.qwen3_moe import Model, ModelArgs
        from transformers import AutoTokenizer

        with open(CONFIG_PATH) as f:
            config = json.load(f)
        self.config = config
        self.model = Model(ModelArgs.from_dict(config))
        q = config["quantization"]
        def cp(p, m):
            if p in config.get("quantization", {}): return config["quantization"][p]
            if not hasattr(m, "to_quantized"): return False
            return True
        nn.quantize(self.model, group_size=q["group_size"], bits=q["bits"],
                    mode=q.get("mode", "affine"), class_predicate=cp)

        pinned = mx.load(PINNED_PATH)
        if hasattr(self.model, "sanitize"): pinned = self.model.sanitize(pinned)
        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params); del pinned; gc.collect()

        self.reader = NemotronExpertReader(BIN_DIR, num_workers=4)
        self.expert_cache = PerExpertCache(max_experts=2000)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
        mx.set_memory_limit(12 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)

    def generate_stream(self, messages, max_tokens=500, temperature=0.7):
        """Yields tokens one at a time."""
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.models.base import create_attention_mask

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        tokens = self.tokenizer.encode(text)
        input_ids = mx.array([tokens])
        kv_cache = make_prompt_cache(self.model)

        for tidx in range(max_tokens + 1):
            logits = self._forward(input_ids, kv_cache)
            mx.eval(logits)

            if temperature <= 0.01:
                next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            else:
                probs = mx.softmax(logits[:, -1, :] / temperature, axis=-1)
                next_token = int(mx.random.categorical(mx.log(probs + 1e-10)).item())

            if next_token in {151643, 151645}:
                break
            word = self.tokenizer.decode([next_token])
            if "<|im_end|>" in word or "<|endoftext|>" in word:
                break
            if tidx > 0:
                yield word
            input_ids = mx.array([[next_token]])

    def _forward(self, input_ids, kv_cache):
        from mlx_lm.models.base import create_attention_mask

        h = self.model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, kv_cache[0])

        for i, layer in enumerate(self.model.model.layers):
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=kv_cache[i])
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
                c = self.expert_cache.get(i, eid)
                if c: ce[eid] = c
                else: miss.append(eid)
            if miss:
                ld = self.reader.get_experts(i, miss)
                for eid in miss: self.expert_cache.put(i, eid, ld[eid]); ce[eid] = ld[eid]
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

        h = self.model.model.norm(h)
        return self.model.lm_head(h)


# ─── HTTP Server ─────────────────────────────────────────────────────

engine = None  # Global engine instance


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, data):
        self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
        self.wfile.flush()

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "model": MODEL_NAME})
        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{
                    "id": MODEL_NAME,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "jitem-sniper",
                }]
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json({"error": "not found"}, 404)
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length))

        messages = body.get("messages", [])
        max_tokens = body.get("max_tokens", 500)
        temperature = body.get("temperature", 0.7)
        stream = body.get("stream", False)
        request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            # Initial role chunk
            self._send_sse({
                "id": request_id, "object": "chat.completion.chunk",
                "model": MODEL_NAME, "created": int(time.time()),
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]
            })

            total_tokens = 0
            for token_text in engine.generate_stream(messages, max_tokens, temperature):
                total_tokens += 1
                self._send_sse({
                    "id": request_id, "object": "chat.completion.chunk",
                    "model": MODEL_NAME, "created": int(time.time()),
                    "choices": [{"index": 0, "delta": {"content": token_text}, "finish_reason": None}]
                })

            # Final chunk
            self._send_sse({
                "id": request_id, "object": "chat.completion.chunk",
                "model": MODEL_NAME, "created": int(time.time()),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        else:
            # Non-streaming
            t0 = time.time()
            full_text = ""
            token_count = 0
            for token_text in engine.generate_stream(messages, max_tokens, temperature):
                full_text += token_text
                token_count += 1
            elapsed = time.time() - t0

            self._send_json({
                "id": request_id, "object": "chat.completion",
                "model": MODEL_NAME, "created": int(time.time()),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": full_text},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": token_count,
                    "total_tokens": token_count,
                },
                "timings": {
                    "total_ms": elapsed * 1000,
                    "tokens_per_second": token_count / elapsed if elapsed > 0 else 0,
                }
            })


def main():
    global engine

    parser = argparse.ArgumentParser(description="MoE Sniper — OpenAI-compatible server")
    parser.add_argument("--port", type=int, default=8899)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print("=" * 60)
    print("  MoE Expert Sniper — OpenAI-compatible API Server")
    print("  Qwen3-30B-A3B · 17.2 GB model · 0.87 GB RAM")
    print("=" * 60)

    print("\n  Loading sniper engine...")
    engine = SniperEngine()
    engine.load()
    print("  Engine loaded!\n")

    print(f"  Server: http://{args.host}:{args.port}")
    print(f"  API:    http://localhost:{args.port}/v1/chat/completions")
    print(f"  Models: http://localhost:{args.port}/v1/models")
    print(f"\n  Compatible with: OpenAI SDK, Hermes, Open Interpreter,")
    print(f"  Continue.dev, Cursor, any MCP client, curl\n")
    print(f"  Example:")
    print(f'  curl http://localhost:{args.port}/v1/chat/completions \\')
    print(f'    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"messages": [{{"role": "user", "content": "Hello"}}], "max_tokens": 50}}\'')
    print(f"\n{'='*60}")
    print(f"  Listening...\n")

    server = HTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
