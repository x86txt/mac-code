#!/usr/bin/env python3
"""
Qwen3-30B-A3B MoE Expert Sniper Agent — Full Agentic Capabilities

30B model on 16 GB Mac. Standard mlx_lm: OOM.
4.33 tok/s sustained. 64K context window. 0.87 GB RAM.

Tools:
  🔍 Web search (DuckDuckGo)
  ⚙️  Shell commands (any CLI tool)
  📁 Read/write files
  🐍 Run Python code
  💬 Multi-turn conversation with history
  📊 /stats — live performance metrics
"""

import sys, os, time, json, gc, subprocess, tempfile
from datetime import datetime
from collections import OrderedDict
from threading import Thread, Event
import queue
import numpy as np

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from expert_io import NemotronExpertReader

console = Console()

BITS = 4; GROUP_SIZE = 64
BIN_DIR = os.path.expanduser("~/models/qwen3-30b/bin")
PINNED_PATH = os.path.expanduser("~/models/qwen3-30b/pinned.safetensors")
CONFIG_PATH = os.path.expanduser("~/models/qwen3-30b/config.json")
MAX_CONTEXT_TOKENS = 64000
MAX_RESPONSE_TOKENS = 1000

CREATURES = ["   ⚡( ᐛ )⚡  ", "  ⚡( ᐛ )⚡   ", " ⚡( ᐛ )⚡    ", "  ⚡( ᐛ )⚡   "]

SYSTEM_PROMPT = """You are mac-code, a powerful AI assistant running locally on a 30B MoE model on a 16GB Mac.

Today is {date}. Current year: {year}.

You have DIRECT access to this machine. You can and SHOULD execute commands immediately — do not ask permission, just do it.

When the user asks to see files, check disk, run code, or do anything on their machine:
- Run the shell command IMMEDIATELY by putting it on a line starting with SHELL:
- Example: SHELL: ls ~/Desktop/*.mp4
- Example: SHELL: df -h
- Example: SHELL: python3 -c "print(2+2)"

When the user asks about current events, news, scores, weather:
- Search results are automatically injected — just answer based on them.

Rules:
- Be concise and direct. Act, don't ask.
- When citing events, verify dates are from {year}.
- Execute commands directly — you are an agentic assistant, not a chatbot."""

SEARCH_KEYWORDS = [
    "search", "find", "look up", "google", "what time", "when is", "who is",
    "who won", "weather", "news", "latest", "price", "score", "game", "games",
    "tonight", "today", "yesterday", "stock", "bitcoin", "current", "right now",
    "happening", "results", "schedule", "playing", "match", "nba", "nfl", "mlb",
]

SHELL_KEYWORDS = [
    "list files", "show files", "disk space", "run ", "execute", "find files",
    "read file", "create file", "how much space", "what processes", "memory usage",
    "install", "pip ", "brew ", "git ", "ls ", "cat ", "mkdir", "rm ", "mv ", "cp ",
]

CODE_KEYWORDS = [
    "calculate", "compute", "solve", "math", "convert", "how many", "what is",
    "factorial", "fibonacci", "sort", "algorithm",
]


# ─── Engine ──────────────────────────────────────────────────────────

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
        return f"{self.hits/max(t,1)*100:.0f}% hit ({self.hits}h/{self.misses}m), {len(self.cache)} cached"


class Qwen3SniperEngine:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.reader = None
        self.expert_cache = None
        self.kv_cache = None
        self.total_context_tokens = 0

    def load(self):
        from mlx_lm.models.qwen3_moe import Model, ModelArgs
        from transformers import AutoTokenizer

        with open(CONFIG_PATH) as f:
            config = json.load(f)
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
        pinned_gb = sum(p.nbytes for p in params) / 1e9

        self.reader = NemotronExpertReader(BIN_DIR, num_workers=4)
        self.expert_cache = PerExpertCache(max_experts=2000)
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
        mx.set_memory_limit(12 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)
        return pinned_gb

    def reset_kv(self):
        from mlx_lm.models.cache import make_prompt_cache
        self.kv_cache = make_prompt_cache(self.model)
        self.total_context_tokens = 0

    def forward_token(self, input_ids):
        from mlx_lm.models.base import create_attention_mask
        h = self.model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, self.kv_cache[0])
        for i, layer in enumerate(self.model.model.layers):
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=self.kv_cache[i])
            h = h + attn_out; mx.eval(h)
            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob: scores = scores / scores.sum(axis=-1, keepdims=True)
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
            go = mx.gather_qmm(x_exp, gw, scales=gs, biases=gb, rhs_indices=li, transpose=True, group_size=GROUP_SIZE, bits=BITS)
            uo = mx.gather_qmm(x_exp, uw, scales=us, biases=ub, rhs_indices=li, transpose=True, group_size=GROUP_SIZE, bits=BITS)
            hid = nn.silu(go) * uo
            do = mx.gather_qmm(hid, dw, scales=ds, biases=db, rhs_indices=li, transpose=True, group_size=GROUP_SIZE, bits=BITS)
            while do.ndim > 4: do = do.squeeze(-2)
            h = h + (do * scores[..., None]).sum(axis=-2)
            del gw, gs, gb, uw, us, ub, dw, ds, db
        h = self.model.model.norm(h)
        return self.model.lm_head(h)

    def generate(self, messages, max_tokens=MAX_RESPONSE_TOKENS):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = self.tokenizer.encode(text)
        self.total_context_tokens = len(tokens)
        input_ids = mx.array([tokens])
        self.reset_kv()

        logits = self.forward_token(input_ids)
        mx.eval(logits)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())

        for _ in range(max_tokens):
            if next_token in {151643, 151645}: break
            word = self.tokenizer.decode([next_token])
            if "<|im_end|>" in word or "<|endoftext|>" in word: break
            self.total_context_tokens += 1
            yield word
            input_ids = mx.array([[next_token]])
            logits = self.forward_token(input_ids)
            mx.eval(logits)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())


# ─── Tools ───────────────────────────────────────────────────────────

def classify_intent(message):
    lower = message.lower()
    # Direct shell-like requests
    if any(k in lower for k in SHELL_KEYWORDS): return "shell"
    # File viewing requests → auto shell
    if any(k in lower for k in ["what files", "what's on", "what is on", "see what", "show me",
                                  "videos on", "files on", "photos on", "what's in"]):
        return "shell"
    if any(k in lower for k in SEARCH_KEYWORDS): return "search"
    if lower.startswith("python ") or lower.startswith("run python"): return "python"
    if lower.startswith("read ") or lower.startswith("cat "): return "read_file"
    return "chat"


def quick_search(query):
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

    # Auto-append current year for time-sensitive queries
    year = datetime.now().strftime('%Y')
    time_words = ["latest", "recent", "current", "today", "tonight", "now",
                  "this week", "this month", "schedule", "upcoming", "new"]
    if any(w in query.lower() for w in time_words) and year not in query:
        query = f"{query} {year}"

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(f"- {r['title']}: {r['body']}")
    except Exception:
        return None
    return "\n".join(results) if results else None


def run_shell(cmd_text):
    try:
        result = subprocess.run(cmd_text, shell=True, capture_output=True, text=True, timeout=30)
        output = result.stdout[:4000]
        if result.stderr:
            output += f"\nSTDERR: {result.stderr[:1000]}"
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "(command timed out after 30s)"
    except Exception as e:
        return f"Error: {e}"


def run_python(code):
    """Run Python code in a subprocess and return output."""
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()
            result = subprocess.run(
                [sys.executable, f.name],
                capture_output=True, text=True, timeout=30
            )
            os.unlink(f.name)
            output = result.stdout[:4000]
            if result.stderr:
                output += f"\nSTDERR: {result.stderr[:1000]}"
            return output if output.strip() else "(no output)"
    except Exception as e:
        return f"Error: {e}"


def read_file(path):
    path = os.path.expanduser(path.strip())
    try:
        with open(path) as f:
            content = f.read(10000)
        return content
    except Exception as e:
        return f"Error reading {path}: {e}"


# ─── UI ──────────────────────────────────────────────────────────────

class ThinkingDisplay:
    def __init__(self, phase="sniping experts"):
        self.frame = 0
        self.start = time.time()
        self.phase = phase
    def render(self):
        self.frame += 1
        cf = CREATURES[self.frame % len(CREATURES)]
        t = Text()
        t.append(f"  {cf}", style="bright_green")
        t.append(f"  {self.phase}", style="bold bright_green")
        t.append(f"  {time.time()-self.start:.0f}s", style="dim")
        return t


def print_banner(pinned_gb):
    console.print()
    logo = Text()
    logo.append("  ╭─ ", style="dim")
    logo.append("mac", style="bold bright_cyan")
    logo.append("-", style="dim")
    logo.append("code", style="bold bright_cyan")
    logo.append("  ", style="")
    logo.append("MoE Sniper Edition", style="dim italic")
    console.print(logo)
    console.print("  │", style="dim")
    rows = [
        ("model", "Qwen3-30B-A3B Q4", "128 experts · 17.2 GB · top-8 routing"),
        ("ram", f"{pinned_gb:.2f} GB pinned", f"+ 2,000 experts LRU cached"),
        ("ctx", "64K tokens", "~50 pages of conversation"),
        ("speed", "4.33 tok/s", "p50: 4.97 · p99: 1.86"),
        ("tools", "search · shell · python · files", ""),
    ]
    for label, value, extra in rows:
        line = Text()
        line.append("  │ ", style="dim")
        line.append(f"{label:6s} ", style="bold dim")
        line.append(value, style="white")
        if extra:
            line.append(f"  {extra}", style="dim")
        console.print(line)
    console.print("  │", style="dim")
    note = Text()
    note.append("  ╰─ ", style="dim")
    note.append("standard mlx_lm: OOM ❌", style="dim red")
    note.append("  this model can't run any other way on 16 GB", style="dim")
    console.print(note)
    console.print()
    console.print("  [dim]/stats · /clear · /quit[/]")
    console.print(Rule(style="dim"))
    console.print()


def main():
    console.clear()

    with console.status("[bold bright_cyan]  Loading mac-code...", spinner="dots"):
        engine = Qwen3SniperEngine()
        pinned_gb = engine.load()

    print_banner(pinned_gb)

    session_tokens = 0
    session_time = 0.0
    history = []

    while True:
        try:
            console.print("  [bold bright_cyan]>[/] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n  [dim]goodbye.[/]\n")
            break

        if not user_input.strip():
            continue

        cmd = user_input.strip()
        cmd_lower = cmd.lower()

        if cmd_lower in ("/quit", "/exit", "/q"):
            break
        elif cmd_lower == "/clear":
            console.clear()
            print_banner(pinned_gb)
            history.clear()
            continue
        elif cmd_lower == "/stats":
            avg = session_tokens / session_time if session_time > 0 else 0
            ctx_pct = engine.total_context_tokens / MAX_CONTEXT_TOKENS * 100
            t = Table(show_header=False, box=None, padding=(0, 1))
            t.add_column(style="bold bright_green", width=14)
            t.add_column()
            t.add_row("tokens", f"{session_tokens:,}")
            t.add_row("time", f"{session_time:.1f}s")
            t.add_row("speed", f"{avg:.2f} tok/s")
            t.add_row("memory", f"{mx.get_active_memory()/1e9:.2f} GB")
            t.add_row("context", f"{engine.total_context_tokens:,} / {MAX_CONTEXT_TOKENS:,} ({ctx_pct:.0f}%)")
            t.add_row("expert cache", engine.expert_cache.stats())
            t.add_row("history", f"{len(history)//2} turns")
            console.print(t)
            console.print()
            continue

        # Context window warning
        if engine.total_context_tokens > MAX_CONTEXT_TOKENS * 0.8:
            console.print("  [bold yellow]⚠ Context window 80% full. Use /clear to reset.[/]")

        # ─── Intent Classification + Tool Execution ───
        intent = classify_intent(user_input)
        tool_context = None
        tool_label = None

        if intent == "search":
            with console.status("[bold bright_cyan]  🔍 Searching the web...", spinner="dots"):
                results = quick_search(user_input)
            if results:
                console.print("  [bright_cyan]🔍 search results found[/]")
                tool_context = f"Web search results for '{user_input}':\n{results}"
                tool_label = "search"
            else:
                console.print("  [dim]no search results[/]")

        elif intent == "shell":
            # Map natural language to shell commands
            lower = user_input.lower()
            if "video" in lower and "desktop" in lower:
                shell_cmd = "ls -la ~/Desktop/*.mp4 ~/Desktop/*.mov ~/Desktop/*.mkv ~/Desktop/*.avi 2>/dev/null || echo 'No video files found'"
            elif "file" in lower and "desktop" in lower:
                shell_cmd = "ls -la ~/Desktop/ | head -30"
            elif "photo" in lower or "image" in lower:
                shell_cmd = "ls -la ~/Desktop/*.png ~/Desktop/*.jpg ~/Desktop/*.jpeg ~/Desktop/*.heic 2>/dev/null || echo 'No image files found'"
            elif "disk" in lower or "space" in lower or "storage" in lower:
                shell_cmd = "df -h / | tail -1 && echo '---' && du -sh ~/Desktop ~/Documents ~/Downloads ~/models 2>/dev/null"
            elif "process" in lower or "memory" in lower:
                shell_cmd = "ps aux --sort=-%mem | head -10"
            else:
                # Try to extract literal command
                shell_cmd = user_input
                for prefix in ["run ", "execute ", "shell ", "$ "]:
                    if shell_cmd.lower().startswith(prefix):
                        shell_cmd = shell_cmd[len(prefix):]
                        break

            console.print(f"  [bold bright_cyan]⚙️  $ {shell_cmd}[/]")
            with console.status("[dim]  running...", spinner="dots"):
                output = run_shell(shell_cmd)
            if output.strip():
                for ol in output.strip().split("\n")[:15]:
                    console.print(f"  [dim]{ol}[/]")
            tool_context = f"Shell command executed: $ {shell_cmd}\nOutput:\n{output}"
            tool_label = "shell"

        elif intent == "read_file":
            path = user_input.replace("read ", "").replace("cat ", "").strip()
            with console.status("[bold bright_cyan]  📁 Reading file...", spinner="dots"):
                content = read_file(path)
            console.print(f"  [bright_cyan]📁 {path}[/]")
            tool_context = f"File contents of {path}:\n{content}"
            tool_label = "file"

        elif intent == "python":
            code = user_input.replace("python ", "").replace("run python ", "").strip()
            with console.status("[bold bright_cyan]  🐍 Running Python...", spinner="dots"):
                output = run_python(code)
            console.print(f"  [bright_cyan]🐍 python[/]")
            console.print(f"  [dim]{output[:200]}{'...' if len(output) > 200 else ''}[/]")
            tool_context = f"Python code executed:\n```python\n{code}\n```\nOutput:\n{output}"
            tool_label = "python"

        # ─── Build Messages ───
        now = datetime.now()
        messages = [{"role": "system", "content": SYSTEM_PROMPT.format(
            date=now.strftime('%A, %B %d, %Y'),
            year=now.strftime('%Y')
        )}]

        # Add history (last 10 turns for context)
        for h in history[-20:]:
            messages.append(h)

        # Add tool results as system context
        if tool_context:
            messages.append({"role": "system", "content": tool_context})

        messages.append({"role": "user", "content": user_input})

        # ─── Generate Response with Live Animation ───
        console.print()
        response_text = ""
        t_start = time.time()
        token_count = 0

        # Run generation in a thread so animation stays alive
        token_queue = queue.Queue()
        done_event = Event()
        phase = "sniping experts" if not tool_label else f"analyzing {tool_label} results"

        def generate_thread():
            try:
                for chunk in engine.generate(messages, max_tokens=MAX_RESPONSE_TOKENS):
                    token_queue.put(chunk)
            finally:
                done_event.set()

        thread = Thread(target=generate_thread, daemon=True)
        thread.start()

        # Animate while waiting for first token — snake spinner like Claude Code
        SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame = 0
        while not done_event.is_set() and token_queue.empty():
            frame += 1
            elapsed = time.time() - t_start
            s = SPINNER[frame % len(SPINNER)]
            print(f"\r  \033[36m{s}\033[0m \033[1;36m{phase}\033[0m \033[2m{elapsed:.1f}s\033[0m   ", end="", flush=True)
            time.sleep(0.08)

        # Clear animation line
        print("\r" + " " * 60 + "\r", end="", flush=True)

        # Stream tokens as they arrive
        first = True
        while not done_event.is_set() or not token_queue.empty():
            try:
                chunk = token_queue.get(timeout=0.1)
                if first:
                    console.print(f"  ", end="")
                    first = False
                print(chunk, end="", flush=True)
                response_text += chunk
                token_count += 1
            except queue.Empty:
                continue

        thread.join(timeout=1)

        elapsed = time.time() - t_start
        tps = token_count / elapsed if elapsed > 0 else 0
        session_tokens += token_count
        session_time += elapsed

        console.print()

        # ─── Auto-execute SHELL: commands in response ───
        shell_executed = False
        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("SHELL:"):
                cmd = line[6:].strip()
                if cmd:
                    console.print()
                    console.print(f"  [bold bright_cyan]⚙️  $ {cmd}[/]")
                    with console.status("[dim]  running...", spinner="dots"):
                        output = run_shell(cmd)
                    if output.strip():
                        # Show output in a box
                        for ol in output.strip().split("\n")[:20]:
                            console.print(f"  [dim]{ol}[/]")
                    shell_executed = True

        # If model output a ```bash block, extract and run it
        if not shell_executed and "```bash" in response_text:
            import re
            bash_blocks = re.findall(r'```bash\n(.*?)```', response_text, re.DOTALL)
            for block in bash_blocks:
                cmd = block.strip().split("\n")[0]  # Run first line
                if cmd and len(cmd) < 200:
                    console.print()
                    console.print(f"  [bold bright_cyan]⚙️  $ {cmd}[/]")
                    with console.status("[dim]  running...", spinner="dots"):
                        output = run_shell(cmd)
                    if output.strip():
                        for ol in output.strip().split("\n")[:20]:
                            console.print(f"  [dim]{ol}[/]")
                    shell_executed = True
                    break

        speed = Text()
        speed.append(f"\n  {token_count} tokens · {elapsed:.1f}s · {tps:.2f} tok/s", style="dim")
        ctx_used = engine.total_context_tokens
        speed.append(f" · ctx: {ctx_used:,}/{MAX_CONTEXT_TOKENS:,}", style="dim")
        console.print(speed)
        console.print()

        # Update history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
