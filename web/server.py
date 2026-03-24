#!/usr/bin/env python3
"""
mac code web backend — bridges the browser UI to PicoClaw + llama-server.
Run this, then open index.html. The browser talks to this server,
which routes to PicoClaw (for tool calls) or llama-server (for streaming).
"""

import json, os, subprocess, re, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse
import urllib.request

PORT = 8080
LLM_URL = "http://localhost:8000/v1/chat/completions"
PICOCLAW = os.path.expanduser("~/Desktop/qwen/picoclaw/build/picoclaw-darwin-arm64")
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m|\r')

class Handler(SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/agent":
            self._handle_agent()
        else:
            self.send_error(404)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _handle_chat(self):
        """Proxy streaming chat to llama-server."""
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            req = urllib.request.Request(
                LLM_URL, data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self._cors_headers()
                self.end_headers()

                while True:
                    chunk = resp.read(1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_agent(self):
        """Route through PicoClaw agent for tool use (web search, etc.)."""
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        message = body.get("message", "")
        session = body.get("session", f"web-{int(time.time())}")

        try:
            result = subprocess.run(
                [PICOCLAW, "agent", "-m", message, "-s", session],
                capture_output=True, text=True, timeout=120,
            )

            # Parse: strip ANSI, find last lobster emoji
            clean = ANSI_RE.sub('', result.stdout)
            idx = clean.rfind("\U0001f99e")
            if idx >= 0:
                response = clean[idx:].lstrip("\U0001f99e").strip()
            else:
                response = clean.strip()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({
                "response": response,
                "session": session,
            }).encode())

        except subprocess.TimeoutExpired:
            self.send_response(504)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Agent timeout"}).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self._cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def do_GET(self):
        """Serve static files from web/ directory."""
        if self.path == "/" or self.path == "":
            self.path = "/index.html"
        return SimpleHTTPRequestHandler.do_GET(self)

    def log_message(self, format, *args):
        """Minimal logging."""
        msg = format % args
        if "favicon" not in msg:
            print(f"  {msg}")

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print(f"\n  🍎 mac code web server")
    print(f"  http://localhost:{PORT}")
    print(f"  LLM:   localhost:8000 (llama-server)")
    print(f"  Agent: PicoClaw ({PICOCLAW})")
    print()
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
