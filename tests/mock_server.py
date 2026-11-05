"""A scripted stand-in for an OpenAI-compatible endpoint, for testing the harness plumbing.

It is not a model: it plays a fixed sequence of tool calls (build a plate with a hole,
screenshot, submit, finish), streams them like vLLM does, and reports usage.

    python tests/mock_server.py 8765 [--slow S]   (S seconds of fake decode per call)
"""
from __future__ import annotations

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SLOW = float(sys.argv[sys.argv.index("--slow") + 1]) if "--slow" in sys.argv else 0.05

SCRIPT = [
    ("batch", {"commands": [
        {"command": "solid.box", "params": {"length": 80, "width": 60, "height": 10}},
        {"command": "solid.cylinder", "params": {"diameter": 20, "height": 30, "base": [0, 0, -10], "operation": "cut"}},
    ]}),
    ("measure", {}),
    ("screenshot", {"view": "iso", "width": 320, "height": 240}),
    ("submit", {}),
    ("finish", {"note": "mock run"}),
]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            return self._json({"object": "list", "data": [{"id": "mock-model", "object": "model"}]})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        tools = [t["function"]["name"] for t in req.get("tools", [])]
        msgs = req["messages"]
        if "report_color" in tools:
            name, args = "report_color", {"color": "red"}
        else:
            n = sum(1 for m in msgs if m["role"] == "assistant")
            name, args = SCRIPT[min(n, len(SCRIPT) - 1)]
        prompt_tokens = sum(len(json.dumps(m)) for m in msgs) // 4
        call = {"index": 0, "id": f"call_{time.time_ns()}", "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)}}
        if not req.get("stream"):
            return self._json({
                "id": "x", "object": "chat.completion", "model": req["model"],
                "choices": [{"index": 0, "finish_reason": "tool_calls",
                             "message": {"role": "assistant", "content": None, "tool_calls": [
                                 {k: v for k, v in call.items() if k != "index"}]}}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 20, "total_tokens": prompt_tokens + 20},
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        base = {"id": "x", "object": "chat.completion.chunk", "model": req["model"]}
        time.sleep(0.05)  # fake time to first token
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "reasoning_content": "Plan: plate."}}]})
        argstr = call["function"]["arguments"]
        pieces = [argstr[i:i + 16] for i in range(0, len(argstr), 16)]
        send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": call["id"], "type": "function", "function": {"name": name, "arguments": ""}}]}}]})
        for p in pieces:
            time.sleep(SLOW / max(1, len(pieces)))
            send({**base, "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": p}}]}}]})
        send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]})
        send({**base, "choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(pieces) + 10,
                                              "total_tokens": prompt_tokens + len(pieces) + 10}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
