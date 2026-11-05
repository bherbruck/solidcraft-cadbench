"""Minimal MCP stdio client for `solidcraft-cli mcp --headless`.

One JSON-RPC message per line on stdin/stdout. A reader thread collects
replies so every call can have its own deadline.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path


class MCPError(RuntimeError):
    pass


class MCPTimeout(MCPError):
    pass


class MCPClient:
    def __init__(self, cli: Path, allow_dirs: list[Path], stderr_log: Path):
        args = [str(cli), "mcp", "--headless"]
        for d in allow_dirs:
            args += ["--allow-path", str(d)]
        self._stderr = open(stderr_log, "ab")
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr,
            bufsize=0,
        )
        self._replies: dict[int, queue.Queue] = {}
        self._lock = threading.Lock()
        self._next_id = 1
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self.server_info: dict = {}
        self.instructions = ""

    def _read_loop(self) -> None:
        assert self.proc.stdout is not None
        for raw in self.proc.stdout:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            mid = msg.get("id")
            if mid is None:
                continue
            with self._lock:
                q = self._replies.get(mid)
            if q is not None:
                q.put(msg)

    def _send(self, obj: dict) -> None:
        assert self.proc.stdin is not None
        if self.proc.poll() is not None:
            raise MCPError(f"SolidCraft MCP server exited with code {self.proc.returncode}")
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())
        self.proc.stdin.flush()

    def request(self, method: str, params: dict | None = None, timeout: float = 120) -> dict:
        with self._lock:
            mid = self._next_id
            self._next_id += 1
            q: queue.Queue = queue.Queue()
            self._replies[mid] = q
        self._send({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})
        deadline = time.monotonic() + timeout
        try:
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise MCPTimeout(f"{method} took longer than {timeout:.0f}s")
                try:
                    msg = q.get(timeout=min(left, 1.0))
                except queue.Empty:
                    if self.proc.poll() is not None:
                        raise MCPError(f"SolidCraft MCP server exited with code {self.proc.returncode}")
                    continue
                if "error" in msg:
                    raise MCPError(json.dumps(msg["error"]))
                return msg.get("result", {})
        finally:
            with self._lock:
                self._replies.pop(mid, None)

    def initialize(self) -> None:
        res = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "solidcraft-cadbench", "version": "0.1"},
            },
            timeout=60,
        )
        self.server_info = res.get("serverInfo", {})
        self.instructions = res.get("instructions", "")
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def list_tools(self) -> list[dict]:
        return self.request("tools/list", timeout=60).get("tools", [])

    def call(self, name: str, arguments: dict, timeout: float = 300) -> dict:
        """Returns the MCP tool result: {content: [...], isError?: bool}."""
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)

    def close(self) -> None:
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()
        finally:
            self._stderr.close()
