"""Run CADGenBench fixtures with a local OpenAI-compatible model driving SolidCraft over MCP.

For each fixture: start a fresh headless SolidCraft MCP session, give the
model the benchmark's own task text (plus the drawing image or the input
STEP already loaded), let it call SolidCraft tools until it finishes or
runs out of time, then export output.step and run the benchmark's validity
gate on it. Records tokens, tok/s, time to first token and wall time.

Usage is in README.md; run.sh calls this.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from mcp_client import MCPClient, MCPError, MCPTimeout  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# MCP tools the model does not get: the harness owns files and the design's
# lifetime (export/save/open/new_design), and these are app-only.
MODELING = re.compile(r'"(solid|surface)\.[a-z_.]+"')
HIDDEN_TOOLS = {"export", "save", "open", "new_design", "crash_reports", "running", "pending"}

MAX_TOOL_TEXT = 30_000      # characters of one tool result kept in the conversation
KEEP_IMAGES = 2             # most recent rendered images kept in the conversation
GATE_TIMEOUT_S = 240
CUT_OFF_KEEP = 6_000       # characters of a cut-off reply kept so the model can continue

SYSTEM_PROMPT = """\
You are an expert mechanical CAD engineer. You build precise 3D solids in SolidCraft, a \
parametric CAD program, by calling its tools. Every SolidCraft tool runs real modeling \
commands on one design; lengths are in millimetres.

Your goal: produce ONE closed, valid solid that matches the task as accurately as possible. \
It is scored on shape similarity, interface features (holes, bores, mounting faces) and \
topology against a hidden reference part, but only if it first passes a strict validity \
check (BRep valid, watertight, closed manifold). An invalid part scores zero.

How to work:
1. Read every dimension off the drawing (or inspect the loaded part) and write a short plan \
with numbers before modeling.
2. Build in a few large `batch` calls. Check each stage with `measure` against volumes and \
bounding boxes you computed yourself. Use `screenshot` to look at the part.
3. Call `submit` whenever you have a complete part. It exports the current design, runs the \
benchmark's validity check and shows you the result and a render. You may submit as often \
as you like; the last submission counts.
4. When the part is as good as you can make it and the last `submit` passed, call `finish`.

Rules:
- Keep everything as one body. Avoid features that only touch tangentially (a face exactly \
tangent to a cylinder, coincident faces): overlap joined pieces slightly instead, because \
tangent contacts often produce invalid faces.
- Do not leave sealed internal voids.
- If a feature fails repeatedly, simplify it or leave it out; a valid simpler part beats an \
invalid detailed one.
- You have a hard time limit of {minutes:.0f} minutes for this task. Each reply is limited to \
{reply_tokens} tokens, and long deliberation uses up that time: think briefly, act in small steps, \
call `submit` as soon as a first complete part exists, then improve it. Tool results show the time left.

Editing an imported part (a STEP file already loaded in the design):
- It has no feature history, so there is nothing to re-parameterize. Change its geometry directly.
- Find features with `body_topology` (`faces: "cylinders"` is short; `faces: "all"` can be huge) and `screenshot`.
- Use `solid.offset_face` or `solid.replace_face` to move faces, or add and remove material with \
`solid.cylinder`, `solid.extrude` and similar using `operation: "join"` or `"cut"`.
- Always pass `targets: ["<imported body name>"]`. Without it a join creates a separate body and \
the edit does not touch the part.
- If you cannot make the edit, say so with `finish`; do not submit the input unchanged as if it were edited.

SolidCraft server notes:
{instructions}

SolidCraft command reference (from the solidcraft-modeling skill built into this SolidCraft):
{reference}
"""

EXTRA_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit",
            "description": (
                "Export the current design as the task's output.step, run the CADGenBench validity "
                "check on it and return the result plus an iso render. Call it whenever you have a "
                "complete part; the last submission counts."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "End the task. Call after your final `submit`.",
            "parameters": {
                "type": "object",
                "properties": {"note": {"type": "string", "description": "One line on what is missing, if anything"}},
                "additionalProperties": False,
            },
        },
    },
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")[:80]


def data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


def mcp_text(result: dict) -> tuple[str, list[bytes]]:
    texts, images = [], []
    for item in result.get("content", []) or []:
        if item.get("type") == "text":
            texts.append(item.get("text", ""))
        elif item.get("type") == "image":
            try:
                images.append(base64.b64decode(item.get("data", "")))
            except Exception:
                pass
    return "\n".join(texts), images


def clip(text: str, n: int = MAX_TOOL_TEXT) -> str:
    if len(text) <= n:
        return text
    return text[:n] + f"\n... [{len(text) - n} more characters cut by the harness; ask for a narrower result]"


def mcp_tool_to_openai(tool: dict) -> dict:
    schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": (tool.get("description") or "")[:1500],
            "parameters": schema,
        },
    }


def run_gate(python: str, step: Path, checker_src: Path) -> dict:
    env = dict(os.environ, PYTHONPATH=str(checker_src))
    try:
        p = subprocess.run(
            [python, str(HERE / "gate.py"), str(step)],
            capture_output=True, text=True, timeout=GATE_TIMEOUT_S, env=env,
        )
        line = [l for l in p.stdout.splitlines() if l.startswith("{")]
        if line:
            return json.loads(line[-1])
        return {"is_valid": False, "crash": f"gate printed nothing (rc {p.returncode}): {p.stderr[-800:]}"}
    except subprocess.TimeoutExpired:
        return {"is_valid": False, "crash": f"gate timed out after {GATE_TIMEOUT_S}s"}


# ---------------------------------------------------------------------------
# LLM call with streaming metrics
# ---------------------------------------------------------------------------

def chat(client: OpenAI, model: str, messages: list, tools: list, args, timeout: float) -> tuple[dict, dict]:
    """One streamed chat completion. Returns (assistant_message, metrics)."""
    t0 = time.monotonic()
    t_first = None
    content, reasoning = [], []
    calls: dict[int, dict] = {}
    usage = None
    n_chunks = 0
    finish_reason = None
    kwargs = dict(
        model=model, messages=messages, tools=tools, tool_choice="auto",
        stream=True, stream_options={"include_usage": True},
        temperature=args.temperature, max_tokens=args.max_tokens_per_call, timeout=timeout,
    )
    stream = client.chat.completions.create(**kwargs)
    for chunk in stream:
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not chunk.choices:
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            finish_reason = ch.finish_reason
        d = ch.delta
        piece = False
        if d is None:
            continue
        if d.content:
            content.append(d.content); piece = True
        r = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None)
        if r:
            reasoning.append(r); piece = True
        for tc in d.tool_calls or []:
            piece = True
            slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function:
                if tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function.arguments:
                    slot["arguments"] += tc.function.arguments
        if piece:
            n_chunks += 1
            if t_first is None:
                t_first = time.monotonic()
        if time.monotonic() - t0 > timeout:
            raise TimeoutError("LLM call ran past the task deadline")
    t_end = time.monotonic()

    msg: dict = {"role": "assistant", "content": "".join(content) or None}
    tool_calls = []
    for i in sorted(calls):
        c = calls[i]
        tool_calls.append({
            "id": c["id"] or f"call_{i}_{int(t0*1000)}",
            "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"] or "{}"},
        })
    if tool_calls:
        msg["tool_calls"] = tool_calls

    completion = getattr(usage, "completion_tokens", None) if usage else None
    prompt = getattr(usage, "prompt_tokens", None) if usage else None
    decode_s = (t_end - t_first) if t_first else None
    m = {
        "wall_s": round(t_end - t0, 3),
        "ttft_s": round(t_first - t0, 3) if t_first else None,
        "decode_s": round(decode_s, 3) if decode_s else None,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "usage_reported": usage is not None,
        "stream_chunks": n_chunks,
        "finish_reason": finish_reason,
        "reasoning_chars": sum(len(r) for r in reasoning),
    }
    if completion and decode_s and decode_s > 0 and completion > 1:
        # first token arrives at t_first, so the rest decode over decode_s
        m["decode_tok_s"] = round((completion - 1) / decode_s, 2)
    if completion and t_end > t0:
        m["end_to_end_tok_s"] = round(completion / (t_end - t0), 2)
    return {"message": msg, "reasoning": "".join(reasoning)}, m


# ---------------------------------------------------------------------------
# one task
# ---------------------------------------------------------------------------

class Task:
    def __init__(self, fixture_dir: Path):
        import yaml

        self.id = fixture_dir.name
        self.dir = fixture_dir
        desc = yaml.safe_load((fixture_dir / "description.yaml").read_text())
        self.description = " ".join(str(desc["description"]).split())
        self.inputs = [fixture_dir / f for f in desc.get("input_files", [])]
        self.task_type = desc.get("task_type", "generation")
        self.images = [p for p in self.inputs if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        self.steps = [p for p in self.inputs if p.suffix.lower() in (".step", ".stp")]


def run_task(task: Task, args, client: OpenAI, out_dir: Path, reference: str) -> dict:
    tdir = out_dir / task.id
    renders = tdir / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    transcript = open(tdir / "transcript.jsonl", "w")
    output_step = tdir / "output.step"
    design_sc = tdir / "design.sc"

    def log(kind: str, **kw):
        transcript.write(json.dumps({"t": round(time.monotonic() - t0, 3), "kind": kind, **kw}, default=str) + "\n")
        transcript.flush()

    t0 = time.monotonic()
    deadline = t0 + args.task_timeout
    res: dict = {
        "task": task.id, "task_type": task.task_type, "started": now_iso(),
        "stopped_reason": None, "turns": 0, "llm_calls": [], "tool_calls": 0,
        "tool_errors": 0, "submits": 0, "harness_exports": 0, "modeling_calls": 0, "submit_results": [],
    }
    img_counter = [0]

    def save_png(png: bytes, label: str) -> str:
        img_counter[0] += 1
        name = f"{img_counter[0]:03d}_{slug(label)}.png"
        (renders / name).write_bytes(png)
        return name

    mcp = MCPClient(args.solidcraft_cli, [tdir.resolve(), task.dir.resolve()], tdir / "solidcraft_stderr.log")
    try:
        mcp.initialize()
        hidden = HIDDEN_TOOLS | ({"screenshot", "compare_picture"} if args.no_vision else set())
        tools = [mcp_tool_to_openai(t) for t in mcp.list_tools() if t["name"] not in hidden]
        tools += EXTRA_TOOLS
        res["solidcraft"] = mcp.server_info

        # Editing tasks: load the input part before the model starts.
        setup_note = ""
        for step in task.steps:
            r = mcp.call("open", {"path": str(step.resolve())}, timeout=300)
            text, _ = mcp_text(r)
            log("setup_open", path=str(step), result=clip(text, 2000), is_error=r.get("isError"))
            if r.get("isError"):
                raise MCPError(f"could not open {step}: {text[:500]}")
            m = mcp.call("measure", {}, timeout=300)
            try:
                bodies = json.loads(mcp_text(m)[0]).get("bodies", [])
                res["input_import"] = [{"closed": b.get("closed"), "faces": b.get("faces")} for b in bodies]
                if any(b.get("closed") is False for b in bodies):
                    res["input_import_warning"] = (
                        "SolidCraft imported the input as an open body; any export will fail the gate. "
                        + "; ".join(json.loads(text).get("warnings", [])[:3])
                    )[:1500]
                    print(f"[{task.id}] WARNING: {res['input_import_warning'][:300]}", flush=True)
            except (ValueError, AttributeError):
                pass
            setup_note = (
                f"\n\nThe starting part `{step.name}` is already loaded in the design. Its measurement:\n"
                + clip(mcp_text(m)[0], 4000)
                + "\nApply the requested edit to this part, keep everything else unchanged, then `submit`."
            )

        system = SYSTEM_PROMPT.format(
            minutes=args.task_timeout / 60, reply_tokens=args.max_tokens_per_call, instructions=mcp.instructions.strip(), reference=reference.strip()
        )
        user_text = task.description + setup_note
        if task.images:
            user_text += "\n\nThe drawing is attached."
        user_content: list | str
        if task.images:
            user_content = [{"type": "text", "text": user_text}] + [
                {"type": "image_url", "image_url": {"url": data_url(p.read_bytes())}} for p in task.images
            ]
        else:
            user_content = user_text
        messages: list = [{"role": "system", "content": system}, {"role": "user", "content": user_content}]
        log("system", chars=len(system))
        log("user", text=user_text, images=[p.name for p in task.images])

        finished = False
        idle_turns = 0
        while not finished:
            left = deadline - time.monotonic()
            if left < 5:
                res["stopped_reason"] = "timeout"
                break
            if res["turns"] >= args.max_turns:
                res["stopped_reason"] = "max_turns"
                break
            prune_images(messages)
            try:
                out, m = chat(client, args.model, messages, tools, args, timeout=left)
            except TimeoutError:
                res["stopped_reason"] = "timeout"
                log("llm_timeout")
                break
            except Exception as exc:  # noqa: BLE001
                if time.monotonic() >= deadline - 5 or "timed out" in str(exc).lower() or "Timeout" in type(exc).__name__:
                    res["stopped_reason"] = "timeout"
                    log("llm_timeout", error=f"{type(exc).__name__}: {exc}"[:500])
                    break
                res["stopped_reason"] = "llm_error"
                res["llm_error"] = f"{type(exc).__name__}: {exc}"[:2000]
                log("llm_error", error=res["llm_error"])
                break
            res["turns"] += 1
            res["llm_calls"].append(m)
            msg = out["message"]
            cut_off = m.get("finish_reason") == "length" and not msg.get("tool_calls")
            if cut_off:
                res["cut_off_replies"] = res.get("cut_off_replies", 0) + 1
                # Keep the end of what it worked out, so the next reply continues instead of restarting.
                tail = (msg.get("content") or out["reasoning"] or "")[-CUT_OFF_KEEP:]
                msg = {"role": "assistant", "content": tail or "(reply cut off)"}
            elif msg.get("content") is None and not msg.get("tool_calls"):
                msg = {"role": "assistant", "content": ""}
            messages.append(msg)
            log("assistant", content=msg.get("content"), reasoning=out["reasoning"], tool_calls=msg.get("tool_calls"), metrics=m)

            calls = msg.get("tool_calls") or []
            if cut_off:
                messages.append({"role": "user", "content": (
                    f"Your reply hit the {args.max_tokens_per_call}-token limit before any tool call. The end of it is "
                    "above. Do not start over: stop deliberating and make a tool call now, with what you have."
                )})
                continue
            if not calls:
                idle_turns += 1
                if idle_turns >= 2:
                    res["stopped_reason"] = "model_stopped"
                    break
                messages.append({"role": "user", "content": (
                    "Keep going with tool calls. Call `submit` when you have a complete part, then `finish`."
                )})
                continue
            idle_turns = 0

            pending_images: list[tuple[str, bytes]] = []
            for call in calls:
                name = call["function"]["name"]
                raw_args = call["function"]["arguments"] or "{}"
                res["tool_calls"] += 1
                tc0 = time.monotonic()
                try:
                    targs = json.loads(raw_args) if raw_args.strip() else {}
                    if not isinstance(targs, dict):
                        raise ValueError("arguments must be a JSON object")
                except Exception as exc:  # noqa: BLE001
                    reply, is_err, imgs = f"Invalid JSON arguments: {exc}", True, []
                else:
                    if name == "finish":
                        reply, is_err, imgs = "Finished.", False, []
                        finished = True
                        res["finish_note"] = targs.get("note")
                    elif name == "submit":
                        reply, is_err, imgs = do_submit(mcp, args, output_step, design_sc, res, deadline)
                    else:
                        left = max(5.0, deadline - time.monotonic())
                        try:
                            r = mcp.call(name, targs, timeout=left)
                            reply, imgs = mcp_text(r)
                            is_err = bool(r.get("isError"))
                        except MCPTimeout as exc:
                            reply, is_err, imgs = f"Tool timed out: {exc}", True, []
                            res["stopped_reason"] = "timeout_in_tool"
                            finished = True
                        except MCPError as exc:
                            reply, is_err, imgs = f"Tool error: {exc}", True, []
                if is_err:
                    res["tool_errors"] += 1
                saved = [save_png(p, f"{name}") for p in imgs]
                log("tool", name=name, arguments=raw_args, is_error=is_err, ms=round((time.monotonic() - tc0) * 1000),
                    result=clip(reply), images=saved)
                if not is_err and name in ("batch", "execute") and MODELING.search(raw_args):
                    res["modeling_calls"] = res.get("modeling_calls", 0) + 1
                note = f"\n[{len(imgs)} image(s) follow in the next message]" if imgs and not args.no_vision else ""
                left_s = max(0, int(deadline - time.monotonic()))
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": clip(reply) + note + f"\n[time left: {left_s // 60} min {left_s % 60} s]"})
                for p in imgs:
                    pending_images.append((name, p))
                if finished and res.get("stopped_reason") == "timeout_in_tool":
                    break
            elapsed = time.monotonic() - t0
            if not finished and not res["submits"] and not res.get("nudged") and elapsed > args.task_timeout * 0.5:
                res["nudged"] = True
                messages.append({"role": "user", "content": (
                    "Half the time is gone and nothing has been submitted. Finish a complete part and call `submit` now."
                )})
            if pending_images and not finished and not args.no_vision:
                content = [{"type": "text", "text": "Images returned by your last tool call(s): "
                            + ", ".join(n for n, _ in pending_images)}]
                content += [{"type": "image_url", "image_url": {"url": data_url(p)}} for _, p in pending_images]
                messages.append({"role": "user", "content": content})
        if finished and not res["stopped_reason"]:
            res["stopped_reason"] = "finished"
    except Exception as exc:  # noqa: BLE001
        res["stopped_reason"] = res.get("stopped_reason") or "harness_error"
        res["harness_error"] = f"{type(exc).__name__}: {exc}"
        log("harness_error", error=traceback.format_exc())
    finally:
        # Always leave the last state of the design behind, even on timeout.
        if res.get("stopped_reason") != "timeout_in_tool" and mcp.proc.poll() is None:
            if not res["submits"] or res["stopped_reason"] not in ("finished",):
                try:
                    reply, _, _ = do_submit(mcp, args, output_step, design_sc, res, time.monotonic() + 300, final=True)
                    log("final_export", result=clip(reply, 3000))
                except Exception as exc:  # noqa: BLE001
                    log("final_export_error", error=str(exc))
        mcp.close()
        transcript.close()

    res["wall_s"] = round(time.monotonic() - t0, 1)
    res["finished_at"] = now_iso()
    res["output_exists"] = output_step.exists()
    res["final_gate"] = res["submit_results"][-1]["gate"] if res["submit_results"] else None
    res["summary"] = summarize_calls(res["llm_calls"])
    (tdir / "metrics.json").write_text(json.dumps(res, indent=2, default=str))
    return res


def do_submit(mcp: MCPClient, args, output_step: Path, design_sc: Path, res: dict, deadline: float, final: bool = False):
    left = max(30.0, deadline - time.monotonic())
    if output_step.exists():
        output_step.unlink()
    r = mcp.call("export", {"path": str(output_step.resolve()), "format": "step"}, timeout=left)
    text, _ = mcp_text(r)
    if r.get("isError") or not output_step.exists():
        res["submit_results"].append({"final": final, "export_error": text[:2000], "gate": {"is_valid": False, "nothing_exported": True}})
        return f"Export failed: {text}", True, []
    try:
        mcp.call("save", {"path": str(design_sc.resolve())}, timeout=left)
    except MCPError:
        pass
    gate = run_gate(args.checker_python, output_step, args.checker_src)
    meas = ""
    try:
        meas = mcp_text(mcp.call("measure", {}, timeout=min(left, 120)))[0]
    except MCPError:
        pass
    imgs: list[bytes] = []
    if not final and not args.no_vision:
        try:
            shot = mcp.call("screenshot", {"view": "iso", "width": 640, "height": 480}, timeout=min(left, 120))
            imgs = mcp_text(shot)[1]
        except MCPError:
            pass
    res["harness_exports" if final else "submits"] += 1
    res["submit_results"].append({"final": final, "t": now_iso(), "gate": gate})
    verdict = "PASSED" if gate.get("is_valid") else "FAILED (this part would score 0)"
    reply = (
        f"Submitted output.step. Validity check {verdict}.\nGate result: {json.dumps(gate)}\n"
        f"SolidCraft measure:\n{clip(meas, 3000)}"
    )
    return reply, not gate.get("is_valid"), imgs


def prune_images(messages: list) -> None:
    """Keep the task's drawing and the last KEEP_IMAGES rendered images."""
    seen = 0
    for msg in reversed(messages[2:]):
        if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
            continue
        new = []
        for part in msg["content"]:
            if part.get("type") == "image_url":
                seen += 1
                if seen > KEEP_IMAGES:
                    new.append({"type": "text", "text": "[older render removed by the harness]"})
                    continue
            new.append(part)
        msg["content"] = new


def summarize_calls(calls: list[dict]) -> dict:
    def med(key):
        vals = [c[key] for c in calls if c.get(key) is not None]
        return round(statistics.median(vals), 3) if vals else None

    def tot(key):
        vals = [c[key] for c in calls if c.get(key) is not None]
        return sum(vals) if vals else None

    return {
        "llm_calls": len(calls),
        "llm_wall_s": round(sum(c["wall_s"] for c in calls), 1),
        "prompt_tokens_total": tot("prompt_tokens"),
        "completion_tokens_total": tot("completion_tokens"),
        "prompt_tokens_max": max((c["prompt_tokens"] for c in calls if c.get("prompt_tokens")), default=None),
        "ttft_s_median": med("ttft_s"),
        "decode_tok_s_median": med("decode_tok_s"),
        "end_to_end_tok_s_median": med("end_to_end_tok_s"),
    }


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

def preflight(client: OpenAI, args) -> dict:
    """Check the endpoint: model list, one tool call, one image."""
    out: dict = {}
    try:
        models = client.models.list()
        out["models"] = [m.id for m in models.data]
    except Exception as exc:  # noqa: BLE001
        out["models_error"] = str(exc)
    # 32x32 solid red PNG
    red = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAKElEQVR4nO3NsQ0AAAzCMP5/un0CNkuZ41wybXsHAAAAAAAAAAAAxR4yw/wuPL6QkAAAAABJRU5ErkJggg==")
    tool = [{
        "type": "function",
        "function": {
            "name": "report_color",
            "description": "Report the main color of the image.",
            "parameters": {"type": "object", "properties": {"color": {"type": "string"}}, "required": ["color"]},
        },
    }]
    content: list | str = [
        {"type": "text", "text": "What color is this image? Answer by calling report_color."},
        {"type": "image_url", "image_url": {"url": data_url(red)}},
    ]
    if args.no_vision:
        content = "Call report_color with color 'red'."
    try:
        t0 = time.monotonic()
        r = client.chat.completions.create(
            model=args.model, messages=[{"role": "user", "content": content}], tools=tool,
            tool_choice="auto", max_tokens=1024, temperature=0, timeout=300,
        )
        msg = r.choices[0].message
        calls = msg.tool_calls or []
        out["tool_call_ok"] = bool(calls) and calls[0].function.name == "report_color"
        out["tool_call_args"] = calls[0].function.arguments if calls else None
        out["reply_text"] = (msg.content or "")[:500]
        out["image_ok"] = None if args.no_vision else ("red" in (out["tool_call_args"] or "") + out["reply_text"].lower())
        out["seconds"] = round(time.monotonic() - t0, 2)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        out["tool_call_ok"] = False
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000/v1")
    ap.add_argument("--model", required=True, help="model name as the endpoint lists it")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "none"))
    ap.add_argument("--tasks", default="101,102,218", help="comma-separated fixture ids")
    ap.add_argument("--data-dir", type=Path, default=ROOT / "work" / "data")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "results")
    ap.add_argument("--task-timeout", type=float, default=1800, help="seconds per task (default 1800)")
    ap.add_argument("--max-turns", type=int, default=80)
    ap.add_argument("--max-tokens-per-call", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--no-vision", action="store_true", help="model has no image input: skip drawing tasks")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--solidcraft-cli", type=Path, required=True)
    ap.add_argument("--checker-python", default=sys.executable)
    ap.add_argument("--checker-src", type=Path, required=True, help="cadgenbench/src for the validity gate")
    args = ap.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, max_retries=0)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"{stamp}_{slug(args.model)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"results: {out_dir}", flush=True)

    run = {
        "harness": "solidcraft-cadbench 0.2",
        "started": now_iso(),
        "model": args.model,
        "base_url": args.base_url,
        "settings": {k: str(v) for k, v in vars(args).items() if k != "api_key"},
        "host": {"platform": platform.platform(), "python": platform.python_version(), "cpus": os.cpu_count()},
    }
    if not args.skip_preflight:
        print("preflight: checking tool calls" + ("" if args.no_vision else " and image input") + " ...", flush=True)
        pf = preflight(client, args)
        run["preflight"] = pf
        print("preflight:", json.dumps(pf), flush=True)
        if not pf.get("tool_call_ok"):
            print("STOP: the model did not return a native tool call. Check the server's tool-call parser "
                  "(vLLM: --enable-auto-tool-choice --tool-call-parser ...).", flush=True)
            run["stopped"] = "preflight_failed"
            (out_dir / "run.json").write_text(json.dumps(run, indent=2))
            return 2
        if pf.get("image_ok") is False:
            print("WARNING: the model did not say the test image was red; image input may not work. "
                  "Continuing; rerun with --no-vision to skip drawing tasks.", flush=True)

    # SolidCraft's built-in skill reference, so the prompt matches this binary.
    probe = MCPClient(args.solidcraft_cli, [out_dir.resolve()], out_dir / "solidcraft_probe.log")
    try:
        probe.initialize()
        reference = mcp_text(probe.call("skill_reference", {"skill": "solidcraft-modeling"}, timeout=60))[0]
        run["solidcraft"] = probe.server_info
    finally:
        probe.close()

    results = []
    for tid in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        task = Task(args.data_dir / tid)
        if task.images and args.no_vision:
            print(f"[{tid}] skipped: drawing task and --no-vision", flush=True)
            results.append({"task": tid, "stopped_reason": "skipped_no_vision"})
            continue
        print(f"[{tid}] {task.task_type}: {task.description[:100]}", flush=True)
        r = run_task(task, args, client, out_dir, reference)
        g = r.get("final_gate") or {}
        s = r.get("summary", {})
        print(f"[{tid}] {r['stopped_reason']} in {r['wall_s']}s, turns {r['turns']}, submits {r['submits']}, "
              f"gate {'PASS' if g.get('is_valid') else 'FAIL'}, decode {s.get('decode_tok_s_median')} tok/s, "
              f"ttft {s.get('ttft_s_median')}s", flush=True)
        results.append(r)

    run["finished"] = now_iso()
    run["tasks"] = [{k: v for k, v in r.items() if k not in ("llm_calls",)} for r in results]
    (out_dir / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (out_dir / "summary.md").write_text(summary_md(run, results))
    print((out_dir / "summary.md").read_text(), flush=True)
    return 0


def summary_md(run: dict, results: list[dict]) -> str:
    lines = [
        f"# SolidCraft x CADGenBench: {run['model']}",
        "",
        f"- Started {run['started']}, finished {run.get('finished')}",
        f"- SolidCraft {run.get('solidcraft', {}).get('version')}, endpoint {run['base_url']}, harness {run['harness']}",
        f"- Preflight: {json.dumps(run.get('preflight'))}",
        "",
        "| Task | Type | Stopped | Wall s | Turns | Model submits | Modeling calls | Gate | Decode tok/s (median) | TTFT s (median) | Completion tokens | Max prompt tokens | Cut-off replies | Tool errors |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        g = r.get("final_gate") or {}
        s = r.get("summary") or {}
        if r.get("stopped_reason") == "skipped_no_vision":
            gate = "skipped"
        elif g.get("nothing_exported") or not g:
            gate = "nothing exported"
        else:
            gate = "PASS" if g.get("is_valid") else "FAIL"
        lines.append(
            f"| {r['task']} | {r.get('task_type', '')} | {r.get('stopped_reason')} | {r.get('wall_s', '')} | "
            f"{r.get('turns', '')} | {r.get('submits', '')} | {r.get('modeling_calls', '')} | {gate} | "
            f"{s.get('decode_tok_s_median')} | {s.get('ttft_s_median')} | {s.get('completion_tokens_total')} | "
            f"{s.get('prompt_tokens_max')} | {r.get('cut_off_replies', 0) if 'turns' in r else ''} | {r.get('tool_errors', '')} |"
        )
    lines += ["", "The gate is the benchmark's validity check only; accuracy scores need the private ground truth.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
