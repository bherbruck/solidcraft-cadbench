# solidcraft-cadbench

Runs a few [CADGenBench](https://github.com/huggingface/cadgenbench) tasks with a **local model** driving
[SolidCraft](https://github.com/tangency-labs/solidcraft) (parametric CAD) through its MCP tools, and
measures how fast and how well it goes: tokens/s, time to first token, time to finish each task,
and whether the part it exports passes the benchmark's validity check.

## Run it

You need Linux (x86_64 or aarch64), `git`, `curl`, about 1.5 GB of free disk, and a model served
behind an OpenAI-compatible endpoint with **native tool calls** (vLLM: `--enable-auto-tool-choice`
and the model's `--tool-call-parser`). Two of the three default tasks start from a drawing, so the
model also needs **image input**.

```sh
git clone https://github.com/bherbruck/solidcraft-cadbench
cd solidcraft-cadbench
./run.sh --base-url http://localhost:8000/v1 --model <model name as /v1/models lists it>
```

That's all. It takes 90 minutes at most (3 tasks, 30-minute limit each), plus a few minutes
of setup on the first run. When it ends it prints the path of one file to send back:

```
results/<date>_<model>.tar.gz
```

Run it again for another model; each run gets its own results folder and tarball.

## What it does

1. Installs everything inside `./work` (nothing outside this folder changes):
   [uv](https://github.com/astral-sh/uv) if it isn't installed, a Python 3.12 env, the benchmark's
   code at a pinned commit (for the validity check), and the newest **SolidCraft 0.40** release
   from GitHub, checked against its published SHA-256.
2. Downloads the task inputs from the public
   [CADGenBench dataset](https://huggingface.co/datasets/HuggingAI4Engineering/cadgenbench-data)
   (pinned revision, ODC-By): 101 and 102 are engineering drawings to model, 218 is a STEP part to edit.
3. Checks the endpoint: one tool call and one test image. It stops if tool calls don't work.
4. For each task, starts a fresh headless SolidCraft (`solidcraft-cli mcp --headless`), gives the
   model the benchmark's own task text plus the drawing (or the part already loaded), and lets it
   call SolidCraft tools. The model calls `submit` to export and validity-check its part, and
   `finish` when done. At the time limit the harness exports whatever the design holds.
5. Writes the results and packs them into the tarball.

Nothing is uploaded anywhere.

## Options

| Option | Default | |
|---|---|---|
| `--tasks` | `101,102,218` | Any CADGenBench fixture ids |
| `--task-timeout` | `1800` | Seconds per task |
| `--no-vision` | off | For text-only models: skips drawing tasks and sends no images |
| `--api-key` | `none` | If the endpoint needs one (or set `OPENAI_API_KEY`) |
| `--temperature` | `0.6` | Use `0` if your server only has a fast path without sampling |
| `--max-tokens-per-call` | `32768` | |
| `--max-turns` | `80` | Model calls per task |

Environment: `SOLIDCRAFT_TAG=v0.40.0-rc.2` pins a SolidCraft release instead of taking the newest 0.40.

## What's in the results

```
results/<date>_<model>/
  summary.md          one table: time, turns, gate pass/fail, tok/s, TTFT, tokens per task
  run.json            settings, preflight, every task's metrics
  console.log
  <task>/
    output.step       the exported part
    design.sc         the SolidCraft design (open it in SolidCraft 0.40)
    transcript.jsonl  every model message, reasoning, tool call and tool result, with timings
    metrics.json      per-call TTFT, decode tok/s, prompt/completion tokens
    renders/*.png     every image the model saw
```

**Decode tok/s** is completion tokens after the first, divided by the time from first to last
streamed token, using the server's reported usage. **TTFT** includes prompt processing, which
grows as the conversation does.

The validity check is the benchmark's real gate, but the accuracy score needs the benchmark's
private ground truth, so this harness can't produce a CADGenBench score.

A cut-off reply (one that hits the per-reply token limit) keeps its last 6,000 characters in the
conversation and the model is told to act. Tool results end with the time left, and a model that
hasn't submitted at half time is reminded once. `Model submits` counts only the model's own `submit`
calls; a part exported by the harness at the end shows up under `harness_exports` in `metrics.json`.

## Known limits

- SolidCraft 0.40 uses its new solidcore kernel. Some STEP inputs don't import as closed solids
  on it yet (201, 206, 208, 209, 211, 212 and 224 of the 17 editing tasks tried); the harness warns
  when that happens, because any export of such a part fails the gate. Most of the others import
  closed but refuse booleans on rc.2 (`SMLib's check refuses the result`), so those edits can't be
  done by adding or removing material. The default edit task 218 accepts them.
- A SolidCraft call that runs past the task's time limit ends the task without a final export.

## License

MIT. The task inputs are downloaded from the CADGenBench dataset (ODC-By) at run time and are not
part of this repository.
