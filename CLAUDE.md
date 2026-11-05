# Notes for a coding agent running this

Goal: run the benchmark against a locally served model and hand back one tarball.

1. Find the endpoint and model name: `curl -s http://localhost:<port>/v1/models`. Use the `id` exactly.
2. Run from this folder: `./run.sh --base-url http://localhost:<port>/v1 --model <id>`.
   It installs everything into `./work` and needs no sudo. Don't install things system-wide.
3. If preflight says the tool call failed, the server is missing its tool-call parser
   (vLLM: `--enable-auto-tool-choice --tool-call-parser <parser>`). Fix the server, not the harness.
4. If the model has no image input, add `--no-vision` (drawing tasks are skipped).
5. When it finishes, report the `results/<date>_<model>.tar.gz` path and paste `summary.md`.

Don't edit the harness to make a task pass, don't change the prompts, and don't rerun a task just
because it failed: failures and timeouts are results. If the harness itself crashes, report the
traceback from `console.log`.
