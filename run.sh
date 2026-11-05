#!/usr/bin/env bash
# One command: set up, run CADGenBench tasks through SolidCraft with your model, package results.
#
#   ./run.sh --base-url http://localhost:8000/v1 --model <name as /v1/models lists it> [more options]
#
# Everything is installed inside this folder (./work); nothing outside it is changed.
# Options after --base-url/--model are passed to harness/run_bench.py (see README.md):
#   --tasks 101,102,204   --task-timeout 600   --no-vision   --api-key KEY
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
WORK=$ROOT/work
mkdir -p "$WORK"

CADGENBENCH_COMMIT=33304cf771fc5639144b1df9611e347251052cf8   # benchmark code for the validity gate
SOLIDCRAFT_SERIES=${SOLIDCRAFT_SERIES:-0.40}                    # newest 0.40.x release, rc or final

TASKS=101,102,204
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ ${args[i]} == --tasks ]] && TASKS=${args[i+1]}
  [[ ${args[i]} == --tasks=* ]] && TASKS=${args[i]#--tasks=}
done
if ! printf '%s\n' "$@" | grep -q -- '--base-url' || ! printf '%s\n' "$@" | grep -q -- '--model'; then
  sed -n 2,8p "$0"; exit 1
fi

say() { printf '\n== %s\n' "$*"; }

# 1. uv (a self-contained Python package manager), kept in ./work/uv
if command -v uv >/dev/null 2>&1; then UV=$(command -v uv); else
  UV=$WORK/uv/uv
  if [[ ! -x $UV ]]; then
    say "installing uv into $WORK/uv"
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$WORK/uv" UV_NO_MODIFY_PATH=1 INSTALLER_NO_MODIFY_PATH=1 sh
  fi
fi

# 2. Python env: harness + the benchmark's validity checker (pins match the benchmark's tested stack)
VENV=$WORK/venv
if [[ ! -x $VENV/bin/python ]]; then
  say "creating Python 3.12 env in $VENV (about 1 GB)"
  "$UV" venv -q --python 3.12 "$VENV"
fi
"$UV" pip install -q --python "$VENV/bin/python" -r "$ROOT/requirements.txt"

# 3. benchmark code (only its src/ is used, for the validity gate)
if [[ ! -d $WORK/cadgenbench/.git ]]; then
  say "fetching CADGenBench code at $CADGENBENCH_COMMIT"
  git init -q "$WORK/cadgenbench"
  git -C "$WORK/cadgenbench" remote add origin https://github.com/huggingface/cadgenbench.git
fi
if [[ $(git -C "$WORK/cadgenbench" rev-parse -q --verify HEAD 2>/dev/null) != "$CADGENBENCH_COMMIT" ]]; then
  git -C "$WORK/cadgenbench" fetch -q --depth 1 origin "$CADGENBENCH_COMMIT"
  git -C "$WORK/cadgenbench" checkout -q --detach FETCH_HEAD
fi

# 4. SolidCraft (newest $SOLIDCRAFT_SERIES release; SOLIDCRAFT_TAG=v0.40.0-rc.2 pins one)
say "SolidCraft $SOLIDCRAFT_SERIES"
CLI=$("$VENV/bin/python" "$ROOT/harness/fetch.py" solidcraft --series "$SOLIDCRAFT_SERIES" --dest "$WORK/solidcraft")
"$CLI" --version

# 5. task inputs from the public CADGenBench dataset (pinned revision)
say "fixtures $TASKS"
"$VENV/bin/python" "$ROOT/harness/fetch.py" data --tasks "$TASKS" --dest "$WORK/data"

# 6. run
say "running"
set +e
"$VENV/bin/python" "$ROOT/harness/run_bench.py" \
  --solidcraft-cli "$CLI" \
  --checker-python "$VENV/bin/python" \
  --checker-src "$WORK/cadgenbench/src" \
  --data-dir "$WORK/data" \
  --out-dir "$ROOT/results" \
  "$@" | tee "$WORK/last_run.log"
rc=${PIPESTATUS[0]}
set -e

# 7. package the newest results folder
latest=$(ls -td "$ROOT"/results/*/ 2>/dev/null | head -1)
if [[ -n $latest ]]; then
  cp "$WORK/last_run.log" "$latest/console.log"
  name=$(basename "$latest")
  tar -czf "$ROOT/results/$name.tar.gz" -C "$ROOT/results" "$name"
  say "done (exit $rc). Send this file back: $ROOT/results/$name.tar.gz ($(du -h "$ROOT/results/$name.tar.gz" | cut -f1))"
fi
exit "$rc"
