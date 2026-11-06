"""Download what a run needs: the CADGenBench fixture inputs and SolidCraft 0.40.

    python fetch.py data   --tasks 101,102,218 --dest work/data
    python fetch.py solidcraft --series 0.40 --dest work/solidcraft   (prints the CLI path)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

DATA_REPO = "HuggingAI4Engineering/cadgenbench-data"
DATA_REVISION = "f76f965585817c621d6ea0d150d745adf670e66e"   # pinned so every run sees the same inputs
SOLIDCRAFT_REPO = "tangency-labs/solidcraft"


def fetch_data(tasks: list[str], dest: Path) -> None:
    import yaml
    from huggingface_hub import hf_hub_download

    for t in tasks:
        desc = Path(hf_hub_download(DATA_REPO, f"{t}/description.yaml", repo_type="dataset",
                                    revision=DATA_REVISION, local_dir=dest))
        files = yaml.safe_load(desc.read_text()).get("input_files", [])
        for f in files:
            hf_hub_download(DATA_REPO, f"{t}/{f}", repo_type="dataset", revision=DATA_REVISION, local_dir=dest)
        print(f"fixture {t}: {', '.join(files)}", file=sys.stderr)


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "solidcraft-cadbench", "Accept": "application/vnd.github+json"})
    tok = os.environ.get("GITHUB_TOKEN")
    if tok and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {tok}")
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def _version_key(tag: str) -> tuple:
    # v0.40.0 > v0.40.0-rc.3 > v0.40.0-rc.2 ; v0.40.1-rc.1 > v0.40.0
    m = re.match(r"v?(\d+)\.(\d+)\.(\d+)(?:-rc\.(\d+))?$", tag)
    if not m:
        return (-1,)
    a, b, c, rc = m.groups()
    return (int(a), int(b), int(c), 1 if rc is None else 0, int(rc or 0))


def fetch_solidcraft(series: str, dest: Path, tag: str | None) -> Path:
    arch = {"x86_64": "x86_64", "AMD64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}[platform.machine()]
    if not tag:
        releases = json.loads(_get(f"https://api.github.com/repos/{SOLIDCRAFT_REPO}/releases?per_page=50"))
        tags = [r["tag_name"] for r in releases
                if r["tag_name"].lstrip("v").startswith(series + ".") and not r.get("draft")]
        if not tags:
            raise SystemExit(f"no SolidCraft {series}.x release found on GitHub")
        tag = max(tags, key=_version_key)
    rel = json.loads(_get(f"https://api.github.com/repos/{SOLIDCRAFT_REPO}/releases/tags/{tag}"))
    version = tag.lstrip("v")
    name = f"solidcraft-{version}-linux-{arch}.tar.gz"
    assets = {a["name"]: a["browser_download_url"] for a in rel["assets"]}
    if name not in assets:
        raise SystemExit(f"{tag} has no {name}")
    folder = dest / f"solidcraft-{version}-linux-{arch}"
    cli = folder / "bin" / "solidcraft-cli"
    if not cli.exists():
        dest.mkdir(parents=True, exist_ok=True)
        print(f"downloading SolidCraft {tag} ({name}) ...", file=sys.stderr)
        sums = _get(assets["SHA256SUMS.txt"]).decode()
        want = next((l.split()[0] for l in sums.splitlines() if l.strip().endswith(name)), None)
        blob = _get(assets[name])
        got = hashlib.sha256(blob).hexdigest()
        if want != got:
            raise SystemExit(f"checksum mismatch for {name}: expected {want}, got {got}")
        tar_path = dest / name
        tar_path.write_bytes(blob)
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest, filter="data")
        tar_path.unlink()
    if not cli.exists():
        raise SystemExit(f"{cli} not found after unpacking")
    (dest / "VERSION").write_text(tag + "\n")
    print(f"SolidCraft {tag}", file=sys.stderr)
    return cli


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["data", "solidcraft"])
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--tasks", default="101,102,218")
    ap.add_argument("--series", default="0.40")
    ap.add_argument("--tag", default=os.environ.get("SOLIDCRAFT_TAG") or None)
    a = ap.parse_args()
    if a.what == "data":
        fetch_data([t.strip() for t in a.tasks.split(",") if t.strip()], a.dest)
    else:
        print(fetch_solidcraft(a.series, a.dest, a.tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
