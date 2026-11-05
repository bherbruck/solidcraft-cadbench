"""CADGenBench validity gate on one STEP file, printed as JSON.

Runs the benchmark's own check (cadgenbench.common.validity.analyze_step),
which the leaderboard uses before scoring. Run it in its own process: OCP
can crash or hang on a bad file.

    python gate.py path/to/output.step
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    step = Path(sys.argv[1])
    out: dict = {"file": str(step)}
    try:
        from cadgenbench.common.validity import analyze_step

        result = analyze_step(step)
        val = result.validation
        out.update(
            is_valid=bool(val.is_valid),
            is_watertight=bool(val.is_watertight),
            errors=list(val.topology_errors)[:20],
        )
        m = result.measurements
        b = m.bounding_box
        out.update(
            solids=m.solid_count,
            shells=m.shell_count,
            faces=m.face_count,
            volume_mm3=round(float(m.volume), 2),
            bbox_size_mm=[round(b.x_max - b.x_min, 3), round(b.y_max - b.y_min, 3), round(b.z_max - b.z_min, 3)],
        )
    except Exception as exc:  # noqa: BLE001
        out.update(is_valid=False, crash=f"{type(exc).__name__}: {exc}")
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
