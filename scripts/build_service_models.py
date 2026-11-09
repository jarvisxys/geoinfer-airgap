"""Assemble the service model registry folder (Phase 3; mounted at /models).

For every entry of ``registry:`` in configs/service.yaml, copies ``<run>/model.onnx`` to
``OUT/<id>/model.onnx`` and records its id, sensor mode, channels, tuned threshold
(``eval.json``), normalisation arms (``run.json``) and SHA-256 in ``OUT/registry.yaml``;
copies the train-split statistics to ``OUT/stats/stats.json``. Refuses runs whose ONNX
export failed or skipped the parity check. ``python -m geoinfer.service.registry sync``
then registers the folder inside the stack.

Usage: uv run python scripts/build_service_models.py [OUT]   (default /mnt/d/geo/models/service)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from geoinfer.service.registry import MANIFEST, sha256_file

CHANNELS = {"optical": 4, "sar": 4, "fused": 8}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out", type=Path, nargs="?", default=Path("/mnt/d/geo/models/service"))
    p.add_argument("--stats", type=Path, default=Path("data/processed/stats/stats.json"))
    a = p.parse_args()
    cfg: dict[str, Any] = yaml.safe_load(Path("configs/service.yaml").read_text())
    entries = []
    for model_id, spec in cfg["registry"].items():
        run = Path(spec["run"])
        exp = json.loads((run / "export.json").read_text())
        if not exp["parity"]["ok"]:
            raise SystemExit(f"{run}: ONNX parity check failed")
        run_cfg = json.loads((run / "run.json").read_text())["config"]
        mode = spec["sensor_mode"]
        if run_cfg["modality"] != mode:
            raise SystemExit(f"{run} is a {run_cfg['modality']} run, registry says {mode}")
        dst = a.out / model_id / "model.onnx"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run / "model.onnx", dst)
        entries.append(
            {
                "id": model_id,
                "path": f"{model_id}/model.onnx",
                "sensor_mode": mode,
                "in_channels": CHANNELS[mode],
                "threshold": float(json.loads((run / "eval.json").read_text())["threshold"]),
                "norm": dict(run_cfg["norm"]),
                "sha256": sha256_file(dst),
                "run_name": exp["run_name"],
                "parity_max_abs_logit": exp["parity"]["max_abs_logit"],
            }
        )
        print(
            f"{model_id}: {run} thr {entries[-1]['threshold']} sha256 {entries[-1]['sha256'][:12]}"
        )
    (a.out / "stats").mkdir(parents=True, exist_ok=True)
    shutil.copy2(a.stats, a.out / "stats" / "stats.json")
    (a.out / MANIFEST).write_text(yaml.safe_dump({"models": entries}, sort_keys=False))
    print(f"wrote {a.out / MANIFEST}")


if __name__ == "__main__":
    main()
