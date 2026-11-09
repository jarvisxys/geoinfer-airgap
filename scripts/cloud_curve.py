"""S3 robustness curve: pixel IoU vs synthetic cloud cover on the optical input.

Each run is evaluated on the val split with clouds blended into the raw optical bands
(``preprocess.clouds`` in configs/preprocess.yaml: fractal noise, per-tile seeds, so all
models see identical clouds; brightness = train-split p98 white point). Runs keep their
tuned threshold (``eval.json``). Models were trained without clouds, so the curve shows
how gracefully each one degrades; a SAR-only run is evaluated once (clouds do not touch
it) and drawn flat.

Usage: uv run python scripts/cloud_curve.py RUN_DIR [RUN_DIR ...] [--data-root PATH]
           [--interim-root PATH] [--out docs/figures/s3_clouds]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import yaml

from geoinfer.eval.metrics import PixelCounts, pixel_scores
from geoinfer.preprocess.clouds import TileClouds
from geoinfer.train.evaluate import load_run, predict_tiles


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--interim-root", type=Path, default=None)
    p.add_argument("--sar-cache", default=None, help="override data.sar_cache of the run")
    p.add_argument("--test-list", type=Path, default=Path("data/splits/val.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=Path("docs/figures/s3_clouds"))
    a = p.parse_args()
    pre: dict[str, Any] = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    cc = pre["clouds"]
    stats = json.loads((Path(pre["stats"]["out_dir"]) / "stats.json").read_text())
    brightness = cc["brightness"] or stats["optical"]["p_hi"]

    curves: dict[str, Any] = {}
    for run_dir in a.run_dirs:
        lr = load_run(run_dir, a.device, a.data_root, a.interim_root, a.sar_cache)
        if lr.threshold is None:
            raise SystemExit(f"{run_dir}: no eval.json; run evaluate_run.py first")
        name, modality = lr.run["run_name"], lr.cfg["modality"]
        coverages = [0.0] if modality == "sar" else [float(c) for c in cc["coverages"]]
        rows = []
        for cov in coverages:
            t0 = time.perf_counter()
            corrupt = TileClouds(cov, cc["beta"], cc["softness"], brightness, int(cc["seed"]))
            ds = lr.dataset(a.test_list, optical_corrupt=corrupt)
            counts = PixelCounts()
            for pr in predict_tiles(lr.net, ds, a.device, lr.amp):
                counts += PixelCounts.from_masks(pr.prob >= lr.threshold, pr.mask, pr.valid)
            s = pixel_scores(counts)
            rows.append(
                {
                    "coverage": cov,
                    "iou": round(s["iou"], 4),
                    "f1": round(s["f1"], 4),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            print(name, rows[-1], flush=True)
        curves[name] = {"modality": modality, "threshold": lr.threshold, "points": rows}

    meta = {"test_list": str(a.test_list), "clouds": cc, "brightness": brightness}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "curves": curves}, indent=2) + "\n"
    )
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=110)
    xs = [float(c) * 100 for c in cc["coverages"]]
    for name, c in curves.items():
        pts = c["points"]
        if c["modality"] == "sar":
            ax.plot(xs, [pts[0]["iou"]] * len(xs), "--", label=f"{name} (no optical)")
        else:
            ax.plot([q["coverage"] * 100 for q in pts], [q["iou"] for q in pts], "o-", label=name)
    ax.set_xlabel("synthetic cloud cover on the optical input (%)")
    ax.set_ylabel("val pixel IoU")
    ax.set_title("S3: robustness to clouds (models trained without clouds)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"))
    print(f"wrote {a.out.with_suffix('.json')} and .png")


if __name__ == "__main__":
    main()
