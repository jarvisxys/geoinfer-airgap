"""Evaluate a finished training run on the held-out split (study S3).

Rebuilds the run's dataset settings from its ``run.json``, loads ``best.pt``, picks the
threshold with the best pixel IoU on the tune split, then scores the val split at that
threshold: pixel IoU / F1 before and after the post-processing, object F1, and the same
per SAR look direction. Writes ``eval.json`` into the run directory.

Usage: uv run python scripts/evaluate_run.py RUN_DIR [--data-root PATH]
           [--interim-root PATH] [--test-list data/splits/val.txt]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from geoinfer.eval.metrics import pixel_scores
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.train.evaluate import (
    best_threshold,
    load_run,
    predict_tiles,
    score_predictions,
    threshold_sweep,
)

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data-root", type=Path, default=None, help="override data.root")
    p.add_argument("--interim-root", type=Path, default=None)
    p.add_argument("--sar-cache", default=None, help="override data.sar_cache of the run")
    p.add_argument("--test-list", type=Path, default=Path("data/splits/val.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    lr = load_run(a.run_dir, a.device, a.data_root, a.interim_root, a.sar_cache)
    run, cfg, net, amp, dataset = lr.run, lr.cfg, lr.net, lr.amp, lr.dataset
    post = load_postprocess_config()

    t0 = time.perf_counter()
    tune_ds = dataset(Path(cfg["data"]["val_list"]))
    sweep = threshold_sweep(predict_tiles(net, tune_ds, a.device, amp), THRESHOLDS)
    thr = best_threshold(sweep)
    test_ds = dataset(a.test_list)
    recs = {r.tile_id: r for r in test_ds.records}
    scores = score_predictions(predict_tiles(net, test_ds, a.device, amp), recs, thr, post)
    out = {
        "run_name": run["run_name"],
        "checkpoint_epoch": lr.epoch,
        "modality": cfg["modality"],
        "norm": cfg["norm"],
        "tune_list": cfg["data"]["val_list"],
        "test_list": str(a.test_list),
        "tune_sweep_iou": {str(t): round(pixel_scores(c)["iou"], 4) for t, c in sweep.items()},
        **scores,
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (a.run_dir / "eval.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "run_name",
                    "threshold",
                    "pixel_raw",
                    "pixel_clean",
                    "object",
                    "by_orientation",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
