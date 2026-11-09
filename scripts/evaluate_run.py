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
from typing import Any

import numpy as np
import torch
import yaml

from geoinfer.eval.metrics import pixel_scores
from geoinfer.io.config import load_io_config
from geoinfer.models.unet import build_unet
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.train.dataset import Normalizer, SN6Tiles, index_tiles
from geoinfer.train.evaluate import (
    best_threshold,
    predict_tiles,
    score_predictions,
    threshold_sweep,
)
from geoinfer.train.train import CHANNELS, OPTICAL_DIR, SAR_DIR, TILE_REGEX, resolve_amp

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", type=Path)
    p.add_argument("--data-root", type=Path, default=None, help="override data.root")
    p.add_argument("--interim-root", type=Path, default=None)
    p.add_argument("--test-list", type=Path, default=Path("data/splits/val.txt"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    run = json.loads((a.run_dir / "run.json").read_text())
    cfg: dict[str, Any] = run["config"]
    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    pcfg_yaml = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    mcfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    interim = (a.interim_root or Path(dcfg["interim_root"])).expanduser()
    lines = (interim / dcfg["orientations"]).read_text().splitlines()
    orient = {k: int(v) for k, v in (ln.split() for ln in lines if ln.strip())}
    root = (a.data_root or Path(cfg["data"]["root"])).expanduser()
    stats = Path(pcfg_yaml["stats"]["out_dir"])
    norm = Normalizer.from_files(
        stats / "stats.json",
        stats / "strip_stats.json",
        cfg["norm"]["optical"],
        cfg["norm"]["sar"],
        pcfg_yaml["sar"]["arms"],
    )
    io_cfg, post = load_io_config(), load_postprocess_config()

    def dataset(list_path: Path) -> SN6Tiles:
        ids = Path(list_path).read_text().split()
        recs = index_tiles(root, ids, SAR_DIR, OPTICAL_DIR, TILE_REGEX, orient)
        return SN6Tiles(
            recs,
            cfg["modality"],
            norm,
            root / cfg["data"]["masks_dir"],
            io_cfg,
            crop=None,
            train=False,
            hflip=False,
            align_orientation=bool(cfg["data"]["align_orientation"]),
            seed=int(cfg.get("seed", 0)),
        )

    arch = mcfg.get("unet", {})
    net = build_unet(
        CHANNELS[cfg["modality"]],
        encoder=arch.get("encoder", "resnet34"),
        decoder_channels=tuple(arch.get("decoder_channels", (256, 128, 64, 32, 16))),
    )
    state = torch.load(a.run_dir / "best.pt", map_location="cpu", weights_only=True)
    net.load_state_dict(state["model"])
    net.to(a.device)
    amp, _ = resolve_amp(
        cfg["optim"]["amp"], net, CHANNELS[cfg["modality"]], a.device, size=int(cfg["data"]["crop"])
    )

    t0 = time.perf_counter()
    tune_ds = dataset(Path(cfg["data"]["val_list"]))
    sweep = threshold_sweep(predict_tiles(net, tune_ds, a.device, amp), THRESHOLDS)
    thr = best_threshold(sweep)
    test_ds = dataset(a.test_list)
    recs = {r.tile_id: r for r in test_ds.records}
    scores = score_predictions(predict_tiles(net, test_ds, a.device, amp), recs, thr, post)
    out = {
        "run_name": run["run_name"],
        "checkpoint_epoch": int(state["epoch"]),
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
