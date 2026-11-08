"""Post-processing smoke test on D4 with synthetic noise (module d evidence).

No model exists yet, so a probability map is faked from the D4 labels: buildings at
0.85, background at 0.10, then speckle (isolated false positives), roof holes and small
spurious blobs are added. The scene is cleaned block-wise from disk (``clean_raster``),
compared with the labels before and after cleaning, and checked against cleaning the
whole scene in RAM (they must be identical). Memory of the cleaning step is sampled.

Usage: uv run python scripts/postprocess_check.py [--seed 42] [--figure PATH]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader, RasterWriter
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.postprocess.morphology import binarize, clean_mask, clean_raster, label_components
from geoinfer.tiling.config import load_tiling_config


def _anon_mib() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("RssAnon:"):
            return int(line.split()[1]) / 1024
    return float("nan")


class _PeakAnon:
    """Sample anonymous RSS in a thread; ``peak - start`` is the step's own memory."""

    def __enter__(self) -> _PeakAnon:
        self.start = self.peak = _anon_mib()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, _anon_mib())

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._t.join()


def _read_rows(path: Path, step: int) -> np.ndarray:
    """Band 1 of a raster, read in row blocks (no single whole-raster read)."""
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        out = np.empty((h, w), dtype=r.info.dtypes[0])
        for r0 in range(0, h, step):
            win = Window(0, r0, w, min(step, h - r0))
            out[r0 : r0 + win.height] = r.read(win, [1]).data[0]
    return out


def _scores(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    p, g = pred & valid, gt & valid
    tp = float((p & g).sum())
    fp, fn = float((p & ~g).sum()), float((~p & g).sum())
    return {
        "iou": round(tp / max(tp + fp + fn, 1.0), 4),
        "f1": round(2 * tp / max(2 * tp + fp + fn, 1.0), 4),
        "precision": round(tp / max(tp + fp, 1.0), 4),
        "recall": round(tp / max(tp + fn, 1.0), 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--figure", type=Path, default=Path("docs/figures/postprocess_d4.png"))
    p.add_argument("--json", type=Path, default=Path("docs/figures/postprocess_d4.json"))
    a = p.parse_args()
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    d4 = Path(dcfg["scenes"]["d4"])
    cfg, io_cfg = load_postprocess_config(), load_io_config()
    rng = np.random.default_rng(a.seed)

    step = io_cfg.cog.blocksize
    with RasterReader(d4 / "d4_labels.tif") as r:
        info = r.info
    gt = _read_rows(d4 / "d4_labels.tif", step) > 0
    valid = _read_rows(d4 / "d4_strips.tif", step) > 0
    gt &= valid
    h, w = gt.shape

    prob = np.where(gt, 0.85, 0.10).astype(np.float32)
    bg = valid & ~gt
    speck = bg & (rng.random((h, w)) < 0.015)  # isolated false positives
    holes = gt & (rng.random((h, w)) < 0.03)  # roof holes
    prob[speck] = 0.9
    prob[holes] = 0.2
    n_blobs = int(valid.sum() / 2000)  # small spurious blobs, all below 80 px
    ys, xs = rng.integers(0, h - 8, n_blobs), rng.integers(0, w - 8, n_blobs)
    sizes = rng.integers(2, 8, n_blobs)
    for y, x, s in zip(ys, xs, sizes, strict=True):
        region = (slice(y, y + s), slice(x, x + s))
        prob[region] = np.where(bg[region], 0.9, prob[region])
    prob[~valid] = np.nan

    scratch = load_tiling_config().scratch_dir
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pp-", dir=scratch))
    try:
        src, dst = tmp / "prob.tif", tmp / "mask.tif"
        with RasterWriter(
            src, info, io_cfg, count=1, dtype="float32", nodata=np.nan, cog=False
        ) as wr:
            for r0 in range(0, h, step):
                win = Window(0, r0, w, min(step, h - r0))
                wr.write(prob[None, r0 : r0 + win.height], win)

        raw = binarize(prob, cfg.threshold, valid)
        t0 = time.perf_counter()
        whole = clean_mask(prob, valid, cfg, info.transform)
        t_whole = time.perf_counter() - t0

        del prob
        t0 = time.perf_counter()
        with _PeakAnon() as mem:
            clean_raster(src, dst, cfg, io_cfg)
        t_block = time.perf_counter() - t0
        out = _read_rows(dst, step)
        cleaned = out == 1
        identical = bool(np.array_equal(cleaned, whole) and np.all((out == 255) == ~valid))

        n = {
            "labels": label_components(gt, cfg.connectivity)[1],
            "raw": label_components(raw, cfg.connectivity)[1],
            "cleaned": label_components(cleaned, cfg.connectivity)[1],
        }
        s_raw, s_clean = _scores(raw, gt, valid), _scores(cleaned, gt, valid)
        result: dict[str, Any] = {
            "scene": f"D4 {w} x {h} px, {valid.mean():.1%} valid",
            "noise": {
                "speck_px": int(speck.sum()),
                "hole_px": int(holes.sum()),
                "blobs": n_blobs,
                "seed": a.seed,
            },
            "config": {
                "threshold": cfg.threshold,
                "opening": cfg.opening,
                "closing": cfg.closing,
                "min_area_m2": cfg.min_area_m2,
                "connectivity": cfg.connectivity,
                "block": cfg.block,
            },
            "pixel_raw": s_raw,
            "pixel_cleaned": s_clean,
            "components": n,
            "blockwise_equals_whole": identical,
            "seconds": {
                "blockwise_from_disk": round(t_block, 1),
                "whole_in_ram": round(t_whole, 1),
            },
            "blockwise_extra_anon_mib": round(mem.peak - mem.start, 0),
        }
        print(json.dumps(result, indent=2))
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(result, indent=2) + "\n")

        r0, c0, s = 7400, 1000, 400  # same crop as the D4 quick-look
        crop = (slice(r0, r0 + s), slice(c0, c0 + s))
        fig, ax = plt.subplots(1, 3, figsize=(12, 4.4), dpi=100)
        for x, img, title in (
            (ax[0], raw[crop], f"thresholded noisy input (IoU {s_raw['iou']})"),
            (ax[1], cleaned[crop], f"cleaned (IoU {s_clean['iou']})"),
            (ax[2], gt[crop], "labels"),
        ):
            x.imshow(img, cmap="gray", interpolation="nearest")
            x.set_title(title, fontsize=10)
            x.set_xticks([])
            x.set_yticks([])
        fig.suptitle("D4 post-processing on synthetic noise (open 3, close 3, < 20 m2 removed)")
        fig.tight_layout()
        fig.savefig(a.figure)
        print(f"figure: {a.figure}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
