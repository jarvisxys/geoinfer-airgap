"""Study S1 harness: stitching strategies on a large scene (D4).

For each strategy, tiles are fed through a "model", stitched, and scored:

- scene pixel IoU / F1 of the thresholded probability map;
- border-band IoU: the same, only on pixels within ``border_band_px`` of a seam line;
- seam score (eval.seams) on the probability map, skipping pixels near D4 strip joins;
- object F1 after the module (d) clean-up and vectorizing;
- tile + stitch time.

All strategies are measured on the same lines: the union of the seam lines of the
O = 0 grid (naive) and the O = ``overlap`` grid (crop / hann / tukey).

Until Phase 2 provides a network, ``--model synthetic`` stands in for one: it reads the
label tile and adds two CNN-like defects, confidence that fades towards the tile edges
and a per-tile calibration offset (configs/eval.yaml, s1.synthetic). Results from it
test the harness and show how the strategies react to edge defects; they are not model
accuracy.

Usage: uv run python scripts/s1_stitching.py [--model synthetic]
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import cv2
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.eval.metrics import (
    PixelCounts,
    filter_by_valid,
    object_scores,
    pixel_scores,
)
from geoinfer.eval.seams import border_band, label_edges, seam_lines, seam_score
from geoinfer.io.raster import RasterReader
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.postprocess.morphology import binarize, clean_mask_blockwise
from geoinfer.postprocess.vectorize import to_geodataframe, vectorize_mask_blockwise
from geoinfer.tiling.config import load_tiling_config
from geoinfer.tiling.stitcher import Stitcher
from geoinfer.tiling.tiler import TileSpec, read_tile, tile_grid


class SyntheticEdgeModel:
    """Labels plus CNN-like border defects (see module docstring)."""

    def __init__(self, cfg: dict[str, Any], tile: int) -> None:
        self.cfg = cfg
        y, x = np.mgrid[:tile, :tile]
        d = np.minimum(np.minimum(y, x), np.minimum(tile - 1 - y, tile - 1 - x)).astype(float)
        self.conf = 1.0 - cfg["edge_amplitude"] * np.exp(-d / cfg["edge_length_px"])

    def __call__(self, labels: np.ndarray, spec: TileSpec) -> np.ndarray:
        p = np.where(labels > 0, 0.9, 0.1).astype(np.float32)
        if self.cfg["label_blur_px"] > 0:
            p = np.asarray(cv2.GaussianBlur(p, (0, 0), self.cfg["label_blur_px"]), np.float32)
        # a tile's offset depends on where it sits, not on the strategy (fair comparison)
        rng = np.random.default_rng([self.cfg["seed"], spec.y0 + 10**6, spec.x0 + 10**6])
        bias = rng.normal(0.0, self.cfg["tile_bias_sd"])
        out = np.clip(0.5 + (p - 0.5) * self.conf + bias, 0.0, 1.0)
        return np.asarray(out, dtype=np.float32)


def _read_rows(path: Path, step: int = 1024) -> np.ndarray:
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        out = np.empty((h, w), dtype=r.info.dtypes[0])
        for r0 in range(0, h, step):
            win = Window(0, r0, w, min(step, h - r0))
            out[r0 : r0 + win.height] = r.read(win, [1]).data[0]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["synthetic"], default="synthetic")
    p.add_argument("--out", type=Path, default=Path("docs/figures/s1_d4_synthetic"))
    a = p.parse_args()
    ecfg: dict[str, Any] = yaml.safe_load(Path("configs/eval.yaml").read_text())
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    s1 = ecfg["s1"]
    scene = Path(dcfg["scenes"][s1["scene"]])
    labels_path, strips_path = scene / "d4_labels.tif", scene / "d4_strips.tif"
    pcfg, tcfg = load_postprocess_config(), load_tiling_config()
    tile, ov = int(s1["tile"]), int(s1["overlap"])

    gt_ids = _read_rows(labels_path)
    strips = _read_rows(strips_path)
    valid = strips > 0
    gt = (gt_ids > 0) & valid
    h, w = gt.shape
    with RasterReader(labels_path) as r:
        info = r.info
    r0_, c0_ = seam_lines(h, w, tile, 0)
    r1_, c1_ = seam_lines(h, w, tile, ov)
    rows, cols = np.union1d(r0_, r1_), np.union1d(c0_, c1_)
    band = border_band((h, w), rows, cols, ecfg["border_band_px"]) & valid
    excl = label_edges(strips, ecfg["strip_join_exclude_px"])
    gt_poly = filter_by_valid(gpd.read_file(scene / "d4_buildings.geojson"), valid, info.transform)
    model = SyntheticEdgeModel(s1["synthetic"], tile)

    results: dict[str, Any] = {}
    probs: dict[str, np.ndarray] = {}
    for strategy in s1["strategies"]:
        o = 0 if strategy == "naive" else ov
        t0 = time.perf_counter()
        n = 0
        with ExitStack() as st_:
            rl = st_.enter_context(RasterReader(labels_path, max_window_px=tile * tile))
            rs = st_.enter_context(RasterReader(strips_path, max_window_px=tile * tile))
            st = st_.enter_context(
                Stitcher(
                    h,
                    w,
                    tile,
                    o,
                    strategy,
                    memmap_min_px=tcfg.memmap_min_px,
                    scratch_dir=tcfg.scratch_dir,
                )
            )
            for spec in tile_grid(h, w, tile, o):
                tv = read_tile(rs, spec, None, tcfg.pad_mode)
                tvalid = tv.valid & (tv.data[0] > 0)
                if not tvalid.any():
                    continue
                lab = read_tile(rl, spec, None, tcfg.pad_mode).data[0]
                st.add(spec, model(lab, spec), tvalid)
                n += 1
            prob = st.read(Window(0, 0, w, h))[0]
        t_stitch = time.perf_counter() - t0
        probs[strategy] = prob
        covered = np.isfinite(prob) & valid
        pred = binarize(prob, pcfg.threshold, covered)
        scene_c = PixelCounts.from_masks(pred, gt, covered)
        band_c = PixelCounts.from_masks(pred, gt, covered & band)
        seam = seam_score(prob, rows, cols, covered, excl)
        clean = clean_mask_blockwise(prob, covered, pcfg, info.transform)
        polys = vectorize_mask_blockwise(clean, info.transform, pcfg.block, 4)
        pred_poly = filter_by_valid(to_geodataframe(polys, info.crs), valid, info.transform)
        obj = object_scores(pred_poly, gt_poly, pcfg.min_area_m2)
        results[strategy] = {
            "overlap": o,
            "tiles": n,
            "scene_iou": round(pixel_scores(scene_c)["iou"], 4),
            "scene_f1": round(pixel_scores(scene_c)["f1"], 4),
            "border_band_iou": round(pixel_scores(band_c)["iou"], 4),
            "seam_score": round(seam, 3),
            "object_f1": round(obj.f1, 4),
            "tile_and_stitch_s": round(t_stitch, 1),
        }
        print(strategy, results[strategy])

    meta = {
        "scene": f"{s1['scene']} {w} x {h} px",
        "model": a.model,
        "tile": tile,
        "overlap": ov,
        "seam_lines": {"rows": len(rows), "cols": len(cols)},
        "border_band_px": ecfg["border_band_px"],
        "strip_join_exclude_px": ecfg["strip_join_exclude_px"],
        "synthetic": s1["synthetic"],
        "threshold": pcfg.threshold,
        "note": "synthetic model = labels + edge-confidence loss + per-tile bias; "
        "tests the harness, not model accuracy",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out_json = a.out.with_suffix(".json")
    out_json.write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")
    print(f"wrote {out_json}")

    # seam close-ups: a window around a tile corner of the O = 0 grid
    yc = int(rows[np.argmin(np.abs(rows - h // 2))])
    xc = int(cols[np.argmin(np.abs(cols - w // 2))])
    win = (slice(yc - 128, yc + 128), slice(xc - 128, xc + 128))
    fig, ax = plt.subplots(1, len(probs), figsize=(4 * len(probs), 4.4), dpi=100)
    for x, (name, pr) in zip(np.atleast_1d(ax), probs.items(), strict=True):
        x.imshow(pr[win], cmap="magma", vmin=0, vmax=1, interpolation="nearest")
        x.axhline(127.5, color="cyan", lw=0.4, ls=":")
        x.axvline(127.5, color="cyan", lw=0.4, ls=":")
        r = results[name]
        x.set_title(f"{name}: seam {r['seam_score']}, band IoU {r['border_band_iou']}", fontsize=9)
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle("S1 on D4 with the synthetic edge-defect model: probability at a tile corner")
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"))
    print(f"figure: {a.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
