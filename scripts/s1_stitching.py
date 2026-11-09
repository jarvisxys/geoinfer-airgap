"""Study S1 harness: stitching strategies on a large scene (D4).

For each strategy, tiles are fed through a "model", stitched, and scored:

- scene pixel IoU / F1 of the thresholded probability map;
- border-band IoU: the same, only on pixels within ``border_band_px`` of a seam line;
- seam score (eval.seams) on the probability map, skipping pixels near D4 strip joins;
- object F1 after the module (d) clean-up and vectorizing;
- tile + stitch time.

All strategies are measured on the same lines: the union of the seam lines of the
O = 0 grid (naive) and the O = ``overlap`` grid (crop / hann / tukey).

``--model run --run-dir models/s3/optical`` uses a trained run: the D4 optical tile is
normalised as in training and passed through its ``best.pt``; the binarisation threshold
is the run's tuned one (``eval.json``). Only optical runs are accepted: SAR and fused
models need a per-strip north-south flip (decision D5), and a tile on the D4 mosaic can
straddle strips with opposite look directions.

``--model synthetic`` stands in for a network: it reads the
label tile and adds two CNN-like defects, confidence that fades towards the tile edges
and a per-tile calibration offset (configs/eval.yaml, s1.synthetic). Results from it
test the harness and show how the strategies react to edge defects; they are not model
accuracy.

Usage: uv run python scripts/s1_stitching.py [--model synthetic]
       uv run python scripts/s1_stitching.py --model run --run-dir models/s3/optical
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from dataclasses import replace
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


class RunModel:
    """A trained optical run applied to D4 optical tiles (see module docstring)."""

    def __init__(self, run_dir: Path, optical_path: Path, pad_mode: str, device: str) -> None:
        import torch

        from geoinfer.models.unet import build_unet
        from geoinfer.train.dataset import Normalizer
        from geoinfer.train.train import CHANNELS, resolve_amp

        self.torch = torch
        run = json.loads((run_dir / "run.json").read_text())
        cfg = run["config"]
        if cfg["modality"] != "optical":
            raise SystemExit("S1 on the D4 mosaic takes optical runs only (see docstring)")
        ev = json.loads((run_dir / "eval.json").read_text())
        self.threshold = float(ev["threshold"])
        pre = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
        stats = Path(pre["stats"]["out_dir"])
        self.norm = Normalizer.from_files(
            stats / "stats.json",
            stats / "strip_stats.json",
            cfg["norm"]["optical"],
            cfg["norm"]["sar"],
            pre["sar"]["arms"],
        )
        arch = yaml.safe_load(Path("configs/models.yaml").read_text()).get("unet", {})
        self.net = build_unet(
            CHANNELS["optical"],
            encoder=arch.get("encoder", "resnet34"),
            decoder_channels=tuple(arch.get("decoder_channels", (256, 128, 64, 32, 16))),
        )
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        self.net.load_state_dict(state["model"])
        self.net.to(device).eval()
        self.device = device
        self.amp, self.amp_reason = resolve_amp(
            cfg["optim"]["amp"], self.net, 4, device, size=int(cfg["data"]["crop"])
        )
        # held open for the whole run (closed when the process exits)
        self.reader = RasterReader(optical_path, max_window_px=4 * 1024 * 1024).__enter__()
        self.pad_mode = pad_mode
        self.name = run["run_name"]

    def __call__(self, spec: TileSpec) -> np.ndarray:
        torch = self.torch
        t = read_tile(self.reader, spec, None, self.pad_mode)
        # reflected padding is real imagery: normalise every non-zero pixel; the stitcher
        # gets the scene's valid mask separately
        x = self.norm.optical(t.data, np.any(t.data != 0, axis=0))
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = self.net(torch.from_numpy(x[None]).to(self.device))
        return np.asarray(torch.sigmoid(logits.float()).cpu().numpy()[0, 0], np.float32)


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
    p.add_argument("--model", choices=["synthetic", "run"], default="synthetic")
    p.add_argument("--run-dir", type=Path, default=None, help="trained run (--model run)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--scene", default=None, help="key under scenes: in data.yaml (d4, d3val)")
    p.add_argument("--scene-dir", type=Path, default=None, help="override the scene folder")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    ecfg: dict[str, Any] = yaml.safe_load(Path("configs/eval.yaml").read_text())
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    s1 = ecfg["s1"]
    key = a.scene or s1["scene"]
    scene = a.scene_dir or Path(dcfg["scenes"][key])
    labels_path, strips_path = scene / f"{key}_labels.tif", scene / f"{key}_strips.tif"
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
    gt_all = gpd.read_file(scene / f"{key}_buildings.geojson")
    gt_poly = filter_by_valid(gt_all, valid, info.transform)
    threshold = pcfg.threshold
    if a.model == "run":
        if a.run_dir is None:
            raise SystemExit("--model run needs --run-dir")
        run_model = RunModel(a.run_dir, scene / f"{key}_optical.tif", tcfg.pad_mode, a.device)
        threshold = run_model.threshold
        model_name = run_model.name
        print(f"run {model_name}: threshold {threshold}, amp {run_model.amp_reason}")
    else:
        synth = SyntheticEdgeModel(s1["synthetic"], tile)
        model_name = "synthetic"
    out = a.out or Path(f"docs/figures/s1_{key}_{model_name}")

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
                if a.model == "run":
                    tile_prob = run_model(spec)
                else:
                    tile_prob = synth(read_tile(rl, spec, None, tcfg.pad_mode).data[0], spec)
                st.add(spec, tile_prob, tvalid)
                n += 1
            prob = st.read(Window(0, 0, w, h))[0]
        t_stitch = time.perf_counter() - t0
        probs[strategy] = prob
        covered = np.isfinite(prob) & valid
        pred = binarize(prob, threshold, covered)
        scene_c = PixelCounts.from_masks(pred, gt, covered)
        band_c = PixelCounts.from_masks(pred, gt, covered & band)
        seam = seam_score(prob, rows, cols, covered, excl)
        clean = clean_mask_blockwise(
            prob, covered, replace(pcfg, threshold=threshold), info.transform
        )
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
        "scene": f"{key} {w} x {h} px",
        "model": model_name,
        "tile": tile,
        "overlap": ov,
        "seam_lines": {"rows": len(rows), "cols": len(cols)},
        "border_band_px": ecfg["border_band_px"],
        "strip_join_exclude_px": ecfg["strip_join_exclude_px"],
        "synthetic": s1["synthetic"] if a.model == "synthetic" else None,
        "run_dir": str(a.run_dir) if a.run_dir else None,
        "threshold": threshold,
        "note": "synthetic model = labels + edge-confidence loss + per-tile bias; "
        "tests the harness, not model accuracy"
        if a.model == "synthetic"
        else "trained optical run on D4 optical (held-out val tiles)",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out_json = out.with_suffix(".json")
    out_json.write_text(json.dumps({"meta": meta, "results": results}, indent=2) + "\n")
    print(f"wrote {out_json}")

    # seam close-ups: the crossing of seam lines with the most labelled building pixels
    # within 128 px (so the panel shows roofs, not water)
    cands = [
        (int(r), int(c)) for r in r0_ for c in c0_ if 128 <= r < h - 128 and 128 <= c < w - 128
    ]
    yc, xc = max(
        cands, key=lambda rc: int(gt[rc[0] - 128 : rc[0] + 128, rc[1] - 128 : rc[1] + 128].sum())
    )
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
    fig.suptitle(f"S1 on {key.upper()}, model {model_name}: probability at a tile corner")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"))
    print(f"figure: {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
