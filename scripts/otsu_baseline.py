"""Classical SAR baseline on D4: Otsu threshold on dB, then the module (d) clean-up.

Feature: band-mean dB of SAR-Intensity, optionally after a Lee filter on linear
intensity (the S2 arms). Thresholds: one Otsu threshold for the whole scene, or one per
acquisition strip (SN6 SAR brightness differs between strips). Pixels above the
threshold are building. Scores (pixel IoU / F1 before and after clean-up, object F1)
are against the D4 labels inside the valid area. This is the "no deep learning" lower
bound of plan step 6.

Usage: uv run python scripts/otsu_baseline.py [--out docs/figures/otsu_d4]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio.windows import Window

from geoinfer.eval.metrics import PixelCounts, filter_by_valid, object_scores, pixel_scores
from geoinfer.io.raster import RasterReader
from geoinfer.models.classical import otsu_per_label, otsu_threshold
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.postprocess.morphology import clean_mask_blockwise
from geoinfer.postprocess.vectorize import to_geodataframe, vectorize_mask_blockwise
from geoinfer.preprocess.sar import db_to_linear, lee_filter, linear_to_db


def _read_rows(path: Path, step: int = 1024) -> np.ndarray:
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        out = np.empty((h, w), dtype=r.info.dtypes[0])
        for r0 in range(0, h, step):
            win = Window(0, r0, w, min(step, h - r0))
            out[r0 : r0 + win.height] = r.read(win, [1]).data[0]
    return out


def sar_feature(path: Path, lee: int | None, enl: float, block_rows: int) -> np.ndarray:
    """Band-mean dB (after an optional Lee filter), computed in row blocks with a halo."""
    halo = (lee or 1) // 2
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        out = np.zeros((h, w), dtype=np.float32)
        for r0 in range(0, h, block_rows):
            r1 = min(r0 + block_rows, h)
            e0, e1 = max(r0 - halo, 0), min(r1 + halo, h)
            db = r.read(Window(0, e0, w, e1 - e0)).data.astype(np.float64)
            valid = np.all(db != 0, axis=0)
            if lee:
                lin = lee_filter(db_to_linear(db, valid), valid, lee, enl)
                db = linear_to_db(lin, valid).astype(np.float64)
            feat = np.where(valid, db.mean(axis=0), 0.0)
            out[r0:r1] = feat[r0 - e0 : r0 - e0 + (r1 - r0)]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("docs/figures/otsu_d4"))
    a = p.parse_args()
    ecfg: dict[str, Any] = yaml.safe_load(Path("configs/eval.yaml").read_text())
    oc = ecfg["otsu"]
    dcfg: dict[str, Any] = yaml.safe_load(Path("configs/data.yaml").read_text())
    scene = Path(dcfg["scenes"][oc["scene"]])
    enl = float(json.loads(Path(oc["stats_json"]).read_text())["sar"]["enl"]["value"])
    pcfg = load_postprocess_config()
    lo, hi = (float(v) for v in oc["range_db"])

    strips = _read_rows(scene / "d4_strips.tif")
    valid = strips > 0
    gt = (_read_rows(scene / "d4_labels.tif") > 0) & valid
    with RasterReader(scene / "d4_labels.tif") as r:
        info = r.info
    gt_poly = filter_by_valid(gpd.read_file(scene / "d4_buildings.geojson"), valid, info.transform)
    n_labels = int(strips.max()) + 1

    results: list[dict[str, Any]] = []
    best: tuple[float, str, np.ndarray, np.ndarray, np.ndarray] | None = None
    for lee in oc["lee"]:
        t0 = time.perf_counter()
        feat = sar_feature(scene / "d4_sar.tif", lee, enl, int(oc["block_rows"]))
        t_feat = time.perf_counter() - t0
        t_scene = otsu_threshold(feat, valid, int(oc["bins"]), lo, hi)
        for scope in oc["scopes"]:
            if scope == "scene":
                thr = np.full(feat.shape, t_scene, dtype=np.float32)
                spread = 0.0
            else:
                per = otsu_per_label(feat, strips, valid, n_labels, int(oc["bins"]), lo, hi)
                per = np.where(np.isnan(per), t_scene, per)
                thr = per[strips].astype(np.float32)
                used = per[np.unique(strips[valid])]
                spread = float(used.max() - used.min())
            raw = (feat > thr) & valid
            clean = clean_mask_blockwise(raw.astype(np.float32), valid, pcfg, info.transform)
            polys = vectorize_mask_blockwise(clean, info.transform, pcfg.block, 4)
            pred_poly = filter_by_valid(to_geodataframe(polys, info.crs), valid, info.transform)
            obj = object_scores(pred_poly, gt_poly, pcfg.min_area_m2)
            s_raw = pixel_scores(PixelCounts.from_masks(raw, gt, valid))
            s_cln = pixel_scores(PixelCounts.from_masks(clean, gt, valid))
            name = f"{'raw dB' if lee is None else f'Lee {lee}'} / {scope} Otsu"
            row = {
                "variant": name,
                "threshold_db": round(t_scene, 2) if scope == "scene" else "per strip",
                "strip_threshold_range_db": round(spread, 2),
                "iou_raw": round(s_raw["iou"], 4),
                "iou_clean": round(s_cln["iou"], 4),
                "f1_clean": round(s_cln["f1"], 4),
                "precision_clean": round(s_cln["precision"], 4),
                "recall_clean": round(s_cln["recall"], 4),
                "object_f1": round(obj.f1, 4),
                "feature_s": round(t_feat, 1),
            }
            results.append(row)
            print(row)
            if best is None or s_cln["iou"] > best[0]:
                best = (s_cln["iou"], name, feat.copy(), raw, clean)

    meta = {
        "scene": f"D4 {info.width} x {info.height} px, {valid.mean():.1%} valid",
        "enl": enl,
        "bins": oc["bins"],
        "range_db": [lo, hi],
        "postprocess": {
            "opening": pcfg.opening,
            "closing": pcfg.closing,
            "min_area_m2": pcfg.min_area_m2,
            "connectivity": pcfg.connectivity,
        },
        "foreground": "feature > threshold (bright returns)",
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2) + "\n"
    )
    print(f"wrote {a.out.with_suffix('.json')}")

    assert best is not None
    _, name, feat, raw, clean = best
    r0, c0, s = 7400, 1000, 400  # same crop as the D4 quick-look
    win = (slice(r0, r0 + s), slice(c0, c0 + s))
    f = feat[win]
    fv = f[valid[win]]
    lo_p, hi_p = np.percentile(fv, [2, 98]) if fv.size else (0.0, 1.0)
    fig, ax = plt.subplots(1, 4, figsize=(16, 4.4), dpi=100)
    for x, img, title, cmap, vr in (
        (ax[0], f, f"SAR feature ({name.split(' / ')[0]})", "gray", (lo_p, hi_p)),
        (ax[1], raw[win], "Otsu mask", "gray", (0, 1)),
        (ax[2], clean[win], "after clean-up", "gray", (0, 1)),
        (ax[3], gt[win], "labels", "gray", (0, 1)),
    ):
        x.imshow(img, cmap=cmap, vmin=vr[0], vmax=vr[1], interpolation="nearest")
        x.set_title(title, fontsize=10)
        x.set_xticks([])
        x.set_yticks([])
    fig.suptitle(f"Classical SAR baseline on D4, best variant: {name}")
    fig.tight_layout()
    fig.savefig(a.out.with_suffix(".png"))
    print(f"figure: {a.out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
