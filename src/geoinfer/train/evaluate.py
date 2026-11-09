"""S3 evaluation of a trained model on whole tiles (plan section 10).

1. :func:`predict_tiles` runs the model on whole (padded) tiles, removes the padding and
   flips look-direction-1 predictions back north-south (decision D5), so every
   prediction is in map orientation.
2. :func:`threshold_sweep` / :func:`best_threshold` pick the probability threshold with
   the highest pixel IoU on the tune split.
3. :func:`score_predictions` reports pixel IoU / F1 before and after the module (d)
   clean-up, and object F1 (SpaceNet matching, IoU >= 0.5, < 20 m2 ignored) against
   each tile's label polygons, overall and per look direction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import torch

from geoinfer.eval.metrics import PixelCounts, object_scores, pixel_scores
from geoinfer.io.raster import inspect_raster
from geoinfer.postprocess.config import PostprocessConfig
from geoinfer.postprocess.morphology import clean_mask
from geoinfer.postprocess.vectorize import polygonize, to_geodataframe
from geoinfer.train.dataset import SN6Tiles, TileRecord

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class TilePrediction:
    """Model output for one tile, in map orientation."""

    tile_id: str
    strip: str
    orientation: int
    prob: npt.NDArray[np.float32]
    mask: BoolArray
    valid: BoolArray


def predict_tiles(
    net: Any, ds: SN6Tiles, device: str, amp: bool, batch_size: int = 1
) -> Iterator[TilePrediction]:
    """Run ``net`` on every sample of an evaluation dataset (``train=False``)."""
    if ds.train:
        raise ValueError("predict_tiles needs an evaluation dataset (train=False)")
    net.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    k = 0
    with torch.no_grad():
        for b in loader:
            x = b["image"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = net(x)
            prob = torch.sigmoid(logits.float()).cpu().numpy()[:, 0]
            for j in range(prob.shape[0]):
                rec = ds.records[k]
                k += 1
                h, w = (int(v[j]) for v in b["size"])
                p = prob[j, :h, :w]
                m = b["mask"][j].numpy()[:h, :w] > 0
                v = b["valid"][j].numpy()[:h, :w]
                if bool(b["flipped_ns"][j]):
                    p, m, v = p[::-1], m[::-1], v[::-1]
                yield TilePrediction(
                    rec.tile_id,
                    rec.strip,
                    rec.orientation,
                    np.ascontiguousarray(p, dtype=np.float32),
                    np.ascontiguousarray(m),
                    np.ascontiguousarray(v),
                )


def threshold_sweep(
    preds: Iterable[TilePrediction], thresholds: Sequence[float]
) -> dict[float, PixelCounts]:
    """Pixel counts for each threshold, in one pass over the predictions."""
    counts = {t: PixelCounts() for t in thresholds}
    for p in preds:
        for t in thresholds:
            counts[t] += PixelCounts.from_masks(p.prob >= t, p.mask, p.valid)
    return counts


def best_threshold(counts: Mapping[float, PixelCounts]) -> float:
    """Threshold with the highest pixel IoU (lowest threshold on ties)."""

    def iou(t: float) -> float:
        v = pixel_scores(counts[t])["iou"]
        return -1.0 if math.isnan(v) else v

    return max(sorted(counts), key=iou)


def _round(d: Mapping[str, float]) -> dict[str, float | None]:
    return {k: (None if math.isnan(v) else round(v, 4)) for k, v in d.items()}


def score_predictions(
    preds: Iterable[TilePrediction],
    records: Mapping[str, TileRecord],
    threshold: float,
    pcfg: PostprocessConfig,
) -> dict[str, Any]:
    """Pixel and object scores at ``threshold``, overall and per look direction."""
    groups: dict[str, dict[str, Any]] = {}
    n = 0
    for p in preds:
        n += 1
        rec = records[p.tile_id]
        info = inspect_raster(rec.sar)
        raw = (p.prob >= threshold) & p.valid
        clean = clean_mask(p.prob, p.valid, pcfg, info.transform)
        pred_gdf = to_geodataframe(polygonize(clean, info.transform, pcfg.connectivity), info.crs)
        if rec.labels is not None:
            gt = gpd.read_file(rec.labels)
            gt = gt.to_crs(info.crs) if len(gt) and gt.crs is not None else gt.set_crs(info.crs)
        else:
            gt = gpd.GeoDataFrame(geometry=[], crs=info.crs)
        obj = object_scores(pred_gdf, gt, pcfg.min_area_m2)
        for key in ("all", str(p.orientation)):
            g = groups.setdefault(
                key,
                {
                    "raw": PixelCounts(),
                    "clean": PixelCounts(),
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tiles": 0,
                },
            )
            g["raw"] += PixelCounts.from_masks(raw, p.mask, p.valid)
            g["clean"] += PixelCounts.from_masks(clean, p.mask, p.valid)
            g["tp"] += obj.tp
            g["fp"] += obj.fp
            g["fn"] += obj.fn
            g["tiles"] += 1

    def summary(g: Mapping[str, Any]) -> dict[str, Any]:
        tp, fp, fn = g["tp"], g["fp"], g["fn"]
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else math.nan
        return {
            "tiles": g["tiles"],
            "pixel_raw": _round(pixel_scores(g["raw"])),
            "pixel_clean": _round(pixel_scores(g["clean"])),
            "object": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": None if math.isnan(f1) else round(f1, 4),
            },
        }

    overall = summary(groups["all"]) if "all" in groups else {}
    return {
        **overall,
        "threshold": threshold,
        "tiles": n,
        "by_orientation": {k: summary(v) for k, v in sorted(groups.items()) if k != "all"},
    }
