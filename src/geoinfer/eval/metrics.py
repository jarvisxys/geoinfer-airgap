"""Accuracy metrics: pixel IoU / F1 / precision / recall and object F1 (section 10).

Pixel scores come from TP / FP / FN / TN counts that add up, so scene-size rasters are
scored block by block. Object F1 follows the SpaceNet convention: a predicted polygon
matches a ground-truth polygon if their IoU is at least 0.5, each polygon matches at
most once (greedy, best IoU first), and polygons under ``min_area_m2`` are ignored on
both sides. Candidate pairs come from a shapely STRtree.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import shapely
from rasterio.transform import Affine, rowcol
from rasterio.windows import Window

from geoinfer.io.raster import RasterReader

BoolArray = npt.NDArray[np.bool_]
DEFAULT_BLOCK = 2048


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


@dataclass(frozen=True)
class PixelCounts:
    """Confusion counts of a binary building mask."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @classmethod
    def from_masks(
        cls, pred: BoolArray, gt: BoolArray, valid: BoolArray | None = None
    ) -> PixelCounts:
        """Count over ``valid`` pixels (all by default)."""
        if pred.shape != gt.shape or (valid is not None and valid.shape != pred.shape):
            raise ValueError(f"shape mismatch: {pred.shape}, {gt.shape}")
        v = np.ones(pred.shape, bool) if valid is None else valid
        p, g = pred & v, gt & v
        tp = int(np.count_nonzero(p & g))
        fp = int(np.count_nonzero(p & ~g))
        fn = int(np.count_nonzero(~p & g))
        return cls(tp, fp, fn, int(np.count_nonzero(v)) - tp - fp - fn)

    def __add__(self, other: PixelCounts) -> PixelCounts:
        return PixelCounts(
            self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn
        )


def pixel_scores(c: PixelCounts) -> dict[str, float]:
    """IoU, F1, precision and recall of the building class; NaN where undefined."""
    return {
        "iou": _ratio(c.tp, c.tp + c.fp + c.fn),
        "f1": _ratio(2 * c.tp, 2 * c.tp + c.fp + c.fn),
        "precision": _ratio(c.tp, c.tp + c.fp),
        "recall": _ratio(c.tp, c.tp + c.fn),
    }


def score_rasters(
    pred: str | Path,
    gt: str | Path,
    *,
    pred_value: int = 1,
    valid: str | Path | None = None,
    block: int = DEFAULT_BLOCK,
) -> PixelCounts:
    """Pixel counts of a mask raster against a label raster, block by block.

    Parameters
    ----------
    pred
        Mask raster; ``pred_value`` = building, nodata = no prediction (not scored).
    gt
        Label raster; any value > 0 is building (instance rasters work), nodata is not
        scored.
    valid
        Optional raster whose values > 0 mark the area to score (e.g. D4 strips).
    block
        Block side in pixels.
    """
    paths = [Path(pred), Path(gt)] + ([Path(valid)] if valid is not None else [])
    with ExitStack() as stack:
        readers = [stack.enter_context(RasterReader(p)) for p in paths]
        ref = readers[0].info
        for r in readers[1:]:
            i = r.info
            same = (i.width, i.height, i.crs) == (ref.width, ref.height, ref.crs)
            if not (same and i.transform.almost_equals(ref.transform, precision=1e-6)):
                raise ValueError(f"{i.path} is on another grid than {ref.path}")
        total = PixelCounts()
        for r0 in range(0, ref.height, block):
            for c0 in range(0, ref.width, block):
                win = Window(c0, r0, min(block, ref.width - c0), min(block, ref.height - r0))
                p = readers[0].read(win, [1])
                g = readers[1].read(win, [1])
                v = p.valid & g.valid
                if valid is not None:
                    v &= readers[2].read(win, [1]).data[0] > 0
                total += PixelCounts.from_masks(p.data[0] == pred_value, g.data[0] > 0, v)
        return total


@dataclass(frozen=True)
class ObjectScores:
    """Object-level matching result."""

    tp: int
    fp: int
    fn: int
    ious: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return _ratio(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        return _ratio(self.tp, self.tp + self.fn)

    @property
    def f1(self) -> float:
        return _ratio(2 * self.tp, 2 * self.tp + self.fp + self.fn)


def object_scores(
    pred: gpd.GeoDataFrame,
    gt: gpd.GeoDataFrame,
    min_area_m2: float,
    iou_threshold: float = 0.5,
) -> ObjectScores:
    """Match predicted to ground-truth polygons (SpaceNet object F1).

    Both frames must share one projected CRS (areas in m2).
    """
    if pred.crs is None or gt.crs is None or pred.crs != gt.crs:
        raise ValueError(f"CRS mismatch: {pred.crs} vs {gt.crs}")
    if not pred.crs.is_projected:
        raise ValueError("object scores need a projected CRS (areas in m2)")
    p = np.asarray([g for g in pred.geometry if g is not None and g.area >= min_area_m2])
    g = np.asarray([x for x in gt.geometry if x is not None and x.area >= min_area_m2])
    if len(p) == 0 or len(g) == 0:
        return ObjectScores(0, len(p), len(g))
    tree = shapely.STRtree(g)
    pi, gi = tree.query(p, predicate="intersects")
    inter = shapely.area(shapely.intersection(p[pi], g[gi]))
    union = shapely.area(p[pi]) + shapely.area(g[gi]) - inter
    iou = inter / np.maximum(union, 1e-12)
    keep = iou >= iou_threshold
    order = np.argsort(-iou[keep], kind="stable")
    used_p: set[int] = set()
    used_g: set[int] = set()
    ious: list[float] = []
    for a, b, v in zip(pi[keep][order], gi[keep][order], iou[keep][order], strict=True):
        if a in used_p or b in used_g:
            continue
        used_p.add(int(a))
        used_g.add(int(b))
        ious.append(float(v))
    tp = len(ious)
    return ObjectScores(tp, len(p) - tp, len(g) - tp, ious)


def filter_by_valid(gdf: gpd.GeoDataFrame, valid: BoolArray, transform: Affine) -> gpd.GeoDataFrame:
    """Keep polygons whose centroid falls on a valid pixel of ``valid``."""
    if len(gdf) == 0:
        return gdf
    c = gdf.geometry.centroid
    rows, cols = rowcol(transform, c.x.to_numpy(), c.y.to_numpy())
    r, k = np.asarray(rows), np.asarray(cols)
    h, w = valid.shape
    inside = (r >= 0) & (r < h) & (k >= 0) & (k < w)
    ok = np.zeros(len(gdf), dtype=bool)
    ok[inside] = valid[r[inside], k[inside]]
    return gdf[ok].reset_index(drop=True)
