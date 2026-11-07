"""Geographic train / tune / val splits for SpaceNet 6 tiles.

Neighbouring SN6 tiles overlap, and tiles from different collection strips cover the
same ground, so a random split leaks buildings between splits. Here a tile joins a
split only if its whole footprint lies inside that split's region; tiles crossing a
region boundary, or lying inside more than one region, are dropped.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.crs import CRS
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

Footprints = dict[str, BaseGeometry]


def split_regions(
    train_extent: BaseGeometry, val_extent: BaseGeometry, tune_fraction: float
) -> dict[str, BaseGeometry]:
    """Cut the eastern ``tune_fraction`` of the Train extent off as a tuning strip.

    Parameters
    ----------
    train_extent, val_extent
        Official SN6 extents, in the tiles' projected CRS (metres).
    tune_fraction
        Share of the Train extent's x-range (east side) used for threshold tuning.

    Returns
    -------
    dict
        ``{"train": core, "tune": strip, "val": val_extent}``.
    """
    if not 0.0 < tune_fraction < 1.0:
        raise ValueError(f"tune_fraction must be in (0, 1), got {tune_fraction}")
    minx, miny, maxx, maxy = train_extent.bounds
    cut = maxx - tune_fraction * (maxx - minx)
    return {
        "train": train_extent.intersection(box(minx, miny, cut, maxy)),
        "tune": train_extent.intersection(box(cut, miny, maxx, maxy)),
        "val": val_extent,
    }


def tile_footprints(paths: Iterable[Path], id_pattern: str, expected_crs: str | CRS) -> Footprints:
    """Read tile footprints (bounding boxes) from raster headers; no pixels are read.

    Parameters
    ----------
    paths
        Tile GeoTIFFs.
    id_pattern
        Regex whose first group is the tile ID (``tile_id_regex`` in ``data.yaml``).
    expected_crs
        Every tile must be in this CRS.

    Raises
    ------
    ValueError
        If a file name has no tile ID, an ID repeats, or a tile has another CRS.
    """
    rx = re.compile(id_pattern)
    want = CRS.from_user_input(expected_crs)
    out: Footprints = {}
    for p in paths:
        m = rx.search(Path(p).name)
        if m is None:
            raise ValueError(f"no tile ID in file name {p}")
        tid = m.group(1)
        if tid in out:
            raise ValueError(f"duplicate tile ID {tid}")
        with rasterio.open(p) as src:
            if src.crs != want:
                raise ValueError(f"{p}: CRS {src.crs} differs from expected {want}")
            out[tid] = box(*src.bounds)
    return out


def assign_splits(
    footprints: Mapping[str, BaseGeometry],
    regions: Mapping[str, BaseGeometry],
    tolerance_m: float,
) -> tuple[dict[str, list[str]], list[str]]:
    """Assign each tile to the single region that fully contains it.

    Parameters
    ----------
    footprints
        Tile ID to footprint.
    regions
        Split name to region polygon.
    tolerance_m
        Regions are grown by this much before the test, so tiles whose edges sit on a
        region edge (up to float noise) still count as inside.

    Returns
    -------
    tuple
        ``(assigned, dropped)``: split name to sorted tile IDs, and the sorted IDs that
        straddle a boundary, fall outside every region, or fit in more than one.
    """
    grown = {name: reg.buffer(tolerance_m) if tolerance_m else reg for name, reg in regions.items()}
    assigned: dict[str, list[str]] = {name: [] for name in regions}
    dropped: list[str] = []
    for tid in sorted(footprints):
        hits = [name for name, reg in grown.items() if reg.covers(footprints[tid])]
        if len(hits) == 1:
            assigned[hits[0]].append(tid)
        else:
            dropped.append(tid)
    return assigned, dropped


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_splits(
    assigned: Mapping[str, list[str]],
    dropped: list[str],
    out_dir: Path,
    meta: Mapping[str, Any],
) -> dict[str, Any]:
    """Write ``<split>.txt`` (sorted tile IDs) and ``splits.json`` (rule, counts, hashes).

    Returns
    -------
    dict
        The content written to ``splits.json``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for name, ids in assigned.items():
        p = out_dir / f"{name}.txt"
        p.write_text("".join(f"{t}\n" for t in sorted(ids)))
        hashes[p.name] = _sha256(p)
    info: dict[str, Any] = {
        **meta,
        "counts": {**{k: len(v) for k, v in assigned.items()}, "dropped": len(dropped)},
        "sha256": hashes,
    }
    (out_dir / "splits.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    return info


def select_subset(
    footprints: Mapping[str, BaseGeometry],
    assigned: Mapping[str, list[str]],
    counts: Mapping[str, int],
) -> dict[str, list[str]]:
    """Pick a compact cluster of tiles per split for the laptop working subset.

    For each split, the tiles nearest to the mean tile centre are taken (ties broken by
    tile ID), so the subset is a contiguous patch that mixes collection strips.

    Parameters
    ----------
    footprints
        Tile ID to footprint.
    assigned
        Output of :func:`assign_splits`.
    counts
        Tiles wanted per split; capped at what is available.

    Returns
    -------
    dict
        Split name to sorted chosen tile IDs.
    """
    chosen: dict[str, list[str]] = {}
    for name, want in counts.items():
        ids = sorted(assigned.get(name, []))
        if not ids:
            chosen[name] = []
            continue
        xy = np.array([[footprints[t].centroid.x, footprints[t].centroid.y] for t in ids])
        d = np.hypot(*(xy - xy.mean(axis=0)).T)
        order = np.lexsort((np.arange(len(ids)), d))  # distance, then ID order
        chosen[name] = sorted(ids[i] for i in order[:want])
    return chosen
