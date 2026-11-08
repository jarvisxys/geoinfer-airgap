"""Geographic train / tune / val splits for SpaceNet 6 tiles.

Neighbouring SN6 tiles overlap, and tiles from different collection strips cover the
same ground, so a random split leaks buildings between splits. Here a tile joins a
split only if its whole footprint lies inside that split's region; tiles crossing a
region boundary, or lying inside more than one region, are dropped.

The SN6 train tarball has no tiles in the official Val extent (it ends near x=596.7 km),
so all three tile splits are cut from the covered part of the Train extent. The official
Val extent is kept for scene-level optical evaluation on the PS-RGBNIR mosaic.
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
    extent: BaseGeometry,
    x_range: tuple[float, float],
    tune_fraction: float,
    val_fraction: float,
) -> dict[str, BaseGeometry]:
    """Cut an extent west to east into train | tune | val by vertical lines.

    The cuts are placed on ``x_range`` (the easting range the tiles actually cover),
    not on the extent itself: the SN6 train tarball covers only the western part of
    the official Train extent, so cutting the extent would leave tune and val empty.

    Parameters
    ----------
    extent
        Region the splits must stay inside (official Train extent), in metres.
    x_range
        ``(xmin, xmax)`` of the tile coverage, see :func:`footprint_x_range`.
    tune_fraction, val_fraction
        Shares of ``x_range`` for tune (middle strip) and val (eastern strip).

    Returns
    -------
    dict
        ``{"train": west, "tune": middle, "val": east}``, each clipped to ``extent``.
    """
    if not (0.0 < tune_fraction < 1.0 and 0.0 < val_fraction < 1.0):
        raise ValueError(f"fractions must be in (0, 1), got {tune_fraction}, {val_fraction}")
    if tune_fraction + val_fraction >= 1.0:
        raise ValueError("tune_fraction + val_fraction must be < 1")
    x0, x1 = x_range
    if not x1 > x0:
        raise ValueError(f"x_range must be increasing, got {x_range}")
    minx, miny, maxx, maxy = extent.bounds
    lo, hi = min(minx, x0), max(maxx, x1)
    val_cut = x1 - val_fraction * (x1 - x0)
    tune_cut = val_cut - tune_fraction * (x1 - x0)
    return {
        "train": extent.intersection(box(lo, miny, tune_cut, maxy)),
        "tune": extent.intersection(box(tune_cut, miny, val_cut, maxy)),
        "val": extent.intersection(box(val_cut, miny, hi, maxy)),
    }


def footprint_x_range(footprints: Mapping[str, BaseGeometry]) -> tuple[float, float]:
    """Return ``(xmin, xmax)`` over all tile footprints.

    Raises
    ------
    ValueError
        If ``footprints`` is empty.
    """
    if not footprints:
        raise ValueError("no footprints")
    b = np.array([fp.bounds for fp in footprints.values()])
    return float(b[:, 0].min()), float(b[:, 2].max())


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
