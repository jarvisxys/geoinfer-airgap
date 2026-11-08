from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from geoinfer.train.splits import (
    assign_splits,
    footprint_x_range,
    select_subset,
    split_regions,
    tile_footprints,
    write_splits,
)
from tests.conftest import CRS

TILE = 450.0  # 900 px at 0.5 m


def _tile(x: float, y: float) -> box:
    return box(x, y, x + TILE, y + TILE)


def test_split_regions_cuts_west_to_east_over_covered_range() -> None:
    extent = box(-100, 0, 1500, 500)  # official Train extent, wider than the tiles
    regions = split_regions(extent, (0.0, 1000.0), tune_fraction=0.15, val_fraction=0.20)
    assert regions["train"].bounds == (-100, 0, 650, 500)
    assert regions["tune"].bounds == (650, 0, 800, 500)
    assert regions["val"].bounds == (800, 0, 1500, 500)
    total = sum(r.area for r in regions.values())
    assert total == pytest.approx(extent.area)
    assert regions["train"].intersection(regions["val"]).area == 0


def test_split_regions_stays_inside_extent() -> None:
    extent = box(0, 0, 1000, 500).union(box(0, 500, 300, 800))  # L-shaped extent
    regions = split_regions(extent, (0.0, 1000.0), tune_fraction=0.1, val_fraction=0.1)
    for reg in regions.values():
        assert extent.covers(reg)


@pytest.mark.parametrize(
    ("tune", "val"), [(0.0, 0.2), (0.15, 0.0), (0.5, 0.5), (-0.1, 0.2), (0.15, 1.2)]
)
def test_split_regions_rejects_bad_fractions(tune: float, val: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        split_regions(box(0, 0, 1, 1), (0.0, 1.0), tune_fraction=tune, val_fraction=val)


def test_split_regions_rejects_empty_range() -> None:
    with pytest.raises(ValueError, match="x_range"):
        split_regions(box(0, 0, 1, 1), (1.0, 1.0), tune_fraction=0.1, val_fraction=0.1)


def test_footprint_x_range_spans_all_tiles() -> None:
    fps = {"a": _tile(100, 0), "b": _tile(-50, 900), "c": _tile(2000, 10)}
    assert footprint_x_range(fps) == (-50.0, 2000.0 + TILE)
    with pytest.raises(ValueError, match="no footprints"):
        footprint_x_range({})


def test_assign_keeps_only_fully_inside_tiles() -> None:
    regions = {"train": box(0, 0, 2000, 1000), "val": box(2000, 0, 4000, 1000)}
    fps = {
        "a": _tile(100, 100),  # inside train
        "b": _tile(2500, 100),  # inside val
        "c": _tile(1800, 100),  # straddles the boundary
        "d": _tile(5000, 100),  # outside every region
    }
    assigned, dropped = assign_splits(fps, regions, tolerance_m=0.0)
    assert assigned == {"train": ["a"], "val": ["b"]}
    assert dropped == ["c", "d"]


def test_assign_tolerance_accepts_edge_aligned_tiles() -> None:
    regions = {"train": box(0, 0, 450, 450)}
    fps = {"edge": box(-0.2, 0, 450.1, 450)}  # float noise at the extent edge
    assert assign_splits(fps, regions, tolerance_m=0.0)[0]["train"] == []
    assert assign_splits(fps, regions, tolerance_m=0.5)[0]["train"] == ["edge"]


def test_assign_never_puts_a_tile_in_two_splits() -> None:
    # overlapping regions (Train and Val extents overlap by ~52 m in SN6)
    regions = {"train": box(0, 0, 1000, 1000), "val": box(0, 0, 1000, 1000)}
    assigned, dropped = assign_splits({"a": _tile(10, 10)}, regions, tolerance_m=0.0)
    assert assigned == {"train": [], "val": []}
    assert dropped == ["a"]


def test_tile_footprints_reads_bounds_and_ids(tmp_path: Path) -> None:
    pattern = r"_(\d{14}_\d{14}_tile_\d+)\.tif$"
    paths = []
    for i, x0 in enumerate([594_000.0, 594_450.0]):
        p = (
            tmp_path
            / f"SN6_Train_AOI_11_Rotterdam_SAR-Intensity_20190804120223_20190804120456_tile_{i}.tif"
        )
        with rasterio.open(
            p,
            "w",
            driver="GTiff",
            width=900,
            height=900,
            count=1,
            dtype="uint8",
            crs=CRS,
            transform=from_origin(x0, 5_757_000.0, 0.5, 0.5),
        ) as dst:
            dst.write(np.zeros((1, 900, 900), dtype=np.uint8))
        paths.append(p)
    fps = tile_footprints(paths, pattern, expected_crs=CRS)
    assert sorted(fps) == [
        "20190804120223_20190804120456_tile_0",
        "20190804120223_20190804120456_tile_1",
    ]
    assert fps["20190804120223_20190804120456_tile_1"].bounds == (
        594_450.0,
        5_756_550.0,
        594_900.0,
        5_757_000.0,
    )
    with pytest.raises(ValueError, match="CRS"):
        tile_footprints(paths, pattern, expected_crs="EPSG:4326")


def test_tile_footprints_rejects_unmatched_name(tmp_path: Path) -> None:
    p = tmp_path / "bad.tif"
    p.touch()
    with pytest.raises(ValueError, match="tile ID"):
        tile_footprints([p], r"_(\d+)\.tif$", expected_crs=CRS)


def test_write_splits_is_sorted_and_hashed(tmp_path: Path) -> None:
    assigned = {"train": ["b", "a"], "val": ["c"]}
    meta = write_splits(assigned, ["x"], tmp_path, {"rule": "inside"})
    assert (tmp_path / "train.txt").read_text() == "a\nb\n"
    assert (tmp_path / "val.txt").read_text() == "c\n"
    saved = json.loads((tmp_path / "splits.json").read_text())
    assert saved == meta
    assert saved["counts"] == {"train": 2, "val": 1, "dropped": 1}
    assert saved["rule"] == "inside"
    assert len(saved["sha256"]["train.txt"]) == 64


def test_select_subset_takes_compact_cluster_per_split() -> None:
    fps = {f"t{i}": _tile(i * 100.0, 0) for i in range(20)}
    fps |= {f"v{i}": _tile(10_000 + i * 100.0, 0) for i in range(10)}
    assigned = {"train": [f"t{i}" for i in range(20)], "val": [f"v{i}" for i in range(10)]}
    chosen = select_subset(fps, assigned, {"train": 5, "val": 3})
    assert len(chosen["train"]) == 5
    assert len(chosen["val"]) == 3
    xs = sorted(fps[t].centroid.x for t in chosen["train"])
    assert xs[-1] - xs[0] == pytest.approx(400.0)  # 5 neighbouring tiles, no gaps
    assert set(chosen["val"]) <= set(assigned["val"])


def test_select_subset_caps_at_available_and_is_deterministic() -> None:
    fps = {f"t{i}": _tile(i * 100.0, 0) for i in range(4)}
    assigned = {"train": list(fps)}
    a = select_subset(fps, assigned, {"train": 10})
    assert sorted(a["train"]) == sorted(fps)
    assert a == select_subset(fps, assigned, {"train": 10})
