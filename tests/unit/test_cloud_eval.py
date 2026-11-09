from __future__ import annotations

import numpy as np
import pytest

from geoinfer.io.config import IOConfig
from geoinfer.preprocess.clouds import TileClouds, cloud_fraction
from geoinfer.train.dataset import SN6Tiles
from tests.conftest import _norm, _records


def _clouds(coverage: float) -> TileClouds:
    return TileClouds(coverage, beta=2.2, softness=0.15, brightness=[900.0] * 4, seed=42)


def test_tile_clouds_are_deterministic_per_tile() -> None:
    x = np.full((4, 64, 64), 300, dtype=np.uint16)
    v = np.ones((64, 64), bool)
    c = _clouds(0.4)
    a, b = c(x, v, "tile_1"), c(x, v, "tile_1")
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c(x, v, "tile_2"))  # each tile its own clouds
    assert a.dtype == np.uint16


def test_tile_clouds_hit_the_coverage_and_zero_is_identity() -> None:
    x = np.full((4, 128, 128), 300, dtype=np.uint16)
    v = np.ones((128, 128), bool)
    out = _clouds(0.4)(x, v, "t")
    covered = (out[0].astype(float) - 300) / (900 - 300)  # = alpha
    assert cloud_fraction(covered) == pytest.approx(0.4, abs=0.03)
    np.testing.assert_array_equal(_clouds(0.0)(x, v, "t"), x)
    with pytest.raises(ValueError, match="coverage"):
        _clouds(1.5)


def test_dataset_clouds_touch_optical_channels_only(scene: dict, io_cfg: IOConfig) -> None:
    kw = dict(
        modality="fused",
        normalizer=_norm(scene),
        masks_dir=scene["root"] / "masks",
        io_cfg=io_cfg,
        crop=None,
        train=False,
        hflip=False,
        align_orientation=True,
        seed=0,
    )
    clean = SN6Tiles(_records(scene), **kw)[1]
    cloudy = SN6Tiles(_records(scene), optical_corrupt=_clouds(0.6), **kw)[1]
    assert not np.array_equal(cloudy["image"][:4], clean["image"][:4])
    np.testing.assert_array_equal(cloudy["image"][4:], clean["image"][4:])  # SAR unchanged
    np.testing.assert_array_equal(cloudy["mask"], clean["mask"])


def _train_ds(scene: dict, io_cfg: IOConfig, aug: dict | None, seed: int = 0) -> SN6Tiles:
    return SN6Tiles(
        _records(scene),
        "fused",
        _norm(scene),
        scene["root"] / "masks",
        io_cfg,
        crop=32,
        train=True,
        hflip=False,
        align_orientation=False,
        seed=seed,
        vary_by_epoch=True,
        cloud_aug=aug,
    )


def test_cloud_augmentation_is_reproducible_and_optical_only(scene: dict, io_cfg: IOConfig) -> None:
    aug = {"p": 1.0, "max_coverage": 0.8, "beta": 2.2, "softness": 0.15}
    plain = _train_ds(scene, io_cfg, None)[0]
    a = _train_ds(scene, io_cfg, aug)[0]
    b = _train_ds(scene, io_cfg, aug)[0]
    np.testing.assert_array_equal(a["image"], b["image"])  # same (seed, epoch, index)
    assert a["offset"] == plain["offset"]  # crops drawn as without augmentation
    np.testing.assert_array_equal(a["image"][4:], plain["image"][4:])  # SAR untouched
    np.testing.assert_array_equal(a["mask"], plain["mask"])
    assert a["cloud_coverage"] > 0
    assert plain["cloud_coverage"] == 0


def test_cloud_augmentation_probability_zero_is_off(scene: dict, io_cfg: IOConfig) -> None:
    aug = {"p": 0.0, "max_coverage": 0.8, "beta": 2.2, "softness": 0.15}
    np.testing.assert_array_equal(
        _train_ds(scene, io_cfg, aug)[0]["image"], _train_ds(scene, io_cfg, None)[0]["image"]
    )


def test_cloud_augmentation_needs_optical(scene: dict, io_cfg: IOConfig) -> None:
    with pytest.raises(ValueError, match="optical"):
        SN6Tiles(
            _records(scene),
            "sar",
            _norm(scene),
            scene["root"] / "masks",
            io_cfg,
            crop=32,
            train=True,
            hflip=False,
            align_orientation=False,
            seed=0,
            cloud_aug={"p": 0.5, "max_coverage": 0.8, "beta": 2.2, "softness": 0.15},
        )
