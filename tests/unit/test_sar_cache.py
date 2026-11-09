from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import geoinfer.train.dataset as ds_mod
from geoinfer.io.config import IOConfig
from geoinfer.train.dataset import SN6Tiles, chain_cache_path, ensure_sar_chain
from tests.conftest import _norm, _records

LEE_ARMS = {
    "db": {"lee": None, "domain": "db", "clip": True},
    "db_lee5": {"lee": 5, "domain": "db", "clip": True},
}


def _norm_lee(scene: dict):
    import json

    from geoinfer.train.dataset import Normalizer

    stats = json.loads((scene["stats"] / "stats.json").read_text())
    stats["sar"]["arms"]["db_lee5"] = dict(stats["sar"]["arms"]["db"], lee=5)
    return Normalizer(stats, None, "global", "db_lee5", LEE_ARMS)


def _ds(scene: dict, io_cfg: IOConfig, norm, cache: Path | None, modality: str = "sar"):
    return SN6Tiles(
        _records(scene),
        modality,
        norm,
        scene["root"] / "masks",
        io_cfg,
        crop=None,
        train=False,
        hflip=False,
        align_orientation=True,
        seed=0,
        sar_cache=cache,
    )


@pytest.mark.parametrize("modality", ["sar", "fused"])
def test_cached_samples_equal_uncached(
    scene: dict, io_cfg: IOConfig, tmp_path: Path, modality: str
) -> None:
    norm = _norm_lee(scene)
    plain = _ds(scene, io_cfg, norm, None, modality)
    cached = _ds(scene, io_cfg, norm, tmp_path / "cache", modality)
    for i in range(len(plain)):
        np.testing.assert_array_equal(cached[i]["image"], plain[i]["image"])  # cache miss
        np.testing.assert_array_equal(cached[i]["image"], plain[i]["image"])  # cache hit
    assert len(list((tmp_path / "cache").glob("*.npy"))) == 2


def test_cache_hit_does_not_refilter(
    scene: dict, io_cfg: IOConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    norm = _norm_lee(scene)
    ds = _ds(scene, io_cfg, norm, tmp_path / "c")
    first = ds[0]["image"]

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("Lee filter ran again")

    monkeypatch.setattr(ds_mod, "sar_chain", boom)
    np.testing.assert_array_equal(ds[0]["image"], first)


def test_arms_without_lee_do_not_use_the_cache(
    scene: dict, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ds = _ds(scene, io_cfg, _norm(scene), tmp_path / "c")
    ds[0]
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())


def test_cache_key_includes_lee_size_and_enl(tmp_path: Path) -> None:
    a = chain_cache_path(tmp_path, "t_tile_1", 5, 2.124)
    assert a != chain_cache_path(tmp_path, "t_tile_1", 7, 2.124)
    assert a != chain_cache_path(tmp_path, "t_tile_1", 5, 2.5)
    assert a.parent == tmp_path


def test_ensure_sar_chain_writes_atomically(scene: dict, tmp_path: Path) -> None:
    rec = _records(scene)[0]
    cache = tmp_path / "cache"
    db = np.full((4, 16, 16), 30.0, np.float32)
    valid = np.ones((16, 16), bool)
    out = ensure_sar_chain(rec.tile_id, db, valid, 5, 2.0, cache)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 30.0, rtol=1e-6)  # Lee keeps a constant image
    expected = chain_cache_path(cache, rec.tile_id, 5, 2.0).name
    assert [q.name for q in cache.iterdir()] == [expected]  # no temporary file left


def test_resolve_sar_cache(tmp_path: Path) -> None:
    from geoinfer.train.dataset import resolve_sar_cache

    assert resolve_sar_cache(tmp_path, None) is None
    assert resolve_sar_cache(tmp_path, "c") == tmp_path / "c"
    assert resolve_sar_cache(tmp_path, "/abs/c") == Path("/abs/c")
