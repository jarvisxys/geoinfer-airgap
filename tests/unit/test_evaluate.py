from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from geoinfer.io.config import IOConfig  # noqa: E402
from geoinfer.postprocess.config import PostprocessConfig  # noqa: E402
from geoinfer.train.dataset import SN6Tiles  # noqa: E402
from geoinfer.train.evaluate import (  # noqa: E402
    TilePrediction,
    best_threshold,
    predict_tiles,
    score_predictions,
    threshold_sweep,
)
from tests.conftest import _norm, _records  # noqa: E402

PCFG = PostprocessConfig(
    threshold=0.5,
    opening=3,
    closing=3,
    min_area_m2=20.0,
    connectivity=4,
    simplify_tolerance_m=0.0,
    block=2048,
    geojson_crs="EPSG:4326",
)


class FirstChannel(torch.nn.Module):
    """Fake model: logits = the first input channel (so orientation can be checked)."""

    def forward(self, x):
        return x[:, :1] * 4.0


def _eval_ds(scene: dict, io_cfg: IOConfig, align: bool) -> SN6Tiles:
    return SN6Tiles(
        _records(scene),
        "sar",
        _norm(scene),
        scene["root"] / "masks",
        io_cfg,
        crop=None,
        train=False,
        hflip=False,
        align_orientation=align,
        seed=0,
    )


def test_dataset_reports_the_unpadded_size(scene: dict, io_cfg: IOConfig) -> None:
    s = _eval_ds(scene, io_cfg, False)[0]
    assert s["size"] == (64, 64)


def test_predictions_come_back_in_map_orientation(scene: dict, io_cfg: IOConfig) -> None:
    plain = list(predict_tiles(FirstChannel(), _eval_ds(scene, io_cfg, False), "cpu", amp=False))
    aligned = list(predict_tiles(FirstChannel(), _eval_ds(scene, io_cfg, True), "cpu", amp=False))
    assert [p.tile_id for p in aligned] == [p.tile_id for p in plain]
    for a, b in zip(aligned, plain, strict=True):
        assert a.prob.shape == (64, 64)
        np.testing.assert_allclose(a.prob, b.prob, atol=1e-6)  # flipped in, flipped back
        np.testing.assert_array_equal(a.mask, b.mask)
        np.testing.assert_array_equal(a.valid, b.valid)
    assert [p.orientation for p in aligned] == [0, 1]


def _pred(prob: np.ndarray, mask: np.ndarray, orientation: int = 0) -> TilePrediction:
    return TilePrediction(
        "t", "s", orientation, prob.astype(np.float32), mask.astype(bool), np.ones(mask.shape, bool)
    )


def test_threshold_sweep_finds_the_best_threshold() -> None:
    mask = np.zeros((20, 20), bool)
    mask[5:15, 5:15] = True
    prob = np.where(mask, 0.7, 0.4)  # threshold in (0.4, 0.7] is perfect
    counts = threshold_sweep([_pred(prob, mask)], [0.3, 0.5, 0.8])
    assert counts[0.5].fp == 0
    assert counts[0.5].fn == 0
    assert counts[0.3].fp == 300
    assert counts[0.8].fn == 100
    assert best_threshold(counts) == 0.5


def test_perfect_predictions_score_one(scene: dict, io_cfg: IOConfig) -> None:
    preds = [
        TilePrediction(
            p.tile_id, p.strip, p.orientation, p.mask.astype(np.float32), p.mask, p.valid
        )
        for p in predict_tiles(FirstChannel(), _eval_ds(scene, io_cfg, False), "cpu", amp=False)
    ]
    recs = {r.tile_id: r for r in _records(scene)}
    out = score_predictions(preds, recs, threshold=0.5, pcfg=PCFG)
    assert out["pixel_raw"]["iou"] == 1.0
    assert out["pixel_clean"]["iou"] == 1.0
    assert out["object"]["f1"] == 1.0
    assert out["object"]["tp"] == 2  # one 48 m2 building per tile
    assert set(out["by_orientation"]) == {"0", "1"}
    assert out["tiles"] == 2


def test_load_run_rebuilds_model_and_dataset(scene: dict, tmp_path) -> None:
    import json

    from geoinfer.models.unet import build_unet
    from geoinfer.preprocess.clouds import TileClouds
    from geoinfer.train.evaluate import load_run

    if not Path("data/processed/stats/stats.json").exists():
        pytest.skip("needs the DVC-tracked train statistics")
    root = scene["root"]
    (root / "SummaryData").mkdir()
    (root / "SummaryData" / "SAR_orientations.txt").write_text(
        "".join(f"{k} {v}\n" for k, v in scene["orient"].items())
    )
    run = tmp_path / "run"
    run.mkdir()
    cfg = {
        "modality": "fused",
        "seed": 0,
        "norm": {"optical": "global", "sar": "db"},
        "data": {
            "root": str(root),
            "masks_dir": "masks",
            "align_orientation": True,
            "crop": 64,
            "sar_cache": None,
        },
        "optim": {"amp": "auto"},
    }
    (run / "run.json").write_text(json.dumps({"run_name": "r", "config": cfg}))
    (run / "eval.json").write_text(json.dumps({"threshold": 0.45}))
    torch.save({"model": build_unet(8).state_dict(), "epoch": 3}, run / "best.pt")
    lst = tmp_path / "ids.txt"
    lst.write_text("\n".join(scene["ids"]) + "\n")
    lr = load_run(run, "cpu", interim_root=root)
    assert (lr.epoch, lr.threshold, lr.amp) == (3, 0.45, False)
    ds = lr.dataset(lst)
    assert ds[0]["image"].shape == (8, 64, 64)
    cloudy = lr.dataset(lst, optical_corrupt=TileClouds(0.5, 2.2, 0.15, [900.0] * 4, 1))[0]
    assert not np.array_equal(cloudy["image"][:4], ds[0]["image"][:4])
