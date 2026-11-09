from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from geoinfer.io.config import IOConfig  # noqa: E402
from geoinfer.train.dataset import SN6Tiles  # noqa: E402
from geoinfer.train.train import (  # noqa: E402
    load_train_config,
    masked_bce_dice,
    resolve_amp,
    train,
)
from tests.conftest import _norm, _records  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def test_masked_loss_ignores_invalid_pixels() -> None:
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 8, 8)
    target = (torch.rand(2, 1, 8, 8) > 0.5).float()
    valid = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    valid[..., :4] = False
    a, _ = masked_bce_dice(logits, target, valid, bce=1.0, dice=1.0)
    other = logits.clone()
    other[..., :4] = 100.0  # garbage where invalid
    b, _ = masked_bce_dice(other, target, valid, bce=1.0, dice=1.0)
    torch.testing.assert_close(a, b)


def test_masked_loss_is_small_for_a_perfect_prediction() -> None:
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1
    logits = (target * 2 - 1) * 20.0
    valid = torch.ones_like(target, dtype=torch.bool)
    loss, parts = masked_bce_dice(logits, target, valid, bce=1.0, dice=1.0)
    assert float(loss) < 1e-3
    assert set(parts) == {"bce", "dice"}
    bad, _ = masked_bce_dice(-logits, target, valid, bce=1.0, dice=1.0)
    assert float(bad) > 1.0


def test_masked_loss_has_finite_gradients_with_no_valid_pixels() -> None:
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    loss, _ = masked_bce_dice(
        logits, torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4, dtype=torch.bool), 1.0, 1.0
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_fixed_crops_do_not_change_with_the_epoch(scene: dict, io_cfg: IOConfig) -> None:
    ds = SN6Tiles(
        _records(scene),
        "sar",
        _norm(scene),
        scene["root"] / "masks",
        io_cfg,
        crop=32,
        train=True,
        hflip=True,
        align_orientation=False,
        seed=5,
        vary_by_epoch=False,
    )
    first = ds[0]["offset"], ds[0]["hflipped"]
    for e in range(5):
        ds.set_epoch(e)
        assert (ds[0]["offset"], ds[0]["hflipped"]) == first


def test_repo_train_configs_load() -> None:
    for name in ("train_optical", "train_sar", "train_fused", "train_overfit"):
        cfg = load_train_config(REPO / "configs" / f"{name}.yaml")
        assert cfg["modality"] in {"optical", "sar", "fused"}
        assert cfg["eval"]["target"] in {"whole_tiles", "train_crops"}
    assert load_train_config(REPO / "configs/train_overfit.yaml")["eval"]["target"] == "train_crops"


def test_config_rejects_unknown_or_non_string_keys(tmp_path: Path) -> None:
    base = yaml.safe_load((REPO / "configs/train_fused.yaml").read_text())
    base["eval"]["on"] = "train_crops"  # YAML would even turn a bare `on:` into True
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(base))
    with pytest.raises(ValueError, match="eval"):
        load_train_config(p)
    raw = (REPO / "configs/train_fused.yaml").read_text().replace("  target:", "  on:")
    p.write_text(raw)
    with pytest.raises(ValueError, match="key"):
        load_train_config(p)


def test_train_loop_writes_run_json_and_checkpoints(
    scene: dict, io_cfg: IOConfig, tmp_path: Path
) -> None:
    ids = scene["ids"]
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "train.txt").write_text("\n".join(ids) + "\n")
    (lists / "val.txt").write_text(ids[0] + "\n")
    cfg = {
        "run_name": "smoke",
        "modality": "fused",
        "seed": 0,
        "data": {
            "root": str(scene["root"]),
            "train_list": str(lists / "train.txt"),
            "val_list": str(lists / "val.txt"),
            "masks_dir": "masks",
            "crop": 32,
            "hflip": True,
            "align_orientation": False,
            "vary_by_epoch": True,
            "workers": 0,
        },
        "norm": {"optical": "global", "sar": "db"},
        "model": {"pretrained": None},
        "optim": {
            "epochs": 2,
            "batch_size": 2,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "amp": False,
            "grad_clip": 1.0,
        },
        "loss": {"bce": 1.0, "dice": 1.0},
        "eval": {"threshold": 0.5, "target": "whole_tiles"},
        "out_dir": str(tmp_path / "runs"),
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run = train(
        load_train_config(cfg_path),
        device="cpu",
        io_cfg=io_cfg,
        orientations=scene["orient"],
        sar_arms=scene["arms"],
        stats_dir=scene["stats"],
    )
    out = tmp_path / "runs" / "smoke"
    info = json.loads((out / "run.json").read_text())
    assert info["run_name"] == "smoke"
    assert len(info["epochs"]) == 2
    assert {"train_loss", "val_iou", "val_loss"} <= set(info["epochs"][0])
    assert info["best_epoch"] in (0, 1)
    state = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    assert "model" in state
    assert run["best_epoch"] == info["best_epoch"]


def test_resolve_amp_modes() -> None:
    net = torch.nn.Conv2d(2, 1, 3)
    assert resolve_amp(False, net, 2, "cuda") == (False, "off in config")
    assert resolve_amp(True, net, 2, "cpu") == (False, "no CUDA device")
    assert resolve_amp("auto", net, 2, "cpu") == (False, "no CUDA device")
    with pytest.raises(ValueError, match="amp"):
        resolve_amp("sometimes", net, 2, "cpu")


@pytest.mark.gpu
def test_resolve_amp_probe_on_gpu() -> None:
    from geoinfer.models.unet import build_unet

    net = build_unet(8).cuda()
    before = {k: v.clone() for k, v in net.state_dict().items()}
    on, why = resolve_amp("auto", net, 8, "cuda", size=128)
    assert isinstance(on, bool)
    assert why.startswith("auto")
    if "GTX 16" in torch.cuda.get_device_name(0):
        assert not on
    for k, v in net.state_dict().items():  # probe ran on a copy
        torch.testing.assert_close(v, before[k])


def test_git_commit_falls_back_to_revision_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess

    from geoinfer.train import train as train_mod

    def no_git(*a: object, **k: object) -> None:
        raise subprocess.CalledProcessError(128, "git")

    monkeypatch.setattr(subprocess, "run", no_git)
    monkeypatch.chdir(tmp_path)
    assert train_mod._git_commit() == "unknown"
    (tmp_path / "REVISION").write_text("abc1234\n")
    assert train_mod._git_commit() == "abc1234"
