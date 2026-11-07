from __future__ import annotations

from pathlib import Path

import pytest

from geoinfer.io.config import load_io_config

REPO_CFG = Path(__file__).resolve().parents[2] / "configs" / "io.yaml"


def test_repo_config_loads() -> None:
    cfg = load_io_config(REPO_CFG)
    assert cfg.cog.blocksize == 512
    assert cfg.cog.predictor == "YES"
    assert cfg.max_window_px == 4096 * 4096
    assert Path("/mnt/d/geo/raw") in cfg.read_only_roots


def test_rejects_blocksize_not_multiple_of_16(tmp_path: Path) -> None:
    text = REPO_CFG.read_text().replace("blocksize: 512", "blocksize: 500")
    bad = tmp_path / "io.yaml"
    bad.write_text(text)
    with pytest.raises(ValueError, match="blocksize"):
        load_io_config(bad)


def test_read_only_roots_expand_user(tmp_path: Path) -> None:
    text = REPO_CFG.read_text().replace("/mnt/d/geo/raw", "~/raw")
    p = tmp_path / "io.yaml"
    p.write_text(text)
    assert load_io_config(p).read_only_roots == (Path.home() / "raw",)
