"""install_offline.sh on a tiny fake bundle with a stub `docker` on PATH."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "install_offline.sh"

STUB = """#!/usr/bin/env bash
echo "$@" >> "$DOCKER_LOG"
case "$1 $2" in
  "compose version") echo "Docker Compose version v5" ;;
  "image inspect") echo "sha256:$(echo -n "$5" | sha256sum | cut -c1-12)" ;;
esac
exit 0
"""


def _fake_bundle(root: Path, tamper: bool = False, wrong_id: bool = False) -> Path:
    b = root / "geoinfer-9.9.9"
    for d in ("images", "models", "demo", "scripts"):
        (b / d).mkdir(parents=True)
    (b / "images" / "images.tar.gz").write_bytes(b"not really images")
    tags = ["geoinfer-api:9.9.9", "geoinfer-worker:9.9.9"]
    ids = [hashlib.sha256(t.encode()).hexdigest()[:12] for t in tags]
    if wrong_id:
        ids[1] = "0" * 12
    (b / "images" / "images.txt").write_text(
        "".join(f"{t} sha256:{i}\n" for t, i in zip(tags, ids, strict=True))
    )
    (b / "models" / "registry.yaml").write_text("models: []\n")
    (b / "demo" / "demo_optical.tif").write_bytes(b"tif")
    (b / "demo" / "demo_scene.json").write_text('{"orientation": 1}')
    (b / "compose.yaml").write_text("services: {}\n")
    (b / "compose.airgap.yaml").write_text("services: {}\n")
    (b / "VERSION").write_text("9.9.9\n")
    files = sorted(p for p in b.rglob("*") if p.is_file())
    lines = [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  ./{p.relative_to(b)}\n" for p in files]
    (b / "MANIFEST.sha256").write_text("".join(lines))
    if tamper:
        (b / "models" / "registry.yaml").write_text("models: [evil]\n")
    tar = root / "geoinfer-9.9.9.tar"
    with tarfile.open(tar, "w") as t:
        t.add(b, arcname=b.name)
    shutil.rmtree(b)
    return tar


@pytest.fixture
def stub_env(tmp_path: Path) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(STUB)
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_LOG": str(tmp_path / "docker.log"),
    }


def _run(
    tar: Path, target: Path, env: dict[str, str], *args: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), str(tar), str(target), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_install_verifies_loads_and_writes_env(tmp_path: Path, stub_env: dict[str, str]) -> None:
    tar = _fake_bundle(tmp_path)
    target = tmp_path / "install"
    r = _run(tar, target, stub_env)
    assert r.returncode == 0, r.stdout + r.stderr
    env = dict(
        line.split("=", 1) for line in (target / ".env").read_text().split("\n") if "=" in line
    )
    assert len(env["GEOINFER_DB_PASSWORD"]) >= 32
    assert env["GEOINFER_TAG"] == "9.9.9"
    assert env["GEOINFER_MODELS"] == str(target / "models")
    assert env["GEOINFER_DATA"] == str(target / "data")
    assert (target / "data" / "inbox" / "demo_optical.tif").exists()
    assert (target / "data" / "demo_scene.json").exists()
    assert (target / "data" / "outputs").is_dir()
    log = Path(stub_env["DOCKER_LOG"]).read_text()
    assert "load -i images/images.tar.gz" in log
    assert "compose" not in log.replace("compose version", "")  # no --up: nothing started
    # a second run keeps the existing password
    pw = env["GEOINFER_DB_PASSWORD"]
    assert _run(tar, target, stub_env).returncode == 0
    assert f"GEOINFER_DB_PASSWORD={pw}" in (target / ".env").read_text()


def test_install_refuses_a_tampered_bundle(tmp_path: Path, stub_env: dict[str, str]) -> None:
    r = _run(_fake_bundle(tmp_path, tamper=True), tmp_path / "i", stub_env)
    assert r.returncode != 0
    assert "checksum" in (r.stdout + r.stderr).lower()
    assert "load" not in Path(stub_env["DOCKER_LOG"]).read_text()


def test_install_refuses_unexpected_image_ids(tmp_path: Path, stub_env: dict[str, str]) -> None:
    r = _run(_fake_bundle(tmp_path, wrong_id=True), tmp_path / "i", stub_env)
    assert r.returncode != 0
    assert "geoinfer-worker:9.9.9" in r.stdout + r.stderr


def test_install_up_starts_the_airgapped_stack(tmp_path: Path, stub_env: dict[str, str]) -> None:
    r = _run(_fake_bundle(tmp_path), tmp_path / "i", stub_env, "--up", "--airgap")
    assert r.returncode == 0, r.stdout + r.stderr
    log = Path(stub_env["DOCKER_LOG"]).read_text()
    assert "compose -f compose.yaml -f compose.airgap.yaml up -d --wait" in log
