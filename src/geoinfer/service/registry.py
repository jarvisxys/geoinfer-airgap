"""Model registry: put ONNX models into the ``models`` table with their SHA-256.

The service never downloads weights (air-gap rule 3): models are files under the
registry folder, and ``/v1/models`` reports exactly which file (by hash) served a job.

Usage: python -m geoinfer.service.registry register ID PATH --sensor-mode fused
           --in-channels 8 --threshold 0.4 [--norm '{"optical": "global", "sar": "db_lee7"}']
       python -m geoinfer.service.registry sync     (every entry of MODELS_DIR/registry.yaml)
       python -m geoinfer.service.registry list
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from geoinfer.service.db.schema import Model


def sha256_file(path: Path) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_model(
    session: Session,
    model_id: str,
    path: Path,
    models_dir: Path,
    sensor_mode: str,
    in_channels: int,
    threshold: float,
    norm: dict[str, Any] | None = None,
) -> Model:
    """Insert or update a registry entry; ``path`` is stored relative to ``models_dir``."""
    path = Path(path).resolve()
    root = Path(models_dir).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{path} is not inside the model registry {root}")
    if not path.is_file():
        raise FileNotFoundError(path)
    m = session.get(Model, model_id) or Model(id=model_id)
    m.path = str(path.relative_to(root))
    m.sha256 = sha256_file(path)
    m.sensor_mode, m.in_channels, m.threshold, m.norm = sensor_mode, in_channels, threshold, norm
    session.add(m)
    return m


MANIFEST = "registry.yaml"


def sync_manifest(session: Session, models_dir: Path) -> list[Model]:
    """Register every model listed in ``models_dir/registry.yaml``.

    The manifest is written by ``scripts/build_service_models.py``: a ``models`` list of
    ``id``, ``path`` (relative to ``models_dir``), ``sensor_mode``, ``in_channels``,
    ``threshold``, ``norm`` and ``sha256``. A file whose hash differs from the manifest
    is refused (the bundle was altered or is incomplete).
    """
    raw = yaml.safe_load((Path(models_dir) / MANIFEST).read_text())
    out = []
    for e in raw["models"]:
        m = register_model(
            session,
            e["id"],
            Path(models_dir) / e["path"],
            models_dir,
            e["sensor_mode"],
            int(e["in_channels"]),
            float(e["threshold"]),
            e.get("norm"),
        )
        if m.sha256 != e["sha256"]:
            raise ValueError(f"{e['id']}: {e['path']} sha256 differs from {MANIFEST}")
        out.append(m)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("model_id")
    r.add_argument("path", type=Path)
    r.add_argument(
        "--models-dir", type=Path, default=Path(os.environ.get("GEOINFER_MODELS_DIR", "/models"))
    )
    r.add_argument("--sensor-mode", choices=["optical", "sar", "fused"], required=True)
    r.add_argument("--in-channels", type=int, required=True)
    r.add_argument("--threshold", type=float, required=True)
    r.add_argument("--norm", type=json.loads, default=None)
    sy = sub.add_parser("sync")
    sy.add_argument(
        "--models-dir", type=Path, default=Path(os.environ.get("GEOINFER_MODELS_DIR", "/models"))
    )
    sub.add_parser("list")
    a = p.parse_args()
    engine = create_engine(os.environ["GEOINFER_DB_URL"])
    with Session(engine) as s:
        if a.cmd == "register":
            m = register_model(
                s,
                a.model_id,
                a.path,
                a.models_dir,
                a.sensor_mode,
                a.in_channels,
                a.threshold,
                a.norm,
            )
            s.commit()
            print(f"registered {m.id}: {m.path} sha256 {m.sha256}")
        elif a.cmd == "sync":
            for m in sync_manifest(s, a.models_dir):
                print(f"registered {m.id}: {m.path} sha256 {m.sha256}")
            s.commit()
        else:
            for m in s.scalars(select(Model).order_by(Model.id)):
                print(f"{m.id}\t{m.sensor_mode}\t{m.in_channels}ch\tthr {m.threshold}\t{m.sha256}")
    engine.dispose()


if __name__ == "__main__":
    main()
