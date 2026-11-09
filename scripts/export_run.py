"""Export a trained run to ONNX and check parity with PyTorch (plan Phase 2).

Writes ``<run>/model.onnx`` (dynamic batch, height, width) and ``<run>/export.json`` with
the max absolute logit and probability difference between PyTorch fp32 and ONNX Runtime
fp32 on random inputs of ``configs/export.yaml`` ``parity.sizes``. Fails if the logit
difference exceeds ``parity.tolerance``.

Usage: uv run python scripts/export_run.py RUN_DIR [RUN_DIR ...] [--device cuda]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from geoinfer.models.export_onnx import export_onnx, onnx_parity
from geoinfer.models.unet import build_unet
from geoinfer.train.train import CHANNELS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs", type=Path, nargs="+")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    ecfg: dict[str, Any] = yaml.safe_load(Path("configs/export.yaml").read_text())
    arch = yaml.safe_load(Path("configs/models.yaml").read_text()).get("unet", {})
    failed = []
    for run_dir in a.run_dirs:
        run = json.loads((run_dir / "run.json").read_text())
        channels = CHANNELS[run["config"]["modality"]]
        net = build_unet(
            channels,
            encoder=arch.get("encoder", "resnet34"),
            decoder_channels=tuple(arch.get("decoder_channels", (256, 128, 64, 32, 16))),
        )
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"])
        t0 = time.perf_counter()
        path = export_onnx(net, channels, run_dir / "model.onnx", int(ecfg["opset"]))
        t_export = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        inputs = [
            rng.standard_normal((b, channels, h, w)).astype(np.float32)
            for b, h, w in ecfg["parity"]["sizes"]
            for _ in range(int(ecfg["parity"]["random_batches"]))
        ]
        par = onnx_parity(net, path, inputs, device=a.device)
        ok = par["max_abs_logit"] <= float(ecfg["parity"]["tolerance"])
        info = {
            "run_name": run["run_name"],
            "checkpoint_epoch": int(state["epoch"]),
            "onnx": path.name,
            "onnx_mib": round(path.stat().st_size / 2**20, 1),
            "opset": ecfg["opset"],
            "inputs": {"name": "image", "shape": ["batch", channels, "h", "w"]},
            "outputs": {"name": "logits", "shape": ["batch", 1, "h", "w"]},
            "parity": {
                **par,
                "device": a.device,
                "sizes": ecfg["parity"]["sizes"],
                "tolerance": ecfg["parity"]["tolerance"],
                "ok": ok,
            },
            "export_seconds": round(t_export, 1),
        }
        (run_dir / "export.json").write_text(json.dumps(info, indent=2) + "\n")
        print(json.dumps({k: info[k] for k in ("run_name", "onnx_mib", "parity")}))
        if not ok:
            failed.append(run["run_name"])
    if failed:
        raise SystemExit(f"parity above tolerance: {failed}")


if __name__ == "__main__":
    main()
