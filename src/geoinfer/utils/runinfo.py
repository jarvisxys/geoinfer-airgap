"""Environment fingerprint for reproducibility (written into every run.json).

``python -m geoinfer.utils.runinfo`` prints it as JSON; this is also the Gate 0
check that PyTorch sees the GPU.
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import shutil
import subprocess
from typing import Any

PACKAGES = (
    "numpy",
    "rasterio",
    "shapely",
    "opencv-python-headless",
    "torch",
    "torchvision",
    "segmentation-models-pytorch",
    "onnxruntime-gpu",
)


def _run(cmd: list[str]) -> str | None:
    """Run a local command and return stripped stdout, or None if it fails."""
    if shutil.which(cmd[0]) is None:
        return None
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=15, check=True)
    except (subprocess.SubprocessError, OSError):
        return None
    return res.stdout.strip()


def package_versions(names: tuple[str, ...] = PACKAGES) -> dict[str, str | None]:
    """Installed version of each distribution, None if not installed."""
    out: dict[str, str | None] = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = None
    return out


def torch_info() -> dict[str, Any]:
    """CUDA visibility as seen by PyTorch; empty dict if torch is not installed."""
    try:
        import torch
    except ImportError:
        return {}
    cudnn = torch.backends.cudnn
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": cudnn.version() if cudnn.is_available() else None,
    }
    if info["cuda_available"]:
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["arch_list"] = torch.cuda.get_arch_list()
    return info


def collect() -> dict[str, Any]:
    """Collect git SHA, platform, GPU, driver and key package versions."""
    gpu = _run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]
    )
    return {
        "git_sha": _run(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_run(["git", "status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": gpu,
        "gdal": _run(["gdalinfo", "--version"]),
        "packages": package_versions(),
        "torch": torch_info(),
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
