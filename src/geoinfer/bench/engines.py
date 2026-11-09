"""Inference engines of study S4 / experiment E5, one call signature for all.

An engine maps a float32 batch ``(n, C, h, w)`` (host memory) to building probabilities
``(n, h, w)`` (host memory). Engines:

- ``torch_fp32`` / ``torch_amp``: the trained PyTorch U-Net (``best.pt``), eager, AMP fp16
  autocast for ``torch_amp``;
- ``ort_fp32`` / ``ort_fp16``: ONNX Runtime CUDA on the export / its fp16 conversion
  (weights and activations fp16, float32 input and output);
- ``trt_fp16``: ONNX Runtime's TensorRT provider with fp16 enabled, engines cached on disk
  per shape profile (the first call per GPU builds them; build time is reported apart).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import yaml

from geoinfer.service.worker.predictor import sigmoid

FloatArray = npt.NDArray[np.float32]
ENGINES = ("torch_fp32", "torch_amp", "ort_fp32", "ort_fp16", "trt_fp16")


class Engine(Protocol):
    name: str

    def __call__(self, x: FloatArray) -> FloatArray: ...


def fp16_model(src: Path, dst: Path) -> Path:
    """fp16 copy of an ONNX model with float32 inputs and outputs (cached at ``dst``)."""
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst
    import onnx
    from onnxruntime.transformers import float16

    m = float16.convert_float_to_float16(onnx.load(src), keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, dst)
    return dst


def preload_tensorrt() -> list[str]:
    """Load the TensorRT libraries of the ``tensorrt-cu13`` wheel (RTLD_GLOBAL).

    ONNX Runtime's TensorRT provider resolves ``libnvinfer.so.10`` and the ONNX parser at
    session creation; ``onnxruntime.preload_dlls`` covers CUDA and cuDNN but not TensorRT.
    """
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("tensorrt-cu13 is not installed (uv sync --extra trt)")
    root = Path(next(iter(spec.submodule_search_locations)))
    loaded = []
    for name in ("libnvinfer.so.10", "libnvinfer_plugin.so.10", "libnvonnxparser.so.10"):
        f = root / name
        if f.exists():
            ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)
            loaded.append(name)
    if "libnvinfer.so.10" not in loaded:
        raise RuntimeError(f"no libnvinfer.so.10 in {root}")
    return loaded


class _Ort:
    def __init__(self, name: str, path: Path, providers: Sequence[Any]) -> None:
        import onnxruntime as ort

        ort.preload_dlls()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.name = name
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
        self.providers = self.session.get_providers()
        want = providers[0] if isinstance(providers[0], str) else providers[0][0]
        if self.providers[0] != want:
            raise RuntimeError(f"{name}: {want} unavailable, session runs on {self.providers}")
        self.input = self.session.get_inputs()[0].name

    def __call__(self, x: FloatArray) -> FloatArray:
        return sigmoid(np.asarray(self.session.run(None, {self.input: x})[0])[:, 0])


class _Torch:
    def __init__(self, name: str, run_dir: Path, amp: bool) -> None:
        import torch

        from geoinfer.models.unet import build_unet
        from geoinfer.train.train import CHANNELS

        cfg = json.loads((run_dir / "run.json").read_text())["config"]
        arch = yaml.safe_load(Path("configs/models.yaml").read_text()).get("unet", {})
        net = build_unet(
            CHANNELS[cfg["modality"]],
            encoder=arch.get("encoder", "resnet34"),
            decoder_channels=tuple(arch.get("decoder_channels", (256, 128, 64, 32, 16))),
        )
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state["model"])
        self.net = net.cuda().eval()
        self.torch, self.amp, self.name = torch, amp, name

    def __call__(self, x: FloatArray) -> FloatArray:
        torch = self.torch
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=self.amp):
            logits = self.net(torch.from_numpy(x).cuda(non_blocking=True))
        return np.asarray(torch.sigmoid(logits.float())[:, 0].cpu().numpy(), np.float32)


def make_engine(
    name: str,
    *,
    onnx_path: Path | None = None,
    run_dir: Path | None = None,
    cache_dir: Path,
    trt_shapes: tuple[str, str, str] | None = None,
) -> Engine:
    """Build an engine by name (see module docstring).

    Parameters
    ----------
    onnx_path, run_dir
        The ONNX export (ORT / TRT engines) and the training run (PyTorch engines).
    cache_dir
        fp16 models and TensorRT engine caches.
    trt_shapes
        TensorRT profile ``(min, opt, max)`` as ``"image:NxCxHxW"`` strings.
    """
    if name not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {name!r}")
    if name.startswith("torch"):
        if run_dir is None:
            raise ValueError(f"{name} needs run_dir")
        return _Torch(name, run_dir, amp=name == "torch_amp")
    if onnx_path is None:
        raise ValueError(f"{name} needs onnx_path")
    if name == "ort_fp32":
        return _Ort(name, onnx_path, ["CUDAExecutionProvider"])
    if name == "ort_fp16":
        return _Ort(
            name,
            fp16_model(onnx_path, cache_dir / "fp16" / onnx_path.name),
            ["CUDAExecutionProvider"],
        )
    opts: dict[str, Any] = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": str(cache_dir / "trt"),
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": str(cache_dir / "trt"),
    }
    if trt_shapes is not None:
        lo, opt, hi = trt_shapes
        opts |= {
            "trt_profile_min_shapes": lo,
            "trt_profile_opt_shapes": opt,
            "trt_profile_max_shapes": hi,
        }
    (cache_dir / "trt").mkdir(parents=True, exist_ok=True)
    import onnxruntime as ort

    ort.preload_dlls()  # CUDA / cuDNN first: TensorRT links against them
    preload_tensorrt()
    return _Ort(name, onnx_path, [("TensorrtExecutionProvider", opts), "CUDAExecutionProvider"])
