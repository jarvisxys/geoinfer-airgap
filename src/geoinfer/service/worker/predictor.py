"""ONNX Runtime inference for the worker (Phase 3 decision 1: no torch in the images).

Models are the Phase 2 exports: input ``image`` (batch, C, h, w) float32, output
``logits`` (batch, 1, h, w). The predictor returns building probabilities.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float32]


class Predictor(Protocol):
    """Batch of normalised tiles ``(n, C, T, T)`` in, probabilities ``(n, T, T)`` out."""

    def __call__(self, x: FloatArray) -> FloatArray: ...


def sigmoid(x: npt.NDArray[np.floating]) -> FloatArray:
    """Logistic function, overflow-safe, as float32."""
    out = np.empty(x.shape, np.float32)
    np.divide(1.0, 1.0 + np.exp(-np.clip(x, -88.0, 88.0)), out=out, casting="unsafe")
    return out


class OnnxPredictor:
    """An ONNX Runtime session on the first available of ``providers``.

    Parameters
    ----------
    path
        ``.onnx`` file (already checked against the registry hash by the caller).
    providers
        Execution providers in order of preference; unavailable ones are skipped and
        the CPU provider is the last resort.
    """

    def __init__(self, path: Path, providers: Sequence[str]) -> None:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in providers and hasattr(ort, "preload_dlls"):
            # CUDA / cuDNN come from the nvidia-* wheels, not a system toolkit: load them
            # before the session, or the CUDA provider silently falls back to the CPU
            with contextlib.suppress(Exception):  # no wheels: the provider list decides
                ort.preload_dlls()
        available = set(ort.get_available_providers())
        use = [p for p in providers if p in available] or ["CPUExecutionProvider"]
        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(path), sess_options=opts, providers=use)
        inp = self.session.get_inputs()[0]
        self.input_name: str = inp.name
        self.in_channels = int(inp.shape[1])
        self.providers: list[str] = list(self.session.get_providers())

    def __call__(self, x: FloatArray) -> FloatArray:
        logits = self.session.run(None, {self.input_name: x})[0]
        return sigmoid(np.asarray(logits)[:, 0])
