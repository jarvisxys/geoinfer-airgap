"""ONNX export of a trained U-Net and a PyTorch / ONNX Runtime parity check.

The graph takes ``image`` ``(batch, C, h, w)`` float32 and returns ``logits``
``(batch, 1, h, w)``; batch, height and width are dynamic (h and w must be multiples of
32, as for the network itself). Export uses torch's TorchScript-based exporter
(``dynamo=False``): it handles this plain convolutional network without the extra
``onnxscript`` dependency of the newer exporter. Files are written to a temporary name
and renamed, so a model file is complete or absent.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def export_onnx(net: Any, channels: int, path: Path, opset: int) -> Path:
    """Export ``net`` (eval mode, on CPU) to ``path``."""
    in_ch = int(net.encoder.conv1.weight.shape[1])
    if in_ch != channels:
        raise ValueError(f"model has {in_ch} input channels, expected {channels}")
    net = net.eval().to("cpu")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    x = torch.zeros(1, channels, 256, 256)
    axes = {0: "batch", 2: "h", 3: "w"}
    try:
        with torch.no_grad():
            torch.onnx.export(
                net,
                x,
                str(tmp),
                dynamo=False,
                opset_version=opset,
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": axes, "logits": axes},
            )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def onnx_parity(
    net: Any, path: Path, inputs: Sequence[npt.NDArray[np.float32]], device: str = "cpu"
) -> dict[str, float | int]:
    """Max absolute difference between PyTorch and ONNX Runtime (both fp32).

    Returns
    -------
    dict
        ``max_abs_logit``, ``max_abs_prob`` over all inputs, and the number of inputs.
    """
    import onnxruntime as ort

    providers = ["CPUExecutionProvider"]
    if device.startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    net = net.eval().to(device)
    d_logit = d_prob = 0.0
    for x in inputs:
        with torch.no_grad():
            ref = net(torch.from_numpy(x).to(device)).float().cpu().numpy()
        got = sess.run(None, {"image": x.astype(np.float32)})[0]
        d_logit = max(d_logit, float(np.abs(got - ref).max()))
        d_prob = max(d_prob, float(np.abs(1 / (1 + np.exp(-got)) - 1 / (1 + np.exp(-ref))).max()))
    return {"max_abs_logit": d_logit, "max_abs_prob": d_prob, "inputs": len(inputs)}
