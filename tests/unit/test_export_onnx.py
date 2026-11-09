from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")
ort = pytest.importorskip("onnxruntime")

from geoinfer.models.export_onnx import export_onnx, onnx_parity  # noqa: E402

# the TorchScript-based exporter is deprecated but used on purpose (see module docstring)
pytestmark = [
    pytest.mark.filterwarnings("ignore:You are using the legacy TorchScript-based ONNX export"),
    pytest.mark.filterwarnings("ignore:The feature will be removed"),
]
from geoinfer.models.unet import build_unet  # noqa: E402


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory: pytest.TempPathFactory) -> tuple[object, Path]:
    torch.manual_seed(0)
    net = build_unet(8).eval()
    path = tmp_path_factory.mktemp("onnx") / "m.onnx"
    export_onnx(net, 8, path, opset=18)
    return net, path


def test_export_writes_one_file_with_named_io(onnx_model: tuple[object, Path]) -> None:
    _, path = onnx_model
    assert [p.name for p in path.parent.iterdir()] == ["m.onnx"]  # no temporary left
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["image"]
    assert [o.name for o in s.get_outputs()] == ["logits"]
    shape = s.get_inputs()[0].shape
    assert shape[1] == 8
    assert all(isinstance(d, str) for d in (shape[0], shape[2], shape[3]))  # dynamic axes


@pytest.mark.parametrize(("b", "h", "w"), [(1, 64, 64), (3, 96, 160)])
def test_dynamic_batch_and_size(onnx_model: tuple[object, Path], b: int, h: int, w: int) -> None:
    _, path = onnx_model
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    y = s.run(None, {"image": np.zeros((b, 8, h, w), np.float32)})[0]
    assert y.shape == (b, 1, h, w)


def test_parity_with_pytorch(onnx_model: tuple[object, Path]) -> None:
    net, path = onnx_model
    x = np.random.default_rng(1).standard_normal((2, 8, 128, 96)).astype(np.float32)
    out = onnx_parity(net, path, [x], device="cpu")
    assert out["max_abs_logit"] < 1e-3
    assert out["max_abs_prob"] < 1e-4
    assert out["inputs"] == 1


def test_export_rejects_wrong_channels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="channels"):
        export_onnx(build_unet(4).eval(), 8, tmp_path / "x.onnx", opset=18)
