from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from geoinfer.models.unet import adapt_stem, build_unet, load_encoder_weights  # noqa: E402


def _fake_resnet34_state(tmp_path: Path) -> Path:
    """A torchvision-format resnet34 state dict with random values (no download)."""
    from torchvision.models import resnet34

    sd = resnet34(weights=None).state_dict()
    path = tmp_path / "resnet34.pth"
    torch.save(sd, path)
    return path


@pytest.mark.parametrize("channels", [1, 3, 4, 8])
def test_adapt_stem_follows_the_plan_rule(channels: int) -> None:
    w3 = torch.randn(64, 3, 7, 7)
    w = adapt_stem(w3, channels)
    assert w.shape == (64, channels, 7, 7)
    if channels == 3:
        assert torch.equal(w, w3)
    else:
        for i in range(channels):
            torch.testing.assert_close(w[:, i], w3[:, i % 3] * 3 / channels)


def test_adapt_stem_keeps_the_flat_response_for_multiples_of_three() -> None:
    # an all-ones input gives the same stem response as the RGB stem when C = 3k
    w3 = torch.randn(64, 3, 7, 7)
    torch.testing.assert_close(adapt_stem(w3, 6).sum(dim=1), w3.sum(dim=1))


@pytest.mark.parametrize("channels", [4, 8])
def test_unet_output_shape(channels: int) -> None:
    net = build_unet(channels, encoder="resnet34", decoder_channels=(256, 128, 64, 32, 16))
    net.eval()
    with torch.no_grad():
        y = net(torch.zeros(2, channels, 64, 96))
    assert y.shape == (2, 1, 64, 96)


def test_encoder_weights_load_from_a_local_file(tmp_path: Path) -> None:
    path = _fake_resnet34_state(tmp_path)
    ref = torch.load(path, weights_only=True)
    net = build_unet(8, encoder="resnet34", decoder_channels=(256, 128, 64, 32, 16))
    load_encoder_weights(net, path)
    enc = net.encoder.state_dict()
    torch.testing.assert_close(enc["layer4.2.conv2.weight"], ref["layer4.2.conv2.weight"])
    torch.testing.assert_close(enc["conv1.weight"], adapt_stem(ref["conv1.weight"], 8))
    assert "fc.weight" not in enc


def test_missing_weight_file_is_an_error(tmp_path: Path) -> None:
    net = build_unet(4, encoder="resnet34", decoder_channels=(256, 128, 64, 32, 16))
    with pytest.raises(FileNotFoundError, match="registry"):
        load_encoder_weights(net, tmp_path / "nope.pth")


def test_build_never_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.request

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("network access")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    monkeypatch.setattr(torch.hub, "load_state_dict_from_url", boom)
    net = build_unet(4, encoder="resnet34", decoder_channels=(256, 128, 64, 32, 16))
    assert sum(p.numel() for p in net.parameters()) > 20_000_000
    assert np.isfinite(sum(float(p.detach().sum()) for p in net.parameters()))
