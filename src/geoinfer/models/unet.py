"""U-Net with a ResNet encoder for building segmentation (plan step 6).

Built with ``segmentation_models_pytorch`` but never with its downloader: encoder
weights come only from a local file in the model registry (air-gapped rule). The
pretrained RGB stem is adapted to ``C`` input channels with the plan's rule, the same
one smp uses: channel ``i`` gets the RGB weight ``i mod 3`` scaled by ``3 / C``
(``3 / 8`` for the 8-channel fused input), so a flat input keeps its response when
``C`` is a multiple of 3 and activations stay at the pretrained scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import segmentation_models_pytorch as smp
import torch


def adapt_stem(w3: Any, channels: int) -> Any:
    """``(out, 3, k, k)`` RGB stem weights -> ``(out, channels, k, k)``."""
    if w3.ndim != 4 or w3.shape[1] != 3:
        raise ValueError(f"expected (out, 3, k, k) stem weights, got {tuple(w3.shape)}")
    if channels == 3:
        return w3.clone()
    idx = torch.arange(channels) % 3
    return w3[:, idx] * (3.0 / channels)


def build_unet(
    in_channels: int,
    encoder: str = "resnet34",
    decoder_channels: Sequence[int] = (256, 128, 64, 32, 16),
    classes: int = 1,
) -> Any:
    """Randomly initialised U-Net; load pretrained weights with :func:`load_encoder_weights`."""
    return smp.Unet(
        encoder_name=encoder,
        encoder_weights=None,  # never download; see load_encoder_weights
        in_channels=in_channels,
        classes=classes,
        decoder_channels=list(decoder_channels),
    )


def load_encoder_weights(net: Any, path: Path) -> None:
    """Load a torchvision-format ResNet state dict into ``net.encoder``.

    The classifier head is dropped and the stem adapted to the model's input channels.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist (fill the registry with ``make weights``).
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not in the model registry; run `make weights`")
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = {k: v for k, v in sd.items() if not k.startswith("fc.")}
    in_ch = int(net.encoder.conv1.weight.shape[1])
    sd["conv1.weight"] = adapt_stem(sd["conv1.weight"], in_ch)
    # smp's encoder overrides load_state_dict without a return value; use the base method,
    # strict, so any missing or extra key is an error
    torch.nn.Module.load_state_dict(net.encoder, sd, strict=True)
