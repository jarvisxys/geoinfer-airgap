"""Training loop for the S3 U-Nets (plan step 6): masked BCE + Dice, fp16 on CUDA.

``python -m geoinfer.train.train --config configs/train_fused.yaml [--set k=v ...]``
(``make train CFG=...``). Each run writes ``<out_dir>/<run_name>/`` with ``run.json``
(config, environment, per-epoch losses and validation IoU, best epoch, timings, peak
GPU memory), ``best.pt`` (highest validation IoU) and ``last.pt``. Checkpoints go to DVC,
not git.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from geoinfer.eval.metrics import PixelCounts, pixel_scores
from geoinfer.io.config import IOConfig, load_io_config
from geoinfer.models.unet import build_unet, load_encoder_weights
from geoinfer.train.dataset import Normalizer, SN6Tiles, index_tiles

SAR_DIR, OPTICAL_DIR = "SAR-Intensity", "PS-RGBNIR"
TILE_REGEX = r"_(\d{14}_\d{14}_tile_\d+)\.tif$"
CHANNELS = {"optical": 4, "sar": 4, "fused": 8}
EVAL_TARGETS = ("whole_tiles", "train_crops")
SCHEMA: dict[str, set[str] | None] = {
    "run_name": None,
    "modality": None,
    "seed": None,
    "out_dir": None,
    "data": {
        "root",
        "train_list",
        "val_list",
        "masks_dir",
        "crop",
        "hflip",
        "align_orientation",
        "vary_by_epoch",
        "workers",
    },
    "norm": {"optical", "sar"},
    "model": {"pretrained"},
    "optim": {"epochs", "batch_size", "lr", "weight_decay", "amp", "grad_clip"},
    "loss": {"bce", "dice"},
    "eval": {"threshold", "target"},
}


def _check_keys(node: Any, where: str) -> None:
    """Every mapping key must be a string (YAML 1.1 reads a bare `on:` as True)."""
    if isinstance(node, dict):
        for k, val in node.items():
            if not isinstance(k, str):
                raise ValueError(f"non-string key {k!r} under '{where}' (quote it in YAML)")
            _check_keys(val, f"{where}.{k}" if where else k)


def masked_bce_dice(
    logits: Any, target: Any, valid: Any, bce: float, dice: float, eps: float = 1.0
) -> tuple[Any, dict[str, float]]:
    """BCE-with-logits plus soft Dice, both over valid pixels only.

    Parameters
    ----------
    logits, target, valid
        ``(B, 1, H, W)``; ``target`` in {0, 1}, ``valid`` boolean.
    bce, dice
        Weights of the two terms.
    eps
        Dice smoothing; with no valid pixels both terms are 0 (finite gradients).
    """
    v = valid.float()
    t = target.float()
    per_px = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.float(), t, reduction="none"
    )
    l_bce = (per_px * v).sum() / v.sum().clamp(min=1.0)
    p = torch.sigmoid(logits.float()) * v
    inter = (p * t * v).sum()
    l_dice = 1.0 - (2.0 * inter + eps) / (p.sum() + (t * v).sum() + eps)
    parts = {"bce": float(l_bce.detach()), "dice": float(l_dice.detach())}
    return bce * l_bce + dice * l_dice, parts


FP16_UNRELIABLE = ("GTX 16",)  # TU116/TU117: erratic non-finite fp16 convolutions


def resolve_amp(
    setting: bool | str, net: Any, channels: int, device: str, size: int = 256
) -> tuple[bool, str]:
    """Decide fp16 autocast: ``True``, ``False`` or ``"auto"``.

    ``auto`` uses fp32 on GPUs whose fp16 convolutions are known to be unreliable (the
    laptop's GTX 1650 gives non-finite outputs depending on input size and mode, so a
    single probe cannot clear it); elsewhere it runs a copy of the model in training
    mode on one random ``size`` batch under autocast and keeps fp16 only if the output
    is finite. The copy keeps BatchNorm statistics of ``net`` untouched.

    Returns
    -------
    tuple
        ``(enabled, reason)``; the reason is recorded in ``run.json``.
    """
    if setting not in (True, False, "auto"):
        raise ValueError(f"amp must be true, false or auto, got {setting!r}")
    if setting is False:
        return False, "off in config"
    if not device.startswith("cuda"):
        return False, "no CUDA device"
    if setting is True:
        return True, "on in config"
    name = torch.cuda.get_device_name(0)
    if any(tag in name for tag in FP16_UNRELIABLE):
        return False, f"auto: fp16 unreliable on {name}, using fp32"
    probe = copy.deepcopy(net).train()
    x = torch.randn(2, channels, size, size, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        ok = bool(torch.isfinite(probe(x).float()).all())
    del probe
    if ok:
        return True, f"auto: fp16 probe finite at {size} px on {name}"
    return False, f"auto: fp16 probe not finite on {name}, using fp32"


def load_train_config(path: Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Read a training YAML and apply ``key.sub=value`` overrides (values parsed as YAML)."""
    cfg: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        node[parts[-1]] = yaml.safe_load(raw)
    _check_keys(cfg, "")
    for k in ("run_name", "modality", "data", "norm", "model", "optim", "loss", "eval"):
        if k not in cfg:
            raise ValueError(f"{path}: missing '{k}'")
    for k, val in cfg.items():
        if k not in SCHEMA:
            raise ValueError(f"{path}: unknown key '{k}'")
        allowed = SCHEMA[k]
        if allowed is not None:
            extra = set(val) - allowed
            if extra:
                raise ValueError(f"{path}: unknown keys under '{k}': {sorted(extra)}")
    if cfg["modality"] not in CHANNELS:
        raise ValueError(f"modality must be one of {list(CHANNELS)}")
    if cfg["eval"].get("target", "whole_tiles") not in EVAL_TARGETS:
        raise ValueError(f"eval.target must be one of {EVAL_TARGETS}")
    return cfg


def _git_commit() -> str:
    """Short HEAD hash, with ``-dirty`` when tracked files have uncommitted changes."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{head}-dirty" if status else head
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _batch(b: Mapping[str, Any], device: str) -> tuple[Any, Any, Any]:
    x = b["image"].to(device, non_blocking=True)
    y = b["mask"].to(device, non_blocking=True).unsqueeze(1).float()
    v = b["valid"].to(device, non_blocking=True).unsqueeze(1)
    return x, y, v


def evaluate(
    net: Any, loader: Any, device: str, threshold: float, loss_cfg: Mapping[str, float]
) -> tuple[float, PixelCounts]:
    """Mean loss and pixel counts (prediction ``sigmoid >= threshold``) over a loader."""
    net.eval()
    counts, total, n = PixelCounts(), 0.0, 0
    with torch.no_grad():
        for b in loader:
            x, y, v = _batch(b, device)
            logits = net(x)
            loss, _ = masked_bce_dice(logits, y, v, loss_cfg["bce"], loss_cfg["dice"])
            total += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
            pred = (torch.sigmoid(logits) >= threshold).cpu().numpy()[:, 0]
            counts += PixelCounts.from_masks(
                pred, y.cpu().numpy()[:, 0] > 0.5, v.cpu().numpy()[:, 0]
            )
    return total / max(n, 1), counts


def train(
    cfg: Mapping[str, Any],
    device: str,
    io_cfg: IOConfig,
    orientations: Mapping[str, int],
    sar_arms: Mapping[str, Mapping[str, Any]],
    stats_dir: Path,
    registry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train one model; returns the content written to ``run.json``."""
    t_start = time.perf_counter()
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    d, o, lc = cfg["data"], cfg["optim"], cfg["loss"]
    root = Path(d["root"]).expanduser()
    masks = root / d["masks_dir"]
    norm = Normalizer.from_files(
        Path(stats_dir) / "stats.json",
        Path(stats_dir) / "strip_stats.json",
        cfg["norm"]["optical"],
        cfg["norm"]["sar"],
        sar_arms,
    )

    def records(list_path: str) -> Any:
        ids = Path(list_path).read_text().split()
        return index_tiles(root, ids, SAR_DIR, OPTICAL_DIR, TILE_REGEX, orientations)

    common = {
        "modality": cfg["modality"],
        "normalizer": norm,
        "masks_dir": masks,
        "io_cfg": io_cfg,
        "align_orientation": bool(d["align_orientation"]),
        "seed": seed,
    }
    train_ds = SN6Tiles(
        records(d["train_list"]),
        crop=int(d["crop"]),
        train=True,
        hflip=bool(d["hflip"]),
        vary_by_epoch=bool(d["vary_by_epoch"]),
        **common,
    )
    if cfg["eval"].get("target", "whole_tiles") == "train_crops":
        val_ds = SN6Tiles(
            records(d["val_list"]),
            crop=int(d["crop"]),
            train=True,
            hflip=False,
            vary_by_epoch=False,
            **common,
        )
        val_bs = int(o["batch_size"])
    else:
        val_ds = SN6Tiles(records(d["val_list"]), crop=None, train=False, hflip=False, **common)
        val_bs = 1
    cuda = device.startswith("cuda")
    gen = torch.Generator().manual_seed(seed)
    workers = int(d.get("workers", 0))
    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(o["batch_size"]),
        shuffle=True,
        generator=gen,
        num_workers=workers,
        pin_memory=cuda,
        drop_last=len(train_ds) > int(o["batch_size"]),
    )
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=val_bs, shuffle=False, num_workers=workers, pin_memory=cuda
    )

    arch = (registry or {}).get("unet", {})
    net = build_unet(
        CHANNELS[cfg["modality"]],
        encoder=arch.get("encoder", "resnet34"),
        decoder_channels=tuple(arch.get("decoder_channels", (256, 128, 64, 32, 16))),
        classes=int(arch.get("classes", 1)),
    )
    weights = cfg["model"].get("pretrained")
    if weights:
        if registry is None:
            raise ValueError("pretrained weights need the model registry config")
        entry = registry["pretrained"][weights]
        load_encoder_weights(net, Path(registry["registry"]) / entry["file"])
    net.to(device)
    opt = torch.optim.AdamW(
        net.parameters(), lr=float(o["lr"]), weight_decay=float(o["weight_decay"])
    )
    epochs = int(o["epochs"])
    steps = max(epochs * len(train_dl), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps))
    )
    amp, amp_reason = resolve_amp(
        o["amp"], net, CHANNELS[cfg["modality"]], device, size=int(d["crop"])
    )
    print(f"amp: {amp} ({amp_reason})", flush=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    out = Path(cfg["out_dir"]) / cfg["run_name"]
    out.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, Any]] = []
    best_iou, best_epoch = -1.0, 0
    for epoch in range(epochs):
        train_ds.set_epoch(epoch)
        net.train()
        t0, tot, n = time.perf_counter(), 0.0, 0
        for b in train_dl:
            x, y, v = _batch(b, device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                logits = net(x)
            loss, _ = masked_bce_dice(logits, y, v, float(lc["bce"]), float(lc["dice"]))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(o["grad_clip"]))
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += float(loss.detach()) * x.shape[0]
            n += x.shape[0]
        val_loss, counts = evaluate(net, val_dl, device, float(cfg["eval"]["threshold"]), lc)
        iou = pixel_scores(counts)["iou"]
        rec = {
            "epoch": epoch,
            "train_loss": round(tot / max(n, 1), 5),
            "val_loss": round(val_loss, 5),
            "val_iou": None if math.isnan(iou) else round(iou, 4),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)
        state = {"model": net.state_dict(), "epoch": epoch}
        torch.save(state, out / "last.pt")
        score = -1.0 if math.isnan(iou) else iou
        if score > best_iou or epoch == 0:
            best_iou, best_epoch = max(score, best_iou), epoch
            torch.save(state, out / "best.pt")

    info: dict[str, Any] = {
        "run_name": cfg["run_name"],
        "config": copy.deepcopy(dict(cfg)),
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if cuda else device,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "params": sum(p.numel() for p in net.parameters()),
        "epochs": history,
        "best_epoch": best_epoch,
        "best_val_iou": round(best_iou, 4),
        "seconds": round(time.perf_counter() - t_start, 1),
        "amp": {"enabled": amp, "reason": amp_reason},
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20) if cuda else None,
    }
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    cfg = load_train_config(a.config, a.set)
    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    pcfg = yaml.safe_load(Path("configs/preprocess.yaml").read_text())
    mcfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    interim = Path(dcfg["interim_root"]).expanduser()
    lines = (interim / dcfg["orientations"]).read_text().splitlines()
    orient = {k: int(v) for k, v in (ln.split() for ln in lines if ln.strip())}
    info = train(
        cfg,
        a.device,
        load_io_config(),
        orient,
        pcfg["sar"]["arms"],
        Path(pcfg["stats"]["out_dir"]),
        mcfg,
    )
    print(
        f"best epoch {info['best_epoch']}, val IoU {info['best_val_iou']}, "
        f"{info['seconds']} s, peak GPU {info['peak_gpu_mib']} MiB"
    )


if __name__ == "__main__":
    main()
