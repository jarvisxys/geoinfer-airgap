"""SpaceNet 6 tile dataset (numpy; the torch wrapper lives in ``train.torch_data``).

- :func:`index_tiles` pairs SAR-Intensity, PS-RGBNIR and label GeoJSON files by tile ID.
- :func:`ensure_mask` rasterises a tile's building footprints once into a uint8 GeoTIFF
  cache on the SAR tile's grid (CRS and transform kept).
- :class:`Normalizer` applies the S2 arms: optical ``minmax`` (per tile, baseline) or
  ``global`` (train-split 2-98 % stretch then z-score); SAR any arm of
  ``configs/preprocess.yaml``, including the per-strip arm of decision D4.
- :class:`SN6Tiles` returns normalised ``optical`` (4), ``sar`` (4) or ``fused`` (8)
  channel images with mask and valid pixels. Training samples are random crops with an
  optional east-west flip (it keeps the SAR look direction, which is north or south);
  evaluation samples are whole tiles padded to a multiple of 32. ``align_orientation``
  flips look-direction-1 strips north-south so every strip looks the same way.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import numpy.typing as npt
from rasterio.features import rasterize
from rasterio.windows import Window

from geoinfer.io.config import IOConfig
from geoinfer.io.raster import RasterInfo, RasterReader, RasterWriter, inspect_raster
from geoinfer.preprocess.clouds import apply_clouds, cloud_mask
from geoinfer.preprocess.normalize import (
    apply_sar_arm,
    minmax_per_tile,
    percentile_stretch,
    select_arm_stats,
    zscore,
)
from geoinfer.preprocess.sar import sar_chain

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float32]
MODALITIES = ("optical", "sar", "fused")
OPTICAL_ARMS = ("minmax", "global")
LABEL_DIR = "geojson_buildings"


@dataclass(frozen=True)
class TileRecord:
    """Files and metadata of one SN6 tile."""

    tile_id: str
    strip: str
    orientation: int
    sar: Path
    optical: Path
    labels: Path | None


def _by_id(folder: Path, rx: re.Pattern[str], suffix: str) -> dict[str, Path]:
    return {m.group(1): f for f in folder.glob(f"*{suffix}") if (m := rx.search(f.name))}


def index_tiles(
    root: Path,
    ids: Sequence[str],
    sar_dir: str,
    optical_dir: str,
    tile_regex: str,
    orientations: Mapping[str, int],
) -> list[TileRecord]:
    """Pair the files of ``ids`` under ``root`` (one folder per modality).

    Raises
    ------
    FileNotFoundError
        If a tile has no SAR or no optical file.
    """
    rx = re.compile(tile_regex)
    geo_rx = re.compile(tile_regex.replace(r"\.tif$", r"\.geojson$"))
    sar = _by_id(root / sar_dir, rx, ".tif")
    opt = _by_id(root / optical_dir, rx, ".tif")
    lab = _by_id(root / LABEL_DIR, geo_rx, ".geojson")
    out = []
    for tid in ids:
        for name, idx in ((sar_dir, sar), (optical_dir, opt)):
            if tid not in idx:
                raise FileNotFoundError(f"{tid}: no {name} file under {root}")
        strip = tid.split("_tile_")[0]
        out.append(
            TileRecord(tid, strip, int(orientations[strip]), sar[tid], opt[tid], lab.get(tid))
        )
    return out


def rasterize_labels(geojson: Path | None, info: RasterInfo) -> npt.NDArray[np.uint8]:
    """Building footprints burnt as 1 on the grid of ``info`` (pixel centres inside)."""
    shape = (info.height, info.width)
    if geojson is None:
        return np.zeros(shape, np.uint8)
    gdf = gpd.read_file(geojson)
    if len(gdf) == 0:
        return np.zeros(shape, np.uint8)
    if gdf.crs is not None and info.crs is not None and gdf.crs != info.crs:
        gdf = gdf.to_crs(info.crs)
    geoms = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(shape, np.uint8)
    return np.asarray(
        rasterize(geoms, out_shape=shape, transform=info.transform, fill=0, dtype="uint8")
    )


def ensure_mask(rec: TileRecord, masks_dir: Path, io_cfg: IOConfig) -> Path:
    """Path of the cached mask of ``rec``, rasterised on first use."""
    path = masks_dir / f"{rec.tile_id}.tif"
    if path.exists():
        return path
    info = inspect_raster(rec.sar)
    m = rasterize_labels(rec.labels, info)
    with RasterWriter(path, info, io_cfg, count=1, dtype="uint8", cog=False) as wr:
        wr.write(m[None], Window(0, 0, info.width, info.height))
    return path


class Normalizer:
    """S2 normalisation arms with train-split (and per-strip) statistics."""

    def __init__(
        self,
        stats: Mapping[str, Any],
        strip_stats: Mapping[str, Any] | None,
        optical_arm: str,
        sar_arm: str,
        sar_arms: Mapping[str, Mapping[str, Any]],
    ) -> None:
        if optical_arm not in OPTICAL_ARMS:
            raise ValueError(f"optical arm must be one of {OPTICAL_ARMS}, got {optical_arm!r}")
        if sar_arm not in sar_arms:
            raise ValueError(f"unknown SAR arm {sar_arm!r}")
        self.optical_arm, self.sar_arm = optical_arm, sar_arm
        self.opt_stats = stats["optical"]
        self.enl = float(stats["sar"]["enl"]["value"])
        self.arm = dict(sar_arms[sar_arm])
        if self.arm.get("scope") == "strip":
            if strip_stats is None:
                raise ValueError(f"arm {sar_arm!r} needs per-strip statistics")
            self.global_sar = stats["sar"]["arms"][self.arm["fallback"]]
            self.strip_sar: Mapping[str, Any] = strip_stats["arms"][sar_arm]["strips"]
        else:
            self.global_sar = stats["sar"]["arms"][sar_arm]
            self.strip_sar = {}

    @classmethod
    def from_files(
        cls,
        stats_json: Path,
        strip_stats_json: Path | None,
        optical_arm: str,
        sar_arm: str,
        sar_arms: Mapping[str, Mapping[str, Any]],
    ) -> Normalizer:
        """Load ``stats.json`` (and ``strip_stats.json`` if it exists)."""
        stats = json.loads(Path(stats_json).read_text())
        strip = None
        if strip_stats_json is not None and Path(strip_stats_json).exists():
            strip = json.loads(Path(strip_stats_json).read_text())
        return cls(stats, strip, optical_arm, sar_arm, sar_arms)

    def optical(self, x: npt.NDArray[Any], valid: BoolArray) -> FloatArray:
        """Normalise ``(4, h, w)`` uint16 optical."""
        if self.optical_arm == "minmax":
            return minmax_per_tile(x, valid)
        s = self.opt_stats
        stretched = percentile_stretch(x, valid, s["p_lo"], s["p_hi"])
        return zscore(stretched, valid, s["mean"], s["std"])

    @property
    def lee(self) -> int | None:
        """Lee window of the SAR arm (None: no speckle filter)."""
        lee = self.arm.get("lee")
        return None if lee is None else int(lee)

    def sar_chain(self, db: npt.NDArray[Any], valid: BoolArray) -> FloatArray:
        """The arm's speckle filter: dB in, (Lee-filtered) dB out."""
        return sar_chain(db, valid, self.lee, self.enl)

    def sar_from_chain(
        self, chain: npt.NDArray[Any], valid: BoolArray, strip: str | None
    ) -> FloatArray:
        """Clip and z-score filtered dB with the arm's global or per-strip statistics."""
        stats, _ = select_arm_stats(strip, self.global_sar, self.strip_sar)
        return apply_sar_arm(chain, valid, stats, self.arm["domain"], bool(self.arm["clip"]))

    def sar(self, db: npt.NDArray[Any], valid: BoolArray, strip: str | None) -> FloatArray:
        """Normalise ``(4, h, w)`` SAR-Intensity dB with the configured arm."""
        return self.sar_from_chain(self.sar_chain(db, valid), valid, strip)


def resolve_sar_cache(root: Path, value: str | None) -> Path | None:
    """``data.sar_cache`` from a training config: None, absolute, or relative to root."""
    if value is None:
        return None
    return Path(value) if Path(value).is_absolute() else Path(root) / value


def chain_cache_path(cache_dir: Path, tile_id: str, lee: int, enl: float) -> Path:
    """Cache file of a tile's Lee-filtered dB (the key holds the window and the ENL)."""
    return Path(cache_dir) / f"{tile_id}.lee{lee}.enl{enl:.3f}.npy"


def ensure_sar_chain(
    tile_id: str,
    db: npt.NDArray[Any],
    valid: BoolArray,
    lee: int,
    enl: float,
    cache_dir: Path,
) -> FloatArray:
    """Lee-filtered dB of a tile from the cache, computed and stored on first use.

    Files are float32 ``.npy`` (identical to filtering on the fly), written to a temporary
    name and renamed, so parallel loader workers never read a partial file.
    """
    path = chain_cache_path(cache_dir, tile_id, lee, enl)
    if path.exists():
        return np.asarray(np.load(path), dtype=np.float32)
    out = np.asarray(sar_chain(db, valid, lee, enl), dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, out)
    os.replace(tmp, path)
    return out


def _read(path: Path) -> npt.NDArray[Any]:
    with RasterReader(path) as r:
        return r.read(Window(0, 0, r.info.width, r.info.height)).data


class SN6Tiles:
    """Indexable SN6 samples (see module docstring).

    Parameters
    ----------
    records
        Tiles from :func:`index_tiles`.
    modality
        ``optical``, ``sar`` or ``fused`` (optical channels first).
    normalizer
        S2 arms to apply.
    masks_dir, io_cfg
        Mask cache location and I/O config.
    crop
        Training crop side (multiple of 32); ``None`` for whole tiles.
    train
        Random crops and flips when True; whole padded tiles when False.
    hflip
        Random east-west flip in training.
    align_orientation
        Flip look-direction-1 strips north-south (image, mask and valid).
    seed
        Base seed; draws depend on ``(seed, epoch, index)`` only.
    vary_by_epoch
        False keeps every tile's crop and flip fixed across epochs (overfit checks).
    sar_cache
        Folder for cached Lee-filtered SAR (arms with a Lee filter only); None filters
        on the fly. The filter always uses the SAR valid mask, so SAR and fused models
        share one cache.
    optical_corrupt
        Optional ``f(optical, valid, tile_id) -> optical`` applied to raw optical before
        normalisation (synthetic clouds for the S3 robustness curve).
    cloud_aug
        Training-time cloud augmentation ``{p, max_coverage, beta, softness}``: with
        probability ``p`` a sample's raw optical gets clouds of uniform random coverage
        in ``[0, max_coverage)`` (train-split p98 white point). Drawn from its own random
        stream, so crops and flips are the same as without it.
    """

    def __init__(
        self,
        records: Sequence[TileRecord],
        modality: str,
        normalizer: Normalizer,
        masks_dir: Path,
        io_cfg: IOConfig,
        crop: int | None,
        train: bool,
        hflip: bool,
        align_orientation: bool,
        seed: int,
        vary_by_epoch: bool = True,
        sar_cache: Path | None = None,
        optical_corrupt: Callable[[npt.NDArray[Any], BoolArray, str], npt.NDArray[Any]]
        | None = None,
        cloud_aug: Mapping[str, float] | None = None,
    ) -> None:
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        if train and (crop is None or crop % 32):
            raise ValueError(f"training needs a crop that is a multiple of 32, got {crop}")
        self.records = list(records)
        self.modality, self.norm = modality, normalizer
        self.masks_dir, self.io_cfg = masks_dir, io_cfg
        self.crop, self.train, self.hflip = crop, train, hflip
        self.align_orientation, self.seed = align_orientation, seed
        self.vary_by_epoch = vary_by_epoch
        self.sar_cache = sar_cache
        self.optical_corrupt = optical_corrupt
        if cloud_aug is not None and modality == "sar":
            raise ValueError("cloud augmentation needs optical input (optical or fused)")
        self.cloud_aug = dict(cloud_aug) if cloud_aug is not None else None
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        """Change the random draws (crops, flips) for a new epoch."""
        self.epoch = epoch

    def _clouds(
        self, opt: npt.NDArray[Any], valid: BoolArray, i: int
    ) -> tuple[npt.NDArray[Any], float]:
        """Training cloud augmentation from its own random stream (see class docstring)."""
        c = self.cloud_aug
        assert c is not None
        rng = np.random.default_rng([self.seed, self.epoch if self.vary_by_epoch else 0, i, 7])
        if rng.random() >= float(c["p"]):
            return opt, 0.0
        coverage = float(rng.uniform(0.0, float(c["max_coverage"])))
        alpha = cloud_mask(
            opt.shape[1:],
            coverage,
            float(c["beta"]),
            float(c["softness"]),
            int(rng.integers(0, 2**31)),
        )
        return apply_clouds(opt, valid, alpha, self.norm.opt_stats["p_hi"]), coverage

    def __getitem__(self, i: int) -> dict[str, Any]:
        rec = self.records[i]
        parts, valid = [], None
        coverage = 0.0
        if self.modality in ("optical", "fused"):
            opt = _read(rec.optical)
            ov = np.any(opt != 0, axis=0)
            if self.optical_corrupt is not None:
                opt = self.optical_corrupt(opt, ov, rec.tile_id)
            if self.train and self.cloud_aug is not None:
                opt, coverage = self._clouds(opt, ov, i)
            valid = ov
        if self.modality in ("sar", "fused"):
            sar = _read(rec.sar)
            sv = np.all(sar != 0, axis=0)
            valid = sv if valid is None else valid & sv
        assert valid is not None
        if self.modality in ("optical", "fused"):
            parts.append(self.norm.optical(opt, valid))
        if self.modality in ("sar", "fused"):
            lee = self.norm.lee
            if self.sar_cache is not None and lee is not None:
                chain = ensure_sar_chain(rec.tile_id, sar, sv, lee, self.norm.enl, self.sar_cache)
            else:
                chain = self.norm.sar_chain(sar, sv)
            parts.append(self.norm.sar_from_chain(chain, valid, rec.strip))
        image = np.concatenate(parts, axis=0)
        mask = _read(ensure_mask(rec, self.masks_dir, self.io_cfg))[0]

        flipped = self.align_orientation and rec.orientation == 1
        if flipped:
            image, mask, valid = image[:, ::-1], mask[::-1], valid[::-1]
        offset, hflipped = (0, 0), False
        size = tuple(mask.shape)
        if self.train:
            assert self.crop is not None
            rng = np.random.default_rng([self.seed, self.epoch if self.vary_by_epoch else 0, i])
            h, w = mask.shape
            r0 = int(rng.integers(0, h - self.crop + 1))
            c0 = int(rng.integers(0, w - self.crop + 1))
            sl = (slice(r0, r0 + self.crop), slice(c0, c0 + self.crop))
            image, mask, valid = image[:, sl[0], sl[1]], mask[sl], valid[sl]
            offset = (r0, c0)
            size = (self.crop, self.crop)
            hflipped = self.hflip and bool(rng.random() < 0.5)
            if hflipped:
                image, mask, valid = image[:, :, ::-1], mask[:, ::-1], valid[:, ::-1]
        else:
            h, w = mask.shape
            size = (h, w)
            ph, pw = (-h) % 32, (-w) % 32
            image = np.pad(image, ((0, 0), (0, ph), (0, pw)))
            mask = np.pad(mask, ((0, ph), (0, pw)))
            valid = np.pad(valid, ((0, ph), (0, pw)))
        return {
            "image": np.ascontiguousarray(image, dtype=np.float32),
            "mask": np.ascontiguousarray(mask, dtype=np.uint8),
            "valid": np.ascontiguousarray(valid, dtype=bool),
            "tile_id": rec.tile_id,
            "strip": rec.strip,
            "offset": offset,
            "size": size,
            "hflipped": hflipped,
            "flipped_ns": flipped,
            "cloud_coverage": coverage,
        }
