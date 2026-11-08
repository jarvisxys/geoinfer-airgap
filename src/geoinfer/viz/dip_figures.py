"""Figures for notebook 02 (S2 normalisation, speckle filtering, co-registration, tiling).

Each function takes arrays (no file access) and returns ``(figure, metrics)`` so the
notebook prints the numbers that the figure shows.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from geoinfer.eval.seams import seam_lines, seam_score
from geoinfer.preprocess.coreg import edge_magnitude
from geoinfer.preprocess.normalize import minmax_per_tile, percentile_stretch
from geoinfer.preprocess.sar import db_to_linear, edge_preservation_index, enl, sar_chain
from geoinfer.preprocess.stats import BandHistogram

BoolArray = npt.NDArray[np.bool_]
Metrics = dict[str, Any]


def _bin_values(h: BandHistogram) -> npt.NDArray[np.float64]:
    off = 0.0 if h.discrete else 0.5
    return np.asarray(h.lo + h.step * (np.arange(h.n_bins) + off), np.float64)


def _rgb(x: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
    return np.clip(np.moveaxis(np.asarray(x[:3], np.float64), 0, -1), 0.0, 1.0)


def _off(ax: Any) -> None:
    ax.set_xticks([])
    ax.set_yticks([])


def optical_seam_figure(
    a: npt.NDArray[Any],
    b: npt.NDArray[Any],
    offset: tuple[int, int],
    stats: Mapping[str, Sequence[float]],
) -> tuple[Figure, Metrics]:
    """Two overlapping optical tiles pasted together, per-tile min-max vs global stretch.

    Parameters
    ----------
    a, b
        ``(bands, h, w)`` uint16 tiles on one grid; ``b`` starts at ``offset`` (rows,
        cols) relative to ``a`` and is pasted on top.
    stats
        Train-split optical ``p_lo`` / ``p_hi`` per band.
    """
    (h, w), (dr, dc) = a.shape[1:], offset
    hh, ww = max(h, dr + b.shape[1]), max(w, dc + b.shape[2])
    out: Metrics = {}
    fig, ax = plt.subplots(1, 2, figsize=(11, 6), dpi=100)
    names = ("per-tile min-max (baseline)", "global 2-98 % stretch (train split)")

    def norm(name: str, t: npt.NDArray[Any], v: BoolArray) -> npt.NDArray[np.float32]:
        if name == names[0]:
            return minmax_per_tile(t, v)
        return percentile_stretch(t, v, stats["p_lo"], stats["p_hi"])

    for k, name in enumerate(names):
        canvas = np.zeros((a.shape[0], hh, ww), np.float32)
        for t, (r0, c0) in ((a, (0, 0)), (b, (dr, dc))):
            v = np.any(t != 0, axis=0)
            canvas[:, r0 : r0 + t.shape[1], c0 : c0 + t.shape[2]] = norm(name, t, v)
        lum = canvas[:3].mean(axis=0)
        # seams = the edges of the pasted tile b inside the canvas
        rows = np.array([v for v in (dr, dr + b.shape[1]) if 0 < v < hh])
        cols = np.array([v for v in (dc, dc + b.shape[2]) if 0 < v < ww])
        s = seam_score(lum, rows, cols, valid=lum > 0)
        out[name] = round(s, 2)
        any_v = np.any(canvas != 0, axis=0)
        rr, cc = np.nonzero(any_v.any(axis=1))[0], np.nonzero(any_v.any(axis=0))[0]
        ax[k].imshow(_rgb(canvas))
        if rr.size:
            ax[k].set_ylim(rr.max() + 0.5, rr.min() - 0.5)
            ax[k].set_xlim(cc.min() - 0.5, cc.max() + 0.5)
        ax[k].add_patch(
            Rectangle(
                (dc - 0.5, dr - 0.5), b.shape[2], b.shape[1], fill=False, ec="cyan", lw=0.6, ls=":"
            )
        )
        ax[k].set_title(f"{name}\nseam score {s:.2f}", fontsize=10)
        _off(ax[k])
    fig.suptitle(
        "S2 optical: two adjacent tiles of one strip, normalised separately (dotted: tile 2 edges)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, {"seam_score": out}


def histogram_figure(
    opt_hist: BandHistogram,
    opt_stats: Mapping[str, Sequence[float]],
    sar_hists: Mapping[str, BandHistogram],
    sar_arms: Mapping[str, Mapping[str, Any]],
    band_names: Sequence[str] = ("R", "G", "B", "NIR"),
    sar_names: Sequence[str] = ("HH", "HV", "VH", "VV"),
) -> tuple[Figure, Metrics]:
    """Histograms before and after normalisation (optical and the SAR arms).

    ``sar_hists`` maps an arm name to the histogram of its dB values (after Lee for the
    Lee arms); ``sar_arms`` holds each arm's ``p_lo``, ``p_hi``, ``mean``, ``std`` and
    ``domain`` / ``clip`` from stats.json.
    """
    fig, ax = plt.subplots(2, 3, figsize=(15, 8), dpi=100)
    v = _bin_values(opt_hist)
    for b, name in enumerate(band_names):
        ax[0, 0].plot(v, opt_hist.counts[b], lw=0.8, label=name)
        lo, hi = opt_stats["p_lo"][b], opt_stats["p_hi"][b]
        y = np.clip((v - lo) / (hi - lo), 0, 1)
        y = (y - opt_stats["mean"][b]) / opt_stats["std"][b]
        ax[0, 1].hist(y, bins=120, weights=opt_hist.counts[b], histtype="step", label=name)
    ax[0, 0].set_xlim(0, 2 * max(opt_stats["p_hi"]))
    ax[0, 0].set_title("optical raw DN (train split)")
    ax[0, 1].set_title("optical: 2-98 % stretch + z-score")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].set_yscale("log")
    ax[0, 2].axis("off")
    lines = ["optical 2-98 % (DN):"] + [
        f"  {n}: {opt_stats['p_lo'][b]:.0f} - {opt_stats['p_hi'][b]:.0f}"
        for b, n in enumerate(band_names)
    ]

    metrics: Metrics = {}
    first = next(iter(sar_hists.values()))
    vdb = _bin_values(first)
    for b, name in enumerate(sar_names):
        ax[1, 0].plot(np.power(10.0, vdb / 10.0), first.counts[b], lw=0.8, label=name)
    ax[1, 0].set_xscale("log")
    ax[1, 0].set_title("SAR linear intensity (baseline arm)")
    ax[1, 0].legend(fontsize=8)
    for name, h in sar_hists.items():
        arm = sar_arms[name]
        vb = _bin_values(h)
        b = 0
        lo, hi = arm["p_lo"][b], arm["p_hi"][b]
        x = np.clip(vb, lo, hi) if arm.get("clip", True) else vb
        z = (x - arm["mean"][b]) / arm["std"][b]
        ax[1, 1].plot(vb, h.counts[b] / h.counts[b].sum(), lw=0.8, label=name)
        ax[1, 2].hist(
            z, bins=150, weights=h.counts[b] / h.counts[b].sum(), histtype="step", label=name
        )
        metrics[name] = {
            "mean_db": round(float(arm["mean"][b]), 2),
            "std_db": round(float(arm["std"][b]), 2),
        }
    lines += ["", f"SAR arms ({sar_names[0]}), clipped mean / std dB:"] + [
        f"  {k}: {v['mean_db']:.2f} / {v['std_db']:.2f}" for k, v in metrics.items()
    ]
    ax[0, 2].text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9)
    ax[1, 1].set_title(f"SAR dB per arm ({sar_names[0]})")
    ax[1, 1].set_xlim(0, 70)
    ax[1, 1].legend(fontsize=8)
    ax[1, 2].set_title("SAR after clip 1-99 % + z-score")
    ax[1, 2].legend(fontsize=8)
    fig.suptitle("S2 histograms before / after normalisation (train-split statistics)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, metrics


def homogeneous_window(lin: npt.NDArray[Any], valid: BoolArray, size: int) -> tuple[int, int]:
    """Top-left of the fully valid ``size`` block with the lowest coefficient of variation."""
    best, pos = np.inf, (0, 0)
    h, w = valid.shape
    for r in range(0, h - size + 1, size // 2):
        for c in range(0, w - size + 1, size // 2):
            if not valid[r : r + size, c : c + size].all():
                continue
            blk = lin[r : r + size, c : c + size]
            cv = float(blk.std() / blk.mean()) if blk.mean() > 0 else np.inf
            if cv < best:
                best, pos = cv, (r, c)
    if not np.isfinite(best):
        raise ValueError("no fully valid window")
    return pos


def sar_patch_figure(
    db: npt.NDArray[Any], valid: BoolArray, enl_value: float, band: int = 0, window: int = 32
) -> tuple[Figure, Metrics]:
    """One SAR patch as linear intensity, dB, Lee 5 and Lee 7, with ENL and EPI."""
    chains = {"dB": sar_chain(db, valid, None, enl_value)}
    for k in (5, 7):
        chains[f"Lee {k}"] = sar_chain(db, valid, k, enl_value)
    lin = db_to_linear(chains["dB"], valid)[band]
    r, c = homogeneous_window(lin, valid, window)
    hom = np.zeros(valid.shape, bool)
    hom[r : r + window, c : c + window] = True
    metrics: Metrics = {"homogeneous_window": [r, c, window]}
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.8), dpi=100)
    v = valid
    lo, hi = np.percentile(chains["dB"][band][v], [2, 98])
    lin_hi = np.percentile(lin[v], 98)
    ax[0].imshow(np.where(v, lin, 0), cmap="gray", vmin=0, vmax=lin_hi)
    ax[0].set_title(f"linear intensity\nENL {enl(lin, hom):.2f}", fontsize=10)
    metrics["linear"] = {"enl": round(enl(lin, hom), 2)}
    for k, (name, x) in enumerate(chains.items(), 1):
        img = x[band]
        e = enl(db_to_linear(x, valid)[band], hom)
        epi = edge_preservation_index(chains["dB"][band], img, valid)
        metrics[name] = {"enl": round(e, 2), "epi": round(epi, 3)}
        ax[k].imshow(np.where(v, img, lo), cmap="gray", vmin=lo, vmax=hi)
        ax[k].set_title(f"{name}\nENL {e:.2f}, EPI {epi:.3f}", fontsize=10)
    for x in ax:
        x.add_patch(Rectangle((c - 0.5, r - 0.5), window, window, fill=False, ec="yellow", lw=1))
        _off(x)
    fig.suptitle(
        "S2 SAR speckle: ENL on the most homogeneous 32 px window (yellow; biased "
        "high by selection), EPI vs unfiltered dB"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig, metrics


def strip_calibration_figure(
    strip_means: Mapping[str, Sequence[float]],
    global_mean: Sequence[float],
    samples: Mapping[str, npt.NDArray[Any]],
    strip_stats: Mapping[str, Mapping[str, Sequence[float]]],
    global_stats: Mapping[str, Sequence[float]],
    band: int = 0,
) -> tuple[Figure, Metrics]:
    """Why decision D4: per-strip mean dB, and sample strips under global vs strip z-score.

    ``samples`` maps a strip ID to its (Lee-filtered) dB values of ``band``.
    """
    ids = sorted(strip_means, key=lambda s: strip_means[s][band])
    m = np.array([strip_means[s] for s in ids])
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), dpi=100)
    for b in range(m.shape[1]):
        ax[0].plot(m[:, b], lw=0.9, label=f"band {b + 1}")
        ax[0].axhline(global_mean[b], lw=0.6, ls="--", color=f"C{b}")
    ax[0].set_xlabel("strips, sorted by band-1 mean")
    ax[0].set_ylabel("mean dB (Lee 7, clipped)")
    ax[0].set_title(f"{len(ids)} strips: mean spread {np.ptp(m[:, band]):.1f} dB (band 1)")
    ax[0].legend(fontsize=8)
    for name, x in samples.items():
        g = (
            np.clip(x, global_stats["p_lo"][band], global_stats["p_hi"][band])
            - global_stats["mean"][band]
        ) / global_stats["std"][band]
        st = strip_stats[name]
        s = (np.clip(x, st["p_lo"][band], st["p_hi"][band]) - st["mean"][band]) / st["std"][band]
        ax[1].hist(g, bins=120, histtype="step", density=True, label=name[:15])
        ax[2].hist(s, bins=120, histtype="step", density=True, label=name[:15])
    ax[1].set_title("sample strips, global z-score (db_lee7)")
    ax[2].set_title("same strips, per-strip z-score (db_lee7_strip)")
    for x in ax[1:]:
        x.set_xlim(-4, 4)
        x.legend(fontsize=7)
    fig.suptitle("Decision D4: SAR calibration differs between acquisition strips")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig, {"strips": len(ids), "mean_spread_db": round(float(np.ptp(m[:, band])), 2)}


def coreg_overlay_figure(
    sar_db: npt.NDArray[Any],
    optical: npt.NDArray[Any],
    valid: BoolArray,
    ksize: int,
    blur_sigma: float,
    crop: int = 300,
) -> tuple[Figure, Metrics]:
    """Edge maps of SAR (green) and optical (magenta) over each other on one tile."""
    sar = sar_db.astype(np.float64).mean(axis=0)
    opt = optical[:3].astype(np.float64).mean(axis=0)
    if valid.any():  # as in coregistration_shift: no blurred step at the nodata edge
        sar = np.where(valid, sar, sar[valid].mean())
        opt = np.where(valid, opt, opt[valid].mean())
    if blur_sigma > 0:
        sar = np.asarray(cv2.GaussianBlur(sar, (0, 0), blur_sigma), np.float64)
        opt = np.asarray(cv2.GaussianBlur(opt, (0, 0), blur_sigma), np.float64)
    k = 2 * (ksize // 2 + int(np.ceil(3 * blur_sigma))) + 1
    keep = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    es, eo = edge_magnitude(sar, keep, ksize), edge_magnitude(opt, keep, ksize)
    valid = keep

    def norm(e: npt.NDArray[Any]) -> npt.NDArray[np.float64]:
        p = np.percentile(e[valid], 99) if valid.any() else 1.0
        return np.clip(e / max(float(p), 1e-9), 0, 1)

    h, w = valid.shape
    r0, c0 = max((h - crop) // 2, 0), max((w - crop) // 2, 0)
    sl = (slice(r0, r0 + crop), slice(c0, c0 + crop))
    ns, no = norm(es)[sl], norm(eo)[sl]
    rgb = np.stack([no, ns, no], axis=-1)
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.8), dpi=100)
    ax[0].imshow(no, cmap="magma")
    ax[0].set_title("optical edges (Sobel, smoothed)")
    ax[1].imshow(ns, cmap="viridis")
    ax[1].set_title("SAR edges (Sobel, smoothed)")
    ax[2].imshow(rgb)
    ax[2].set_title("overlay: optical magenta, SAR green, both white")
    for x in ax:
        _off(x)
    fig.suptitle("Co-registration check: edge maps compared by masked NCC (make coreg)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    corr = float(np.corrcoef(ns.ravel(), no.ravel())[0, 1])
    return fig, {"edge_correlation_at_zero_shift": round(corr, 3)}


def tiling_overlay_figure(
    image: npt.NDArray[Any],
    scale: float,
    shape: tuple[int, int],
    tile: int,
    overlap: int,
    band_px: int,
) -> tuple[Figure, Metrics]:
    """Tile edges, crop edges and the border band over a scene overview.

    ``image`` is an ``(h, w, 3)`` overview of a ``shape`` scene, ``scale`` = overview
    pixels per scene pixel.
    """
    h, w = shape
    rows_t, cols_t = seam_lines(h, w, tile, 0)
    rows_o, cols_o = seam_lines(h, w, tile, overlap)
    aspect = image.shape[0] / image.shape[1]
    fig, ax = plt.subplots(1, 1, figsize=(9, 9 * aspect + 0.9), dpi=100)
    ax.imshow(image)
    for rr in rows_o:
        ax.axhspan((rr - band_px) * scale, (rr + band_px) * scale, color="yellow", alpha=0.08)
    for cc in cols_o:
        ax.axvspan((cc - band_px) * scale, (cc + band_px) * scale, color="yellow", alpha=0.08)
    for rr in rows_t:
        ax.axhline(rr * scale, color="red", lw=0.6)
    for cc in cols_t:
        ax.axvline(cc * scale, color="red", lw=0.6)
    for rr in rows_o:
        ax.axhline(rr * scale, color="cyan", lw=0.4, ls="--")
    for cc in cols_o:
        ax.axvline(cc * scale, color="cyan", lw=0.4, ls="--")
    _off(ax)
    ax.set_title(
        f"T {tile}: red = O 0 tile edges; cyan = O {overlap} tile + crop edges;\n"
        f"yellow = {band_px} px border band",
        fontsize=9,
    )
    fig.tight_layout()
    return fig, {
        "tiles_o0": (len(rows_t) + 1) * (len(cols_t) + 1),
        "lines_o": [len(rows_o), len(cols_o)],
    }
