"""Success and failure gallery for the report (guidelines: at least 5 of each, analysed).

Runs the service models through the worker pipeline (``SegmentJob``) on the demo band of
D4 (4096 x 2621 px, look direction 1 on 97 % of the SAR, labels in ``demo_labels.tif``),
once clean and once with a synthetic 50 % cloud cover on the optical bands (study S3
generator, fixed seed). Probabilities are cleaned at each model's tuned threshold.

The band is scored in 256 px windows; cases are picked by measured criteria (no hand
picking), each from a different window:

- success: best optical IoU in a dense block; best optical IoU on large buildings; best
  SAR IoU; fused (cloud-trained) under cloud vs the clean-trained optical model; the
  cloud-trained optical model under cloud;
- failure: most false-positive pixels where there are no buildings (optical); most
  ground-truth buildings swallowed by single predicted blobs (optical); largest SAR
  offset from the footprints (layover); clean-trained optical under cloud; worst SAR
  IoU in a dense block.

Writes docs/figures/gallery_success.png, gallery_failure.png and gallery.json.

Usage: uv run python scripts/gallery.py
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from rasterio.windows import Window
from shapely.geometry import box

from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader, RasterWriter
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.postprocess.morphology import clean_mask_blockwise, label_components
from geoinfer.preprocess.clouds import apply_clouds, cloud_mask
from geoinfer.service.worker.pipeline import SceneInputs, SegmentJob, to_crs
from geoinfer.service.worker.predictor import OnnxPredictor
from geoinfer.tiling.config import load_tiling_config
from geoinfer.train.dataset import Normalizer

DATA = Path.home() / "geo-data/service"
MODELS = Path("/mnt/d/geo/models/service")
SCRATCH = Path("/mnt/d/geo/service/scratch/gallery")
FIG = Path("docs/figures")
WIN = 256
TP, FP, FN = "#2a78d6", "#eb6834", "#eda100"  # categorical slots 1, 2, 4


def _rows(path: Path, bands: list[int] | None = None, step: int = 512) -> np.ndarray:
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        parts = [r.read(Window(0, y, w, min(step, h - y)), bands).data for y in range(0, h, step)]
    return np.concatenate(parts, axis=1)


def cloudy_copy(src: Path, dst: Path, coverage: float, seed: int, p_hi: list[float]) -> Path:
    """The optical band with a synthetic cloud cover (S3 generator), written as a COG."""
    if dst.exists():
        return dst
    pre = yaml.safe_load(Path("configs/preprocess.yaml").read_text())["clouds"]
    x = _rows(src)
    alpha = cloud_mask(x.shape[1:], coverage, float(pre["beta"]), float(pre["softness"]), seed)
    y = apply_clouds(x, np.any(x != 0, axis=0), alpha, p_hi)
    with RasterReader(src) as r:
        info = r.info
    with RasterWriter(dst, info, load_io_config(), nodata=info.nodata) as w:
        for y0 in range(0, info.height, 512):
            h = min(512, info.height - y0)
            w.write(y[:, y0 : y0 + h], Window(0, y0, info.width, h))
    np.save(dst.with_suffix(".alpha.npy"), (alpha >= 0.5))
    return dst


def run(scene: SceneInputs, onnx: Path, norm: Normalizer, thr: float, tag: str) -> np.ndarray:
    """Cleaned building mask of the whole band through the worker pipeline."""
    tcfg = replace(load_tiling_config(), scratch_dir=SCRATCH / "memmap")
    pcfg = load_postprocess_config()
    pred = OnnxPredictor(onnx, ("CUDAExecutionProvider", "CPUExecutionProvider"))
    ref = scene.optical or scene.sar
    assert ref is not None
    with RasterReader(ref) as r:
        info = r.info
    x0, y0 = info.transform * (0, info.height)
    x1, y1 = info.transform * (info.width, 0)
    e = abs(info.transform.a) / 4
    aoi = to_crs(box(x0 + e, y0 + e, x1 - e, y1 - e), info.crs, "EPSG:4326")
    with SegmentJob(
        scene, aoi, norm, thr, tcfg, pcfg, load_io_config(), SCRATCH / tag, max_aoi_px=50_000_000
    ) as job:
        batch: list[Any] = []
        for item in job.tiles():
            batch.append(item)
            if len(batch) == 8:
                for (s, _, v), p in zip(batch, pred(np.stack([b[1] for b in batch])), strict=True):
                    job.add(s, p, v)
                batch.clear()
        if batch:
            for (s, _, v), p in zip(batch, pred(np.stack([b[1] for b in batch])), strict=True):
                job.add(s, p, v)
        prob = job.stitcher.read(Window(0, 0, job.w, job.h))[0]
        covered = np.isfinite(prob)
        mask = clean_mask_blockwise(prob, covered, replace(pcfg, threshold=thr), job.transform)
        assert (job.h, job.w) == (info.height, info.width), "AOI must cover the whole band"
    print(f"{tag}: {mask.mean():.3f} building share")
    return mask


def _iou(p: np.ndarray, g: np.ndarray) -> float:
    u = (p | g).sum()
    return float((p & g).sum() / u) if u else float("nan")


def window_stats(
    masks: dict[str, np.ndarray], gt_ids: np.ndarray, valid: np.ndarray, cloud: np.ndarray
) -> list[dict[str, Any]]:
    h, w = gt_ids.shape
    out = []
    for r in range(0, h - WIN + 1, WIN):
        for c in range(0, w - WIN + 1, WIN):
            sl = (slice(r, r + WIN), slice(c, c + WIN))
            v = valid[sl]
            if v.mean() < 0.95:
                continue
            ids = gt_ids[sl]
            g = ids > 0
            inst = np.unique(ids[g])
            rec: dict[str, Any] = {
                "row": r,
                "col": c,
                "gt_buildings": int(inst.size),
                "gt_share": round(float(g.mean()), 3),
                "mean_building_px": float(g.sum() / max(1, inst.size)),
                "cloud_share": round(float(cloud[sl].mean()), 3),
            }
            for k, m in masks.items():
                p = m[sl] & v
                rec[f"iou_{k}"] = round(_iou(p, g), 3)
                rec[f"fp_{k}"] = int((p & ~g).sum())
            lab, n = label_components(masks["optical"][sl], 4)
            merged = 0
            for j in range(1, n + 1):
                k = np.unique(ids[(lab == j) & g])
                merged += max(0, k.size - 1) if k.size >= 3 else 0
            rec["optical_merged"] = int(merged)
            if inst.size >= 8:
                (dx, dy), resp = cv2.phaseCorrelate(
                    g.astype(np.float32), masks["sar"][sl].astype(np.float32)
                )
                rec["sar_shift_px"] = [round(dx, 1), round(dy, 1)]
                rec["sar_shift_resp"] = round(float(resp), 3)
            out.append(rec)
    return out


def _pick(stats: list[dict], used: list[tuple[int, int]], key: Any, ok: Any) -> dict:
    def far(s: dict) -> bool:
        return all(abs(s["row"] - r) >= WIN or abs(s["col"] - c) >= WIN for r, c in used)

    cands = [s for s in stats if ok(s) and far(s)]
    best = max(cands, key=key)
    used.append((best["row"], best["col"]))
    return best


def _rgb(x: np.ndarray) -> np.ndarray:
    rgb = x[:3].astype(np.float64)  # PS-RGBNIR band order: R, G, B, NIR
    lo, hi = np.percentile(rgb[rgb > 0], [2, 98]) if (rgb > 0).any() else (0, 1)
    return np.clip((rgb - lo) / max(1e-9, hi - lo), 0, 1).transpose(1, 2, 0)


def _sar(x: np.ndarray) -> np.ndarray:
    v = x[0].astype(np.float64)
    lo, hi = np.percentile(v[v > 0], [2, 98]) if (v > 0).any() else (0, 1)
    return np.clip((v - lo) / max(1e-9, hi - lo), 0, 1)


def _overlay(ax: Any, base: np.ndarray, pred: np.ndarray, gt: np.ndarray, gray: bool) -> None:
    ax.imshow(base, cmap="gray" if gray else None, interpolation="nearest")
    rgba = np.zeros((*gt.shape, 4))
    for m, col in ((pred & gt, TP), (pred & ~gt, FP), (~pred & gt, FN)):
        rgba[m] = (*(int(col[i : i + 2], 16) / 255 for i in (1, 3, 5)), 0.55)
    ax.imshow(rgba, interpolation="nearest")


def render(
    cases: list[dict],
    arrays: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    gt: np.ndarray,
    title: str,
    out: Path,
) -> None:
    fig, axes = plt.subplots(len(cases), 2, figsize=(7.2, 3.3 * len(cases)), dpi=130)
    for row, case in zip(axes, cases, strict=True):
        r, c = case["row"], case["col"]
        sl = (slice(r, r + WIN), slice(c, c + WIN))
        src = case["input"]
        base = (
            _sar(arrays[src][:, sl[0], sl[1]])
            if src == "sar"
            else _rgb(arrays[src][:, sl[0], sl[1]])
        )
        row[0].imshow(base, cmap="gray" if src == "sar" else None, interpolation="nearest")
        row[0].set_title(f"{case['id']}: input ({src.replace('_', ' ')})", fontsize=8, loc="left")
        _overlay(row[1], base, masks[case["model"]][sl], gt[sl], src == "sar")
        row[1].set_title(
            f"{case['model'].replace('_', ' ')}: IoU {case['iou']:.2f}", fontsize=8, loc="left"
        )
        for a in row:
            a.set_xticks([])
            a.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, alpha=0.7) for col in (TP, FP, FN)]
    fig.legend(
        handles,
        ["true positive", "false positive", "missed (false negative)"],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(title, y=1.012, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out, bbox_inches="tight", facecolor="#fcfcfb")
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    reg = {m["id"]: m for m in yaml.safe_load((MODELS / "registry.yaml").read_text())["models"]}
    stats = json.loads((MODELS / "stats" / "stats.json").read_text())
    arms = yaml.safe_load(Path("configs/preprocess.yaml").read_text())["sar"]["arms"]
    norm = Normalizer(stats, None, "global", "db_lee7", arms)
    opt, sar = DATA / "inbox" / "demo_optical.tif", DATA / "inbox" / "demo_sar.tif"
    cloudy = cloudy_copy(
        opt, SCRATCH / "demo_optical_cloud50.tif", 0.5, 7, stats["optical"]["p_hi"]
    )
    clean_opt = Path("models/s3/optical")
    thr_clean = float(json.loads((clean_opt / "eval.json").read_text())["threshold"])
    o, s, f = (
        reg[k]
        for k in ("unet_r34_optical_clouds_v1", "unet_r34_sar_lee7_v1", "unet_r34_fused_clouds_v1")
    )
    masks = {
        "optical": run(
            SceneInputs("optical", opt, None, None),
            MODELS / o["path"],
            norm,
            o["threshold"],
            "optical",
        ),
        "sar": run(
            SceneInputs("sar", None, sar, 1), MODELS / s["path"], norm, s["threshold"], "sar"
        ),
        "fused": run(
            SceneInputs("fused", opt, sar, 1), MODELS / f["path"], norm, f["threshold"], "fused"
        ),
        "optical_cloud": run(
            SceneInputs("optical", cloudy, None, None),
            MODELS / o["path"],
            norm,
            o["threshold"],
            "optical_cloud",
        ),
        "fused_cloud": run(
            SceneInputs("fused", cloudy, sar, 1),
            MODELS / f["path"],
            norm,
            f["threshold"],
            "fused_cloud",
        ),
        "optical_clean_trained_cloud": run(
            SceneInputs("optical", cloudy, None, None),
            clean_opt / "model.onnx",
            norm,
            thr_clean,
            "optical_clean_trained_cloud",
        ),
    }
    gt_ids = _rows(DATA / "demo_labels.tif", [1])[0]
    sar_arr = _rows(sar)
    valid = (gt_ids >= 0) & np.all(sar_arr != 0, axis=0)
    cloud = np.load(cloudy.with_suffix(".alpha.npy"))
    st = window_stats(masks, gt_ids, valid, cloud)
    arrays = {"optical": _rows(opt), "sar": sar_arr, "optical_cloud": _rows(cloudy)}
    gt = gt_ids > 0
    used: list[tuple[int, int]] = []

    def case(cid: str, model: str, inp: str, s: dict, why: str) -> dict:
        return {
            "id": cid,
            "model": model,
            "input": inp,
            "row": s["row"],
            "col": s["col"],
            "iou": s[f"iou_{model}"],
            "why": why,
            "stats": s,
        }

    succ = [
        case(
            "S1",
            "optical",
            "optical",
            _pick(st, used, lambda s: s["iou_optical"], lambda s: s["gt_buildings"] >= 20),
            "dense residential block, optical",
        ),
        case(
            "S2",
            "optical",
            "optical",
            _pick(
                st,
                used,
                lambda s: s["iou_optical"],
                lambda s: s["mean_building_px"] >= 1500 and s["gt_share"] >= 0.2,
            ),
            "large buildings, optical",
        ),
        case(
            "S3",
            "sar",
            "sar",
            _pick(st, used, lambda s: s["iou_sar"], lambda s: s["gt_buildings"] >= 10),
            "SAR only, look direction aligned (D5)",
        ),
        case(
            "S4",
            "fused_cloud",
            "optical_cloud",
            _pick(
                st,
                used,
                lambda s: s["iou_fused_cloud"] - s["iou_optical_clean_trained_cloud"],
                lambda s: s["cloud_share"] >= 0.5 and s["gt_buildings"] >= 10,
            ),
            "fused (cloud-trained) under cloud: SAR sees through",
        ),
        case(
            "S5",
            "optical_cloud",
            "optical_cloud",
            _pick(
                st,
                used,
                lambda s: s["iou_optical_cloud"],
                lambda s: s["cloud_share"] >= 0.3 and s["gt_buildings"] >= 10,
            ),
            "cloud-trained optical under thin cloud",
        ),
    ]
    fail = [
        case(
            "F1",
            "optical",
            "optical",
            _pick(st, used, lambda s: s["fp_optical"], lambda s: s["gt_buildings"] <= 2),
            "false positives without buildings (optical)",
        ),
        case(
            "F2",
            "optical",
            "optical",
            _pick(st, used, lambda s: s["optical_merged"], lambda s: s["gt_buildings"] >= 10),
            "adjacent buildings merged into blobs",
        ),
        case(
            "F3",
            "sar",
            "sar",
            _pick(
                st,
                used,
                lambda s: abs(s["sar_shift_px"][1]) + abs(s["sar_shift_px"][0]),
                lambda s: "sar_shift_px" in s and s["sar_shift_resp"] >= 0.05,
            ),
            "SAR prediction offset from the footprints (layover)",
        ),
        case(
            "F4",
            "optical_clean_trained_cloud",
            "optical_cloud",
            _pick(
                st,
                used,
                lambda s: -s["iou_optical_clean_trained_cloud"],
                lambda s: s["cloud_share"] >= 0.6 and s["gt_buildings"] >= 10,
            ),
            "clean-trained optical under cloud",
        ),
        case(
            "F5",
            "sar",
            "sar",
            _pick(st, used, lambda s: -s["iou_sar"], lambda s: s["gt_buildings"] >= 20),
            "SAR in a dense block",
        ),
    ]
    render(succ, arrays, masks, gt, "Success cases (demo band of D4)", FIG / "gallery_success.png")
    render(fail, arrays, masks, gt, "Failure cases (demo band of D4)", FIG / "gallery_failure.png")
    band = {k: round(_iou(m & valid, gt & valid), 4) for k, m in masks.items()}
    (FIG / "gallery.json").write_text(
        json.dumps({"band_iou": band, "window_px": WIN, "success": succ, "failure": fail}, indent=1)
        + "\n"
    )
    print("band IoU", band)


if __name__ == "__main__":
    main()
