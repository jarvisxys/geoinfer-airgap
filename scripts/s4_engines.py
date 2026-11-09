"""Study S4 / experiments E1 and E5: inference engines, batch sizes, tile / overlap sweep.

Parts (``--part``, default all; settings in configs/bench.yaml):

- ``engines``: each engine on the same ``s4.n_tiles`` real D4 tiles, normalised exactly
  as the worker does; per batch size 50 warm-up batches, then ``repeats`` timed repeats
  of at least ``min_tiles_per_repeat`` tiles. Tiles/s (mean +- std), km^2/min (tile stride
  area), p50 / p95 batch latency, peak VRAM above the idle baseline, GPU utilisation,
  SM clock (throttling) and temperature from a 10 Hz NVML sampler. Engines whose output
  is not finite (fp16 on GTX 16-series) are reported and not timed.
- ``accuracy``: each engine at ``accuracy_batch`` over the whole D4 mosaic through the
  worker pipeline (stitch, clean-up, vectorise): pixel IoU and object F1, their drop vs
  ``torch_fp32``, and end-to-end km^2/min (disk I/O and stitching included).
- ``sweep``: tile x overlap through the worker pipeline with ``sweep.engine``; batch keeps
  pixels per batch constant. IoU / object F1 once, end-to-end time ``repeats`` times.

Writes ``docs/figures/s4_<label>.json``. GPU processes are logged before and after each
part (server: discard runs where someone else was active).

Usage: uv run python scripts/s4_engines.py --label laptop [--gpu-index 0] [--part engines]
       [--scene-dir DIR] [--models-dir DIR] [--run-dir DIR]   (server paths)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import yaml
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from geoinfer.bench.engines import Engine, make_engine
from geoinfer.bench.gpu import GpuSampler, gpu_processes, nvml_reader
from geoinfer.bench.timing import summarize, time_batches
from geoinfer.eval.metrics import PixelCounts, filter_by_valid, object_scores, pixel_scores
from geoinfer.io.config import load_io_config
from geoinfer.io.raster import RasterReader
from geoinfer.postprocess.config import load_postprocess_config
from geoinfer.service.worker.pipeline import SceneInputs, prepare_tile, segment_aoi, to_crs
from geoinfer.tiling.config import load_tiling_config
from geoinfer.tiling.tiler import tile_grid
from geoinfer.train.dataset import Normalizer


def _rows(path: Path, step: int = 1024) -> np.ndarray:
    with RasterReader(path) as r:
        h, w = r.info.height, r.info.width
        return np.concatenate(
            [r.read(Window(0, y, w, min(step, h - y)), [1]).data[0] for y in range(0, h, step)]
        )


class Ctx:
    """Everything the parts share: config, scene, model, normaliser, labels."""

    def __init__(self, a: argparse.Namespace) -> None:
        self.cfg: dict[str, Any] = yaml.safe_load(Path("configs/bench.yaml").read_text())
        s4 = self.cfg["s4"]
        dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
        self.key = s4["scene"]
        self.scene_dir = a.scene_dir or Path(dcfg["scenes"][self.key])
        self.optical = self.scene_dir / f"{self.key}_optical.tif"
        mdir = a.models_dir or Path(s4["models_dir"])
        reg = yaml.safe_load((mdir / "registry.yaml").read_text())["models"]
        self.model = next(m for m in reg if m["id"] == s4["model"])
        self.onnx = mdir / self.model["path"]
        self.run_dir = a.run_dir or Path(s4["run_dir"])
        stats = json.loads((mdir / "stats" / "stats.json").read_text())
        arms = yaml.safe_load(Path("configs/preprocess.yaml").read_text())["sar"]["arms"]
        n = self.model["norm"]
        self.norm = Normalizer(stats, None, n["optical"], n["sar"], arms)
        self.threshold = float(self.model["threshold"])
        self.cache = Path(s4["trt_cache"]).expanduser()
        self.scratch = self.cache.parent / "s4_out"
        self.gpu_index = a.gpu_index
        self.read = nvml_reader(a.gpu_index)
        self.hz = float(self.cfg["protocol"]["gpu_sample_hz"])
        with RasterReader(self.optical) as r:
            self.info = r.info
        self.gsd = abs(self.info.transform.a)
        self._labels: tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame] | None = None

    def labels(self) -> tuple[np.ndarray, np.ndarray, gpd.GeoDataFrame]:
        if self._labels is None:
            valid = _rows(self.scene_dir / f"{self.key}_strips.tif") > 0
            gt = (_rows(self.scene_dir / f"{self.key}_labels.tif") > 0) & valid
            polys = gpd.read_file(self.scene_dir / f"{self.key}_buildings.geojson")
            self._labels = gt, valid, filter_by_valid(polys, valid, self.info.transform)
        return self._labels

    def aoi(self) -> Any:
        b = self.info
        x0, y0 = b.transform * (0, b.height)
        x1, y1 = b.transform * (b.width, 0)
        e = self.gsd / 4  # stay inside the scene edge after the round trip
        return to_crs(box(x0 + e, y0 + e, x1 - e, y1 - e), b.crs, "EPSG:4326")

    def engine(self, name: str, tile: int, max_batch: int, opt_batch: int) -> Engine:
        c = 4
        shapes = (
            f"image:1x{c}x{tile}x{tile}",
            f"image:{opt_batch}x{c}x{tile}x{tile}",
            f"image:{max_batch}x{c}x{tile}x{tile}",
        )
        return make_engine(
            name, onnx_path=self.onnx, run_dir=self.run_dir, cache_dir=self.cache, trt_shapes=shapes
        )


def _tiles(ctx: Ctx, tile: int, overlap: int, n: int) -> np.ndarray:
    """``n`` normalised D4 tiles with at least half their pixels valid, in grid order."""
    scene = SceneInputs("optical", ctx.optical, None, None)
    out = []
    with RasterReader(ctx.optical) as r:
        for spec in tile_grid(ctx.info.height, ctx.info.width, tile, overlap):
            prep = prepare_tile(spec, scene, {"optical": r}, ctx.norm, "reflect", False)
            if prep is not None and prep[1].mean() >= 0.5:
                out.append(prep[0])
            if len(out) == n:
                break
    return np.stack(out)


def part_engines(ctx: Ctx) -> dict[str, Any]:
    s4, proto = ctx.cfg["s4"], ctx.cfg["protocol"]
    tile, ov = int(s4["tile"]), int(s4["overlap"])
    tiles = _tiles(ctx, tile, ov, int(s4["n_tiles"]))
    batches = [int(b) for b in s4["batches"]]
    out: dict[str, Any] = {}
    for name in s4["engines"]:
        idle = ctx.read()
        base = idle.mem_mib if idle else 0.0
        res: dict[str, Any] = {"batches": {}}
        try:
            t0 = time.perf_counter()
            eng = ctx.engine(name, tile, max(batches), int(s4["accuracy_batch"]))
            y = eng(tiles[:1])  # TensorRT builds its engine here
            res["build_s"] = round(time.perf_counter() - t0, 1)
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
            print(name, out[name])
            continue
        if not np.isfinite(y).all():
            out[name] = {"error": "non-finite output (fp16 overflow on this GPU)"}
            print(name, out[name])
            continue
        for b in batches:
            try:
                time_batches(eng, tiles, b, int(proto["warmup_batches"]), 0)
                nb = math.ceil(int(proto["min_tiles_per_repeat"]) / b)
                with GpuSampler(ctx.read, ctx.hz) as gs:
                    reps = [
                        time_batches(eng, tiles, b, 0, nb) for _ in range(int(proto["repeats"]))
                    ]
                r = summarize(reps, b, tile, ov, ctx.gsd) | gs.summary()
                if "mem_peak_mib" in r:
                    r["vram_mib"] = round(r["mem_peak_mib"] - base)
                res["batches"][str(b)] = r
                print(
                    name,
                    b,
                    {k: r[k] for k in ("tiles_per_s", "p95_ms", "util_mean", "vram_mib") if k in r},
                )
            except Exception as e:  # out of memory at large batches
                res["batches"][str(b)] = {"error": f"{type(e).__name__}: {e}"[:200]}
                print(name, b, "failed:", res["batches"][str(b)]["error"][:100])
                break
        out[name] = res
        del eng
    return {"tile": tile, "overlap": ov, "n_tiles": len(tiles), "engines": out}


def _end_to_end(
    ctx: Ctx, eng: Engine, tile: int, overlap: int, batch: int, score: bool
) -> dict[str, Any]:
    tcfg = replace(
        load_tiling_config(),
        tile=tile,
        overlap=overlap,
        strategy="hann" if overlap else "naive",
        scratch_dir=ctx.scratch / "memmap",
    )
    pcfg = load_postprocess_config()
    with GpuSampler(ctx.read, ctx.hz) as gs:
        res = segment_aoi(
            SceneInputs("optical", ctx.optical, None, None),
            ctx.aoi(),
            eng,
            ctx.norm,
            ctx.threshold,
            tcfg,
            pcfg,
            load_io_config(),
            ctx.scratch / f"{tile}_{overlap}",
            batch_size=batch,
            max_aoi_px=200_000_000,
        )
    km2 = ctx.info.width * ctx.info.height * ctx.gsd**2 / 1e6
    out: dict[str, Any] = {
        "tiles": res.n_tiles,
        "timings": res.timings,
        "km2_per_min": round(km2 / res.timings["total_s"] * 60, 3),
        "gpu": gs.summary(),
    }
    if score:
        gt, valid, gt_poly = ctx.labels()
        with RasterReader(res.mask_path) as m:
            win = from_bounds(
                *m.info.transform * (0, m.info.height),
                *m.info.transform * (m.info.width, 0),
                ctx.info.transform,
            )
            mask = m.read(Window(0, 0, m.info.width, m.info.height), [1]).data[0]
        r0, c0 = round(win.row_off), round(win.col_off)
        h, w = mask.shape
        g, v = gt[r0 : r0 + h, c0 : c0 + w], valid[r0 : r0 + h, c0 : c0 + w]
        ok = v & (mask != 255)
        pc = PixelCounts.from_masks(mask == 1, g, ok)
        pred = filter_by_valid(res.buildings.to_crs(ctx.info.crs), valid, ctx.info.transform)
        obj = object_scores(pred, gt_poly, pcfg.min_area_m2)
        out |= {"pixel_iou": round(pixel_scores(pc)["iou"], 4), "object_f1": round(obj.f1, 4)}
    return out


def part_accuracy(ctx: Ctx, engines: dict[str, Any] | None) -> dict[str, Any]:
    s4 = ctx.cfg["s4"]
    tile, ov, b = int(s4["tile"]), int(s4["overlap"]), int(s4["accuracy_batch"])
    out: dict[str, Any] = {}
    for name in s4["engines"]:
        if engines is not None and "error" in engines.get(name, {}):
            out[name] = {"skipped": engines[name]["error"]}
            continue
        try:
            eng = ctx.engine(name, tile, b, b)
            out[name] = _end_to_end(ctx, eng, tile, ov, b, score=True)
            del eng
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"[:300]}
        print(
            name, {k: out[name].get(k) for k in ("pixel_iou", "object_f1", "km2_per_min", "error")}
        )
    ref = out.get("torch_fp32", {})
    for r in out.values():
        if "pixel_iou" in r and "pixel_iou" in ref:
            r["iou_drop_vs_torch_fp32"] = round(ref["pixel_iou"] - r["pixel_iou"], 4)
            r["object_f1_drop_vs_torch_fp32"] = round(ref["object_f1"] - r["object_f1"], 4)
    return {"batch": b, "tile": tile, "overlap": ov, "engines": out}


def part_sweep(ctx: Ctx) -> dict[str, Any]:
    sw, reps = ctx.cfg["sweep"], int(ctx.cfg["protocol"]["repeats"])
    out = []
    for tile in sw["tiles"]:
        batch = max(1, int(sw["pixels_per_batch"]) // (int(tile) ** 2))
        eng = ctx.engine(sw["engine"], int(tile), batch, batch)
        for ov in sw["overlaps"]:
            if ov >= tile:
                continue
            runs = [
                _end_to_end(ctx, eng, int(tile), int(ov), batch, score=(i == 0))
                for i in range(reps)
            ]
            rate = np.array([r["km2_per_min"] for r in runs])
            rec = {
                "tile": tile,
                "overlap": ov,
                "batch": batch,
                "tiles": runs[0]["tiles"],
                "pixel_iou": runs[0]["pixel_iou"],
                "object_f1": runs[0]["object_f1"],
                "km2_per_min": round(float(rate.mean()), 3),
                "km2_per_min_std": round(float(rate.std()), 3),
                "infer_s": runs[0]["timings"]["infer_s"],
                "total_s": round(float(np.mean([r["timings"]["total_s"] for r in runs])), 2),
                "gpu": runs[0]["gpu"],
            }
            out.append(rec)
            print(
                "sweep",
                {k: rec[k] for k in ("tile", "overlap", "pixel_iou", "object_f1", "km2_per_min")},
            )
        del eng
    return {"engine": sw["engine"], "runs": out}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label", required=True, help="laptop | server (output file name)")
    p.add_argument("--gpu-index", type=int, default=0, help="NVML index of the GPU in use")
    p.add_argument("--part", choices=["engines", "accuracy", "sweep", "all"], default="all")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--scene-dir", type=Path, default=None, help="override the D4 folder")
    p.add_argument("--models-dir", type=Path, default=None, help="override s4.models_dir")
    p.add_argument("--run-dir", type=Path, default=None, help="override s4.run_dir")
    a = p.parse_args()
    ctx = Ctx(a)
    out_path = a.out or Path(f"docs/figures/s4_{a.label}.json")
    result: dict[str, Any] = json.loads(out_path.read_text()) if out_path.exists() else {}
    import pynvml

    name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(a.gpu_index))
    result["meta"] = {
        "label": a.label,
        "gpu": name if isinstance(name, str) else name.decode(),
        "scene": f"{ctx.key} {ctx.info.width} x {ctx.info.height} px",
        "model": ctx.model["id"],
        "threshold": ctx.threshold,
        "protocol": ctx.cfg["protocol"],
    }
    parts = ["engines", "accuracy", "sweep"] if a.part == "all" else [a.part]
    for part in parts:
        before = gpu_processes(a.gpu_index)
        t0 = time.perf_counter()
        if part == "engines":
            r = part_engines(ctx)
        elif part == "accuracy":
            r = part_accuracy(ctx, result.get("engines", {}).get("engines"))
        else:
            r = part_sweep(ctx)
        r["gpu_processes"] = {"before": before, "after": gpu_processes(a.gpu_index)}
        r["seconds"] = round(time.perf_counter() - t0)
        result[part] = r
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{part} done in {r['seconds']} s -> {out_path}")


if __name__ == "__main__":
    main()
