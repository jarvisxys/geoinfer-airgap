# geoinfer-airgap

Offline (air-gapped) inference service for large multi-sensor satellite scenes. It takes a
SAR + optical GeoTIFF, tiles and normalises it, runs a U-Net building segmentation model in
GPU batches, stitches the result into a seamless georeferenced mask, and reports accuracy
(IoU / F1) and speed (throughput, p95 latency, GPU utilisation).

DIP course project **HSR-AIML-034** (industry reference: GalaxEye Space). The graded core is
the image-processing pipeline in `src/geoinfer/{preprocess,tiling,postprocess,eval}`; the
FastAPI / PostGIS / Docker service is the delivery vehicle.

## Status

| Phase | Scope | Gate | Status |
|---|---|---|---|
| 0 | Setup: env, repo, CI, data download | CI green; GPU visible from PyTorch and Docker; sample tiles render | done |
| 1 | DIP core: io, normalisation, SAR despeckle, tiling, stitching, morphology, metrics | `v0.1-dip` | done |
| 2 | Models: optical / SAR / fused U-Net, S1-S3 studies | `v0.2-models` | done |
| 3 | Offline service: ONNX, FastAPI, worker, Docker, offline bundle | `v0.3-service` | |
| 4 | Benchmarks: S4 engines, E1-E6 serving experiments | `v0.4-bench` | |
| 5 | Report + demo | `v1.0-submission` | |

## Quick start (WSL2 Ubuntu 24.04)

```bash
git clone https://github.com/jarvisxys/geoinfer-airgap.git ~/geoinfer-airgap
cd ~/geoinfer-airgap
make setup      # uv env (Python 3.12, torch 2.14.1 + CUDA 13) and pre-commit hooks
make check      # ruff, mypy, pytest (CPU)
make env        # torch / CUDA / GPU fingerprint
make gate0      # opens a SAR tile + its optical twin, writes docs/figures/gate0_sar_optical.png
```

Data download (Windows PowerShell, resumable, writes to `D:\geo\raw\sn6`):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\download_sn6.ps1
```

Then extract the sample into WSL: `mkdir -p ~/geo-data/sample && tar -xzf /mnt/d/geo/raw/sn6/SN6_buildings_AOI_11_Rotterdam_train_sample.tar.gz -C ~/geo-data/sample`.

See [docs/setup.md](docs/setup.md) for the full machine setup.

## Layout

```
src/geoinfer/   io, preprocess, tiling, models, postprocess, eval, train, service/{api,worker,db}
configs/        all parameters (data, training, serving, benchmarks)
scripts/        data download, large-scene build, offline bundle
tests/          unit/, integration/, fixtures/ (tiny GeoTIFFs only)
docs/           setup, server environment, experiments, figures
```

## Results (Phase 2)

Held-out val split (637 tiles, eastern block of the covered Train extent, see
`docs/decisions.md` D1); thresholds tuned on the tune split. Full tables, settings and
reading in `docs/experiments.md`; checkpoints, ONNX models and run records in DVC
(`models/`).

| Study | Result | Entry |
|---|---|---|
| S3 sensors | val pixel IoU / object F1: optical 0.682 / 0.508, SAR 0.479 / 0.244, **fused 0.688 / 0.519** | E2 |
| S3 clouds | trained without clouds, optical and fused fall below SAR alone at 20 % cloud cover; cloud augmentation keeps them at 0.61-0.62 IoU at 40 %, and at 80 % only the cloud-trained fused model (0.527) beats SAR alone (0.479) | E6 |
| S2 SAR | dB vs linear +0.07 IoU, +0.11 object F1; over 3 seeds the dB arms tie on pixel IoU, per-strip normalisation is best on object F1 (0.264 vs 0.248 for the same Lee 7 filter), Lee 5 worst; speckle filtering does not help the CNN | E4, E7 |
| S2 optical | per-tile min-max = global stretch on tile IoU (0.681 vs 0.682), but leaves seams after stitching (seam score 2.60 vs 0.92) | E4, S2 figures |
| S1 stitching | real optical model; D4: overlap +0.005 border-band IoU, seam score 2.79 (naive) -> 1.48 (crop) -> 0.98 (Hann / Tukey); official Val extent (D3-val, 121 Mpx): +0.009 border-band IoU, +0.010 object F1, seams 3.23 -> 0.99 | E3, E9 |
| Morphology | clean-up raises SAR object F1 0.223 -> 0.244 (closing joins fragments); neutral for optical and fused; costs up to 0.013 pixel IoU | E8 |
| Baseline | classical Otsu on SAR: best 0.168 pixel IoU (Lee 7, per-strip threshold) | `otsu_d4.json` |
| Export | ONNX (dynamic batch, h, w) matches PyTorch to 2.4e-5 in logits | E5 |

## Reproduce

Data builds (run once, in this order; each writes outside git or into DVC):

| Output | Command | Runtime (laptop, USB drive) |
|---|---|---|
| Split lists `data/splits/*.txt`, `docs/figures/splits_map.png` | `make splits` | ~2 min |
| 200-tile working subset in `~/geo-data/subset` | `make subset` | ~25 min |
| Train-split S2 statistics `data/processed/stats/` (DVC) | `make stats` | ~48 min |
| Per-strip SAR statistics `strip_stats.json` (DVC, decision D4) | `make strip-stats` | ~47 min |
| D4 scene (SAR, optical, labels, strips) | `make d4` | ~17 min |

Figures and tables (`make figures` rebuilds all of them, ~7 min; reruns change only timings):

| Figure / table (`docs/figures/`) | Command | Runtime |
|---|---|---|
| Gate 0 quick-look `gate0_sar_optical.png` | `make gate0` | < 10 s |
| S2 figures `s2_*.png` + numbers `s2_metrics.json` (notebook 02) | `make figures-nb` | ~30 s |
| Co-registration table and scatter `coreg_shifts.{csv,png}` | `make coreg` | ~1 min |
| D4 quick-look `d4_overview.png` | `make d4-quicklook` | < 30 s |
| Post-processing on D4 `postprocess_d4.{png,json}` | `make postprocess-check` | ~1 min |
| S1 dry run (synthetic model) `s1_d4_synthetic.{png,json}` | `make s1` | ~2 min |
| Classical SAR baseline `otsu_d4.{png,json}` | `make otsu` | ~6 min |
| D3 tile -> stitch round trip (console table) | `make tile-check` | ~3 min |
| S1 with the trained optical model `s1_d4_optical.{png,json}` | `make s1-optical` | ~6 min |

Models (Phase 2; training runs on the college server, see `docs/server_env.md` and
`scripts/server/`; the laptop needs only `dvc pull` for the results):

| Output | Command | Where / runtime |
|---|---|---|
| Laptop sanity check (overfit 8 tiles) | `make overfit` | laptop, ~16 min |
| Server: code, stats and weights; environment; SN6 download | `scripts/server/push_code.sh`, `remote_setup.sh`, `remote_data.sh start` / `finish` | ~20 min |
| S3 models (optical, SAR, fused) | `scripts/server/remote_train.sh` | server, 30-75 min |
| S2 arms; seed repeats | `remote_s2.sh`; `remote_s2_seeds.sh` | server, ~35 min each |
| Cloud-trained S3 arm | `remote_s3_clouds.sh` | server, ~40 min |
| Val evaluation (`eval.json`) | `remote_eval.sh <run>` (`scripts/evaluate_run.py`) | server, ~5 min per run |
| Cloud curves `s3_clouds*.json`, `s3_clouds_all.png` | `remote_cloud.sh`; `scripts/plot_cloud_curves.py` | server, ~10 min per model |
| ONNX export + parity (`model.onnx`, `export.json`) | `make export` | laptop, ~1 min |

## Licence

Code: MIT ([LICENSE](LICENSE)). SpaceNet 6 data and anything derived from it: CC BY-SA 4.0
([DATA_LICENSE.md](DATA_LICENSE.md)).
