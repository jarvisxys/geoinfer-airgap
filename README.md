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
| 1 | DIP core: io, normalisation, SAR despeckle, tiling, stitching, morphology, metrics | `v0.1-dip` | in progress |
| 2 | Models: optical / SAR / fused U-Net, S1-S3 studies | `v0.2-models` | |
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

## Reproduce

| Figure / table | Command | Expected runtime |
|---|---|---|
| Gate 0 quick-look | `make gate0` | < 10 s |

## Licence

Code: MIT ([LICENSE](LICENSE)). SpaceNet 6 data and anything derived from it: CC BY-SA 4.0
([DATA_LICENSE.md](DATA_LICENSE.md)).
