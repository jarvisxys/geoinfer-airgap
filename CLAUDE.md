# geoinfer-airgap: rules for Claude Code

## Project
Air-gapped geospatial inference platform (HSR-AIML-034, DIP course project).
Graded core = DIP pipeline in src/geoinfer/{preprocess,tiling,postprocess,eval}.
Data: SpaceNet 6 (SAR + optical, 0.5 m). Task: building segmentation.
The plan of record is "HSR-AIML-034 Research & Development Plan" (phases 0-5, gates 0-4).

## Commands
make setup | make check (ruff, mypy, pytest -m 'not gpu') | make test-gpu
make env | make gate0 | make disk
make train CFG=configs/train_fused.yaml | make serve | make bench | make bundle   (later phases)

## Rules
- Python 3.12, type hints, numpy-style docstrings.
- Never read a whole raster: use rasterio windows / out_shape. Never write under /mnt/d/geo/raw.
- No network access from src/: no URLs, no hub downloads; weights load from models/registry/.
- Every function in preprocess/, tiling/, postprocess/ gets a unit test on a synthetic array.
- Every output raster keeps the input CRS and transform; test it.
- Parameters live in configs/*.yaml, not in code.
- Run make check before proposing a commit.

## Git
- main is always runnable; work on short branches (feat/..., fix/..., exp/..., docs/...) and merge via PR.
- Conventional Commits (feat:, fix:, exp:, docs:, test:, chore:, ci:), short and simple messages.
- No AI co-author or "generated with" trailers in commits or PRs; the repo owner is the only contributor.
- Never commit rasters, tarballs, checkpoints or .env; large files go through DVC.

## Paths
- Repo: ~/geoinfer-airgap (WSL ext4). Working data: ~/geo-data (sample + ~200 tiles only).
- Bulk data: /mnt/d/geo/{raw,interim,processed,models,bundles,dvc-store}. raw/ is read-only.

## Hardware
Laptop: GTX 1650 4 GB (sm_75), about 5 GB RAM inside WSL2.
Server: RTX PRO 6000 Blackwell 96 GB (sm_120), shared with other users.
