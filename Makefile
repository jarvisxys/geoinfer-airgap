.DEFAULT_GOAL := help
SHELL := /bin/bash

SAMPLE ?= $(HOME)/geo-data/sample
CFG ?= configs/train_fused.yaml

.PHONY: help setup setup-ci check lint fmt types test test-gpu test-data cov env gate0 disk cog splits subset stats coreg tile-check d4 vectorize s1 otsu \
        train export serve bench bundle figures clean

help:  ## list targets
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-10s %s\n", $$1, $$2}'

setup:  ## laptop/server env: core + torch + onnx + service + dev tools, git hooks
	uv sync --extra ml --extra export --extra serve
	pre-commit install

setup-ci:  ## CPU-only env used by CI (no torch)
	uv sync --frozen --extra serve

check: lint types test disk  ## everything CI runs, plus a disk check

lint:  ## ruff lint + format check
	uv run ruff check
	uv run ruff format --check

fmt:  ## auto-fix lint and format
	uv run ruff check --fix
	uv run ruff format

types:  ## mypy on src/
	uv run mypy src

test:  ## unit tests without GPU or dataset
	uv run pytest -m "not gpu and not data" --cov=geoinfer

test-gpu:  ## GPU-only tests
	uv run pytest -m gpu

test-data:  ## tests that need the SpaceNet 6 sample under $(SAMPLE)
	SAMPLE=$(SAMPLE) uv run pytest -m data

env:  ## print the environment fingerprint (torch, CUDA, GPU, package versions)
	uv run python -m geoinfer.utils.runinfo

gate0:  ## Gate 0: SAR + optical twin open and render a quick-look
	uv run python scripts/gate0_check.py --sample $(SAMPLE) --out docs/figures

cog:  ## convert a raster to COG: make cog SRC=in.tif DST=out.tif
	@test -n "$(SRC)" -a -n "$(DST)" || { echo "usage: make cog SRC=in.tif DST=out.tif"; exit 2; }
	uv run python -m geoinfer.io.cog "$(SRC)" "$(DST)"

splits:  ## geographic train/tune/val split files from the extracted D2 tiles
	uv run python scripts/build_splits.py

subset:  ## ~200-tile working subset (COGs) in ~/geo-data/subset; needs splits
	uv run python scripts/build_subset.py

stats:  ## S2 train-split normalisation stats (needs data/splits/train.txt)
	uv run python scripts/compute_stats.py

coreg:  ## SAR vs optical shift table and figure on the working subset
	uv run python scripts/coreg_check.py

vectorize:  ## mask COG -> GeoJSON: make vectorize MASK=m.tif OUT=b.geojson [NATIVE=1]
	@test -n "$(MASK)" -a -n "$(OUT)" || { echo "usage: make vectorize MASK=m.tif OUT=b.geojson"; exit 2; }
	uv run python scripts/vectorize_mask.py "$(MASK)" "$(OUT)" $(if $(NATIVE),--native-crs)

s1:  ## S1 stitching study on D4 (synthetic edge-defect model until Phase 2)
	uv run python scripts/s1_stitching.py

otsu:  ## classical SAR baseline on D4 (Otsu on dB, per scene / per strip, Lee arms)
	uv run python scripts/otsu_baseline.py

d4:  ## build D4 (val-tile SAR + optical + labels mosaic) under scenes.d4
	uv run python scripts/build_d4.py

tile-check:  ## tile -> identity -> stitch round trip on D3 (STRATEGY=hann)
	uv run python scripts/tile_roundtrip.py --nodata 0 $(if $(STRATEGY),--strategy $(STRATEGY))

disk:  ## free space on WSL root, C: and D: (C: fills fast)
	@df -h / /mnt/c /mnt/d 2>/dev/null | awk 'NR==1 || /\/$$|\/mnt\/[cd]$$/'

train export serve bench bundle figures:  ## (later phases)
	@echo "'$@' is not implemented yet; see the phase plan in docs/"

clean:  ## remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
