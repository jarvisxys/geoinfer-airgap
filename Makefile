.DEFAULT_GOAL := help
SHELL := /bin/bash

SAMPLE ?= $(HOME)/geo-data/sample
DATA ?= $(HOME)/geo-data/service
VERSION ?= 0.3.0
LABEL ?= laptop
BUNDLE_OUT ?= /mnt/d/geo/bundles
CFG ?= configs/train_fused.yaml

.PHONY: help setup setup-ci check lint fmt types test test-gpu test-data cov env gate0 disk cog splits subset stats strip-stats coreg tile-check d4 vectorize s1 s1-optical otsu test-db testdb-up testdb-down d4-quicklook postprocess-check figures-nb masks weights overfit \
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

testdb-up:  ## start a local PostGIS for the db tests (docker, port 55432)
	docker run -d --name geoinfer-testdb -e POSTGRES_USER=geoinfer -e POSTGRES_PASSWORD=geoinfer-test \
		-e POSTGRES_DB=geoinfer_test -p 127.0.0.1:55432:5432 postgis/postgis:17-3.5

testdb-down:  ## remove the local test PostGIS
	docker rm -f geoinfer-testdb

test-db:  ## service database tests against the local PostGIS (make testdb-up first)
	GEOINFER_TEST_DB_URL=postgresql+psycopg://geoinfer:geoinfer-test@127.0.0.1:55432/geoinfer_test \
		uv run pytest -m db

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

strip-stats:  ## per-strip SAR stats for strip-scoped S2 arms (needs make stats)
	uv run python scripts/compute_strip_stats.py

coreg:  ## SAR vs optical shift table and figure on the working subset
	uv run python scripts/coreg_check.py

vectorize:  ## mask COG -> GeoJSON: make vectorize MASK=m.tif OUT=b.geojson [NATIVE=1]
	@test -n "$(MASK)" -a -n "$(OUT)" || { echo "usage: make vectorize MASK=m.tif OUT=b.geojson"; exit 2; }
	uv run python scripts/vectorize_mask.py "$(MASK)" "$(OUT)" $(if $(NATIVE),--native-crs)

s1:  ## S1 stitching study on D4 (synthetic edge-defect model until Phase 2)
	uv run python scripts/s1_stitching.py

s1-optical:  ## S1 on D4 with the trained optical model (needs dvc pull models/s3)
	uv run python scripts/s1_stitching.py --model run --run-dir models/s3/optical

otsu:  ## classical SAR baseline on D4 (Otsu on dB, per scene / per strip, Lee arms)
	uv run python scripts/otsu_baseline.py

masks:  ## rasterise building masks for the working subset (cache in <subset>/masks)
	uv run python scripts/build_masks.py

weights:  ## download pretrained encoder weights once into models/registry (needs internet)
	uv run python scripts/fetch_weights.py

d4:  ## build D4 (val-tile SAR + optical + labels mosaic) under scenes.d4
	uv run python scripts/build_d4.py

tile-check:  ## tile -> identity -> stitch round trip on D3 (STRATEGY=hann)
	uv run python scripts/tile_roundtrip.py --nodata 0 $(if $(STRATEGY),--strategy $(STRATEGY))

disk:  ## free space on WSL root, C: and D: (C: fills fast)
	@df -h / /mnt/c /mnt/d 2>/dev/null | awk 'NR==1 || /\/$$|\/mnt\/[cd]$$/'

d4-quicklook:  ## D4 overview + full-resolution crop with labels (needs make d4)
	uv run python scripts/d4_quicklook.py

postprocess-check:  ## module (d) clean-up on D4 with synthetic noise
	uv run python scripts/postprocess_check.py

figures-nb:  ## run notebooks/02_dip_figures.ipynb headless (S2 figures); outputs stay out of git
	uv run --group notebook jupyter nbconvert --to notebook --execute --stdout \
		--ExecutePreprocessor.timeout=1800 notebooks/02_dip_figures.ipynb > /dev/null

plot-s4:  ## S4 Pareto figure and table from docs/figures/s4_{laptop,server}.json
	uv run python scripts/plot_s4.py

plot-load:  ## E2-E4 figure and tables from docs/figures/e_load_laptop.json
	uv run python scripts/plot_load.py

gallery:  ## success / failure cases on the D4 demo band (GPU, ~3 min; make demo-scene first)
	uv run python scripts/gallery.py

figures: gate0 splits coreg d4-quicklook postprocess-check s1 s1-optical otsu figures-nb plot-s4 plot-load gallery  ## rebuild every report figure (needs data, stats, D4, demo scene)

TECTONIC ?= tectonic
report:  ## build report/report.pdf (any LaTeX engine; default tectonic)
	cd report && $(TECTONIC) -X compile report.tex

train:  ## train one model; CFG is the config, SET the overrides (see train.py --set)
	uv run python -m geoinfer.train.train --config $(CFG) $(if $(SET),--set $(SET))

overfit:  ## laptop sanity check: overfit 8 fixed crops (configs/train_overfit.yaml)
	uv run python -m geoinfer.train.train --config configs/train_overfit.yaml

export:  ## ONNX export + parity check of the S3 runs (models/s3, needs dvc pull)
	uv run python -W ignore::DeprecationWarning scripts/export_run.py models/s3/optical models/s3/sar models/s3/fused

images:  ## build the service images (api, worker-ort) with tag dev
	docker build -f docker/api.Dockerfile -t geoinfer-api:dev .
	docker build -f docker/worker-ort.Dockerfile -t geoinfer-worker:dev .

service-models:  ## assemble the service model registry folder (configs/service.yaml registry:)
	uv run python scripts/build_service_models.py $(OUT)

demo-scene:  ## cut the D4 demo scene into $(DATA)/inbox for the stack tests
	uv run python scripts/make_demo_scene.py $(DATA)/inbox | tee /dev/stderr | tail -1 > $(DATA)/demo_scene.json

serve:  ## start the stack (docker/.env); API on 127.0.0.1:8000
	docker compose -f docker/compose.yaml up -d --wait

serve-down:  ## stop the stack (keeps the database volume)
	docker compose -f docker/compose.yaml down

airgap-test:  ## air-gapped stack test: --network none + internal network, one job per sensor mode
	scripts/airgap_test.sh

bundle:  ## offline bundle: make bundle VERSION=0.3.0 [BUNDLE_OUT=/mnt/d/geo/bundles]
	scripts/make_offline_bundle.sh $(VERSION) $(BUNDLE_OUT)

load-requests:  ## synthetic analyst requests (configs/loadtest.yaml) -> data/requests/
	uv run python scripts/gen_requests.py

load-experiments:  ## E2-E4 on the local stack (data on D:, ~2 h; needs well over 5 GB free on C:)
	uv run python scripts/load_experiments.py

bench:  ## S4 engine study on this GPU, then the S4 figure and table
	uv run python scripts/s4_engines.py --label $(LABEL)
	uv run python scripts/plot_s4.py

clean:  ## remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
