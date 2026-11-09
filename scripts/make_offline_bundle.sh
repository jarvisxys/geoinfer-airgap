#!/usr/bin/env bash
# Build the offline bundle (plan rule 2): images (docker save), models, wheels, demo scene,
# compose files, installer, and a SHA-256 manifest of every file.
#
#   scripts/make_offline_bundle.sh [VERSION] [OUT]     (make bundle VERSION=0.3.0)
#
# OUT/geoinfer-VERSION.tar  + .sha256, and OUT/install_offline.sh next to it:
#   geoinfer-VERSION/
#     VERSION, bundle.json        commit, build date, image ids, models
#     MANIFEST.sha256             sha256sum of every other file
#     install_offline.sh, README.txt, compose.yaml, compose.airgap.yaml, .env.example
#     images/images.tar.gz        docker save of api, worker and postgis (gzip -1)
#     images/images.txt           tag and image id, checked after docker load
#     models/                     make service-models output (registry.yaml, stats, onnx)
#     wheels/                     runtime wheels (linux x86_64, cp312) + requirements.txt
#     demo/                       demo scene (make demo-scene), if DEMO has one
#     scripts/service_smoke.py    in-network test client (install_offline.sh --test)
# Needs a clean git tree (ALLOW_DIRTY=1 to override); network only for building.
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION=${1:-0.3.0}
OUT=${2:-/mnt/d/geo/bundles}
DEMO=${DEMO:-$HOME/geo-data/service}
WHEELS=${WHEELS:-1}
DB_IMAGE=postgis/postgis:17-3.5

if [[ "${ALLOW_DIRTY:-0}" != 1 ]] && [[ -n "$(git status --porcelain)" ]]; then
  echo "git tree is not clean (commit first, or ALLOW_DIRTY=1)" >&2; exit 1
fi
COMMIT=$(git rev-parse HEAD)
B="$OUT/geoinfer-$VERSION"
echo "== bundle $VERSION from $COMMIT -> $B"
rm -rf "$B"
mkdir -p "$B"/{images,models,wheels,demo,scripts}

echo "== images"
docker build -q -f docker/api.Dockerfile -t "geoinfer-api:$VERSION" .
docker build -q -f docker/worker-ort.Dockerfile -t "geoinfer-worker:$VERSION" .
docker image inspect "$DB_IMAGE" >/dev/null 2>&1 || docker pull "$DB_IMAGE"
IMAGES=("geoinfer-api:$VERSION" "geoinfer-worker:$VERSION" "$DB_IMAGE")
for t in "${IMAGES[@]}"; do
  echo "$t $(docker image inspect -f '{{.Id}}' "$t")"
done > "$B/images/images.txt"
docker save "${IMAGES[@]}" | gzip -1 > "$B/images/images.tar.gz"

echo "== models"
uv run python scripts/build_service_models.py "$B/models"

if [[ "$WHEELS" == 1 ]]; then
  echo "== wheels"
  uv export --frozen --no-dev --no-emit-project --extra serve --extra worker --no-hashes \
    -o "$B/wheels/requirements.txt" -q
  uv build --wheel -o "$B/wheels" -q
  uvx --from 'pip>=25' pip download -q -r "$B/wheels/requirements.txt" -d "$B/wheels" \
    --only-binary=:all: --implementation cp --python-version 3.12 \
    --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
    --platform manylinux_2_17_x86_64 --platform manylinux2014_x86_64 \
    --platform manylinux1_x86_64 --platform linux_x86_64
fi

echo "== demo, compose, installer"
for f in inbox/demo_optical.tif inbox/demo_sar.tif demo_labels.tif demo_scene.json; do
  [[ -f "$DEMO/$f" ]] && cp "$DEMO/$f" "$B/demo/"
done
cp docker/compose.yaml docker/compose.airgap.yaml "$B/"
cp docker/.env.example "$B/.env.example"
cp scripts/install_offline.sh "$B/"
cp scripts/service_smoke.py "$B/scripts/"
echo "$VERSION" > "$B/VERSION"
cat > "$B/README.txt" <<TXT
geoinfer $VERSION offline bundle (HSR-AIML-034), commit $COMMIT

  ./install_offline.sh geoinfer-$VERSION.tar ~/geoinfer --test   # install + air-gapped test jobs
  cd ~/geoinfer && docker compose up -d --wait                  # API on 127.0.0.1:8000

Requires Docker with Compose v2 and the NVIDIA container toolkit (one GPU for the worker).
Every file is listed in MANIFEST.sha256; the installer refuses a bundle that does not match.
TXT
uv run python - "$B" "$VERSION" "$COMMIT" <<'PY'
import json, sys, datetime, pathlib, yaml
b, version, commit = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
images = [dict(zip(("tag", "id"), l.split())) for l in (b / "images/images.txt").read_text().splitlines()]
models = yaml.safe_load((b / "models/registry.yaml").read_text())["models"]
info = {
    "version": version, "commit": commit,
    "built": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "images": images,
    "models": [{k: m[k] for k in ("id", "sensor_mode", "threshold", "sha256", "run_name")} for m in models],
    "wheels": len(list((b / "wheels").glob("*.whl"))),
}
(b / "bundle.json").write_text(json.dumps(info, indent=2) + "\n")
PY

echo "== manifest"
(cd "$B" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
echo "== tar"
tar -cf "$OUT/geoinfer-$VERSION.tar" -C "$OUT" "geoinfer-$VERSION"
(cd "$OUT" && sha256sum "geoinfer-$VERSION.tar" > "geoinfer-$VERSION.tar.sha256")
cp scripts/install_offline.sh "$OUT/install_offline.sh"
du -sh "$B"/* "$OUT/geoinfer-$VERSION.tar" | sed "s|$OUT/||"
