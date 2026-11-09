#!/usr/bin/env bash
# Offline install of the geoinfer service from a bundle (plan rule 5):
#
#   ./install_offline.sh geoinfer-<version>.tar [TARGET] [--up] [--airgap] [--test]
#
#   TARGET    install folder (default ~/geoinfer); re-running keeps its .env and data/
#   --up      start the stack after installing (docker compose up -d --wait)
#   --airgap  with --up: add compose.airgap.yaml (no network but the internal one)
#   --test    with --up: run one job per sensor mode on the bundled demo scene, inside the
#             network, and print the result (implies --up --airgap)
#
# Steps: extract; check every file against MANIFEST.sha256; docker load the images and
# check their ids against images/images.txt; write .env (random database password, paths,
# your uid/gid) unless it exists; create data/{inbox,outputs,scratch} and copy the demo
# scene. Needs only bash, tar, sha256sum and Docker with Compose (and an NVIDIA GPU with
# the container toolkit for the worker). Nothing is downloaded.
set -euo pipefail

die() { echo "install_offline: $*" >&2; exit 1; }

BUNDLE="" TARGET="" UP=0 AIRGAP=0 TEST=0
for a in "$@"; do
  case "$a" in
    --up) UP=1 ;;
    --airgap) AIRGAP=1 ;;
    --test) TEST=1; UP=1; AIRGAP=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    -*) die "unknown option $a" ;;
    *) if [[ -z "$BUNDLE" ]]; then BUNDLE=$a; elif [[ -z "$TARGET" ]]; then TARGET=$a;
       else die "too many arguments"; fi ;;
  esac
done
[[ -n "$BUNDLE" && -f "$BUNDLE" ]] || die "usage: $0 BUNDLE.tar [TARGET] [--up] [--airgap] [--test]"
TARGET=${TARGET:-$HOME/geoinfer}
BUNDLE=$(cd "$(dirname "$BUNDLE")" && pwd)/$(basename "$BUNDLE")

command -v docker >/dev/null || die "docker not found"
docker compose version >/dev/null 2>&1 || die "docker compose (v2) not found"
command -v sha256sum >/dev/null || die "sha256sum not found"

echo "== extract $BUNDLE -> $TARGET"
mkdir -p "$TARGET"
tar -xf "$BUNDLE" -C "$TARGET" --strip-components=1
TARGET=$(cd "$TARGET" && pwd)
cd "$TARGET"
VERSION=$(cat VERSION)

echo "== verify checksums ($(wc -l < MANIFEST.sha256) files)"
sha256sum -c --quiet MANIFEST.sha256 || die "checksum mismatch: the bundle is damaged or altered"

echo "== load images"
docker load -i images/images.tar.gz
bad=0
while read -r tag want; do
  [[ -z "$tag" ]] && continue
  got=$(docker image inspect -f '{{.Id}}' "$tag" 2>/dev/null || true)
  if [[ "$got" != "$want" ]]; then
    echo "image $tag: id ${got:-missing}, bundle says $want" >&2
    bad=1
  else
    echo "image $tag ok"
  fi
done < images/images.txt
[[ $bad == 0 ]] || die "loaded images differ from the bundle"

if [[ -f .env ]]; then
  echo "== keep existing .env"
else
  echo "== write .env"
  pw=$( (command -v openssl >/dev/null && openssl rand -hex 24) || head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
  umask 077
  cat > .env <<EOF
GEOINFER_DB_PASSWORD=$pw
GEOINFER_DATA=$TARGET/data
GEOINFER_MODELS=$TARGET/models
GEOINFER_TAG=$VERSION
GEOINFER_PORT=8000
GEOINFER_GPU=0
GEOINFER_UID=$(id -u)
GEOINFER_GID=$(id -g)
EOF
  umask 022
fi

mkdir -p data/inbox data/outputs data/scratch
if [[ -d demo ]]; then
  for f in demo/*; do
    case "$(basename "$f")" in
      demo_labels.tif|demo_scene.json) cp -n "$f" data/ ;;
      *) cp -n "$f" data/inbox/ ;;
    esac
  done
fi

DC=(docker compose -f compose.yaml)
[[ $AIRGAP == 1 ]] && DC+=(-f compose.airgap.yaml)
if [[ $UP == 1 ]]; then
  echo "== start: ${DC[*]} up -d --wait"
  "${DC[@]}" up -d --wait
fi
if [[ $TEST == 1 ]]; then
  echo "== test job(s) on the demo scene"
  "${DC[@]}" exec -T api python - < scripts/service_smoke.py
fi

cat <<EOF

geoinfer $VERSION installed in $TARGET
  start:     cd $TARGET && docker compose up -d --wait
  air-gap:   docker compose -f compose.yaml -f compose.airgap.yaml up -d --wait
  API:       http://127.0.0.1:8000/docs (not published in air-gap mode)
  scenes:    copy COGs to $TARGET/data/inbox, then POST /v1/scenes
EOF
