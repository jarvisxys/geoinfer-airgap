#!/usr/bin/env bash
# Split an offline bundle into GitHub-release-sized parts (each asset must be under 2 GiB).
#
#   scripts/split_bundle.sh BUNDLE.tar [OUTDIR]
#
# Writes BUNDLE.tar.part0, .part1, ... (1900 MiB each) and SHA256SUMS (the parts and the
# whole tar) into OUTDIR (default: next to the bundle). Users rejoin the parts with
#   cat geoinfer-<version>.tar.part* > geoinfer-<version>.tar
# and check the tar with sha256sum before running install_offline.sh.
set -euo pipefail

BUNDLE=${1:?usage: $0 BUNDLE.tar [OUTDIR]}
[[ -f "$BUNDLE" ]] || { echo "split_bundle: no such file $BUNDLE" >&2; exit 1; }
BUNDLE=$(cd "$(dirname "$BUNDLE")" && pwd)/$(basename "$BUNDLE")
OUT=${2:-$(dirname "$BUNDLE")}
NAME=$(basename "$BUNDLE")
mkdir -p "$OUT"

split -b 1900M -d -a 1 "$BUNDLE" "$OUT/$NAME.part"
(
  cd "$OUT"
  sha256sum "$NAME".part* > SHA256SUMS
  echo "$(sha256sum < "$BUNDLE" | cut -d' ' -f1)  $NAME" >> SHA256SUMS
)
ls -l "$OUT/$NAME".part*
cat "$OUT/SHA256SUMS"
