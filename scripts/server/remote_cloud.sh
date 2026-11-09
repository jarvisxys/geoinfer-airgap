#!/usr/bin/env bash
# On the server: S3 cloud-robustness curve for finished runs (they need eval.json), using
# the Lee-filtered SAR cache. Output: $BASE/runs/s3_clouds.{json,png}, log cloud_curve.log.
# Usage: [CLOUD_OUT=name] remote_cloud.sh optical fused sar  (default name s3_clouds)
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
dirs=""
for r in "$@"; do dirs="$dirs $BASE/runs/$r"; done
setsid nohup bash -lc "cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && \
uv run python scripts/cloud_curve.py $dirs --data-root $ROOT --interim-root $ROOT \
--sar-cache sar_cache --out $BASE/runs/${CLOUD_OUT:-s3_clouds}" > "$BASE/runs/${CLOUD_OUT:-s3_clouds}.log" 2>&1 < /dev/null &
echo "cloud curve for $* -> $BASE/runs/cloud_curve.log"
