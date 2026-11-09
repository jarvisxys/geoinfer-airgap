#!/usr/bin/env bash
# On the server: evaluate finished runs (threshold on tune, scores on val) on GPU, in the
# background; results in $BASE/runs/<run>/eval.json, logs in $BASE/runs/eval_<run>.log.
# Usage: remote_eval.sh optical [sar fused ...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
for r in "$@"; do
    setsid nohup bash -lc "cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && \
uv run python scripts/evaluate_run.py $BASE/runs/$r --data-root $ROOT --interim-root $ROOT" \
        > "$BASE/runs/eval_$r.log" 2>&1 < /dev/null &
    echo "evaluating $r -> $BASE/runs/eval_$r.log"
done
