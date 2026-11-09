#!/usr/bin/env bash
# On the server: masks for all splits, then the S3 models on one GPU, each in its own
# detached screen session (geoinfer_<model>; tmux is not installed). Usage:
# remote_train.sh [optical sar fused]
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
uv run python scripts/build_masks.py --root "$ROOT" --interim-root "$ROOT" \
    --lists data/splits/train.txt data/splits/tune.txt data/splits/val.txt
models=("${@:-optical sar fused}")
mkdir -p "$BASE/runs"
for m in ${models[@]}; do
    cmd="cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && uv run python -m geoinfer.train.train \
--config configs/train_$m.yaml --interim-root $ROOT --set data.root=$ROOT \
data.train_list=data/splits/train.txt data.val_list=data/splits/tune.txt \
data.workers=24 optim.batch_size=16 out_dir=$BASE/runs > $BASE/runs/$m.log 2>&1"
    screen -dmS "geoinfer_$m" bash -lc "$cmd"
done
screen -ls || true
