#!/usr/bin/env bash
# On the server: study S2 (normalisation arms). Fills the Lee-filtered SAR cache once for
# every split tile, then trains the arms not covered by the S3 runs, each in a detached
# screen session on GPU $GPU. The S3 runs `optical` (global stretch) and `sar` (Lee 7,
# global) are the improved arms they are compared with.
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
lists="data/splits/train.txt data/splits/tune.txt data/splits/val.txt"
uv run python scripts/build_sar_cache.py --root "$ROOT" --cache "$ROOT/sar_cache" \
    --lists $lists --lee 5 7 --workers 48
common="--interim-root $ROOT --set data.root=$ROOT data.train_list=data/splits/train.txt \
data.val_list=data/splits/tune.txt data.workers=16 optim.batch_size=16 out_dir=$BASE/runs \
data.sar_cache=sar_cache"
runs=(
    "s2_optical_minmax configs/train_optical.yaml norm.optical=minmax"
    "s2_sar_linear configs/train_sar.yaml norm.sar=linear"
    "s2_sar_db configs/train_sar.yaml norm.sar=db"
    "s2_sar_lee5 configs/train_sar.yaml norm.sar=db_lee5"
    "s2_sar_lee7_strip configs/train_sar.yaml norm.sar=db_lee7_strip"
)
mkdir -p "$BASE/runs"
for r in "${runs[@]}"; do
    read -r name cfg extra <<< "$r"
    cmd="cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && uv run python -m geoinfer.train.train \
--config $cfg $common run_name=$name $extra > $BASE/runs/$name.log 2>&1"
    screen -dmS "geoinfer_$name" bash -lc "$cmd"
done
screen -ls || true
