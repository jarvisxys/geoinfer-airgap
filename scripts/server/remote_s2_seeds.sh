#!/usr/bin/env bash
# On the server: extra seeds for the S2 SAR dB arms, so differences of ~0.01 IoU can be
# told from run-to-run noise. Seed 0 of each arm is the original run (s2_sar_db,
# s2_sar_lee5, sar, s2_sar_lee7_strip). Each run in its own screen session on GPU $GPU.
# Usage: remote_s2_seeds.sh [seeds...]   (default: 1 2)
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
seeds=("${@:-1 2}")
common="--interim-root $ROOT --set data.root=$ROOT data.train_list=data/splits/train.txt \
data.val_list=data/splits/tune.txt data.workers=10 optim.batch_size=16 out_dir=$BASE/runs \
data.sar_cache=sar_cache"
arms=("s2_sar_db norm.sar=db" "s2_sar_lee5 norm.sar=db_lee5" "s2_sar_lee7 norm.sar=db_lee7"
      "s2_sar_lee7_strip norm.sar=db_lee7_strip")
mkdir -p "$BASE/runs"
for s in ${seeds[@]}; do
    for a in "${arms[@]}"; do
        read -r name extra <<< "$a"
        run="${name}_seed$s"
        cmd="cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && uv run python -m geoinfer.train.train \
--config configs/train_sar.yaml $common run_name=$run seed=$s $extra > $BASE/runs/$run.log 2>&1"
        screen -dmS "geoinfer_$run" bash -lc "$cmd"
    done
done
screen -ls || true
