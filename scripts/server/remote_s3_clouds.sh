#!/usr/bin/env bash
# On the server: S3 cloud arm. Fused and optical models trained with cloud augmentation
# (half the samples get synthetic clouds of 0-80 % coverage on the optical bands), each
# in a detached screen session on GPU $GPU. Compare with the clean-trained S3 runs on the
# cloud curve (remote_cloud.sh).
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
cd "$REPO"
aug='{p: 0.5, max_coverage: 0.8, beta: 2.2, softness: 0.15}'
common="--interim-root $ROOT --set data.root=$ROOT data.train_list=data/splits/train.txt \
data.val_list=data/splits/tune.txt data.workers=16 optim.batch_size=16 out_dir=$BASE/runs \
data.sar_cache=sar_cache"
mkdir -p "$BASE/runs"
for r in "s3_fused_clouds configs/train_fused.yaml" "s3_optical_clouds configs/train_optical.yaml"; do
    read -r name cfg <<< "$r"
    cmd="cd $REPO && $SERVER_ENV CUDA_VISIBLE_DEVICES=$GPU && uv run python -m geoinfer.train.train \
--config $cfg $common run_name=$name 'data.cloud_aug=$aug' > $BASE/runs/$name.log 2>&1"
    screen -dmS "geoinfer_$name" bash -lc "$cmd"
done
screen -ls || true
