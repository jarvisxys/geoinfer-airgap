#!/usr/bin/env bash
# Laptop -> server: the committed code (git archive of HEAD) plus the DVC-tracked inputs
# training needs (stats, encoder weights). No git credentials are needed on the server.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/server/env.sh
: "${REMOTE:?set REMOTE=user@host in scripts/server/remote.local}"
ssh "$REMOTE" "mkdir -p $REPO $RAW $DATA $BASE/bin $BASE/cache $BASE/tmp"
git archive --format=tar HEAD | ssh "$REMOTE" "tar -x -C $REPO"
rev="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- || echo -dirty)"
ssh "$REMOTE" "echo $rev > $REPO/REVISION"   # run.json records it (no .git on the server)
rsync -a --info=stats1 data/processed/stats "$REMOTE:$REPO/data/processed/"
rsync -a --info=stats1 models/registry "$REMOTE:$REPO/models/"
echo "pushed $(git rev-parse --short HEAD) to $REMOTE:$REPO"
