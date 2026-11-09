#!/usr/bin/env bash
# Laptop: remove everything we put on the server when the server work is done.
# Stops our screen sessions and downloads and deletes BASE (code, data, env, caches,
# runs). Pull results first! The laptop's SSH key stays unless --remove-key is given
# (later phases still need the GPU).
# Usage: cleanup.sh [--remove-key]
set -euo pipefail
source "$(dirname "$0")/env.sh"
ssh "$REMOTE" "for s in \$(screen -ls | grep -o '[0-9]*\.geoinfer_[a-z]*'); do screen -S \$s -X quit; done; \
pkill -u \$(whoami) -f 'spacenet-dataset' || true; rm -rf $BASE; \
ls -d $BASE 2>/dev/null || echo 'BASE removed'"
if [ "${1:-}" = "--remove-key" ]; then
    key="$(cut -d' ' -f2 ~/.ssh/id_ed25519.pub)"
    ssh "$REMOTE" "grep -vF '$key' ~/.ssh/authorized_keys > ~/.ssh/ak.tmp || true; \
mv ~/.ssh/ak.tmp ~/.ssh/authorized_keys; chmod 600 ~/.ssh/authorized_keys"
    echo "key removed (next ssh asks for the password)"
fi
