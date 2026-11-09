#!/usr/bin/env bash
# On the server: user-space uv + the locked environment (torch cu130 for sm_120).
set -euo pipefail
source "$(dirname "$0")/env.sh"
eval "$SERVER_ENV"
mkdir -p "$BASE/bin" "$BASE/cache" "$BASE/tmp"
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh
fi
cd "$REPO"
uv sync --frozen --extra ml
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
