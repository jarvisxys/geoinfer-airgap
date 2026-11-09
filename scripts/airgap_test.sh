#!/usr/bin/env bash
# Air-gap acceptance test (plan rule 1, Gate 3 rehearsal).
#  1. the worker image runs its import + ONNX Runtime CUDA check under `docker run --network none`;
#  2. the stack comes up with compose.airgap.yaml (internal network only, no published port);
#  3. scripts/service_smoke.py runs inside the api container: egress must be blocked, and one
#     job per sensor mode must finish on the demo scene (make demo-scene);
#  4. the masks are scored against the demo labels on the host.
# Needs docker/.env (see docker/.env.example), make images, make service-models.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source docker/.env; set +a
DC=(docker compose -f docker/compose.yaml -f docker/compose.airgap.yaml)
TAG=${GEOINFER_TAG:-dev}

echo "== 1. worker image, --network none"
docker run --rm --network none --gpus "device=${GEOINFER_GPU:-0}" \
  -v "${GEOINFER_MODELS}:/models:ro" "geoinfer-worker:${TAG}" python -c "
import sys, glob
from geoinfer.service.worker.predictor import OnnxPredictor
p = OnnxPredictor(sorted(glob.glob('/models/*/model.onnx'))[0], ['CUDAExecutionProvider', 'CPUExecutionProvider'])
print('providers', p.providers, 'torch imported:', 'torch' in sys.modules)
assert p.providers[0] == 'CUDAExecutionProvider'
"

echo "== 2. stack up (air-gapped)"
mkdir -p "${GEOINFER_DATA}"/{inbox,outputs,scratch}
trap '"${DC[@]}" logs --tail 20 init worker; "${DC[@]}" down' EXIT
"${DC[@]}" up -d --wait

echo "== 3. smoke test inside the network"
"${DC[@]}" exec -T api python - < scripts/service_smoke.py

echo "== 4. scores"
uv run python scripts/score_service_outputs.py "${GEOINFER_DATA}"
