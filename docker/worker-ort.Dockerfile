# geoinfer-worker (ort): tiling, ONNX Runtime CUDA inference, stitching, post-processing.
# The default serving image (plan section 7): no torch. CUDA 13 / cuDNN 9 come from the
# nvidia-* wheels that ONNX Runtime preloads; the GPU driver comes from the host.
# Build: docker build -f docker/worker-ort.Dockerfile -t geoinfer-worker:dev .
FROM python:3.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra serve --extra worker
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable --extra serve --extra worker

FROM python:3.12-slim-bookworm
# libexpat: needed by the GDAL bundled in the rasterio wheel
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
WORKDIR /app
COPY configs ./configs
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp MPLCONFIGDIR=/tmp/mpl \
    NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 9101
CMD ["python", "-m", "geoinfer.service.worker"]
