# Machine setup (Phase 0)

Record of the laptop environment, so the setup can be repeated on a clean machine.

## Laptop

| Item | Value |
|---|---|
| CPU / RAM | i7-11800H (16 threads) / 7.7 GB |
| GPU | GTX 1650 with Max-Q, 4 GB, compute capability 7.5 (sm_75) |
| NVIDIA driver | 610.62 (CUDA 13.3) |
| OS | Windows 11 + WSL 2.4.13, Ubuntu 24.04.2 LTS |
| Disks | C: ~22 GB free (WSL distro lives here); D: USB HDD, ~740 GB free (bulk data) |

## Windows side

| Software | Status |
|---|---|
| NVIDIA driver with CUDA 13.x | installed (610.62) |
| WSL2 + Ubuntu 24.04 | installed |
| `%USERPROFILE%\.wslconfig` | memory=5GB, processors=12, swap=8GB, swapFile on D: |
| Docker Desktop (WSL2 backend, GPU, disk image on D:) | see "Docker Desktop" below |
| VS Code + WSL extension | installed |
| QGIS 3.x | installed (3.44) |

`.wslconfig`:

```ini
[wsl2]
memory=5GB
processors=12
swap=8GB
swapFile=D:\\wsl-swap.vhdx
```

Apply with `wsl --shutdown` from PowerShell.

### Docker Desktop

1. Install Docker Desktop for Windows (WSL2 backend).
2. Settings > Resources > WSL integration: enable for `Ubuntu-24.04`.
3. Settings > Resources > Advanced > Disk image location: `D:\DockerDesktop`.
4. Check GPU access from WSL:
   `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi`
   (the image's CUDA version must not exceed the driver's, 13.3 here)

## Inside WSL2

```bash
# system packages (needs sudo)
sudo apt update && sudo apt install -y make gdal-bin gh build-essential

# user-space tools
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install awscli
uv tool install "dvc[ssh]"
uv tool install pre-commit
```

Git pushes from WSL reuse the Windows `gh` login:

```bash
git config --global credential.https://github.com.helper \
  "!'/mnt/c/Program Files/GitHub CLI/gh.exe' auth git-credential"
```

## Storage layout

```
/mnt/d/geo/            # D: USB HDD, bulk cold data
  raw/sn6/             # tarballs, mosaics, GeoJSON (never edited)
  interim/             # extracted tiles, COG conversions
  processed/           # normalised tiles / stats JSON
  models/              # checkpoints, ONNX, TensorRT engines
  bundles/             # offline docker save tarballs
  dvc-store/           # DVC local remote ("dlocal")
~/geo-data/            # WSL ext4 on C: (fast): sample + ~200 working tiles only
~/geoinfer-airgap/     # this repo (WSL ext4, never on the USB drive)
```

Do not train from `/mnt/d` directly: WSL2 reads Windows drives through the slow 9P bridge.

## Datasets

`scripts/download_sn6.ps1` (Windows PowerShell) downloads over HTTPS with resume, straight to
`D:\geo\raw\sn6`, and logs to `download.log` there. Re-run it after any interruption.

| # | File | Size |
|---|---|---|
| D1 | `SN6_buildings_AOI_11_Rotterdam_train_sample.tar.gz` | 124 MiB |
| D2 | `SN6_buildings_AOI_11_Rotterdam_train.tar.gz` | 39.0 GiB |
| D3 | `SN6_AOI_11_Rotterdam_PS-RGBNIR_TrainVal.tif` + building / extent GeoJSONs | 2.8 GiB + ~130 MiB |

After the downloads finish, record checksums for the air-gap manifest:

```bash
cd /mnt/d/geo/raw/sn6 && sha256sum *.tar.gz *.tif > SHA256SUMS
```

## Python environment

```bash
make setup     # uv sync --extra ml --extra export --extra serve + pre-commit install
make env       # prints torch / CUDA / GPU fingerprint
```

Pins: Python 3.12, torch 2.14.1 (CUDA 13 wheels, covers sm_75 and sm_120), torchvision 0.29.2,
rasterio 1.5.2, onnxruntime-gpu 1.31.1. CI installs only the CPU stack (`--extra serve`).
