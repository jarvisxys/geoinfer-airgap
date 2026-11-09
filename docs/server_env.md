# College server environment

Run these the first time you log in and paste the output below.

```bash
nvidia-smi                                   # driver version, CUDA version, other users' jobs
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
df -h ~ /scratch /data 2>/dev/null           # where large files may live, quotas
free -h; nproc                               # RAM / CPU you can use
groups | grep -E "docker|sudo"               # Docker allowed?
which apptainer singularity conda module 2>/dev/null
sinfo 2>/dev/null || echo "no SLURM"         # is there a job scheduler?
curl -sI https://pypi.org | head -1          # does the server have internet?
```

## What the answers decide

- **Driver shows CUDA 13.x**: use the same `torch==2.14.1` lockfile. If it shows only 12.8/12.9,
  ask for a driver update or pin `torch==2.12.x` with the `cu128` index on the server.
- **No internet**: build the offline bundle on the laptop, `rsync` it over; the server becomes a
  real air-gapped acceptance test.
- **Shared GPU**: limit yourself with `CUDA_VISIBLE_DEVICES`, train inside `tmux`, and log
  `nvidia-smi` so GPU-utilisation numbers are not polluted by other users' jobs.
- **Data transfer**: `rsync -avP --partial` the 39 GiB tarball (resumable).

## Output (2026-11-09, first login; hostname redacted)

Summary: 2x RTX PRO 6000 Blackwell (96 GB, sm_120), driver CUDA 13.3; GPU 0 busy with
another user's job, GPU 1 free; home 87 % full, `/scratch` 3.6 TB free but only
`/scratch/user6` is writable; 503 GB RAM, 96 cores; no sudo, docker, conda or tmux
(`screen` exists); SLURM partition down; internet available. Runbook: `scripts/server/`.

$ nvidia-smi
Mon Nov  9 12:23:22 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 610.43.02              KMD Version: 610.43.02     CUDA UMD Version: 13.3     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA RTX PRO 6000 Blac...    Off |   00000000:3D:00.0 Off |                    0 |
| N/A   55C    P0            133W /  450W |    4376MiB /  97887MiB |     26%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
|   1  NVIDIA RTX PRO 6000 Blac...    Off |   00000000:AB:00.0 Off |                    0 |
| N/A   36C    P8             38W /  450W |      14MiB /  97887MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A          191964      G   /usr/lib/xorg/Xorg                        4MiB |
|    0   N/A  N/A         2346817      C   ...0_bundle/.venv_gpu/bin/python       4352MiB |
|    1   N/A  N/A          191964      G   /usr/lib/xorg/Xorg                        4MiB |
+-----------------------------------------------------------------------------------------+
$ nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
name, memory.total [MiB], compute_cap
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 12.0
NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887 MiB, 12.0
$ df -h ~ /scratch /data 2>/dev/null
Filesystem                      Size  Used Avail Use% Mounted on
/dev/md126p2                    869G  713G  113G  87% /
/dev/mapper/vg_apps-lv_scratch  5.5T  2.0T  3.6T  36% /scratch
$ free -h; nproc
               total        used        free      shared  buff/cache   available
Mem:           503Gi        26Gi       8.8Gi        62Mi       471Gi       477Gi
Swap:          8.0Gi       9.3Mi       8.0Gi
96
$ groups | grep -E "docker|sudo"
$ which apptainer singularity conda module 2>/dev/null
$ sinfo 2>/dev/null || echo "no SLURM"
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
test*        up   infinite      1  down* <server>
$ curl -sI https://pypi.org | head -1
HTTP/2 200
$
