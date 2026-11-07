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

## Output

_(not run yet)_
