"""Serving experiments E2 (batching), E3 (priority), E4 (offered load) on the laptop stack.

Steps (configs/loadtest.yaml, decision D8):

1. ``setup``: the stack runs with ``GEOINFER_DATA = data_dir`` (on D:); the D3 optical COG
   and the demo fused band are placed in its inbox and registered.
2. ``capacity``: the first ``burst`` requests submitted at once; capacity = ready jobs /
   makespan (jobs/s). It is also the burst scenario.
3. For every experiment arm: the worker is recreated with the arm's scheduling
   (``GEOINFER_BATCHING`` etc.), the job tables are emptied, the request file is replayed
   at ``fraction x capacity`` for ``duration_s`` worth of arrivals (at least 12 jobs), the
   queue drains, and the run is summarised (bench.load) with a 10 Hz NVML sampler.
   Job outputs are deleted after each run, so disk use stays flat.

Writes docs/figures/e_load_laptop.json after every run (resumable with ``--only``).

Usage: uv run python scripts/load_experiments.py [--only capacity,E4,E2,E3]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# isort: off
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # loadtest/ at the repo root
from loadtest.replay import call, replay, wait_all
# isort: on

from geoinfer.bench.gpu import GpuSampler, nvml_reader
from geoinfer.bench.load import summarize_run

API = "http://127.0.0.1:8000"
COMPOSE = ["docker", "compose", "-f", "docker/compose.yaml"]
SCHED_ENV = {
    "batching": "GEOINFER_BATCHING",
    "queue": "GEOINFER_QUEUE",
    "ageing_per_s": "GEOINFER_AGEING_PER_S",
    "max_wait_ms": "GEOINFER_MAX_WAIT_MS",
    "max_active_jobs": "GEOINFER_MAX_ACTIVE_JOBS",
}


def _free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 2**30


def _check_disk() -> None:
    c = _free_gb("/mnt/c") if Path("/mnt/c").exists() else 99.0
    if c < 1.0:
        raise SystemExit(f"stopping: only {c:.1f} GB free on C:")


def compose(env: dict[str, str], *args: str, check: bool = True) -> None:
    r = subprocess.run([*COMPOSE, *args], env=env, capture_output=True, text=True)
    if r.returncode and check:
        raise SystemExit(f"docker compose {' '.join(args)} failed:\n{r.stderr[-2000:]}")
    if r.returncode:
        # `up --wait` also fails when the one-shot init container has exited from an
        # earlier start; readiness is then checked through /readyz instead
        print(f"note: docker compose {' '.join(args)} returned {r.returncode}", flush=True)


def wait_ready(timeout: float = 180) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            if call(API, "GET", "/readyz", timeout=5)[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise SystemExit("stack not ready")


def setup(cfg: dict[str, Any], env: dict[str, str]) -> None:
    data = Path(cfg["data_dir"])
    if not os.path.ismount(data.parents[len(data.parents) - 3]) and str(data).startswith("/mnt/"):
        raise SystemExit(f"{data.parents[len(data.parents) - 3]} is not mounted (USB drive?)")
    inbox = data / "inbox"
    for d in ("inbox", "outputs", "scratch"):
        (data / d).mkdir(parents=True, exist_ok=True)
    dcfg = yaml.safe_load(Path("configs/data.yaml").read_text())
    sources = {
        "d3_optical.tif": Path(dcfg["scenes"]["d3"]),
        "demo_optical.tif": Path.home() / "geo-data/service/inbox/demo_optical.tif",
        "demo_sar.tif": Path.home() / "geo-data/service/inbox/demo_sar.tif",
    }
    for name, src in sources.items():
        if not (inbox / name).exists():
            print(f"copy {src} -> {inbox / name}")
            shutil.copy2(src, inbox / name)
    compose(env, "up", "-d", "--wait", check=False)
    wait_ready()
    for sid, body in cfg["scenes"].items():
        st, out = call(API, "POST", "/v1/scenes", {"id": sid, **body}, timeout=600)
        if st not in (201, 409):
            raise SystemExit(f"register {sid}: {st} {out}")


def configure(env: dict[str, str], arm: dict[str, Any]) -> dict[str, str]:
    e = dict(env)
    for k, var in SCHED_ENV.items():
        e[var] = str(arm[k]) if k in arm else ""
    compose(e, "up", "-d", "--wait", "--no-deps", "--force-recreate", "worker", check=False)
    wait_ready()
    return e


def reset(cfg: dict[str, Any], env: dict[str, str]) -> None:
    compose(
        env,
        "exec",
        "-T",
        "db",
        "psql",
        "-U",
        "geoinfer",
        "-d",
        "geoinfer",
        "-qc",
        "TRUNCATE buildings, jobs",
    )
    out = Path(cfg["data_dir"]) / "outputs"
    for d in out.iterdir():
        if d.is_dir():
            shutil.rmtree(d)


def run_once(
    cfg: dict[str, Any], env: dict[str, str], reqs: list[dict], rate: float | None, label: str
) -> dict[str, Any]:
    _check_disk()
    reset(cfg, env)
    rc = cfg["run"]
    with GpuSampler(nvml_reader(0), 10) as gs:
        t0 = time.monotonic()
        recs = replay(API, reqs, rate)
        wait_all(API, recs, float(rc["drain_timeout_s"]), float(rc["poll_s"]))
        wall = time.monotonic() - t0
    s = summarize_run(recs) | {
        "label": label,
        "rate_per_s": rate,
        "n_requests": len(reqs),
        "wall_s": round(wall, 1),
        "gpu": gs.summary(),
    }
    lat = s.get("latency", {})
    print(
        f"{label}: {s['ready']}/{s['submitted']} ready, p50 {lat.get('p50_s')} s, "
        f"p95 {lat.get('p95_s')} s, {s.get('km2_per_min')} km2/min, "
        f"GPU {s['gpu'].get('util_mean')}%",
        flush=True,
    )
    s["records"] = recs
    return s


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", default="capacity,E4,E2,E3")
    p.add_argument("--out", type=Path, default=Path("docs/figures/e_load_laptop.json"))
    a = p.parse_args()
    cfg = yaml.safe_load(Path("configs/loadtest.yaml").read_text())
    reqs = [json.loads(line) for line in Path(cfg["requests"]["out"]).read_text().splitlines()]
    env = dict(
        os.environ, GEOINFER_DATA=cfg["data_dir"], GEOINFER_WORKER_MEM=str(cfg["worker_mem"])
    )
    res: dict[str, Any] = json.loads(a.out.read_text()) if a.out.exists() else {}
    steps = a.only.split(",")

    def save() -> None:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1) + "\n")

    setup(cfg, env)
    ex = cfg["experiments"]
    base = {
        "batching": ex["E4"]["batching"],
        "max_wait_ms": ex["E4"]["max_wait_ms"],
        "queue": ex["E4"]["queue"],
    }
    if "capacity" in steps:
        e = configure(env, base)
        r = run_once(cfg, e, reqs[: int(cfg["requests"]["burst"])], None, "capacity (burst)")
        r["capacity_jobs_per_s"] = round(r["ready"] / r["makespan_s"], 4)
        res["capacity"] = r
        save()
    cap = float(res["capacity"]["capacity_jobs_per_s"])
    print(f"capacity {cap} jobs/s", flush=True)
    dur = float(cfg["run"]["duration_s"])

    def level(frac: float) -> tuple[float, list[dict]]:
        rate = frac * cap
        n = max(12, min(len(reqs), round(rate * dur)))
        return rate, reqs[:n]

    if "E4" in steps:
        e = configure(env, base)
        res.setdefault("E4", {})
        for frac in ex["E4"]["loads"]:
            rate, sub = level(frac)
            res["E4"][str(frac)] = run_once(cfg, e, sub, rate, f"E4 load {frac}")
            save()
    if "E2" in steps:
        res.setdefault("E2", {})
        rate, sub = level(float(ex["E2"]["load"]))
        for arm in ex["E2"]["arms"]:
            name = f"{arm['batching']}_{arm['max_wait_ms']}ms"
            e = configure(env, base | arm)
            res["E2"][name] = run_once(cfg, e, sub, rate, f"E2 {name}") | {"arm": arm}
            save()
    if "E3" in steps:
        res.setdefault("E3", {})
        for frac in ex["E3"]["loads"]:
            rate, sub = level(float(frac))
            for arm in ex["E3"]["arms"]:
                name = f"{arm['queue']}_load{frac}"
                e = configure(env, base | arm)
                res["E3"][name] = run_once(cfg, e, sub, rate, f"E3 {name}") | {"arm": arm}
                save()
    configure(env, {})  # leave the worker on the config defaults
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
