"""Figures and table of the serving experiments E2-E4 (docs/figures/e_load_laptop.json).

Writes docs/figures/e_load.png (E4 latency vs offered load; E2 throughput and p95 by
batching policy; E3 high-priority p95, FIFO vs priority with ageing),
docs/figures/e_load_table.md and the report table report/load_table.tex. Every run is
re-summarised from its raw records (clock-jump correction, bench.load).

Usage: uv run python scripts/plot_load.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from geoinfer.bench.load import summarize_run

FIG = Path("docs/figures")
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"  # categorical slots 1-4
INK2, GRID = "#52514e", "#e4e3df"


def _lat(r: dict[str, Any], key: str = "p95_s") -> float:
    return float(r.get("latency", {}).get(key, float("nan")))


def _hi(r: dict[str, Any], key: str = "p95_s") -> float:
    return float(r.get("by_priority", {}).get("5", {}).get(key, float("nan")))


def main() -> None:
    d = json.loads((FIG / "e_load_laptop.json").read_text())
    # re-summarise every run from its raw records (clock-jump correction, bench.load)
    for group in ("E4", "E2", "E3"):
        for k, r in d.get(group, {}).items():
            d[group][k] = r | summarize_run(r["records"])
    d["capacity"] = d["capacity"] | summarize_run(d["capacity"]["records"])
    cap = d["capacity"]["capacity_jobs_per_s"]
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
        }
    )
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(12, 3.8), dpi=150)

    e4 = sorted(((float(k), v) for k, v in d.get("E4", {}).items()), key=lambda kv: kv[0])
    x = [k for k, _ in e4]
    for key, col, name in (("p50_s", C1, "p50"), ("p95_s", C2, "p95")):
        y = [_lat(v, key) for _, v in e4]
        a.plot(x, y, color=col, lw=2, marker="o", ms=5, label=name)
    a.axvline(1.0, color=INK2, lw=0.8, ls=":")
    a.text(1.0, a.get_ylim()[1] * 0.55, " burst\n capacity", fontsize=7, color=INK2, va="top")
    a.set_xlabel(f"offered load (x capacity = {cap:.3f} jobs/s)")
    a.set_ylabel("job latency (s), submit to ready")
    a.set_title("E4: latency vs offered load", loc="left", fontsize=10)
    a.legend(frameon=False, loc="upper left")

    e2 = d.get("E2", {})
    names = list(e2)
    b.bar(range(len(names)), [e2[n].get("km2_per_min", 0) for n in names], color=C1, width=0.6)
    for i, n in enumerate(names):
        b.text(
            i,
            e2[n].get("km2_per_min", 0),
            f"p95 {_lat(e2[n]):.0f} s",
            ha="center",
            va="bottom",
            fontsize=7,
            color=INK2,
        )
    labels = [
        n.replace("per_job_0ms", "per job").replace("cross_job_", "cross-job\n") for n in names
    ]
    b.set_xticks(range(len(names)), labels, fontsize=8)
    b.set_ylabel("throughput (km²/min)")
    e2_load = next(iter(e2.values()), {}).get("rate_per_s", 0) / cap if e2 else 0
    b.set_title(f"E2: batching policy ({e2_load:.0f} x capacity)", loc="left", fontsize=10)

    e3 = d.get("E3", {})
    loads = sorted({n.split("_load")[1] for n in e3}, key=float)
    arms = [
        ("fifo", C2, "FIFO"),
        ("priority", C3, "priority, ageing 0.1/s"),
        ("priority_ageing0.01", C4, "priority, ageing 0.01/s"),
    ]
    arms = [a for a in arms if any(f"{a[0]}_load{ld}" in e3 for ld in loads)]
    w = 0.8 / max(1, len(arms))
    for j, (pol, col, lab) in enumerate(arms):
        ys = [_hi(e3.get(f"{pol}_load{ld}", {})) for ld in loads]
        c.bar(
            [i + (j - (len(arms) - 1) / 2) * w for i in range(len(loads))],
            ys,
            width=w * 0.92,
            color=col,
            label=lab,
        )
    c.set_xticks(range(len(loads)), [f"{ld} x capacity" for ld in loads])
    c.set_ylabel("p95 latency of high-priority jobs (s)")
    c.set_title("E3: queue policy", loc="left", fontsize=10)
    c.legend(frameon=False, fontsize=8)
    for ax in (a, b, c):
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "e_load.png", facecolor="#fcfcfb")

    rows = [
        "| Run | Load | Ready / submitted | p50 (s) | p95 (s) | p99 (s) | High-prio p95 (s) | "
        "km²/min | Tiles/s | GPU util |",
        "|" + "---|" * 10,
    ]

    def row(name: str, r: dict[str, Any]) -> None:
        lat = r.get("latency", {})
        rate = r.get("rate_per_s")
        load = "burst" if rate is None else f"{rate / cap:.2f} x"
        rows.append(
            f"| {name} | {load} | {r.get('ready')} / {r.get('submitted')} | "
            f"{lat.get('p50_s')} | {lat.get('p95_s')} | {lat.get('p99_s')} | "
            f"{_hi(r)} | {r.get('km2_per_min')} | {r.get('tiles_per_s')} | "
            f"{r.get('gpu', {}).get('util_mean')}% |"
        )

    row("capacity (burst)", d["capacity"])
    for k, v in e4:
        row(f"E4 {k}", v)
    for n, v in e2.items():
        row(f"E2 {n}", v)
    for n, v in e3.items():
        row(f"E3 {n}", v)
    (FIG / "e_load_table.md").write_text("\n".join(rows) + "\n")

    tex = [
        "\\begin{table}[t]",
        "\\centering\\small",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "Exp. & Arm & Load & p50 (s) & p95 (s) & High-prio p95 (s) & km$^2$/min & GPU \\\\",
        "\\midrule",
    ]

    def trow(exp: str, arm: str, r: dict[str, Any]) -> None:
        lat = r.get("latency", {})
        rate = r.get("rate_per_s")
        load = "burst" if rate is None else f"{rate / cap:.2f}$\\times$"
        tex.append(
            f"{exp} & {arm.replace('_', ' ')} & {load} & {lat.get('p50_s', 0):.1f} & "
            f"{lat.get('p95_s', 0):.1f} & {_hi(r):.1f} & {r.get('km2_per_min', 0):.1f} & "
            f"{r.get('gpu', {}).get('util_mean', 0):.0f}\\,\\% \\\\"
        )

    trow("cap.", "burst of 24", d["capacity"])
    for _k, v in e4:
        trow("E4", "cross-job 25 ms, priority", v)
    tex.append("\\midrule")
    for n, v in e2.items():
        trow("E2", n, v)
    tex.append("\\midrule")
    for n, v in e3.items():
        trow("E3", n.split("_load")[0], v)
    tex += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Serving runs on the laptop stack (re-summarised from the raw job records;",
        f"load in multiples of the burst capacity, {cap:.3f} jobs/s).}}",
        "\\label{tab:load}",
        "\\end{table}",
    ]
    Path("report/load_table.tex").write_text("\n".join(tex) + "\n")
    print(f"wrote {FIG / 'e_load.png'} and {FIG / 'e_load_table.md'}")


if __name__ == "__main__":
    main()
