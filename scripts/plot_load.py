"""Figures and table of the serving experiments E2-E4 (docs/figures/e_load_laptop.json).

Writes docs/figures/e_load.png (E4 latency vs offered load; E2 throughput and p95 by
batching policy; E3 high-priority p95, FIFO vs priority with ageing) and
docs/figures/e_load_table.md.

Usage: uv run python scripts/plot_load.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from geoinfer.bench.load import summarize_run

FIG = Path("docs/figures")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # categorical slots 1-3
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
    a.text(1.0, a.get_ylim()[1] * 0.95, " measured\n capacity", fontsize=7, color=INK2, va="top")
    a.set_xlabel(f"offered load (x capacity = {cap:.3f} jobs/s)")
    a.set_ylabel("job latency (s), submit to ready")
    a.set_title("E4: latency vs offered load", loc="left", fontsize=10)
    a.legend(frameon=False)

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
    b.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7)
    b.set_ylabel("throughput (km²/min)")
    b.set_title("E2: batching policy (0.75 x capacity)", loc="left", fontsize=10)

    e3 = d.get("E3", {})
    loads = sorted({n.split("_load")[1] for n in e3})
    for j, (pol, col) in enumerate((("fifo", C2), ("priority", C3))):
        ys = [_hi(e3.get(f"{pol}_load{ld}", {})) for ld in loads]
        c.bar(
            [i + (j - 0.5) * 0.38 for i in range(len(loads))],
            ys,
            width=0.36,
            color=col,
            label="FIFO" if pol == "fifo" else "priority + ageing",
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
    print(f"wrote {FIG / 'e_load.png'} and {FIG / 'e_load_table.md'}")


if __name__ == "__main__":
    main()
