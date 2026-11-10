"""Figures of study S4: tile/overlap Pareto front and engine throughput on both GPUs.

Reads docs/figures/s4_server.json (sweep and engines) and s4_laptop.json (engines), writes
docs/figures/s4_pareto.png and the table docs/figures/s4_table.md.

Usage: uv run python scripts/plot_s4.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

FIG = Path("docs/figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (dataviz default palette)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ENGINES = ["torch_fp32", "torch_amp", "ort_fp32", "ort_fp16", "trt_fp16"]


def _best(e: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    ok = {b: r for b, r in e.get("batches", {}).items() if "tiles_per_s" in r}
    return max(ok.items(), key=lambda kv: kv[1]["tiles_per_s"]) if ok else None


def main() -> None:
    srv = json.loads((FIG / "s4_server.json").read_text())
    lap = json.loads((FIG / "s4_laptop.json").read_text())
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(11, 4.2), dpi=150, gridspec_kw={"width_ratios": [1.15, 1]}
    )

    runs = srv["sweep"]["runs"]
    for color, tile in zip(SERIES, sorted({r["tile"] for r in runs}), strict=True):
        pts = sorted((r for r in runs if r["tile"] == tile), key=lambda r: r["overlap"])
        x, y = [r["km2_per_min"] for r in pts], [r["pixel_iou"] for r in pts]
        ax.plot(x, y, color=color, lw=2, label=f"tile {tile} px", zorder=2)
        ax.scatter(x, y, s=36, color=color, edgecolor="white", linewidth=1.5, zorder=3)
        if tile == 512:  # selective direct labels: one series carries the overlap values
            for r in pts:
                ax.annotate(
                    f"O={r['overlap']}",
                    (r["km2_per_min"], r["pixel_iou"]),
                    textcoords="offset points",
                    xytext=(7, -3),
                    fontsize=7,
                    color=INK2,
                )
    d = next(r for r in runs if r["tile"] == 512 and r["overlap"] == 64)
    ax.scatter(
        [d["km2_per_min"]],
        [d["pixel_iou"]],
        s=150,
        facecolor="none",
        edgecolor=INK,
        linewidth=1.2,
        zorder=4,
    )
    ax.annotate(
        "service default\n(512 px, O=64)",
        (d["km2_per_min"], d["pixel_iou"]),
        textcoords="data",
        xytext=(43.0, 0.7022),
        fontsize=8,
        arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8},
    )
    ax.text(
        0.02,
        0.22,
        "O = tile overlap in px (labelled on the 512 px line)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK2,
    )
    ax.set_xlabel("end-to-end throughput (km²/min, D4, server GPU)")
    ax.set_ylabel("pixel IoU (D4)")
    ax.set_title("Tile size and overlap: accuracy vs speed", loc="left", fontsize=10)
    ax.legend(frameon=False, loc="center left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)

    gpus = [
        ("laptop GTX 1650", lap["engines"]["engines"]),
        ("server RTX PRO 6000", srv["engines"]["engines"]),
    ]
    h = 0.38
    for k, (name, eng) in enumerate(gpus):
        for i, e in enumerate(ENGINES):
            y = i + (k - 0.5) * h
            best = _best(eng.get(e, {}))
            if best is None:
                bx.text(2, y, "n/a: fp16 output not finite", va="center", fontsize=7, color=INK2)
                continue
            b, r = best
            bx.barh(
                y,
                r["tiles_per_s"],
                height=h - 0.04,
                color=SERIES[k],
                label=name if i == 2 else None,
            )
            bx.text(
                r["tiles_per_s"] + 6,
                y,
                f"{r['tiles_per_s']:.0f} (b{b})",
                va="center",
                fontsize=7,
                color=INK2,
            )
    bx.set_yticks(range(len(ENGINES)), [e.replace("_", " ") for e in ENGINES])
    bx.invert_yaxis()
    bx.set_xlabel("tiles/s, best batch size (512 px tiles)")
    bx.set_title("Inference engines on two GPUs", loc="left", fontsize=10, pad=22)
    bx.legend(
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=2,
        borderaxespad=0.2,
        fontsize=8,
    )
    bx.set_xlim(0, 720)
    bx.grid(axis="x", color=GRID, lw=0.6)
    bx.set_axisbelow(True)
    for a in (ax, bx):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "s4_pareto.png", facecolor="#fcfcfb")
    print(f"wrote {FIG / 's4_pareto.png'}")

    rows = [
        "| GPU | Engine | Batch | Tiles/s | km²/min | p95 batch (ms) | VRAM (MiB) | GPU util | "
        "Pixel IoU | Object F1 | End-to-end km²/min |",
        "|" + "---|" * 11,
    ]
    for name, data in (("GTX 1650", lap), ("RTX PRO 6000", srv)):
        acc = data.get("accuracy", {}).get("engines", {})
        for e in ENGINES:
            best = _best(data["engines"]["engines"].get(e, {}))
            a = acc.get(e, {})
            if best is None:
                rows.append(f"| {name} | {e} | - | non-finite fp16 output | | | | | | | |")
                continue
            b, r = best
            vram = r.get("vram_mib")
            vram_s = "n/m" if not vram else str(vram)  # torch reuses its cached memory
            rows.append(
                f"| {name} | {e} | {b} | {r['tiles_per_s']:.0f} | {r['km2_per_min']:.1f} | "
                f"{r['p95_ms']:.0f} | {vram_s} | {r.get('util_mean', '')}% | "
                f"{a.get('pixel_iou', '')} | {a.get('object_f1', '')} | "
                f"{a.get('km2_per_min', '')} |"
            )
    (FIG / "s4_table.md").write_text("\n".join(rows) + "\n")
    print(f"wrote {FIG / 's4_table.md'}")


if __name__ == "__main__":
    main()
