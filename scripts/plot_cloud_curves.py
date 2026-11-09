"""Draw several cloud-curve results (scripts/cloud_curve.py JSON files) in one figure.

Usage: uv run python scripts/plot_cloud_curves.py docs/figures/s3_clouds.json
           docs/figures/s3_clouds_aug.json --out docs/figures/s3_clouds_all.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = {
    "optical": "optical (clean-trained)",
    "fused": "fused (clean-trained)",
    "sar": "SAR only (no optical)",
    "s3_optical_clouds": "optical (cloud-trained)",
    "s3_fused_clouds": "fused (cloud-trained)",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("jsons", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=Path("docs/figures/s3_clouds_all.png"))
    a = p.parse_args()
    curves: dict[str, dict[str, object]] = {}
    xs: list[float] = []
    for f in a.jsons:
        d = json.loads(f.read_text())
        xs = [float(c) * 100 for c in d["meta"]["clouds"]["coverages"]]
        curves.update(d["curves"])
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=110)
    for name, c in curves.items():
        pts = c["points"]
        assert isinstance(pts, list)
        label = LABELS.get(name, name)
        cloud = "clouds" in name
        if c["modality"] == "sar":
            ax.plot(xs, [pts[0]["iou"]] * len(xs), "k--", lw=1.2, label=label)
        else:
            ax.plot(
                [q["coverage"] * 100 for q in pts],
                [q["iou"] for q in pts],
                "o-" if cloud else "o:",
                color="C1" if "fused" in name else "C0",
                lw=2 if cloud else 1.2,
                label=label,
            )
    ax.set_xlabel("synthetic cloud cover on the optical input (%)")
    ax.set_ylabel("val pixel IoU")
    ax.set_title("S3: robustness to clouds, clean- vs cloud-trained")
    ax.set_ylim(0, 0.75)
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(a.out)
    print(f"figure: {a.out}")


if __name__ == "__main__":
    main()
