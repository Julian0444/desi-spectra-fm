"""Side-by-side z_true vs z_pred scatter for the README (v1 baseline vs v2.1).

Reads the two committed held-out prediction CSVs (canonical 2,000-spectrum
split) and writes docs/img/scatter_v1_v2.png. Offline, no model needed.

Usage:  python3 scripts/plot_scatter_v1_v2.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img" / "scatter_v1_v2.png"

PANELS = [
    (
        "v1 baseline — regression head",
        ROOT / "runs" / "calibration" / "predictions_v1_heldout_canonical.csv",
        "z_pred",
    ),
    (
        "v2.1 shipped — classification head",
        ROOT / "runs" / "desi_80k_classhead_v21" / "predictions.csv",
        "z_pred_map",
    ),
]

OUTLIER = 0.15  # |z_pred - z_true| / (1 + z_true) threshold


def load(path: Path, col: str) -> tuple[np.ndarray, np.ndarray]:
    z_true, z_pred = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            z_true.append(float(row["z_true"]))
            z_pred.append(float(row[col]))
    return np.array(z_true), np.array(z_pred)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lim = (-0.05, 4.05)
    grid = np.linspace(lim[0], lim[1], 200)

    for ax, (title, path, col) in zip(axes, PANELS):
        z_true, z_pred = load(path, col)
        dz = np.abs(z_pred - z_true) / (1 + z_true)
        out = dz > OUTLIER
        eta = out.mean()

        ax.fill_between(
            grid,
            grid - OUTLIER * (1 + grid),
            grid + OUTLIER * (1 + grid),
            color="0.92",
            zorder=0,
            label=f"|Δz|/(1+z) ≤ {OUTLIER}",
        )
        ax.plot(grid, grid, color="0.35", lw=1, zorder=1)
        ax.scatter(
            z_true[~out], z_pred[~out], s=6, alpha=0.35, color="#1f77b4",
            linewidths=0, zorder=2,
        )
        ax.scatter(
            z_true[out], z_pred[out], s=8, alpha=0.6, color="#d62728",
            linewidths=0, zorder=3,
            label=f"catastrophic outliers: η = {eta:.1%}",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("z_true (DESI pipeline)")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.set_aspect("equal")

    axes[0].set_ylabel("z_pred")
    fig.suptitle(
        "Held-out redshift predictions (same 2,000 never-seen DESI spectra)",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
