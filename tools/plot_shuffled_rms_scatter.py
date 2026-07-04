"""Plot RMS scatter after shuffling sample order.

The metrics CSV keeps samples in generation order, so plotting row index
directly can create artificial arcs. This utility shuffles only the display
order and leaves the source CSV unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def infer_spot_column(df: pd.DataFrame) -> int:
    if df.shape[1] == 65:
        return -7
    if df.shape[1] == 66:
        return -8
    raise ValueError(f"Unsupported metrics CSV with {df.shape[1]} columns")


def plot_shuffled_rms(metrics_csv: Path, output_png: Path, seed: int = 2026, dpi: int = 300) -> None:
    df = pd.read_csv(metrics_csv, header=None)
    spot_col = infer_spot_column(df)
    rms_um = df.iloc[:, spot_col].astype(float).to_numpy() * 1000.0

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rms_um))
    shuffled = rms_um[order]
    sample_index = np.arange(1, len(shuffled) + 1)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.labelsize": 16,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig = plt.figure(figsize=(10, 7.2))
    ax = fig.add_subplot(111)
    ax.scatter(
        sample_index,
        shuffled,
        s=13,
        alpha=0.55,
        color="cornflowerblue",
        edgecolors="none",
        label="Network predictions",
    )
    ax.set_xlabel("Shuffled sample index")
    ax.set_ylabel("RMS spot size/um")
    ax.set_xlim(0, len(shuffled) + 1)
    ax.set_ylim(0, 70)
    ax.set_yticks(np.arange(0, 71, 10))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)

    ax_ins = ax.inset_axes([0.62, 0.65, 0.35, 0.30])
    bins = np.array([0.0, 17.0, 33.0, 50.0])
    counts, _, patches = ax_ins.hist(
        shuffled,
        bins=bins,
        color="cornflowerblue",
        alpha=0.9,
        edgecolor="black",
    )
    ax_ins.set_xlim(-2.5, 52.5)
    ax_ins.set_xticks([0.0, 17.0, 33.0, 50.0])
    ax_ins.set_ylim(0, max(float(counts.max()) * 1.18, 1.0))
    ax_ins.set_xlabel("RMS spot size/um", fontsize=9)
    ax_ins.set_ylabel("Count", fontsize=9)
    ax_ins.tick_params(labelsize=8)

    for cnt, patch in zip(counts, patches):
        x = patch.get_x() + patch.get_width() / 2
        ax_ins.text(
            x,
            cnt,
            str(int(cnt)),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            clip_on=True,
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_png, dpi=dpi)
    plt.close(fig)

    summary_csv = output_png.with_suffix(".summary.csv")
    pd.DataFrame(
        [
            {
                "metrics_csv": str(metrics_csv),
                "samples": int(len(rms_um)),
                "seed": int(seed),
                "rms_um_min": float(np.min(rms_um)),
                "rms_um_mean": float(np.mean(rms_um)),
                "rms_um_median": float(np.median(rms_um)),
                "rms_um_max": float(np.max(rms_um)),
                "count_0_17um": int(counts[0]),
                "count_17_33um": int(counts[1]),
                "count_33_50um": int(counts[2]),
            }
        ]
    ).to_csv(summary_csv, index=False, encoding="utf-8")

    print(f"Saved: {output_png}")
    print(f"Saved: {summary_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    plot_shuffled_rms(args.metrics_csv, args.output_png, seed=args.seed, dpi=args.dpi)


if __name__ == "__main__":
    main()
