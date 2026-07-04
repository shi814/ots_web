#!/usr/bin/env python3
"""Compare optical metrics from two metrics_pred CSVs as overlaid histograms."""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_result_summary import MetricThresholds, load_metrics


def _finite(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def _metric_series(metrics):
    return {
        "rms": _finite(metrics["spot"] * 1000.0),
        "dist": _finite(metrics["dist"]),
        "efl": _finite(metrics["efl_error"]),
        "tele": _finite(metrics["tele"]),
    }


def _first_n(series, max_rows):
    if max_rows is None or max_rows <= 0:
        return series
    return {key: values[:max_rows] for key, values in series.items()}


def _bin_edges(values_a, values_b, threshold, bins, pad_ratio=0.04):
    joined = np.concatenate([values_a, values_b])
    joined = joined[np.isfinite(joined)]
    if joined.size == 0:
        right = threshold * 1.25 if threshold > 0 else 1.0
        return np.linspace(0.0, right, bins + 1)

    left = min(0.0, float(np.min(joined)))
    right = max(float(np.max(joined)), float(threshold))
    if right <= left:
        right = left + 1.0
    pad = (right - left) * pad_ratio
    return np.linspace(left, right + pad, bins + 1)


def plot_metrics_hist_compare(
    stage1_csv,
    stage2_csv,
    output_path,
    stage1_label="Stage 1 full test",
    stage2_label="Stage 2 full test",
    max_rows=0,
    bins=30,
    dpi=200,
    thresholds=None,
):
    thresholds = thresholds or MetricThresholds()

    stage1 = _first_n(_metric_series(load_metrics(stage1_csv)), max_rows)
    stage2 = _first_n(_metric_series(load_metrics(stage2_csv)), max_rows)

    specs = [
        {
            "key": "rms",
            "title": "RMS spot radius",
            "xlabel": "RMS spot radius (um)",
            "threshold": thresholds.spot * 1000.0,
            "threshold_label": f"threshold {thresholds.spot * 1000.0:g} um",
        },
        {
            "key": "dist",
            "title": "Distortion",
            "xlabel": "Distortion",
            "threshold": thresholds.dist,
            "threshold_label": f"threshold {thresholds.dist:g}",
        },
        {
            "key": "efl",
            "title": "EFL relative error",
            "xlabel": "|EFL_est - EFL_ideal| / |EFL_ideal|",
            "threshold": thresholds.efl_error,
            "threshold_label": f"threshold {thresholds.efl_error:g}",
        },
        {
            "key": "tele",
            "title": "Telecentricity",
            "xlabel": "Telecentricity (deg)",
            "threshold": thresholds.tele,
            "threshold_label": f"threshold {thresholds.tele:g} deg",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    fig.suptitle("Optical Metrics Histogram Comparison", fontsize=16, fontweight="bold")

    color_stage1 = "#7aa6d9"
    color_stage2 = "#e5a073"

    for ax, spec in zip(axes.ravel(), specs):
        key = spec["key"]
        vals1 = stage1[key]
        vals2 = stage2[key]
        edges = _bin_edges(vals1, vals2, spec["threshold"], bins)

        ax.hist(
            vals1,
            bins=edges,
            alpha=0.65,
            color=color_stage1,
            label=f"{stage1_label} (n={len(vals1)})",
        )
        ax.hist(
            vals2,
            bins=edges,
            alpha=0.65,
            color=color_stage2,
            label=f"{stage2_label} (n={len(vals2)})",
        )
        ax.axvline(
            spec["threshold"],
            color="black",
            linestyle="--",
            linewidth=1.1,
            label=spec["threshold_label"],
        )
        ax.set_title(spec["title"], fontsize=13, fontweight="bold")
        ax.set_xlabel(spec["xlabel"])
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, loc="upper right")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot overlaid optical metrics histograms for two metrics_pred CSV files."
    )
    parser.add_argument("--stage1_csv", required=True)
    parser.add_argument("--stage2_csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage1_label", default="Stage 1 full test")
    parser.add_argument("--stage2_label", default="Stage 2 full test")
    parser.add_argument("--max_rows", type=int, default=0, help="Use first N rows from each CSV; 0 means all rows.")
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    out_path = plot_metrics_hist_compare(
        stage1_csv=args.stage1_csv,
        stage2_csv=args.stage2_csv,
        output_path=args.output,
        stage1_label=args.stage1_label,
        stage2_label=args.stage2_label,
        max_rows=args.max_rows,
        bins=args.bins,
        dpi=args.dpi,
    )
    print(f"[Plot] Saved histogram comparison: {out_path}")


if __name__ == "__main__":
    main()
