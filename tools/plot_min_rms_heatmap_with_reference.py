from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from plot_result_summary import infer_metrics_layout


DEFAULT_METRICS_CSV = (
    "log/260521_1013/stage_1/test_origin_dataset_epoch5000/"
    "test_output_metrics_pred_rmsfilter_on_eflerror_lt_0p1.csv"
)
DEFAULT_REFERENCE_DESIGN_CSV = "data/scan_lens_dataset_surf10_reorder.csv"
DEFAULT_REFERENCE_METRICS_CSV = (
    "data/gt_evaluation/surf10/gt_eval_surf10_rmsfilter_off_metrics.csv"
)
DEFAULT_OUTPUT = "outputs/figures/min_rms_heatmap_with_ga_reference.png"


def _read_metrics_csv(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(path, header=None)
    layout = infer_metrics_layout(df)
    return df, layout


def _load_heatmap_values(metrics_csv: Path) -> pd.DataFrame:
    df, layout = _read_metrics_csv(metrics_csv)
    values = pd.DataFrame(
        {
            "fn": pd.to_numeric(df.iloc[:, 0], errors="coerce").round(2),
            "hfov": pd.to_numeric(df.iloc[:, 1], errors="coerce").round(2),
            "rms_um": pd.to_numeric(df.iloc[:, layout["spot"]], errors="coerce")
            * 1000.0,
        }
    ).dropna()

    grouped = values.groupby(["fn", "hfov"], as_index=False)["rms_um"].min()
    heatmap = grouped.pivot(index="fn", columns="hfov", values="rms_um")
    return heatmap.sort_index(ascending=False).sort_index(axis=1)


def _load_reference_point(
    design_csv: Path,
    metrics_csv: Path,
    row_number: int,
    reference_rms_um: float | None = None,
) -> tuple[float, float, float]:
    row_idx = row_number - 1
    design_df = pd.read_csv(design_csv, header=None)
    if row_idx < 0 or row_idx >= len(design_df):
        raise IndexError(f"Reference row {row_number} is out of range for {design_csv}")

    ref_fn = float(design_df.iloc[row_idx, 0])
    ref_hfov = float(design_df.iloc[row_idx, 1])

    if reference_rms_um is not None:
        return ref_fn, ref_hfov, float(reference_rms_um)

    metrics_df, layout = _read_metrics_csv(metrics_csv)
    if row_idx >= len(metrics_df):
        raise IndexError(f"Reference row {row_number} is out of range for {metrics_csv}")

    ref_rms_um = float(metrics_df.iloc[row_idx, layout["spot"]])

    # GT-evaluation CSVs in this project store the spot radius as the first
    # value in the trailing metrics block, while prediction CSVs store an
    # additional composite loss before the spot column.
    if ref_rms_um < 1.0 and metrics_df.shape[1] >= 8:
        alt_rms_um = float(metrics_df.iloc[row_idx, -8])
        if alt_rms_um > 1.0:
            ref_rms_um = alt_rms_um

    if ref_rms_um < 1.0:
        ref_rms_um *= 1000.0

    return ref_fn, ref_hfov, ref_rms_um


def _nearest_index(values: list[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))


def plot_heatmap(
    metrics_csv: Path,
    reference_design_csv: Path,
    reference_metrics_csv: Path,
    reference_row: int,
    reference_rms_um: float | None,
    output: Path,
    dpi: int,
) -> Path:
    heatmap = _load_heatmap_values(metrics_csv)
    ref_fn, ref_hfov, ref_rms_um = _load_reference_point(
        reference_design_csv,
        reference_metrics_csv,
        reference_row,
        reference_rms_um,
    )

    x_values = [float(v) for v in heatmap.columns.to_list()]
    y_values = [float(v) for v in heatmap.index.to_list()]
    data = heatmap.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))
    im = ax.imshow(data, cmap="viridis", aspect="auto")

    ax.set_box_aspect(1)
    ax.set_xlabel("HFOV (deg)", fontsize=10)
    ax.set_ylabel("F-number", fontsize=10)
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{v:g}" for v in x_values], fontsize=9)
    ax.set_yticks(np.arange(len(y_values)))
    ax.set_yticklabels([f"{v:g}" for v in y_values], fontsize=9)

    ref_x = _nearest_index(x_values, ref_hfov)
    ref_y = _nearest_index(y_values, ref_fn)

    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    threshold = vmin + 0.45 * (vmax - vmin)
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = data[y, x]
            if not np.isfinite(value):
                continue
            text_color = "white" if value < threshold else "#1f2d2d"
            ax.text(
                x,
                y,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
                fontweight="bold",
            )

    badge_x = ref_x - 0.23
    badge_y = ref_y + 0.25
    badge_face = im.cmap(im.norm(ref_rms_um))
    badge_text_color = "white" if ref_rms_um < threshold else "#1f2d2d"
    ax.scatter(
        [badge_x],
        [badge_y],
        s=230,
        marker="o",
        facecolor=badge_face,
        edgecolor="#d62728",
        linewidth=1.4,
        zorder=4,
    )
    ax.text(
        badge_x,
        badge_y,
        f"{ref_rms_um:.1f}",
        ha="center",
        va="center",
        color=badge_text_color,
        fontsize=5.4,
        fontweight="bold",
        zorder=5,
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("um", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the minimum RMS heatmap and mark one external GA reference "
            "design with a circled RMS value."
        )
    )
    parser.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--reference-design-csv", default=DEFAULT_REFERENCE_DESIGN_CSV)
    parser.add_argument("--reference-metrics-csv", default=DEFAULT_REFERENCE_METRICS_CSV)
    parser.add_argument("--reference-row", type=int, default=20)
    parser.add_argument(
        "--reference-rms-um",
        type=float,
        default=None,
        help="Optional RMS value to draw inside the reference circle, in micrometers.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_heatmap(
        metrics_csv=Path(args.metrics_csv),
        reference_design_csv=Path(args.reference_design_csv),
        reference_metrics_csv=Path(args.reference_metrics_csv),
        reference_row=args.reference_row,
        reference_rms_um=args.reference_rms_um,
        output=Path(args.output),
        dpi=args.dpi,
    )
    print(f"Saved heatmap: {output}")


if __name__ == "__main__":
    main()
