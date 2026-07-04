"""Plot combined-test minimum RMS heatmaps with a GA reference marker."""

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


DEFAULT_STAGE1 = (
    "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests/"
    "combined_old_full_and_dense_fulltemplate/stage1/"
    "test_output_metrics_pred_rmsfilter_on_combined_eflerror_lt_0p1.csv"
)
DEFAULT_STAGE2 = (
    "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests/"
    "combined_old_full_and_dense_fulltemplate/stage2_airgap/"
    "test_output_metrics_pred_rmsfilter_on_combined_eflerror_lt_0p1.csv"
)
DEFAULT_REFERENCE_DESIGN = "data/scan_lens_dataset_surf10_reorder.csv"
DEFAULT_REFERENCE_METRICS = "data/gt_evaluation/surf10/gt_eval_surf10_rmsfilter_off_metrics.csv"
DEFAULT_OUTPUT_DIR = (
    "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests/"
    "combined_old_full_and_dense_fulltemplate/heatmaps_with_ga"
)


def read_prediction_min_grid(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv, header=None)
    layout = infer_metrics_layout(df)
    values = pd.DataFrame(
        {
            "fn": pd.to_numeric(df.iloc[:, 0], errors="coerce").round(6),
            "hfov": pd.to_numeric(df.iloc[:, 1], errors="coerce").round(6),
            "rms_um": pd.to_numeric(df.iloc[:, layout["spot"]], errors="coerce") * 1000.0,
        }
    ).dropna()
    grouped = values.groupby(["fn", "hfov"], as_index=False)["rms_um"].min()
    return grouped.pivot(index="fn", columns="hfov", values="rms_um").sort_index(ascending=False).sort_index(axis=1)


def read_ga_reference(
    design_csv: Path,
    metrics_csv: Path,
    row_number: int,
    reference_rms_um: float | None,
) -> tuple[float, float, float]:
    row_idx = row_number - 1
    design_df = pd.read_csv(design_csv, header=None)
    if row_idx < 0 or row_idx >= len(design_df):
        raise IndexError(f"Reference row {row_number} is outside {design_csv}")

    ref_fn = float(design_df.iloc[row_idx, 0])
    ref_hfov = float(design_df.iloc[row_idx, 1])
    if reference_rms_um is not None:
        return ref_fn, ref_hfov, float(reference_rms_um)

    metrics_df = pd.read_csv(metrics_csv, header=None)
    if row_idx < 0 or row_idx >= len(metrics_df):
        raise IndexError(f"Reference row {row_number} is outside {metrics_csv}")

    layout = infer_metrics_layout(metrics_df)
    ref_rms = float(metrics_df.iloc[row_idx, layout["spot"]])

    # The GA evaluation file stores a direct micrometer RMS value in the
    # column before the prediction-style RMS slot. Prefer it when available.
    if ref_rms < 1.0 and metrics_df.shape[1] >= 8:
        alt = float(metrics_df.iloc[row_idx, -8])
        if alt > 1.0:
            ref_rms = alt
    if ref_rms < 1.0:
        ref_rms *= 1000.0

    return ref_fn, ref_hfov, ref_rms


def nearest_index(values: list[float], target: float) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.argmin(np.abs(arr - float(target))))


def draw_heatmap(
    ax: plt.Axes,
    grid: pd.DataFrame,
    *,
    title: str,
    ref_fn: float,
    ref_hfov: float,
    ref_rms_um: float,
    vmin: float,
    vmax: float,
    show_ylabel: bool = True,
) -> object:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ffffff")

    data = np.ma.masked_invalid(grid.to_numpy(dtype=float))
    image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_box_aspect(1)
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("HFOV (deg)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("F-number", fontsize=10)

    x_values = [float(v) for v in grid.columns.to_list()]
    y_values = [float(v) for v in grid.index.to_list()]
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{v:g}" for v in x_values], fontsize=7)
    y_tick_values = [9.0, 11.3, 12.85, 14.4, 15.95, 17.5]
    y_tick_pos = [nearest_index(y_values, value) for value in y_tick_values]
    deduped = []
    for pos, value in zip(y_tick_pos, y_tick_values):
        if pos not in [item[0] for item in deduped]:
            deduped.append((pos, value))
    ax.set_yticks([item[0] for item in deduped])
    ax.set_yticklabels([f"{item[1]:g}" for item in deduped], fontsize=8)
    ax.tick_params(length=2.5, width=0.7)

    ref_x = nearest_index(x_values, ref_hfov)
    ref_y = nearest_index(y_values, ref_fn)
    norm_value = (ref_rms_um - vmin) / (vmax - vmin + 1e-12)
    badge_face = cmap(float(np.clip(norm_value, 0.0, 1.0)))
    text_color = "white" if norm_value < 0.55 else "#1f2d2d"

    ax.scatter(
        [ref_x],
        [ref_y],
        s=245,
        marker="o",
        facecolor=badge_face,
        edgecolor="#d62728",
        linewidth=1.6,
        zorder=4,
    )
    ax.text(
        ref_x,
        ref_y,
        f"{ref_rms_um:.1f}",
        ha="center",
        va="center",
        color=text_color,
        fontsize=6.2,
        fontweight="bold",
        zorder=5,
    )
    return image


def save_single(
    grid: pd.DataFrame,
    output: Path,
    *,
    title: str,
    ref_fn: float,
    ref_hfov: float,
    ref_rms_um: float,
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    image = draw_heatmap(
        ax,
        grid,
        title=title,
        ref_fn=ref_fn,
        ref_hfov=ref_hfov,
        ref_rms_um=ref_rms_um,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Minimum RMS (um)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar_ticks = np.linspace(vmin, vmax, 8)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in cbar_ticks])
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    grid.to_csv(output.with_suffix(".csv"), encoding="utf-8")
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.svg')}")
    print(f"Saved: {output.with_suffix('.csv')}")


def save_pair(
    stage1_grid: pd.DataFrame,
    stage2_grid: pd.DataFrame,
    output: Path,
    *,
    ref_fn: float,
    ref_hfov: float,
    ref_rms_um: float,
    vmin: float,
    vmax: float,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), constrained_layout=True)
    image = draw_heatmap(
        axes[0],
        stage1_grid,
        title="Stage 1",
        ref_fn=ref_fn,
        ref_hfov=ref_hfov,
        ref_rms_um=ref_rms_um,
        vmin=vmin,
        vmax=vmax,
        show_ylabel=True,
    )
    draw_heatmap(
        axes[1],
        stage2_grid,
        title="Stage 2 AirGap",
        ref_fn=ref_fn,
        ref_hfov=ref_hfov,
        ref_rms_um=ref_rms_um,
        vmin=vmin,
        vmax=vmax,
        show_ylabel=False,
    )
    cbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.025)
    cbar.set_label("Minimum RMS (um)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    cbar_ticks = np.linspace(vmin, vmax, 8)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_yticklabels([f"{tick:.1f}" for tick in cbar_ticks])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")
    print(f"Saved: {output.with_suffix('.svg')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-metrics", type=Path, default=Path(DEFAULT_STAGE1))
    parser.add_argument("--stage2-metrics", type=Path, default=Path(DEFAULT_STAGE2))
    parser.add_argument("--reference-design-csv", type=Path, default=Path(DEFAULT_REFERENCE_DESIGN))
    parser.add_argument("--reference-metrics-csv", type=Path, default=Path(DEFAULT_REFERENCE_METRICS))
    parser.add_argument("--reference-row", type=int, default=20)
    parser.add_argument("--reference-rms-um", type=float, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--hide-title", action="store_true")
    parser.add_argument("--skip-pair", action="store_true")
    parser.add_argument(
        "--only-stage",
        choices=["both", "stage1", "stage2"],
        default="both",
        help="Limit which single-stage heatmaps are saved.",
    )
    parser.add_argument(
        "--scale-from-model-only",
        action="store_true",
        help="Do not include the reference RMS value when setting colorbar limits.",
    )
    args = parser.parse_args()

    stage1_grid = read_prediction_min_grid(args.stage1_metrics)
    stage2_grid = read_prediction_min_grid(args.stage2_metrics)
    ref_fn, ref_hfov, ref_rms_um = read_ga_reference(
        args.reference_design_csv,
        args.reference_metrics_csv,
        args.reference_row,
        args.reference_rms_um,
    )

    model_values = [
        stage1_grid.to_numpy(dtype=float).ravel(),
        stage2_grid.to_numpy(dtype=float).ravel(),
    ]
    if not args.scale_from_model_only:
        model_values.append(np.asarray([ref_rms_um], dtype=float))
    combined_values = np.concatenate(model_values)
    finite = combined_values[np.isfinite(combined_values)]
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))

    out_dir = args.output_dir
    suffix = "_notitle" if args.hide_title else ""
    if args.only_stage in ("both", "stage1"):
        save_single(
            stage1_grid,
            out_dir / f"stage1_combined_min_rms_heatmap_with_ga{suffix}.png",
            title="" if args.hide_title else "Stage 1",
            ref_fn=ref_fn,
            ref_hfov=ref_hfov,
            ref_rms_um=ref_rms_um,
            vmin=vmin,
            vmax=vmax,
            dpi=args.dpi,
        )
    if args.only_stage in ("both", "stage2"):
        save_single(
            stage2_grid,
            out_dir / f"stage2_airgap_combined_min_rms_heatmap_with_ga{suffix}.png",
            title="" if args.hide_title else "Stage 2 AirGap",
            ref_fn=ref_fn,
            ref_hfov=ref_hfov,
            ref_rms_um=ref_rms_um,
            vmin=vmin,
            vmax=vmax,
            dpi=args.dpi,
        )
    if not args.skip_pair and args.only_stage == "both":
        save_pair(
            stage1_grid,
            stage2_grid,
            out_dir / f"stage1_stage2_combined_min_rms_heatmap_with_ga{suffix}.png",
            ref_fn=ref_fn,
            ref_hfov=ref_hfov,
            ref_rms_um=ref_rms_um,
            vmin=vmin,
            vmax=vmax,
            dpi=args.dpi,
        )
    print(f"GA reference: F#={ref_fn:g}, HFOV={ref_hfov:g}, RMS={ref_rms_um:.3f} um")


if __name__ == "__main__":
    main()
