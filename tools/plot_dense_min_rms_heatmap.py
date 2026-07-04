from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd


DEFAULT_DETAIL_CSV = (
    "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests/"
    "dense_fulltemplate_fn9_17p5_hfov8_10_12x9/extrapolation_comparison_detail.csv"
)
DEFAULT_OUTPUT = (
    "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests/"
    "dense_fulltemplate_fn9_17p5_hfov8_10_12x9/"
    "stage2_airgap_min_rms_heatmap_continuous.png"
)


def _load_min_rms_grid(detail_csv: Path, stage: str, efl_threshold: float) -> pd.DataFrame:
    detail = pd.read_csv(detail_csv)
    required = {"stage", "fn", "hfov", "rms_mm", "efl_error"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"{detail_csv} is missing required columns: {sorted(missing)}")

    data = detail[
        (detail["stage"].astype(str) == stage)
        & (pd.to_numeric(detail["efl_error"], errors="coerce") < float(efl_threshold))
    ].copy()
    if data.empty:
        raise ValueError(f"No rows found for stage={stage!r} with efl_error < {efl_threshold:g}")

    data["fn"] = pd.to_numeric(data["fn"], errors="coerce").round(6)
    data["hfov"] = pd.to_numeric(data["hfov"], errors="coerce").round(6)
    data["rms_um"] = pd.to_numeric(data["rms_mm"], errors="coerce") * 1000.0
    data = data.dropna(subset=["fn", "hfov", "rms_um"])

    grouped = data.groupby(["fn", "hfov"], as_index=False)["rms_um"].min()
    grid = grouped.pivot(index="fn", columns="hfov", values="rms_um")
    return grid.sort_index().sort_index(axis=1)


def _cell_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        half_step = 0.5
        return np.array([values[0] - half_step, values[0] + half_step], dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate([[first], midpoints, [last]])


def plot_dense_heatmap(
    detail_csv: Path,
    output: Path,
    *,
    stage: str,
    efl_threshold: float,
    dpi: int,
    plot_mode: str,
    show_points: bool,
    save_grid_csv: bool,
) -> Path:
    grid = _load_min_rms_grid(detail_csv, stage, efl_threshold)
    fn_values = grid.index.to_numpy(dtype=float)
    hfov_values = grid.columns.to_numpy(dtype=float)
    z = grid.to_numpy(dtype=float)
    x, y = np.meshgrid(hfov_values, fn_values)
    vmin = float(np.nanmin(z))
    vmax = float(np.nanmax(z))

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    if plot_mode == "cell":
        hfov_edges = _cell_edges(hfov_values)
        fn_edges = _cell_edges(fn_values)
        cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=vmin, vmax=vmax)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                value = z[i, j]
                if not np.isfinite(value):
                    continue
                ax.add_patch(
                    Rectangle(
                        (float(hfov_edges[j]), float(fn_edges[i])),
                        float(hfov_edges[j + 1] - hfov_edges[j]),
                        float(fn_edges[i + 1] - fn_edges[i]),
                        facecolor=cmap(norm(float(value))),
                        edgecolor="none",
                        linewidth=0.0,
                    )
                )
        image = ScalarMappable(norm=norm, cmap=cmap)
        image.set_array([])
        ax.set_xlim(float(hfov_edges[0]), float(hfov_edges[-1]))
        ax.set_ylim(float(fn_edges[0]), float(fn_edges[-1]))
    else:
        levels = np.linspace(vmin, vmax, 36)
        image = ax.contourf(x, y, z, levels=levels, cmap="viridis", extend="both")
        ax.contour(x, y, z, levels=8, colors="white", linewidths=0.35, alpha=0.38)

    if show_points and plot_mode != "cell":
        ax.scatter(x, y, s=7, c="white", alpha=0.45, linewidths=0)

    ax.set_box_aspect(1)
    ax.set_xlabel("HFOV (deg)", fontsize=10)
    ax.set_ylabel("F-number", fontsize=10)
    if plot_mode != "cell":
        ax.set_xlim(float(np.min(hfov_values)), float(np.max(hfov_values)))
        ax.set_ylim(float(np.min(fn_values)), float(np.max(fn_values)))
    ax.set_xticks(np.linspace(float(np.min(hfov_values)), float(np.max(hfov_values)), 5))
    ax.set_yticks([9.0, 11.0, 13.0, 15.0, 17.5])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.tick_params(length=2.5, width=0.7)

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Minimum RMS (um)", fontsize=9)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    cbar.ax.tick_params(labelsize=8)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    if save_grid_csv:
        grid_csv = output.with_suffix(".csv")
        grid.sort_index(ascending=False).to_csv(grid_csv, encoding="utf-8")
        print(f"Saved grid CSV: {grid_csv}")

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a continuous-style minimum RMS heatmap from extrapolation detail CSV."
    )
    parser.add_argument("--detail-csv", default=DEFAULT_DETAIL_CSV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", default="stage2_airgap")
    parser.add_argument("--efl-threshold", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--plot-mode",
        choices=["continuous", "cell"],
        default="continuous",
        help="continuous uses contourf; cell uses nearest-neighbor grid cells without interpolation.",
    )
    parser.add_argument("--hide-points", action="store_true")
    parser.add_argument("--no-grid-csv", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = plot_dense_heatmap(
        detail_csv=Path(args.detail_csv),
        output=Path(args.output),
        stage=args.stage,
        efl_threshold=args.efl_threshold,
        dpi=int(args.dpi),
        plot_mode=args.plot_mode,
        show_points=not args.hide_points,
        save_grid_csv=not args.no_grid_csv,
    )
    print(f"Saved dense heatmap: {output}")


if __name__ == "__main__":
    main()
