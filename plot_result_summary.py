#!/usr/bin/env python3
"""Plot loss/scatter summaries and metrics pass-rate reports."""

import os
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


@dataclass(frozen=True)
class MetricThresholds:
    spot: float = 0.05
    dist: float = 0.02
    efl_error: float = 0.1
    tele: float = 2.0


FIXED_AXIS_CONFIG = {
    "rms": {
        "scatter_ylim": (0.0, 70.0),
        "scatter_yticks": np.arange(0.0, 71.0, 10.0),
        "hist_bins": np.array([0.0, 17.0, 33.0, 50.0]),
        "hist_xlim": (-2.5, 52.5),
        "hist_xticks": [0.0, 17.0, 33.0, 50.0],
    },
    "efl": {
        "scatter_ylim": (0.0, 0.10),
        "scatter_yticks": np.arange(0.0, 0.101, 0.02),
        "hist_bins": np.linspace(0.0, 0.10, 4),
        "hist_xlim": (-0.005, 0.105),
        "hist_xticks": [0.0, 0.05, 0.10],
    },
    "dist": {
        "scatter_ylim": (0.0, 0.10),
        "scatter_yticks": np.arange(0.0, 0.101, 0.02),
        "hist_bins": np.linspace(0.0, 0.10, 4),
        "hist_xlim": (-0.005, 0.105),
        "hist_xticks": [0.0, 0.05, 0.10],
    },
    "tele": {
        "scatter_ylim": (0.0, 4.0),
        "scatter_yticks": np.arange(0.0, 4.1, 1.0),
        "hist_bins": np.linspace(0.0, 4.0, 4),
        "hist_xlim": (-0.2, 4.2),
        "hist_xticks": [0.0, 2.0, 4.0],
    },
}


def infer_metrics_layout(df):
    """
    Infer tail-column layout for metrics_pred CSVs.

    Current train/test metrics layout:
    prefix columns = X(2) + max-11 N_bgr(33) + max-11 CT(22) = 57
    tail columns   = composite, rms, dist, tele, overlap, ray, EFL_est, EFL_ideal

    Some historical optimization CSVs also include loss_EFL before EFL_est/EFL_ideal.
    """
    n_cols = df.shape[1]
    if n_cols == 65:
        return {
            "composite": -8,
            "spot": -7,
            "dist": -6,
            "tele": -5,
            "overlap": -4,
            "ray": -3,
            "efl_est": -2,
            "efl_ideal": -1,
        }
    if n_cols == 66:
        return {
            "composite": -9,
            "spot": -8,
            "dist": -7,
            "tele": -6,
            "overlap": -5,
            "ray": -4,
            "loss_efl": -3,
            "efl_est": -2,
            "efl_ideal": -1,
        }
    raise ValueError(f"Unsupported metrics CSV layout with {n_cols} columns. Expected 65 or 66 columns.")


def plot_loss_curves(
    loss_csv_path,
    output_dir,
    stage1_epochs=None,
    dpi=300,
    plot_every_epochs=None,
    smooth_window=None,
):
    if not loss_csv_path:
        print("[Plot] No loss CSV provided, skip loss curves.")
        return None

    if not os.path.isfile(loss_csv_path):
        print(f"[Plot] Loss CSV not found, skip loss curves: {loss_csv_path}")
        return None

    font2 = {"family": "serif", "weight": "normal", "size": 15}

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(loss_csv_path)

    # Compatible with:
    # 1) optimization_loss_curves.csv with header:
    #    epoch,total_loss,rms_loss,dist_loss,efl_loss,lr
    # 2) stage2 log_loss_airgap_unsupervised.csv with header:
    #    epoch,train_loss,val_loss,...
    # 3) historical stage1 log_loss*.csv without header:
    #    epoch,train_loss,val_loss,train_spot_loss,val_spot_loss
    def _downsample_epoch_series(epochs_arr, *series_list):
        if not plot_every_epochs or float(plot_every_epochs) <= 0:
            return (epochs_arr, *series_list)

        step = float(plot_every_epochs)
        epochs_arr = np.asarray(epochs_arr, dtype=float)
        keep_mask = np.isclose(np.mod(epochs_arr, step), 0.0)
        if keep_mask.size:
            keep_mask[0] = True
            keep_mask[-1] = True
        if not np.any(keep_mask):
            keep_mask = np.ones_like(epochs_arr, dtype=bool)
        downsampled = [epochs_arr[keep_mask]]
        for values in series_list:
            downsampled.append(np.asarray(values, dtype=float)[keep_mask])
        return tuple(downsampled)

    def _smooth_series(values, window=None):
        window = smooth_window if window is None else window
        if not window or int(window) <= 1:
            return np.asarray(values, dtype=float)

        window = int(window)
        values = np.asarray(values, dtype=float)
        if values.size <= 2:
            return values
        window = min(window, values.size)
        if window <= 1:
            return values

        kernel = np.ones(window, dtype=float) / float(window)
        pad_left = window // 2
        pad_right = window - 1 - pad_left
        padded = np.pad(values, (pad_left, pad_right), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    if "epoch" in df.columns:
        epochs = df["epoch"].to_numpy(dtype=float)
        if {"total_loss", "rms_loss"}.issubset(df.columns):
            total_loss = df["total_loss"].to_numpy(dtype=float)
            rms_loss = df["rms_loss"].to_numpy(dtype=float)
            efl_loss = df["efl_loss"].to_numpy(dtype=float) if "efl_loss" in df.columns else None
            dist_loss = df["dist_loss"].to_numpy(dtype=float) if "dist_loss" in df.columns else None
            curve_specs = [
                ([(dist_loss, "Dist Loss", "m")], "Dist Loss", "dist_loss_trend.png"),
                ([(efl_loss, "EFL Loss", "g")], "EFL Loss", "efl_loss_trend.png"),
            ]
        elif {"train_loss", "val_loss"}.issubset(df.columns):
            train_loss = df["train_loss"].to_numpy(dtype=float)
            val_loss = df["val_loss"].to_numpy(dtype=float)
            epochs_plot, train_loss_raw_plot, val_loss_raw_plot = _downsample_epoch_series(
                epochs, train_loss, val_loss
            )
            trend_window = smooth_window if smooth_window and int(smooth_window) > 1 else 31
            train_loss_trend = _smooth_series(train_loss_raw_plot, window=trend_window)
            val_loss_trend = _smooth_series(val_loss_raw_plot, window=trend_window)

            mean_rows = []
            for values, label in (
                (train_loss, "train_filtered_loss"),
                (val_loss, "val_filtered_loss"),
            ):
                finite_values = values[np.isfinite(values)]
                if finite_values.size:
                    mean_rows.append(
                        {
                            "source": "loss_curve",
                            "loss_item": label,
                            "mean": float(np.mean(finite_values)),
                            "median": float(np.median(finite_values)),
                            "min": float(np.min(finite_values)),
                            "max": float(np.max(finite_values)),
                            "count": int(finite_values.size),
                        }
                    )

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            if len(epochs_plot) > 1:
                late_start = epochs_plot[0] + 0.8 * (epochs_plot[-1] - epochs_plot[0])
                ax.axvspan(
                    late_start,
                    epochs_plot[-1],
                    color="#eef1f5",
                    alpha=0.9,
                    linewidth=0,
                    label="late-stage region",
                    zorder=0,
                )
            ax.plot(
                epochs_plot,
                train_loss_raw_plot,
                color="#d62728",
                linewidth=0.55,
                alpha=0.22,
                label="train raw",
                zorder=1,
            )
            ax.plot(
                epochs_plot,
                val_loss_raw_plot,
                color="#1f77b4",
                linewidth=0.55,
                alpha=0.22,
                label="val raw",
                zorder=1,
            )
            ax.plot(
                epochs_plot,
                train_loss_trend,
                color="#d62728",
                linewidth=2.3,
                label=f"train trend ({int(trend_window)}-point MA)",
                zorder=3,
            )
            ax.plot(
                epochs_plot,
                val_loss_trend,
                color="#1f77b4",
                linewidth=2.3,
                label=f"val trend ({int(trend_window)}-point MA)",
                zorder=3,
            )
            finite_loss = np.concatenate(
                [
                    train_loss_raw_plot[np.isfinite(train_loss_raw_plot)],
                    val_loss_raw_plot[np.isfinite(val_loss_raw_plot)],
                ]
            )
            if finite_loss.size:
                y_min = float(np.min(finite_loss))
                y_max = float(np.max(finite_loss))
                y_pad = max((y_max - y_min) * 0.18, 1e-6)
                ax.set_ylim(y_min - y_pad, y_max + y_pad)
            ax.set_xlabel("Epoch", fontsize=18)
            ax.set_ylabel("Unsupervised loss", fontsize=18)
            ax.tick_params(labelsize=12)
            ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.28)
            ax.legend(loc="upper right", fontsize=11, frameon=False)
            fig.tight_layout()
            out_path = os.path.join(output_dir, "AirGap_USL_loss_redraw.png")
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            print(f"[Plot] Saved AirGap loss redraw: {out_path}")

            if mean_rows:
                mean_df = pd.DataFrame(mean_rows)
                mean_csv = os.path.join(output_dir, "loss_curve_mean_summary.csv")
                mean_df.to_csv(mean_csv, index=False)
                print(f"[Analyze] Saved loss curve means: {mean_csv}")
                return mean_df
            return None
        elif {"train_rmse_loss", "val_rmse_loss"}.issubset(df.columns):
            train_rmse_loss = df["train_rmse_loss"].to_numpy(dtype=float)
            val_rmse_loss = df["val_rmse_loss"].to_numpy(dtype=float)
            curve_specs = [
                (
                    [
                        (train_rmse_loss, "Train RMSE Loss", "b"),
                        (val_rmse_loss, "Val RMSE Loss", "r"),
                    ],
                    "RMSE Loss",
                    "rmse_loss_trend.png",
                ),
            ]
        else:
            raise ValueError(
                f"Unsupported loss CSV columns for {loss_csv_path}: {list(df.columns)}"
            )
    else:
        df = pd.read_csv(loss_csv_path, header=None)
        if df.shape[1] < 5:
            raise ValueError(
                f"Unsupported loss CSV layout for {loss_csv_path}. "
                f"Expected at least 5 columns, got {df.shape[1]}."
            )
        epochs = df.iloc[:, 0].to_numpy(dtype=float)
        total_loss = df.iloc[:, 1].to_numpy(dtype=float)
        rms_loss = df.iloc[:, 3].to_numpy(dtype=float)
        dist_loss = None
        efl_loss = None
        curve_specs = [
            ([(dist_loss, "Dist Loss", "m")], "Dist Loss", "dist_loss_trend.png"),
            ([(efl_loss, "EFL Loss", "g")], "EFL Loss", "efl_loss_trend.png"),
        ]

    mean_rows = []
    for series_specs, ylabel, fname in curve_specs:
        series_specs = [spec for spec in series_specs if spec[0] is not None]
        if not series_specs:
            continue
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        for y, label, color in series_specs:
            y = np.asarray(y, dtype=float)
            finite_y = y[np.isfinite(y)]
            if finite_y.size:
                mean_rows.append(
                    {
                        "source": "loss_curve",
                        "loss_item": label,
                        "mean": float(np.mean(finite_y)),
                        "median": float(np.median(finite_y)),
                        "min": float(np.min(finite_y)),
                        "max": float(np.max(finite_y)),
                        "count": int(finite_y.size),
                    }
                )
            ax.plot(epochs, y, f"{color}-", linewidth=1.5, label=label)
        if stage1_epochs is not None:
            ax.axvline(
                x=stage1_epochs,
                color="gray",
                linestyle="-",
                alpha=0.7,
                label=f"Stage boundary ({stage1_epochs})",
            )
        ax.set_xlabel("Epoch", font2)
        ax.set_ylabel(ylabel, font2)
        ax.set_title(f"{ylabel} Trend", fontsize=14, fontweight="bold")
        ax.tick_params(labelsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname("serif")
        ax.grid(True, which="both", axis="y", linestyle="-", linewidth=0.4, alpha=0.5)
        ax.legend(prop={"family": "serif", "size": 12}, loc="upper right")
        fig.tight_layout()
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved {ylabel}: {out_path}")

    if mean_rows:
        mean_df = pd.DataFrame(mean_rows)
        mean_csv = os.path.join(output_dir, "loss_curve_mean_summary.csv")
        mean_df.to_csv(mean_csv, index=False)
        print(f"[Analyze] Saved loss curve means: {mean_csv}")
        return mean_df

    return None


def infer_output_loss_path(metrics_csv_path):
    if not metrics_csv_path:
        return None

    candidates = []
    dirname = os.path.dirname(metrics_csv_path)
    basename = os.path.basename(metrics_csv_path)
    if "metrics" in basename:
        candidates.append(os.path.join(dirname, basename.replace("metrics", "loss")))
    if "metrics" in metrics_csv_path:
        candidates.append(metrics_csv_path.replace("metrics", "loss"))

    seen = set()
    for candidate in candidates:
        norm = os.path.normpath(candidate)
        if norm in seen:
            continue
        seen.add(norm)
        if os.path.isfile(norm):
            return norm

    return None


def infer_output_loss_layout(df):
    """
    Infer tail-column layout for output_loss_pred CSVs.

    Known layouts:
    - train/test loss_pred: ... total_loss, rms_spot, dist_loss, tele_loss, overlap_loss, ray_loss
    - optimization loss_pred: same tail plus efl_loss
    """
    n_cols = df.shape[1]
    if n_cols == 63:
        return {
            "total_loss": -6,
            "rms_spot": -5,
            "dist_loss": -4,
            "tele_loss": -3,
            "overlap_loss": -2,
            "ray_loss": -1,
        }
    if n_cols == 64:
        return {
            "total_loss": -7,
            "rms_spot": -6,
            "dist_loss": -5,
            "tele_loss": -4,
            "overlap_loss": -3,
            "ray_loss": -2,
            "efl_loss": -1,
        }

    raise ValueError(
        f"Unsupported output loss CSV layout with {n_cols} columns. "
        "Expected 63 (train/test) or 64 (optimization)."
    )


def analyze_output_loss_means(metrics_csv_path, output_dir):
    loss_pred_path = infer_output_loss_path(metrics_csv_path)
    if not loss_pred_path:
        print("[Analyze] No matching output_loss_pred CSV found, skip output loss means.")
        return None, None

    df = pd.read_csv(loss_pred_path, header=None)
    layout = infer_output_loss_layout(df)
    rows = []
    for loss_item, col_idx in layout.items():
        values = df.iloc[:, col_idx].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            continue
        rows.append(
            {
                "source": "output_loss_pred",
                "loss_item": loss_item,
                "mean": float(np.mean(finite_values)),
                "median": float(np.median(finite_values)),
                "min": float(np.min(finite_values)),
                "max": float(np.max(finite_values)),
                "count": int(finite_values.size),
            }
        )

    if not rows:
        print(f"[Analyze] No finite output loss values found: {loss_pred_path}")
        return None, loss_pred_path

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(output_dir, "output_loss_mean_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Analyze] Saved output loss means: {summary_csv}")
    return summary_df, loss_pred_path


def load_metrics(metrics_csv_path):
    if not os.path.isfile(metrics_csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv_path}")

    df = pd.read_csv(metrics_csv_path, header=None)
    layout = infer_metrics_layout(df)
    spot = df.iloc[:, layout["spot"]].to_numpy(dtype=float)
    dist = df.iloc[:, layout["dist"]].to_numpy(dtype=float)
    tele = df.iloc[:, layout["tele"]].to_numpy(dtype=float)
    efl_est = df.iloc[:, layout["efl_est"]].to_numpy(dtype=float)
    efl_ideal = df.iloc[:, layout["efl_ideal"]].to_numpy(dtype=float)
    efl_error = np.abs(efl_ideal - efl_est) / (np.abs(efl_ideal) + 1e-10)

    return {
        "df": df,
        "spot": spot,
        "dist": dist,
        "tele": tele,
        "efl_est": efl_est,
        "efl_ideal": efl_ideal,
        "efl_error": efl_error,
    }


def plot_scatter_plots(metrics_csv_path, output_dir, dpi=300, thresholds=None):
    font2 = {"family": "serif", "weight": "normal", "size": 15}
    thresholds = thresholds or MetricThresholds()

    os.makedirs(output_dir, exist_ok=True)
    metrics = load_metrics(metrics_csv_path)
    spot_vals = metrics["spot"]
    dist_vals = metrics["dist"]
    tele_vals = metrics["tele"]
    efl_error = metrics["efl_error"]

    scatter_specs = [
        (
            spot_vals * 1000.0,
            "RMS spot size/um",
            "cornflowerblue",
            "result_rms_scatter.png",
            "rms",
        ),
        (
            efl_error,
            "EFL Loss (relative)",
            "orange",
            "result_efl_scatter.png",
            "efl",
        ),
        (
            dist_vals,
            "Distortion",
            "mediumseagreen",
            "result_dist_scatter.png",
            "dist",
        ),
        (
            tele_vals,
            "Telecentricity/deg",
            "salmon",
            "result_tele_scatter.png",
            "tele",
        ),
    ]

    sys_nums = np.arange(1, len(spot_vals) + 1)

    for vals, ylabel, color, fname, metric_key in scatter_specs:
        axis_cfg = FIXED_AXIS_CONFIG[metric_key]
        vals = vals.astype(float)
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.scatter(sys_nums, vals, alpha=0.6, color=color, s=15, label="Network predictions")
        ax.legend(
            loc="upper left",
            fontsize=12,
            prop={"family": "serif", "size": 12},
            framealpha=0.9,
        )
        ax.set_xlabel("System number", font2)
        ax.set_ylabel(ylabel, font2)
        ax.set_xlim(0, len(vals) + 1)
        ax.set_ylim(*axis_cfg["scatter_ylim"])
        ax.set_yticks(axis_cfg["scatter_yticks"])
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontname("serif")

        if fname == "result_rms_scatter.png":
            ax_ins = ax.inset_axes([0.62, 0.66, 0.35, 0.30])
        else:
            ax_ins = inset_axes(ax, width="35%", height="30%", loc="upper right", borderpad=1.5)

        counts, _, patches = ax_ins.hist(
            vals, bins=axis_cfg["hist_bins"], color=color, alpha=0.9, edgecolor="black"
        )
        ax_ins.set_xlim(*axis_cfg["hist_xlim"])
        ax_ins.set_xticks(axis_cfg["hist_xticks"])
        y_top = max(float(counts.max()) * 1.18, 1.0)
        ax_ins.set_ylim(0, y_top)
        if fname == "result_rms_scatter.png":
            for cnt, patch in zip(counts, patches):
                x = patch.get_x() + patch.get_width() / 2
                ax_ins.text(
                    x,
                    cnt,
                    str(int(cnt)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    clip_on=True,
                )
        else:
            for cnt, patch in zip(counts, patches):
                if cnt <= 0:
                    continue
                x = patch.get_x() + patch.get_width() / 2
                ax_ins.text(
                    x,
                    cnt,
                    str(int(cnt)),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    clip_on=True,
                )
        ax_ins.set_xlabel(ylabel, fontsize=9)
        ax_ins.set_ylabel("Count", fontsize=9)
        ax_ins.tick_params(labelsize=7)

        out_path = os.path.join(output_dir, fname)
        if fname == "result_rms_scatter.png":
            fig.tight_layout()
            fig.savefig(out_path, dpi=dpi)
        else:
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"[Plot] Saved scatter: {out_path}")


def write_loss_mean_table(f, title, summary_df, source_path=None):
    if summary_df is None or summary_df.empty:
        return

    f.write(f"## {title}\n\n")
    if source_path:
        f.write(f"- Source: `{source_path}`\n\n")
    f.write("| Loss item | Mean | Median | Min | Max | Count |\n")
    f.write("|---|---:|---:|---:|---:|---:|\n")
    for row in summary_df.itertuples(index=False):
        f.write(
            f"| {row.loss_item} | {row.mean:.6g} | {row.median:.6g} | "
            f"{row.min:.6g} | {row.max:.6g} | {row.count} |\n"
        )
    f.write("\n")


def analyze_metrics_pass(
    metrics_csv_path,
    output_dir,
    thresholds=None,
    dpi=300,
    loss_curve_summary=None,
    loss_csv_path=None,
    output_loss_summary=None,
    output_loss_path=None,
):
    thresholds = thresholds or MetricThresholds()
    os.makedirs(output_dir, exist_ok=True)
    metrics = load_metrics(metrics_csv_path)
    total = len(metrics["spot"])

    checks = {
        "spot": metrics["spot"] < thresholds.spot,
        "dist": metrics["dist"] < thresholds.dist,
        "efl_error": metrics["efl_error"] < thresholds.efl_error,
        "tele": metrics["tele"] < thresholds.tele,
    }
    checks["all"] = checks["spot"] & checks["dist"] & checks["efl_error"] & checks["tele"]

    summary_rows = [
        ("RMS spot", f"< {thresholds.spot:g} mm", int(checks["spot"].sum()), total),
        ("Distortion", f"< {thresholds.dist:g}", int(checks["dist"].sum()), total),
        ("EFL error", f"< {thresholds.efl_error:g}", int(checks["efl_error"].sum()), total),
        ("Telecentricity", f"< {thresholds.tele:g} deg", int(checks["tele"].sum()), total),
        ("All metrics", "all thresholds", int(checks["all"].sum()), total),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["metric", "threshold", "pass_count", "total"])
    summary_df["pass_rate_percent"] = summary_df["pass_count"] / summary_df["total"] * 100.0

    detail_df = pd.DataFrame(
        {
            "system_number": np.arange(1, total + 1),
            "spot_mm": metrics["spot"],
            "distortion": metrics["dist"],
            "efl_error_relative": metrics["efl_error"],
            "telecentricity_deg": metrics["tele"],
            "spot_pass": checks["spot"],
            "dist_pass": checks["dist"],
            "efl_error_pass": checks["efl_error"],
            "tele_pass": checks["tele"],
            "all_pass": checks["all"],
        }
    )

    summary_csv = os.path.join(output_dir, "metrics_threshold_summary.csv")
    detail_csv = os.path.join(output_dir, "metrics_threshold_per_sample.csv")
    report_md = os.path.join(output_dir, "metrics_threshold_report.md")

    summary_df.to_csv(summary_csv, index=False)
    detail_df.to_csv(detail_csv, index=False)

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Metrics Threshold Report\n\n")
        f.write(f"- Source: `{metrics_csv_path}`\n")
        f.write(f"- Total samples: **{total}**\n\n")
        f.write("| Metric | Threshold | Pass/Total | Pass rate |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in summary_df.itertuples(index=False):
            f.write(
                f"| {row.metric} | {row.threshold} | "
                f"{row.pass_count}/{row.total} | {row.pass_rate_percent:.2f}% |\n"
            )
        f.write("\n")
        fail_count = total - int(checks["all"].sum())
        f.write(f"- Samples passing all thresholds: **{int(checks['all'].sum())}/{total}**\n")
        f.write(f"- Samples failing at least one threshold: **{fail_count}/{total}**\n")
        f.write("\n")
        write_loss_mean_table(
            f,
            "Output Loss Mean Summary",
            output_loss_summary,
            source_path=output_loss_path,
        )
        write_loss_mean_table(
            f,
            "Loss Curve Mean Summary",
            loss_curve_summary,
            source_path=loss_csv_path,
        )

    print(f"[Analyze] Saved threshold summary: {summary_csv}")
    print(f"[Analyze] Saved per-sample detail: {detail_csv}")
    print(f"[Analyze] Saved report: {report_md}")

    return summary_df, detail_df


def plot_result_summary(
    loss_csv_path,
    metrics_csv_path,
    output_dir,
    stage1_epochs=None,
    dpi=300,
    thresholds=None,
):
    os.makedirs(output_dir, exist_ok=True)
    loss_curve_summary = plot_loss_curves(
        loss_csv_path, output_dir, stage1_epochs=stage1_epochs, dpi=dpi
    )
    output_loss_summary, output_loss_path = analyze_output_loss_means(metrics_csv_path, output_dir)
    plot_scatter_plots(metrics_csv_path, output_dir, dpi=dpi, thresholds=thresholds)
    analyze_metrics_pass(
        metrics_csv_path,
        output_dir,
        thresholds=thresholds,
        dpi=dpi,
        loss_curve_summary=loss_curve_summary,
        loss_csv_path=loss_csv_path,
        output_loss_summary=output_loss_summary,
        output_loss_path=output_loss_path,
    )
    print(f"[Plot] Finished result summary: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot loss curves, metrics scatter plots, and threshold pass reports."
    )
    parser.add_argument("--output_dir", type=str, default="results_clean", help="Output directory")
    parser.add_argument("--loss_csv", type=str, default=None, help="Loss CSV path")
    parser.add_argument("--metrics_csv", type=str, default=None, help="Metrics CSV path")
    parser.add_argument("--stage1_epochs", type=int, default=None, help="Stage 1 epoch boundary")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument("--threshold_spot", type=float, default=0.05, help="Spot threshold in mm")
    parser.add_argument("--threshold_dist", type=float, default=0.02, help="Distortion threshold")
    parser.add_argument(
        "--threshold_eflerror", type=float, default=0.1, help="Relative EFL error threshold"
    )
    parser.add_argument("--threshold_tele", type=float, default=2.0, help="Telecentricity threshold")
    args = parser.parse_args()

    default_loss_csv = os.path.join(args.output_dir, "result_loss_curves.csv")
    default_metrics_csv = os.path.join(args.output_dir, "result_metrics_pred.csv")
    loss_csv = args.loss_csv if args.loss_csv is not None else default_loss_csv
    metrics_csv = args.metrics_csv if args.metrics_csv is not None else default_metrics_csv
    thresholds = MetricThresholds(
        spot=args.threshold_spot,
        dist=args.threshold_dist,
        efl_error=args.threshold_eflerror,
        tele=args.threshold_tele,
    )

    plot_result_summary(
        loss_csv,
        metrics_csv,
        args.output_dir,
        stage1_epochs=args.stage1_epochs,
        dpi=args.dpi,
        thresholds=thresholds,
    )
