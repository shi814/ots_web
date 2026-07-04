"""Merge old+dense combined results with missing-grid fill results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = (
    PROJECT_ROOT
    / "log/260521_1013/stage_2/airgap_unsupervised/extrapolation_tests"
)
PREV_ROOT = BASE_ROOT / "combined_old_full_and_dense_fulltemplate"
FILL_ROOT = BASE_ROOT / "missing_grid_fill_fulltemplate_5x5"
OUT_ROOT = BASE_ROOT / "combined_old_dense_fulltemplate_completed_grid"


def efl_mask(metrics: pd.DataFrame, threshold: float = 0.1) -> np.ndarray:
    efl_est = pd.to_numeric(metrics.iloc[:, -2], errors="coerce").to_numpy(dtype=float)
    efl_ideal = pd.to_numeric(metrics.iloc[:, -1], errors="coerce").to_numpy(dtype=float)
    return np.abs(efl_est - efl_ideal) / (np.abs(efl_ideal) + 1e-10) < float(threshold)


def merge_stage(stage: str, fill_stage: str, out_stage: str) -> dict[str, object]:
    out_dir = OUT_ROOT / out_stage
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_metrics_path = PREV_ROOT / stage / "test_output_metrics_pred_rmsfilter_on_combined.csv"
    prev_loss_path = PREV_ROOT / stage / "test_output_loss_pred_rmsfilter_on_combined.csv"
    fill_metrics_path = FILL_ROOT / fill_stage / "test_output_metrics_pred_rmsfilter_on.csv"
    fill_loss_path = FILL_ROOT / fill_stage / "test_output_loss_pred_rmsfilter_on.csv"

    prev_metrics = pd.read_csv(prev_metrics_path, header=None)
    prev_loss = pd.read_csv(prev_loss_path, header=None)
    fill_metrics = pd.read_csv(fill_metrics_path, header=None)
    fill_loss = pd.read_csv(fill_loss_path, header=None)

    if len(prev_metrics) != len(prev_loss):
        raise ValueError(f"Previous metrics/loss row mismatch for {stage}")
    if len(fill_metrics) != len(fill_loss):
        raise ValueError(f"Fill metrics/loss row mismatch for {fill_stage}")

    metrics = pd.concat([prev_metrics, fill_metrics], ignore_index=True)
    loss = pd.concat([prev_loss, fill_loss], ignore_index=True)
    mask = efl_mask(metrics)
    metrics_filtered = metrics.loc[mask].reset_index(drop=True)
    loss_filtered = loss.loc[mask].reset_index(drop=True)

    metrics_path = out_dir / "test_output_metrics_pred_rmsfilter_on_completed.csv"
    loss_path = out_dir / "test_output_loss_pred_rmsfilter_on_completed.csv"
    metrics_filtered_path = out_dir / "test_output_metrics_pred_rmsfilter_on_completed_eflerror_lt_0p1.csv"
    loss_filtered_path = out_dir / "test_output_loss_pred_rmsfilter_on_completed_eflerror_lt_0p1.csv"
    source_index_path = out_dir / "completed_source_index.csv"

    metrics.to_csv(metrics_path, header=False, index=False, encoding="utf-8")
    loss.to_csv(loss_path, header=False, index=False, encoding="utf-8")
    metrics_filtered.to_csv(metrics_filtered_path, header=False, index=False, encoding="utf-8")
    loss_filtered.to_csv(loss_filtered_path, header=False, index=False, encoding="utf-8")

    source_index = pd.DataFrame(
        {
            "source_split": ["old_dense_combined"] * len(prev_metrics)
            + ["missing_grid_fill"] * len(fill_metrics),
            "source_row": list(range(len(prev_metrics))) + list(range(len(fill_metrics))),
            "kept_eflerror_lt_0p1": mask,
        }
    )
    source_index.to_csv(source_index_path, index=False, encoding="utf-8")

    return {
        "stage": out_stage,
        "previous_rows_before_efl_filter": len(prev_metrics),
        "fill_rows_before_efl_filter": len(fill_metrics),
        "completed_rows_before_efl_filter": len(metrics),
        "completed_rows_eflerror_lt_0p1": int(mask.sum()),
        "efl_pass_rate_percent": float(mask.mean() * 100.0),
        "output_dir": str(out_dir),
    }


def main() -> None:
    rows = [
        merge_stage("stage1", "stage1", "stage1"),
        merge_stage("stage2_airgap", "stage2_airgap", "stage2_airgap"),
    ]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary_path = OUT_ROOT / "completed_merge_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
