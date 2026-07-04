#!/usr/bin/env python3
"""Batch-test fully extrapolated ScanLens specifications.

The generated raw test set uses legal OTS material/type templates from the
nearest in-distribution specifications, while replacing F# and HFOV with
requested out-of-range values. It then runs the same Test_Model.py entry point
for stage 1 and stage 2 AirGap checkpoints and writes comparison summaries.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REFERENCE_CSV = PROJECT_ROOT / "data" / "scan_lens_test_ul_20260512.csv"
DEFAULT_RANGE_CSV = PROJECT_ROOT / "data" / "surf10_12_ul_1104.csv"
DEFAULT_STAGE1_CKPT = (
    PROJECT_ROOT
    / "log"
    / "260521_1013"
    / "stage_1"
    / "checkpoints"
    / "SLT_rmsfilter_on_epoch5000_bs512.pth"
)
DEFAULT_STAGE2_CKPT = (
    PROJECT_ROOT
    / "log"
    / "260521_1013"
    / "stage_2"
    / "airgap_unsupervised"
    / "checkpoints"
    / "AirGapUnsupervised_final.pth"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "log"
    / "260521_1013"
    / "stage_2"
    / "airgap_unsupervised"
    / "extrapolation_tests"
)


def _float_tag(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _parse_values(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _spec_range(range_csv: Path) -> tuple[float, float, float, float]:
    data = _load_csv(range_csv)
    return (
        float(np.min(data[:, 0])),
        float(np.max(data[:, 0])),
        float(np.min(data[:, 1])),
        float(np.max(data[:, 1])),
    )


def _default_corner_values(range_csv: Path) -> tuple[list[float], list[float]]:
    fn_min, fn_max, hfov_min, hfov_max = _spec_range(range_csv)
    fn_span = max(fn_max - fn_min, 1.0)
    hfov_span = max(hfov_max - hfov_min, 1.0)
    fn_margin = max(0.15, 0.04 * fn_span)
    hfov_margin = max(0.2, 0.12 * hfov_span)
    return (
        [round(fn_min - fn_margin, 4), round(fn_max + fn_margin, 4)],
        [round(hfov_min - hfov_margin, 4), round(hfov_max + hfov_margin, 4)],
    )


def _nearest_spec(specs: np.ndarray, fn: float, hfov: float) -> tuple[float, float]:
    scale = np.ptp(specs, axis=0)
    scale[scale == 0.0] = 1.0
    dist = np.linalg.norm((specs - np.array([[fn, hfov]], dtype=float)) / scale, axis=1)
    best = specs[int(np.argmin(dist))]
    return float(best[0]), float(best[1])


def _extrapolation_kind(
    fn: float,
    hfov: float,
    fn_min: float,
    fn_max: float,
    hfov_min: float,
    hfov_max: float,
) -> str:
    fn_out = fn < fn_min or fn > fn_max
    hfov_out = hfov < hfov_min or hfov > hfov_max
    if fn_out and hfov_out:
        return "corner_extrapolation"
    if fn_out:
        return "fn_only_extrapolation"
    if hfov_out:
        return "hfov_only_extrapolation"
    return "interpolation"


def build_extrapolation_csv(
    reference_csv: Path,
    range_csv: Path,
    fn_values: list[float],
    hfov_values: list[float],
    output_csv: Path,
    *,
    rows_per_spec: int = 0,
    tol: float = 1e-6,
) -> pd.DataFrame:
    reference = _load_csv(reference_csv)
    if reference.shape[1] < 47:
        raise ValueError(f"Reference CSV should use the raw test layout, got {reference.shape[1]} columns.")

    fn_min, fn_max, hfov_min, hfov_max = _spec_range(range_csv)
    specs = np.unique(reference[:, :2], axis=0)

    out_rows = []
    manifest_rows = []
    for fn in fn_values:
        for hfov in hfov_values:
            template_fn, template_hfov = _nearest_spec(specs, fn, hfov)
            mask = (np.abs(reference[:, 0] - template_fn) <= tol) & (
                np.abs(reference[:, 1] - template_hfov) <= tol
            )
            source_indices = np.flatnonzero(mask)
            rows = reference[source_indices].copy()
            if rows_per_spec and int(rows_per_spec) > 0:
                rows = rows[: int(rows_per_spec)]
                source_indices = source_indices[: int(rows_per_spec)]

            rows[:, 0] = float(fn)
            rows[:, 1] = float(hfov)
            out_rows.append(rows)

            kind = _extrapolation_kind(fn, hfov, fn_min, fn_max, hfov_min, hfov_max)
            for source_idx, row in zip(source_indices, rows):
                manifest_rows.append(
                    {
                        "source_csv": str(reference_csv),
                        "source_row_idx": int(source_idx),
                        "requested_fn": float(fn),
                        "requested_hfov": float(hfov),
                        "template_fn": template_fn,
                        "template_hfov": template_hfov,
                        "used_fn": float(row[0]),
                        "used_hfov": float(row[1]),
                        "seq_length": int(row[2]),
                        "extrapolation_kind": kind,
                    }
                )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output = np.concatenate(out_rows, axis=0)
    pd.DataFrame(output).to_csv(output_csv, header=False, index=False, encoding="utf-8")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_csv.with_name(output_csv.stem + "_manifest.csv"), index=False, encoding="utf-8")
    return manifest


def _run_test_model(
    *,
    test_csv: Path,
    checkpoint: Path,
    output_dir: Path,
    python_exe: str,
    batch_size: int,
    lightweight: bool,
    disable_rms_filter: bool,
    disable_efl_filter: bool,
    extra_args: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SCANLENS_TEST_CSV"] = str(test_csv)
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if lightweight:
        env["SCANLENS_TEST_LIGHTWEIGHT"] = "1"

    cmd = [
        python_exe,
        str(PROJECT_ROOT / "Test_Model.py"),
        "--load_name",
        str(checkpoint),
        "--save_path",
        str(output_dir),
        "--batch_size",
        str(int(batch_size)),
        "--no_export_zmx_json",
    ]
    if disable_rms_filter:
        cmd.append("--disable_rms_filter")
    if disable_efl_filter:
        cmd.append("--disable_efl_filter")
    cmd.extend(extra_args)

    print(f"[Extrapolation] Run: {output_dir.name}", flush=True)
    print("  " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


def _summarize_metrics(metrics_csv: Path, stage: str, efl_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")
    df = pd.read_csv(metrics_csv, header=None)
    if df.empty:
        columns = [
            "stage",
            "fn",
            "hfov",
            "rows_after_hard_filter",
            "rows_efl_lt_threshold",
            "mean_rms_mm",
            "median_rms_mm",
            "min_rms_mm",
            "mean_efl_error",
            "pass_all_count",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame()

    summary_rows = []
    detail = pd.DataFrame(
        {
            "stage": stage,
            "fn": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
            "hfov": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            "composite": pd.to_numeric(df.iloc[:, -8], errors="coerce"),
            "rms_mm": pd.to_numeric(df.iloc[:, -7], errors="coerce"),
            "distortion": pd.to_numeric(df.iloc[:, -6], errors="coerce"),
            "tele_deg": pd.to_numeric(df.iloc[:, -5], errors="coerce"),
            "overlap_loss": pd.to_numeric(df.iloc[:, -4], errors="coerce"),
            "ray_loss": pd.to_numeric(df.iloc[:, -3], errors="coerce"),
            "efl_est": pd.to_numeric(df.iloc[:, -2], errors="coerce"),
            "efl_ideal": pd.to_numeric(df.iloc[:, -1], errors="coerce"),
        }
    )
    detail["efl_error"] = (detail["efl_est"] - detail["efl_ideal"]).abs() / (
        detail["efl_ideal"].abs() + 1e-10
    )
    detail["efl_lt_threshold"] = detail["efl_error"] < float(efl_threshold)
    detail["pass_all"] = (
        (detail["rms_mm"] < 0.05)
        & (detail["distortion"] < 0.02)
        & (detail["tele_deg"] < 2.0)
        & detail["efl_lt_threshold"]
    )

    for (fn, hfov), group in detail.groupby(["fn", "hfov"], dropna=False):
        efl_group = group[group["efl_lt_threshold"]]
        metric_group = efl_group if not efl_group.empty else group
        summary_rows.append(
            {
                "stage": stage,
                "fn": float(fn),
                "hfov": float(hfov),
                "rows_after_hard_filter": int(len(group)),
                "rows_efl_lt_threshold": int(len(efl_group)),
                "mean_rms_mm": float(metric_group["rms_mm"].mean()) if not metric_group.empty else np.nan,
                "median_rms_mm": float(metric_group["rms_mm"].median()) if not metric_group.empty else np.nan,
                "min_rms_mm": float(metric_group["rms_mm"].min()) if not metric_group.empty else np.nan,
                "mean_efl_error": float(metric_group["efl_error"].mean()) if not metric_group.empty else np.nan,
                "pass_all_count": int(group["pass_all"].sum()),
            }
        )

    return pd.DataFrame(summary_rows), detail


def write_comparison(stage_dirs: dict[str, Path], output_dir: Path, efl_threshold: float) -> None:
    summary_frames = []
    detail_frames = []
    for stage, stage_dir in stage_dirs.items():
        metrics_csv = stage_dir / "test_output_metrics_pred_rmsfilter_on.csv"
        summary, detail = _summarize_metrics(metrics_csv, stage, efl_threshold)
        summary_frames.append(summary)
        detail_frames.append(detail)

    summary_df = pd.concat(summary_frames, ignore_index=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_csv = output_dir / "extrapolation_comparison_summary.csv"
    detail_csv = output_dir / "extrapolation_comparison_detail.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    detail_df.to_csv(detail_csv, index=False, encoding="utf-8")

    pivot = summary_df.pivot_table(
        index=["fn", "hfov"],
        columns="stage",
        values=["rows_efl_lt_threshold", "mean_rms_mm", "pass_all_count"],
        aggfunc="first",
    )
    pivot_csv = output_dir / "extrapolation_stage1_vs_stage2_pivot.csv"
    pivot.to_csv(pivot_csv, encoding="utf-8")
    print(f"[Extrapolation] Summary: {summary_csv}", flush=True)
    print(f"[Extrapolation] Detail:  {detail_csv}", flush=True)
    print(f"[Extrapolation] Pivot:   {pivot_csv}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-test fully extrapolated specifications.")
    parser.add_argument("--reference_csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--range_csv", type=Path, default=DEFAULT_RANGE_CSV)
    parser.add_argument("--stage1_checkpoint", type=Path, default=DEFAULT_STAGE1_CKPT)
    parser.add_argument("--stage2_checkpoint", type=Path, default=DEFAULT_STAGE2_CKPT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=None,
        help="Optional direct path for the generated raw test CSV.",
    )
    parser.add_argument("--run_name", default="corner_default")
    parser.add_argument(
        "--fn_values",
        default="",
        help="Comma-separated F# values. Default uses low/high values outside the data range.",
    )
    parser.add_argument(
        "--hfov_values",
        default="",
        help="Comma-separated HFOV values. Default uses low/high values outside the data range.",
    )
    parser.add_argument("--rows_per_spec", type=int, default=0, help="0 keeps all nearest-template rows.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--efl_threshold", type=float, default=0.1)
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only build the raw test CSV and manifest; do not run stage tests.",
    )
    parser.add_argument(
        "--full_analysis",
        action="store_true",
        help="Run Test_Model.py full analysis. Default is lightweight plus custom comparison CSVs.",
    )
    parser.add_argument("--stage", choices=["both", "stage1", "stage2"], default="both")
    parser.add_argument("--disable_rms_filter", action="store_true")
    parser.add_argument("--disable_efl_filter", action="store_true")
    parser.add_argument("extra_test_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reference_csv = args.reference_csv.resolve()
    range_csv = args.range_csv.resolve()
    stage1_ckpt = args.stage1_checkpoint.resolve()
    stage2_ckpt = args.stage2_checkpoint.resolve()
    output_dir = (args.output_root / args.run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")
    if not range_csv.exists():
        raise FileNotFoundError(f"Range CSV not found: {range_csv}")

    if args.fn_values:
        fn_values = _parse_values(args.fn_values)
    else:
        fn_values, _default_hfov_values = _default_corner_values(range_csv)
    if args.hfov_values:
        hfov_values = _parse_values(args.hfov_values)
    else:
        _default_fn_values, hfov_values = _default_corner_values(range_csv)

    spec_tag = "fn" + "-".join(_float_tag(v) for v in fn_values)
    spec_tag += "_hfov" + "-".join(_float_tag(v) for v in hfov_values)
    test_csv = args.output_csv.resolve() if args.output_csv else output_dir / f"extrapolation_input_{spec_tag}.csv"
    manifest = build_extrapolation_csv(
        reference_csv,
        range_csv,
        fn_values,
        hfov_values,
        test_csv,
        rows_per_spec=args.rows_per_spec,
    )
    print(f"[Extrapolation] Input CSV: {test_csv}", flush=True)
    print(f"[Extrapolation] Rows: {len(manifest)}", flush=True)
    print(
        "[Extrapolation] Specs: "
        + ", ".join(f"({fn:g}, {hfov:g})" for fn in fn_values for hfov in hfov_values),
        flush=True,
    )

    if args.build_only:
        print("[Extrapolation] Build only: skip model tests.", flush=True)
        return

    extra_args = list(args.extra_test_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    stage_dirs = {}
    lightweight = not args.full_analysis
    if args.stage in ("both", "stage1"):
        if not stage1_ckpt.exists():
            raise FileNotFoundError(f"Stage-1 checkpoint not found: {stage1_ckpt}")
        stage_dir = output_dir / "stage1"
        _run_test_model(
            test_csv=test_csv,
            checkpoint=stage1_ckpt,
            output_dir=stage_dir,
            python_exe=args.python_exe,
            batch_size=args.batch_size,
            lightweight=lightweight,
            disable_rms_filter=args.disable_rms_filter,
            disable_efl_filter=args.disable_efl_filter,
            extra_args=extra_args,
        )
        stage_dirs["stage1"] = stage_dir

    if args.stage in ("both", "stage2"):
        if not stage2_ckpt.exists():
            raise FileNotFoundError(f"Stage-2 checkpoint not found: {stage2_ckpt}")
        stage_dir = output_dir / "stage2_airgap"
        _run_test_model(
            test_csv=test_csv,
            checkpoint=stage2_ckpt,
            output_dir=stage_dir,
            python_exe=args.python_exe,
            batch_size=args.batch_size,
            lightweight=lightweight,
            disable_rms_filter=args.disable_rms_filter,
            disable_efl_filter=args.disable_efl_filter,
            extra_args=extra_args,
        )
        stage_dirs["stage2_airgap"] = stage_dir

    write_comparison(stage_dirs, output_dir, efl_threshold=float(args.efl_threshold))


if __name__ == "__main__":
    main()
