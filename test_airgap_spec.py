#!/usr/bin/env python3
"""Build and test a single-(F#, HFOV) AirGap test set.

The script copies material-index/type candidates for one system specification
from an existing raw test CSV, writes a dedicated raw test CSV, and then runs
Test_Model.py with the AirGapUnsupervised checkpoint.
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
DEFAULT_AIRGAP_CKPT = (
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
    / "spec_tests"
)


def _float_tag(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _load_csv(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _available_specs(data: np.ndarray) -> pd.DataFrame:
    specs, counts = np.unique(data[:, :2], axis=0, return_counts=True)
    return pd.DataFrame(
        {
            "fn": specs[:, 0],
            "hfov": specs[:, 1],
            "count": counts,
        }
    ).sort_values(["fn", "hfov"], ignore_index=True)


def _nearest_spec(data: np.ndarray, fn: float, hfov: float) -> tuple[float, float]:
    specs = np.unique(data[:, :2], axis=0)
    scale = np.ptp(specs, axis=0)
    scale[scale == 0.0] = 1.0
    dist = np.linalg.norm((specs - np.array([[fn, hfov]], dtype=float)) / scale, axis=1)
    best = specs[int(np.argmin(dist))]
    return float(best[0]), float(best[1])


def build_spec_csv(
    reference_csv: Path,
    fn: float,
    hfov: float,
    output_csv: Path,
    *,
    tol: float = 1e-6,
    max_rows: int = 0,
    allow_nearest: bool = False,
) -> pd.DataFrame:
    data = _load_csv(reference_csv)
    if data.shape[1] < 3 + 11 * 3 + 11:
        raise ValueError(
            f"Reference CSV has {data.shape[1]} columns; expected raw split layout "
            "[fn, hfov, seq_len, 33 refractive-index values, 11 type values]."
        )

    requested_fn = float(fn)
    requested_hfov = float(hfov)
    template_fn = requested_fn
    template_hfov = requested_hfov
    selection_mode = "exact"
    mask = (np.abs(data[:, 0] - template_fn) <= tol) & (np.abs(data[:, 1] - template_hfov) <= tol)

    if not np.any(mask) and allow_nearest:
        template_fn, template_hfov = _nearest_spec(data, requested_fn, requested_hfov)
        selection_mode = "nearest_template"
        mask = (np.abs(data[:, 0] - template_fn) <= tol) & (
            np.abs(data[:, 1] - template_hfov) <= tol
        )

    if not np.any(mask):
        spec_table = _available_specs(data)
        available_csv = output_csv.with_name(output_csv.stem + "_available_specs.csv")
        available_csv.parent.mkdir(parents=True, exist_ok=True)
        spec_table.to_csv(available_csv, index=False, encoding="utf-8")
        raise ValueError(
            f"No rows found for fn={fn:g}, hfov={hfov:g} in {reference_csv}. "
            f"Available specifications were written to {available_csv}."
        )

    source_indices = np.flatnonzero(mask)
    selected = data[source_indices].copy()
    if max_rows and int(max_rows) > 0:
        selected = selected[: int(max_rows)]
        source_indices = source_indices[: int(max_rows)]
    if selection_mode == "nearest_template":
        selected[:, 0] = requested_fn
        selected[:, 1] = requested_hfov

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selected).to_csv(output_csv, header=False, index=False, encoding="utf-8")

    manifest = pd.DataFrame(
        {
            "source_csv": str(reference_csv),
            "source_row_idx": source_indices,
            "selection_mode": selection_mode,
            "requested_fn": requested_fn,
            "requested_hfov": requested_hfov,
            "template_fn": template_fn,
            "template_hfov": template_hfov,
            "used_fn": selected[:, 0],
            "used_hfov": selected[:, 1],
            "seq_length": selected[:, 2].astype(int),
        }
    )
    manifest.to_csv(output_csv.with_name(output_csv.stem + "_manifest.csv"), index=False, encoding="utf-8")
    return manifest


def run_airgap_test(
    test_csv: Path,
    output_dir: Path,
    checkpoint: Path,
    *,
    python_exe: str,
    batch_size: int,
    lightweight: bool,
    keep_export: bool,
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
    ]
    if not keep_export:
        cmd.append("--no_export_zmx_json")
    if disable_rms_filter:
        cmd.append("--disable_rms_filter")
    if disable_efl_filter:
        cmd.append("--disable_efl_filter")
    cmd.extend(extra_args)

    print("[SpecTest] Running AirGap test:", flush=True)
    print(
        "  " + " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd),
        flush=True,
    )
    print(f"[SpecTest] SCANLENS_TEST_CSV={test_csv}", flush=True)
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a dedicated raw test set for one (F#, HFOV) pair and run "
            "the second-stage AirGap unsupervised checkpoint on it."
        )
    )
    parser.add_argument("--fn", type=float, required=True, help="Target F-number.")
    parser.add_argument("--hfov", type=float, required=True, help="Target half field of view.")
    parser.add_argument("--reference_csv", type=Path, default=DEFAULT_REFERENCE_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_AIRGAP_CKPT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument("--tol", type=float, default=1e-6, help="Exact-match tolerance.")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap for quick tests.")
    parser.add_argument(
        "--allow_nearest",
        action="store_true",
        help=(
            "If the exact pair is absent, borrow material/type templates from the "
            "nearest available specification and overwrite fn/hfov with the requested pair."
        ),
    )
    parser.add_argument(
        "--build_only",
        action="store_true",
        help="Only write the dedicated test CSV; do not run Test_Model.py.",
    )
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--lightweight",
        action="store_true",
        help="Run Test_Model.py with SCANLENS_TEST_LIGHTWEIGHT=1.",
    )
    parser.add_argument(
        "--keep_export",
        action="store_true",
        help="Keep ZMX/JSON export enabled. By default it is disabled for spec tests.",
    )
    parser.add_argument("--disable_rms_filter", action="store_true")
    parser.add_argument("--disable_efl_filter", action="store_true")
    parser.add_argument(
        "extra_test_args",
        nargs=argparse.REMAINDER,
        help="Extra arguments passed to Test_Model.py after a standalone '--'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_tag = f"fn{_float_tag(args.fn)}_hfov{_float_tag(args.hfov)}"
    output_dir = args.output_dir or (args.output_root / spec_tag)
    output_csv = args.output_csv or (output_dir / f"custom_test_input_{spec_tag}.csv")

    reference_csv = args.reference_csv.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = output_dir.resolve()
    output_csv = output_csv.resolve()

    if not reference_csv.exists():
        raise FileNotFoundError(f"Reference CSV not found: {reference_csv}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"AirGap checkpoint not found: {checkpoint}")

    manifest = build_spec_csv(
        reference_csv,
        args.fn,
        args.hfov,
        output_csv,
        tol=args.tol,
        max_rows=args.max_rows,
        allow_nearest=args.allow_nearest,
    )
    used_specs = manifest[["used_fn", "used_hfov"]].drop_duplicates()
    print(f"[SpecTest] Wrote dedicated test CSV: {output_csv}", flush=True)
    print(f"[SpecTest] Rows: {len(manifest)}", flush=True)
    for row in used_specs.itertuples(index=False):
        print(f"[SpecTest] Used specification: fn={row.used_fn:g}, hfov={row.used_hfov:g}", flush=True)

    if args.build_only:
        return

    extra_args = list(args.extra_test_args)
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    run_airgap_test(
        output_csv,
        output_dir,
        checkpoint,
        python_exe=args.python_exe,
        batch_size=args.batch_size,
        lightweight=args.lightweight,
        keep_export=args.keep_export,
        disable_rms_filter=args.disable_rms_filter,
        disable_efl_filter=args.disable_efl_filter,
        extra_args=extra_args,
    )

    print("[SpecTest] Done.", flush=True)
    print(f"[SpecTest] Output directory: {output_dir}", flush=True)
    print(f"[SpecTest] Metrics CSV: {output_dir / 'test_output_metrics_pred_rmsfilter_on.csv'}", flush=True)
    print(f"[SpecTest] Loss CSV: {output_dir / 'test_output_loss_pred_rmsfilter_on.csv'}", flush=True)


if __name__ == "__main__":
    main()
