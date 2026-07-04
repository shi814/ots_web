#!/usr/bin/env python3
"""Select low-RMS external Stage-2 AirGap designs and export Markdown/ZMX files."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exports.scanlens_export_from_csv import export_from_config  # noqa: E402
from match_gt_report import (  # noqa: E402
    LensRow,
    _format_compact_row,
    _load_pred_rows,
    _write_compact_table_header,
)


PRED_CSV = (
    REPO_ROOT
    / "log"
    / "260521_1013"
    / "stage_2"
    / "airgap_unsupervised"
    / "extrapolation_tests"
    / "combined_old_dense_fulltemplate_completed_grid"
    / "stage2_airgap"
    / "test_output_metrics_pred_rmsfilter_on_completed_eflerror_lt_0p1.csv"
)
OUT_DIR = (
    PRED_CSV.parent
    / "selected_low_rms_external_10_designs"
)

RMS_MAX = 0.05
DIST_MAX = 0.02
TELE_MAX = 2.0
EFL_ERROR_MAX = 0.1
PER_SURFACE_COUNT = 5


def _is_finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def _passes_metrics(row: LensRow) -> bool:
    return (
        _is_finite(row.rms)
        and _is_finite(row.dist)
        and _is_finite(row.tele)
        and _is_finite(row.loss_efl)
        and float(row.rms) < RMS_MAX
        and abs(float(row.dist)) < DIST_MAX
        and abs(float(row.tele)) < TELE_MAX
        and float(row.loss_efl) < EFL_ERROR_MAX
    )


def _is_external_spec(row: LensRow) -> bool:
    # Original scan-lens specifications covered F# >= 9.75 and HFOV >= 8.5 deg.
    return float(row.fn) < 9.75 or float(row.hfov) < 8.5


def _spec_key(row: LensRow) -> tuple[float, float]:
    return (round(float(row.fn), 6), round(float(row.hfov), 6))


def _sort_key(row: LensRow) -> tuple[float, int]:
    rms = float(row.rms) if _is_finite(row.rms) else float("inf")
    return (rms, int(row.row_idx))


def _select_rows(rows: list[LensRow]) -> list[LensRow]:
    selected: list[LensRow] = []
    used_specs: set[tuple[float, float]] = set()

    for n_surf in (9, 11):
        group = sorted(
            [
                row
                for row in rows
                if row.n_surf == n_surf and _passes_metrics(row) and _is_external_spec(row)
            ],
            key=_sort_key,
        )
        kept = 0
        for row in group:
            key = _spec_key(row)
            if key in used_specs:
                continue
            selected.append(row)
            used_specs.add(key)
            kept += 1
            if kept >= PER_SURFACE_COUNT:
                break
        if kept != PER_SURFACE_COUNT:
            raise RuntimeError(f"Only selected {kept} rows for n_surf={n_surf}.")

    return selected


def _fmt_float_for_name(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _zmx_basename(order: int, row: LensRow) -> str:
    total_surfaces = row.total_surface_count
    fn = _fmt_float_for_name(row.fn)
    hfov = _fmt_float_for_name(row.hfov)
    return f"stage2_sel_{order:02d}_row{row.row_idx:05d}_{total_surfaces}surf_fn{fn}_hfov{hfov}"


def _export_rows(rows: list[LensRow], zmx_dir: Path) -> dict[int, tuple[Path, Path]]:
    zmx_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[int, tuple[Path, Path]] = {}
    for order, row in enumerate(rows, start=1):
        json_path, zmx_path = export_from_config(
            {
                "format": "pred",
                "csv": str(PRED_CSV),
                "row": int(row.row_idx),
                "n_surf": 0,
                "offset_ct": None,
                "ct_is_radius": False,
                "epd": 4.5,
                "aperture_mode": "fnum",
                "semi_diam": 50.0,
                "semi_diam_stop": None,
                "r_sensor": 20.0,
                "glass_matching_csv": "",
                "out_dir": str(zmx_dir),
                "basename": _zmx_basename(order, row),
            }
        )
        exported[int(row.row_idx)] = (Path(json_path), Path(zmx_path))
    return exported


def _write_selected_csv(rows: list[LensRow], exported: dict[int, tuple[Path, Path]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "selection",
                "pred_row",
                "sequence",
                "n_surf",
                "fn",
                "hfov",
                "rms",
                "dist",
                "tele",
                "efl_error",
                "zmx",
                "json",
            ]
        )
        for order, row in enumerate(rows, start=1):
            json_path, zmx_path = exported[int(row.row_idx)]
            writer.writerow(
                [
                    order,
                    int(row.row_idx),
                    f"{row.total_surface_count} surfaces",
                    int(row.n_surf),
                    float(row.fn),
                    float(row.hfov),
                    float(row.rms),
                    float(row.dist),
                    float(row.tele),
                    float(row.loss_efl),
                    str(zmx_path),
                    str(json_path),
                ]
            )


def _write_report(rows: list[LensRow], exported: dict[int, tuple[Path, Path]], out_md: Path) -> None:
    lines: list[str] = [
        "# Stage-2 AirGap Low-RMS External Designs",
        "",
        f"Prediction CSV: `{PRED_CSV}`",
        "",
        "Selection rules:",
        "",
        f"- Source: latest Stage-2 AirGap filtered results, EFL error < `{EFL_ERROR_MAX}`.",
        "- External specification point: `F# < 9.75` or `HFOV < 8.5 deg`.",
        f"- Metric constraints: RMS < `{RMS_MAX}`, distortion < `{DIST_MAX}`, telecentricity < `{TELE_MAX}`.",
        "- Keep different `(F#, HFOV)` pairs.",
        f"- Keep `{PER_SURFACE_COUNT}` 10-surface systems and `{PER_SURFACE_COUNT}` 12-surface systems.",
        "",
        "## Summary",
        "",
        "| No. | Pred row | Sequence | F# | HFOV | RMS | dist | tele | EFL error | ZMX |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for order, row in enumerate(rows, start=1):
        _, zmx_path = exported[int(row.row_idx)]
        rel_zmx = zmx_path.relative_to(OUT_DIR)
        lines.append(
            "| "
            f"{order} | {row.row_idx} | {row.total_surface_count} surfaces | "
            f"{row.fn:.6g} | {row.hfov:.6g} | {row.rms:.6g} | {row.dist:.6g} | "
            f"{row.tele:.6g} | {row.loss_efl:.6g} | `{rel_zmx.as_posix()}` |"
        )
    lines.append("")

    for n_surf, title in ((9, "Selected 10-surface Designs"), (11, "Selected 12-surface Designs")):
        lines.extend([f"## {title}", ""])
        group = [row for row in rows if row.n_surf == n_surf]
        for local_idx, row in enumerate(group, start=1):
            json_path, zmx_path = exported[int(row.row_idx)]
            lines.extend(
                [
                    f"### Design {local_idx}",
                    "",
                    f"- Pred row: `{row.row_idx}`",
                    f"- ZMX: `{zmx_path.relative_to(OUT_DIR).as_posix()}`",
                    f"- JSON: `{json_path.relative_to(OUT_DIR).as_posix()}`",
                    "",
                ]
            )
            _write_compact_table_header(lines)
            lines.append(_format_compact_row("pred", row))
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = _load_pred_rows(str(PRED_CSV))
    selected = _select_rows(rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exported = _export_rows(selected, OUT_DIR / "zmx")
    _write_selected_csv(selected, exported, OUT_DIR / "selected_rows.csv")
    _write_report(selected, exported, OUT_DIR / "selected_low_rms_external_report.md")

    print(f"Selected rows: {[row.row_idx for row in selected]}")
    print(f"Report: {OUT_DIR / 'selected_low_rms_external_report.md'}")
    print(f"ZMX dir: {OUT_DIR / 'zmx'}")


if __name__ == "__main__":
    main()
