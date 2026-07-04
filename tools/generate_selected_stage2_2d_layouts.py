#!/usr/bin/env python3
"""Generate 2D layout figures for the selected Stage-2 AirGap designs."""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lens_visualization.lens_visualization import LensVisualizer  # noqa: E402
from lens_visualization.lens_visualization.json_converter import convert_and_save  # noqa: E402
from web_result_page import (  # noqa: E402
    _crop_image_whitespace,
    _fix_converted_aperture_for_display,
    _unify_aperture_for_display,
)


SELECTED_DIR = (
    REPO_ROOT
    / "log"
    / "260521_1013"
    / "stage_2"
    / "airgap_unsupervised"
    / "extrapolation_tests"
    / "combined_old_dense_fulltemplate_completed_grid"
    / "stage2_airgap"
    / "selected_low_rms_external_10_designs"
)
SELECTED_CSV = SELECTED_DIR / "selected_rows.csv"
OUT_DIR = REPO_ROOT / "outputs" / "figures" / "stage2_selected_external_2d"


def _format_name_number(value: str) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _row_label(row: dict[str, str]) -> str:
    selection = int(float(row["selection"]))
    pred_row = int(float(row["pred_row"]))
    sequence = row["sequence"].split()[0]
    fn = _format_name_number(row["fn"])
    hfov = _format_name_number(row["hfov"])
    return f"sel_{selection:02d}_row{pred_row:05d}_{sequence}surf_fn{fn}_hfov{hfov}"


def _draw_layout(json_path: Path, design_dir: Path, label: str) -> Path:
    design_dir.mkdir(parents=True, exist_ok=True)
    converted_json = design_dir / f"{label}_converted.json"
    convert_and_save(str(json_path), str(converted_json), use_ray_tracing=True)
    _fix_converted_aperture_for_display(converted_json)
    _unify_aperture_for_display(converted_json)

    visualizer = LensVisualizer(str(converted_json), device="cpu")
    try:
        lens_radii = [float(s.r) for s in visualizer.lens.surfaces]
        if lens_radii:
            visualizer.lens.r_sensor = max(lens_radii) * 1.1
    except Exception:
        pass

    layout_path = design_dir / "lens_2d_layout.png"
    visualizer.draw_2d_layout(
        filename=str(layout_path),
        depth=float("inf"),
        zmx_format=True,
        lens_title="",
        show=False,
    )
    _crop_image_whitespace(layout_path)
    return layout_path


def _write_index(rows: list[dict[str, str]], image_paths: list[Path]) -> Path:
    out_md = OUT_DIR / "selected_2d_layouts.md"
    lines = [
        "# Selected Stage-2 AirGap 2D Layouts",
        "",
        f"Selected rows: `{SELECTED_CSV}`",
        "",
        "| No. | Pred row | Sequence | F# | HFOV | RMS | 2D layout |",
        "|---:|---:|---|---:|---:|---:|---|",
    ]
    for row, img_path in zip(rows, image_paths):
        rel_img = img_path.relative_to(OUT_DIR).as_posix()
        lines.append(
            "| "
            f"{int(float(row['selection']))} | "
            f"{int(float(row['pred_row']))} | "
            f"{row['sequence']} | "
            f"{float(row['fn']):.6g} | "
            f"{float(row['hfov']):.6g} | "
            f"{float(row['rms']):.6g} | "
            f"`{rel_img}` |"
        )

    lines.append("")
    for row, img_path in zip(rows, image_paths):
        rel_img = img_path.relative_to(OUT_DIR).as_posix()
        lines.extend(
            [
                f"## Selection {int(float(row['selection']))}: row {int(float(row['pred_row']))}",
                "",
                f"![2D layout]({rel_img})",
                "",
            ]
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out_md


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with SELECTED_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows found in {SELECTED_CSV}")

    image_paths: list[Path] = []
    for row in rows:
        label = _row_label(row)
        json_path = Path(row["json"])
        if not json_path.exists():
            raise FileNotFoundError(json_path)

        design_dir = OUT_DIR / label
        layout_path = _draw_layout(json_path, design_dir, label)
        final_path = OUT_DIR / f"{label}_2d_layout.png"
        shutil.copy2(layout_path, final_path)
        image_paths.append(final_path)
        print(f"Saved 2D layout: {final_path}")

    index_md = _write_index(rows, image_paths)
    print(f"Index: {index_md}")


if __name__ == "__main__":
    main()
