from __future__ import annotations

from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from exports.scanlens_export_from_csv import export_from_config
from web_result_page import DEFAULT_EXPORT_EPD, generate_visuals

OUT_ROOT = ROOT / "outputs" / "figures" / "requested_2d_layouts"


def export_and_draw(
    *,
    label: str,
    fmt: str,
    csv_path: Path,
    row_zero_based: int,
    n_surf: int,
    ct_is_radius: bool,
    glass_matching_csv: Path | None = None,
) -> Path:
    out_dir = OUT_ROOT / label
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, _zmx_path = export_from_config(
        {
            "format": fmt,
            "csv": str(csv_path),
            "row": int(row_zero_based),
            "n_surf": int(n_surf),
            "offset_ct": None,
            "ct_is_radius": bool(ct_is_radius),
            "epd": DEFAULT_EXPORT_EPD,
            "aperture_mode": "fnum",
            "semi_diam": 50.0,
            "semi_diam_stop": None,
            "r_sensor": 20.0,
            "glass_matching_csv": str(glass_matching_csv) if glass_matching_csv else "",
            "out_dir": str(out_dir),
            "basename": label,
        }
    )
    visuals = generate_visuals(
        json_path,
        out_dir,
        metrics_csv=csv_path if fmt == "pred" else None,
        row_idx=row_zero_based if fmt == "pred" else None,
    )
    layout_path = Path(visuals["layout_path"])
    final_path = OUT_ROOT / f"{label}_2d_layout.png"
    shutil.copy2(layout_path, final_path)
    return final_path


def main() -> None:
    jobs = [
        {
            "label": "gt_surf12_reorder_row001_zero_based",
            "fmt": "orig",
            "csv_path": ROOT / "data" / "scan_lens_dataset_surf12_reorder.csv",
            "row_zero_based": 1,
            "n_surf": 11,
            "ct_is_radius": True,
            "glass_matching_csv": None,
        },
        {
            "label": "stage2_full4800_eflfiltered_row535_zero_based",
            "fmt": "pred",
            "csv_path": ROOT
            / "log"
            / "260521_1013"
            / "stage_2"
            / "airgap_unsupervised"
            / "test_epoch5000_full4800_0529"
            / "test_output_metrics_pred_rmsfilter_on_eflerror_lt_0p1.csv",
            "row_zero_based": 535,
            "n_surf": 0,
            "ct_is_radius": False,
            "glass_matching_csv": ROOT
            / "log"
            / "260521_1013"
            / "stage_2"
            / "airgap_unsupervised"
            / "test_epoch5000_full4800_0529"
            / "glass_matching_results_test_output_metrics_pred_rmsfilter_on_eflerror_lt_0p1.csv",
        },
    ]
    for job in jobs:
        out = export_and_draw(**job)
        print(f"Saved 2D layout: {out}")


if __name__ == "__main__":
    main()
