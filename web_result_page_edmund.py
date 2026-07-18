#!/usr/bin/env python3
"""Streamlit page for the Edmund-library training run.

This is a fully self-contained sibling of ``web_result_page.py``. It is kept
separate on purpose so the two deployments never share mutable global state:

* Refractive-index normalization in ``dataset_norm`` is resolved from the
  ``SCANLENS_ORIGIN_CSV`` environment variable **at import time** and cached in
  a module global. The OTS-library site and this Edmund-library site therefore
  need different origin CSVs. We set them here, before any project module is
  imported, and each site runs in its own Streamlit process, so the two never
  collide.
* The model group sizes (OTS candidate count) are built from
  ``SCANLENS_MATERIAL_CSV``. This site pins the Edmund glass catalog.
* All generated artifacts go to a dedicated ``web_outputs_edmund`` folder.

Because every configuration value is set explicitly here (instead of relying on
``utils.set_parser`` defaults), editing the other site's defaults can never
affect this one, and vice versa.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Isolation: pin the Edmund normalization + material catalog BEFORE importing
# any project module (dataset_norm resolves these at import time and caches).
# ---------------------------------------------------------------------------
EDMUND_ORIGIN_CSV = PROJECT_ROOT / "data" / "surf10_12_edmund_only_val960.csv"
EDMUND_MATERIAL_CSV = PROJECT_ROOT / "glass" / "edmund_ots_glass_c_t.csv"

os.environ["SCANLENS_ORIGIN_CSV"] = str(EDMUND_ORIGIN_CSV)
os.environ["SCANLENS_MATERIAL_CSV"] = str(EDMUND_MATERIAL_CSV)

from datetime import datetime
import json
import math
import sys

import numpy as np
import pandas as pd
import streamlit as st

from exports.scanlens_export_from_csv import export_from_config


# Stage-1 (Edmund) checkpoint used for the test run. This is a direct stage-1
# checkpoint (not an AirGap residual), so Test_Model tests it directly.
STAGE1_CKPT = (
    PROJECT_ROOT
    / "log"
    / "edmund"
    / "stage_1.0"
    / "checkpoints"
    / "SLT_rmsfilter_on_epoch5000_bs512.pth"
)

# Entrance pupil diameter used during Edmund training (utils --EPD).
TEST_EPD = 4.0
# Export EPD (matches Edmund run parameters.txt export_epd).
DEFAULT_EXPORT_EPD = 4.5

# Dedicated output tree so artifacts never mix with the OTS-library site.
WEB_OUTPUTS_DIR = PROJECT_ROOT / "web_outputs_edmund"

# Material catalog used to assemble candidate refractive-index arrangements.
OTS_GLASS_CSV = EDMUND_MATERIAL_CSV

# Test-set layout constants (raw split layout: [fn, hfov, seq_len, 33 index, 11 type]).
MAX_SURF = 11          # padded face count
N_WL = 3               # refractive index values per face (B/G/R)
N_INDEX = MAX_SURF * N_WL  # 33

# Air / Glass alternating patterns (mirrors data_process/Generate_USL_dataset.py).
PATTERN_SURF10 = ["A", "G", "A", "G", "A", "G", "A", "G", "A", "A"]  # -> 9 usable faces
PATTERN_SURF12 = ["A", "G", "A", "G", "A", "G", "A", "G", "A", "G", "A", "A"]  # -> 11 usable faces

# How many arrangements to generate for the requested (F#, HFOV).
NUM_SURF10 = 100
NUM_SURF12 = 100
NUM_ARRANGEMENTS = NUM_SURF10 + NUM_SURF12  # 200
GEN_SEED = 42

# Number of best (lowest-RMS, spec-passing) systems to present.
NUM_TOP_SYSTEMS = 3

# Specification thresholds (mirror test_extrapolation_specs.py pass_all criteria).
SPEC_RMS_MAX = 0.05        # mm
SPEC_DIST_MAX = 0.02
SPEC_TELE_MAX = 2.0        # deg
SPEC_EFL_ERR_MAX = 0.1     # relative EFL error


def discover_glass_matching_csv(metrics_csv: Path) -> str:
    parent = metrics_csv.parent
    candidates = sorted(parent.glob("glass_matching_results*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0]) if candidates else ""


def infer_efl_sidecar_csv(metrics_csv: Path) -> Path | None:
    name = metrics_csv.name
    if name.startswith("test_output_metrics_pred"):
        candidate = metrics_csv.with_name(
            name.replace("test_output_metrics_pred", "test_output_efl", 1)
        )
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# 1) Generate a 200-arrangement raw test set for the requested (F#, HFOV).
# ---------------------------------------------------------------------------
def _load_material_catalog() -> np.ndarray:
    """Unique refractive-index triplets from the Edmund glass catalog.

    The Edmund model can only select glasses present in its material CSV, so we
    assemble candidate face sequences from exactly those triplets.
    """
    glass = np.loadtxt(OTS_GLASS_CSV, delimiter=",", dtype=float)
    if glass.ndim == 1:
        glass = glass.reshape(1, -1)
    triplets = np.asarray(glass[:, :3], dtype=np.float64)
    return np.unique(triplets, axis=0)


def _generate_combos(catalog: np.ndarray, pattern: list[str], num_samples: int, seed: int) -> np.ndarray:
    """Randomly assemble face sequences following an Air/Glass pattern."""
    air = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    glass = [np.asarray(row, dtype=np.float64) for row in catalog if not np.allclose(row, air)]
    n_glass = sum(1 for p in pattern if p == "G")

    rng = np.random.default_rng(seed)
    combos = []
    for _ in range(num_samples):
        glass_idx = rng.integers(low=0, high=len(glass), size=n_glass)
        seq = []
        g_iter = iter(glass_idx)
        for p in pattern:
            seq.append(air if p == "A" else glass[next(g_iter)])
        combos.append(np.stack(seq, axis=0).reshape(-1))
    return np.vstack(combos)


def _type_sequence(seq_len: int) -> list[int]:
    """Strict alternating A/G type sequence, padded with zeros to MAX_SURF."""
    type_seq = [i % 2 for i in range(seq_len)]
    type_seq += [0] * (MAX_SURF - seq_len)
    return type_seq


def generate_arrangement_csv(
    target_fn: float,
    target_hfov: float,
    out_csv: Path,
    *,
    num_surf10: int = NUM_SURF10,
    num_surf12: int = NUM_SURF12,
    seed: int = GEN_SEED,
) -> int:
    """Write a raw test CSV with (num_surf10 + num_surf12) refractive-index
    arrangements, all sharing the requested (F#, HFOV). Returns the row count.
    """
    catalog = _load_material_catalog()

    combos10 = _generate_combos(catalog, PATTERN_SURF10, num_surf10, seed)[:, : 9 * 3]
    combos12 = _generate_combos(catalog, PATTERN_SURF12, num_surf12, seed)[:, : 11 * 3]

    rows: list[list[float]] = []
    for arr in combos10:
        material = np.concatenate([arr, np.zeros(N_INDEX - arr.shape[0], dtype=np.float64)])
        rows.append([float(target_fn), float(target_hfov), 9, *material.tolist(), *_type_sequence(9)])
    for arr in combos12:
        material = np.asarray(arr, dtype=np.float64)
        rows.append([float(target_fn), float(target_hfov), 11, *material.tolist(), *_type_sequence(11)])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, header=None, index=False, encoding="utf-8")
    return len(rows)


# ---------------------------------------------------------------------------
# 2) Run the stage-1 (Edmund) test on the generated set.
# ---------------------------------------------------------------------------
def _build_test_opt(out_dir: Path, batch_size: int):
    """Build a Test_Model opt namespace by reusing utils.set_parser.

    Every Edmund-specific value is passed explicitly so this site never relies
    on (or is affected by) shared set_parser defaults. sys.argv is overridden
    temporarily because set_parser() reads argparse from it.
    """
    from utils import set_parser

    argv_backup = sys.argv
    sys.argv = [
        "Test_Model.py",
        "--load_name",
        str(STAGE1_CKPT),
        "--save_path",
        str(out_dir),
        "--material_csv",
        str(EDMUND_MATERIAL_CSV),
        "--origin_csv",
        str(EDMUND_ORIGIN_CSV),
        "--EPD",
        str(TEST_EPD),
        "--batch_size",
        str(int(batch_size)),
        "--no_export_zmx_json",
    ]
    try:
        return set_parser()
    finally:
        sys.argv = argv_backup


def run_stage1_test(test_csv: Path, out_dir: Path, *, batch_size: int = 128) -> Path:
    """Run the Edmund stage-1 test on the generated CSV, in-process.

    The test runs inside the current process (no subprocess) so a single Python
    interpreter holds PyTorch, keeping peak memory low for CPU-only hosting.
    Lightweight mode produces only the metrics CSV; filtering, glass matching,
    export and visuals are handled by the web pipeline. Returns the metrics CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Test_Model reads these at call time.
    os.environ["SCANLENS_TEST_CSV"] = str(test_csv)
    os.environ["SCANLENS_MATERIAL_CSV"] = str(EDMUND_MATERIAL_CSV)
    os.environ["SCANLENS_ORIGIN_CSV"] = str(EDMUND_ORIGIN_CSV)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["SCANLENS_TEST_LIGHTWEIGHT"] = "1"

    import Test_Model
    from utils import set_random_seed

    opt = _build_test_opt(out_dir, batch_size)
    set_random_seed(opt.seed)
    Test_Model.test(opt)

    metrics_csv = out_dir / "test_output_metrics_pred_rmsfilter_on.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Stage-1 test did not produce metrics CSV: {metrics_csv}")
    return metrics_csv


def run_glass_matching(metrics_csv: Path) -> str:
    """Run glass matching on the metrics CSV; return the results CSV path (or "")."""
    try:
        from glass_matching import ensure_material_library_exists, process_test_output_csv

        ensure_material_library_exists()
        return str(process_test_output_csv(str(metrics_csv)))
    except Exception as exc:  # noqa: BLE001 - glass matching is best-effort for export
        print(f"[Web] glass matching failed: {exc}")
        return discover_glass_matching_csv(metrics_csv)


# ---------------------------------------------------------------------------
# 3) Filter by specification and pick the lowest-RMS systems.
# ---------------------------------------------------------------------------
def select_top_systems(metrics_csv: Path, top_n: int = NUM_TOP_SYSTEMS) -> list[dict]:
    """Return the lowest-RMS, spec-passing rows from the metrics CSV.

    The metrics CSV tail columns are:
    [..., composite, rms, dist, tele, ovlp, rays, efl_est, efl_ideal].
    """
    df = pd.read_csv(metrics_csv, header=None)
    if df.empty:
        return []

    rms = pd.to_numeric(df.iloc[:, -7], errors="coerce")
    dist = pd.to_numeric(df.iloc[:, -6], errors="coerce")
    tele = pd.to_numeric(df.iloc[:, -5], errors="coerce")
    efl_est = pd.to_numeric(df.iloc[:, -2], errors="coerce")
    efl_ideal = pd.to_numeric(df.iloc[:, -1], errors="coerce")
    efl_ref = efl_est
    efl_sidecar = infer_efl_sidecar_csv(metrics_csv)
    if efl_sidecar is not None:
        efl_df = pd.read_csv(efl_sidecar)
        if (
            len(efl_df) == len(df)
            and "EFL_first_order" in efl_df.columns
            and "EFL_ideal" in efl_df.columns
        ):
            efl_ref = pd.to_numeric(efl_df["EFL_first_order"], errors="coerce")
            efl_ideal = pd.to_numeric(efl_df["EFL_ideal"], errors="coerce")
    efl_err = (efl_ref - efl_ideal).abs() / (efl_ideal.abs() + 1e-10)

    passing = (
        (rms < SPEC_RMS_MAX)
        & (dist < SPEC_DIST_MAX)
        & (tele < SPEC_TELE_MAX)
        & (efl_err < SPEC_EFL_ERR_MAX)
    )

    candidates = pd.DataFrame(
        {
            "row_idx": df.index,
            "fn": pd.to_numeric(df.iloc[:, 0], errors="coerce"),
            "hfov": pd.to_numeric(df.iloc[:, 1], errors="coerce"),
            "rms": rms,
            "dist": dist,
            "tele": tele,
            "efl_err": efl_err,
        }
    )[passing]
    candidates = candidates.sort_values(["rms", "row_idx"]).head(int(top_n))
    return candidates.to_dict("records")


def export_best_row(
    metrics_csv: Path,
    row_idx: int,
    out_dir: Path,
    glass_matching_csv: str,
    epd: float,
) -> tuple[str, str]:
    cfg = {
        "format": "pred",
        "csv": str(metrics_csv),
        "row": int(row_idx),
        "n_surf": 0,
        "offset_ct": None,
        "ct_is_radius": False,
        "epd": float(epd),
        "semi_diam": 50.0,
        "semi_diam_stop": None,
        "r_sensor": 20.0,
        "out_dir": str(out_dir),
        "basename": f"row{int(row_idx):05d}",
        "glass_matching_csv": glass_matching_csv or "",
    }
    return export_from_config(cfg)


def _fix_converted_aperture_for_display(converted_json_path: Path) -> None:
    """Clamp suspiciously large aperture radii for clearer 2D layout display."""
    try:
        data = json.loads(converted_json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    surfaces = data.get("surfaces", [])
    if not surfaces:
        return

    radii = [float(s.get("r", 0.0) or 0.0) for s in surfaces]
    if not radii:
        return

    stop_idx = next((i for i, s in enumerate(surfaces) if str(s.get("type", "")).lower() == "stop"), 0)
    stop_r = float(surfaces[stop_idx].get("r", 2.0) or 2.0)
    if stop_r <= 0:
        stop_r = 2.0

    # If radii are not suspicious, keep original values.
    median_r = float(pd.Series(radii).median())
    if not (median_r >= 20.0 and stop_r <= 5.0):
        return

    rfov_rad = float(data.get("rfov", 0.0) or 0.0)
    tan_rfov = abs(math.tan(rfov_rad)) if abs(rfov_rad) > 1e-8 else 0.0

    z_positions: list[float] = []
    z = 0.0
    for surf in surfaces:
        z_positions.append(z)
        z += float(surf.get("d_next", 0.0) or 0.0)
    stop_z = z_positions[stop_idx]

    changed = False
    for i, surf in enumerate(surfaces):
        if i == stop_idx:
            continue
        original_r = float(surf.get("r", 0.0) or 0.0)
        dz = abs(z_positions[i] - stop_z)
        est_r = max(stop_r * 1.1, stop_r + dz * tan_rfov * 1.2)
        est_r = min(est_r, 25.0)
        if original_r > est_r:
            surf["r"] = round(est_r, 4)
            changed = True

    if changed:
        converted_json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _draw_spot_diagram_centered(lens, save_path: Path, num_fov: int = 3, num_rays: int = 2048) -> None:
    """Draw Zemax-like field spot diagram (single wavelength, per-field panes)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wvln = 0.587  # Zemax-like single wavelength view (green)
    markers = ["+", "s", "^"]
    colors = ["#0066CC", "#00AA00", "#CC0000"]

    ray = lens.sample_radial_rays(num_field=num_fov, depth=float("inf"), num_rays=num_rays, wvln=wvln)
    ray = lens.trace2sensor(ray)
    x = ray.o[:, :, 0].detach().cpu().numpy()
    y = ray.o[:, :, 1].detach().cpu().numpy()
    valid = np.isfinite(x) & np.isfinite(y)

    rfov_deg = float(np.rad2deg(lens.rfov))
    field_degs = np.linspace(0.0, rfov_deg, num_fov)

    centered_data: list[tuple[np.ndarray, np.ndarray, float]] = []
    all_abs = []
    for field_idx in range(num_fov):
        xv = x[field_idx][valid[field_idx]]
        yv = y[field_idx][valid[field_idx]]
        if xv.size == 0:
            centered_data.append((np.array([], dtype=float), np.array([], dtype=float), float("nan")))
            continue
        image_height_mm = float(np.mean(yv))
        xv_um = (xv - np.mean(xv)) * 1000.0
        yv_um = (yv - np.mean(yv)) * 1000.0
        centered_data.append((xv_um, yv_um, image_height_mm))
        all_abs.append(np.abs(xv_um))
        all_abs.append(np.abs(yv_um))

    if all_abs:
        stacked = np.concatenate(all_abs)
        scale_um = float(np.percentile(stacked, 99.5)) * 1.15
        scale_um = min(max(scale_um, 20.0), 500.0)
    else:
        scale_um = 100.0

    fig, axes = plt.subplots(1, num_fov, figsize=(max(1, num_fov) * 3.6, 4.0))
    if num_fov == 1:
        axes = [axes]

    for i in range(num_fov):
        ax = axes[i]
        xv, yv, image_height_mm = centered_data[i]
        spot_rms_um = float("nan")
        if xv.size > 0:
            ax.scatter(
                xv,
                yv,
                c=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                s=14,
                alpha=0.9,
            )
            spot_rms_um = float(np.sqrt(np.mean(xv**2 + yv**2)))
        ax.set_aspect("equal", "box")
        ax.set_xlim(-scale_um, scale_um)
        ax.set_ylim(-scale_um, scale_um)
        ax.grid(True, color="gainsboro", linewidth=0.8)
        ax.set_xticks(np.linspace(-scale_um, scale_um, 9))
        ax.set_yticks(np.linspace(-scale_um, scale_um, 9))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)
        ax.set_title(f"Field: {field_degs[i]:.2f} deg", fontsize=11, fontweight="bold")
        if np.isfinite(spot_rms_um):
            ax.set_xlabel(f"Spot size (RMS): {spot_rms_um:.2f} um", fontsize=10)

    legend_handles = []
    for i in range(num_fov):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker=markers[i % len(markers)],
                color="none",
                markerfacecolor=colors[i % len(colors)],
                markeredgecolor=colors[i % len(colors)],
                markersize=6,
                label=f"0, {field_degs[i]:.0f}",
            )
        )
    fig.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)
    fig.suptitle(f"Spot Diagram ({num_fov} fields @ {wvln:.3f}um)", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _surface_sag_mm(surface: dict, radius: float) -> float:
    """Return spherical sag at the given semi-aperture in millimeters."""
    c = float(surface.get("c", 0.0) or 0.0)
    if abs(c) < 1e-12 or radius <= 0:
        return 0.0

    arg = 1.0 - (c * radius) ** 2
    if arg <= 1e-8:
        arg = 1e-8
    return c * radius * radius / (1.0 + math.sqrt(arg))


def _surface_radius_cap(surface: dict) -> float:
    """Cap display aperture below the spherical validity radius."""
    c = abs(float(surface.get("c", 0.0) or 0.0))
    if c < 1e-12:
        return float("inf")
    return 0.9 / c


def _edge_clearance_mm(
    surfaces: list[dict],
    z_positions: list[float],
    back_idx: int,
    front_idx: int,
    radius: float,
) -> float:
    """Minimum axial air gap between two adjacent elements up to a radius."""
    if radius <= 0:
        return float(surfaces[back_idx].get("d_next", 0.0) or 0.0)

    min_clearance = float("inf")
    for r in [radius * j / 24.0 for j in range(25)]:
        z_back = z_positions[back_idx] + _surface_sag_mm(surfaces[back_idx], r)
        z_front = z_positions[front_idx] + _surface_sag_mm(surfaces[front_idx], r)
        min_clearance = min(min_clearance, z_front - z_back)
    return min_clearance


def _safe_common_aperture_for_gap(
    surfaces: list[dict],
    z_positions: list[float],
    back_idx: int,
    front_idx: int,
    requested_radius: float,
    min_clearance: float,
) -> float:
    """Shrink an aperture if adjacent lens edges would get too close."""
    if requested_radius <= 0:
        return requested_radius

    requested_radius = min(
        requested_radius,
        _surface_radius_cap(surfaces[back_idx]),
        _surface_radius_cap(surfaces[front_idx]),
    )
    if not math.isfinite(requested_radius) or requested_radius <= 0:
        return 0.0

    if _edge_clearance_mm(surfaces, z_positions, back_idx, front_idx, requested_radius) >= min_clearance:
        return requested_radius

    low = 0.0
    high = requested_radius
    for _ in range(32):
        mid = (low + high) / 2.0
        if _edge_clearance_mm(surfaces, z_positions, back_idx, front_idx, mid) >= min_clearance:
            low = mid
        else:
            high = mid
    return low


def _unify_aperture_for_display(json_path: Path) -> None:
    """Add modest display aperture margin while keeping adjacent lens edges clear."""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return

    surfaces = data.get("surfaces", [])
    if not surfaces:
        return

    z_positions: list[float] = []
    z = 0.0
    for surf in surfaces:
        z_positions.append(z)
        z += float(surf.get("d_next", 0.0) or 0.0)

    element_indices: list[tuple[int, int]] = []
    base_radius_by_front: dict[int, float] = {}
    cap_by_front: dict[int, float] = {}
    margin_factor = 1.08
    margin_min_mm = 0.25
    margin_max_mm = 0.90

    # First pass: identify glass elements and their required clear aperture.
    for i in range(len(surfaces) - 1):
        mat2 = str(surfaces[i].get("mat2", "")).strip().lower()
        if not mat2 or mat2 == "air":
            continue
        if str(surfaces[i].get("type", "")).lower() in ("stop", "aperture"):
            continue

        r1 = float(surfaces[i].get("r", 0.0) or 0.0)
        r2 = float(surfaces[i + 1].get("r", 0.0) or 0.0)
        radii = [r for r in (r1, r2) if r > 0]
        if not radii:
            continue

        base_radius = max(radii)
        element_indices.append((i, i + 1))
        base_radius_by_front[i] = base_radius
        cap_by_front[i] = min(
            _surface_radius_cap(surfaces[i]),
            _surface_radius_cap(surfaces[i + 1]),
        )

    if not element_indices:
        return

    # Use one common display aperture for the whole lens train. This makes the
    # generated layout closer to a manufacturable barrel-style aperture, while
    # still being based on the largest ray-traced clear aperture.
    global_base_radius = max(base_radius_by_front.values())
    extra = min(
        max(global_base_radius * (margin_factor - 1.0), margin_min_mm),
        margin_max_mm,
    )
    global_target = global_base_radius + extra
    target_by_front: dict[int, float] = {
        i: min(global_target, cap_by_front[i]) for i, _back_i in element_indices
    }

    # Second pass: if two adjacent glass elements are separated by a small air
    # gap, keep their enlarged edge radii below the radius where the spherical
    # sags would nearly touch. Repeating lets one tight gap propagate to a small
    # cluster, while unaffected elements keep the global aperture.
    min_edge_clearance_mm = 0.25
    for _ in range(8):
        adjusted = False
        for (front_a, back_a), (front_b, _back_b) in zip(element_indices, element_indices[1:]):
            if front_b != back_a + 1:
                continue

            requested = min(target_by_front[front_a], target_by_front[front_b])
            safe_radius = _safe_common_aperture_for_gap(
                surfaces,
                z_positions,
                back_a,
                front_b,
                requested,
                min_edge_clearance_mm,
            )
            if safe_radius < requested - 1e-4:
                target_by_front[front_a] = min(target_by_front[front_a], safe_radius)
                target_by_front[front_b] = min(target_by_front[front_b], safe_radius)
                adjusted = True
        if not adjusted:
            break

    changed = False
    for i, back_i in element_indices:
        target = round(float(target_by_front[i]), 4)
        r1 = float(surfaces[i].get("r", 0.0) or 0.0)
        r2 = float(surfaces[back_i].get("r", 0.0) or 0.0)
        if r1 != target:
            surfaces[i]["r"] = target
            changed = True
        if r2 != target:
            surfaces[back_i]["r"] = target
            changed = True

    if changed:
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _crop_image_whitespace(image_path: Path, pad: int = 6) -> None:
    """Trim surrounding white margins so only the lenses and rays remain."""
    try:
        from PIL import Image, ImageChops
    except ModuleNotFoundError:
        return
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return

    background = Image.new("RGB", img.size, (255, 255, 255))
    diff = ImageChops.difference(img, background)
    bbox = diff.getbbox()
    if not bbox:
        return

    left, top, right, bottom = bbox
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(img.width, right + pad)
    bottom = min(img.height, bottom + pad)
    img.crop((left, top, right, bottom)).save(image_path)


def _generate_spot_via_script(metrics_csv: Path, row_idx: int, spot_path: Path) -> None:
    """Generate the spot diagram using the project's own raytracer script.

    This uses lens_visualization/spot_diagram.py, which traces the row with
    USL_Loss (same green-chief-ray convention as the statistics pipeline).
    """
    import matplotlib.pyplot as plt
    from lens_visualization.lens_visualization.spot_diagram import plot_spot_diagram_from_csv

    fig, _ = plot_spot_diagram_from_csv(
        csv_path=str(metrics_csv),
        row_idx=int(row_idx),
        save_path=str(spot_path),
    )
    plt.close(fig)


def generate_visuals(
    json_path: str,
    out_dir: Path,
    metrics_csv: Path | None = None,
    row_idx: int | None = None,
) -> dict[str, str]:
    try:
        from lens_visualization.lens_visualization import LensVisualizer
        from lens_visualization.lens_visualization.json_converter import convert_and_save
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency for lens visualization. Please install: "
            "pip install matplotlib"
        ) from exc

    try:
        visualizer = LensVisualizer(json_path, device="cpu")
    except Exception as exc:
        # Exporter JSON is new-format in many runs. Convert to DeepLens standard JSON, then retry.
        converted_json = out_dir / f"{Path(json_path).stem}_converted.json"
        convert_and_save(json_path, str(converted_json), use_ray_tracing=True)
        _fix_converted_aperture_for_display(converted_json)
        _unify_aperture_for_display(converted_json)
        try:
            visualizer = LensVisualizer(str(converted_json), device="cpu")
        except Exception:
            raise RuntimeError(
                "Failed to load lens JSON for visualization. "
                "Both original and converted JSON were not accepted."
            ) from exc

    layout_path = out_dir / "lens_2d_layout.png"
    spot_path = out_dir / "lens_spot_diagram.png"
    distortion_path = out_dir / "lens_distortion.png"

    # Shrink the displayed image plane to just slightly larger than the lens aperture.
    try:
        lens_radii = [float(s.r) for s in visualizer.lens.surfaces]
        if lens_radii:
            visualizer.lens.r_sensor = max(lens_radii) * 1.1
    except Exception:
        pass

    visualizer.draw_2d_layout(
        filename=str(layout_path), depth=float("inf"), zmx_format=True, lens_title="", show=False
    )
    _crop_image_whitespace(layout_path)
    # Spot diagram: use the project raytracer script (consistent with statistics).
    if metrics_csv is not None and row_idx is not None:
        _generate_spot_via_script(metrics_csv, row_idx, spot_path)
    else:
        _draw_spot_diagram_centered(visualizer.lens, spot_path, num_fov=3, num_rays=2048)
    visualizer.draw_distortion_radial(
        rfov=None,
        save_name=str(distortion_path),
        num_points=21,
        wvln=0.587,
        plane="meridional",
        ray_aiming=True,
        show=False,
    )

    lens_info = visualizer.get_lens_info()
    return {
        "layout_path": str(layout_path),
        "spot_path": str(spot_path),
        "distortion_path": str(distortion_path),
        "fnum": str(lens_info.get("fnum", "")),
        "rfov_deg": str(lens_info.get("rfov_deg", "")),
        "num_surfaces": str(lens_info.get("num_surfaces", "")),
    }


def _load_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def _compute_one_system(
    rank: int,
    sys_info: dict,
    metrics_csv: Path,
    glass_csv: str,
    base_out_dir: Path,
) -> dict:
    """Export files and render plots for one system. Returns only paths so the
    result can be cached in st.session_state and re-displayed across reruns."""
    row_idx = int(sys_info["row_idx"])
    out_dir = base_out_dir / f"system_{rank}_row{row_idx:05d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path, zmx_path = export_best_row(
        metrics_csv=metrics_csv,
        row_idx=row_idx,
        out_dir=out_dir,
        glass_matching_csv=glass_csv,
        epd=DEFAULT_EXPORT_EPD,
    )
    vis = generate_visuals(
        json_path=json_path,
        out_dir=out_dir,
        metrics_csv=metrics_csv,
        row_idx=row_idx,
    )

    return {
        "rank": rank,
        "row_idx": row_idx,
        "zmx_path": str(zmx_path),
        "json_path": str(json_path),
        "layout_path": vis["layout_path"],
        "spot_path": vis["spot_path"],
        "distortion_path": vis["distortion_path"],
    }


def _display_one_system(data: dict) -> None:
    rank = data["rank"]
    row_idx = data["row_idx"]

    st.markdown(f"### System {rank}")

    with st.container(border=True):
        d1, d2 = st.columns(2)
        d1.download_button(
            "Download ZMX",
            data=_load_bytes(data["zmx_path"]),
            file_name=Path(data["zmx_path"]).name,
            mime="application/octet-stream",
            use_container_width=True,
            key=f"zmx_{rank}_{row_idx}",
        )
        d2.download_button(
            "Download JSON",
            data=_load_bytes(data["json_path"]),
            file_name=Path(data["json_path"]).name,
            mime="application/json",
            use_container_width=True,
            key=f"json_{rank}_{row_idx}",
        )

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.image(data["layout_path"], caption="2D Layout", use_container_width=True)
        st.image(data["spot_path"], caption="Spot Diagram", use_container_width=True)
    with right:
        st.image(data["distortion_path"], caption="Distortion", use_container_width=True)


def _run_full_pipeline(target_fn: float, target_hfov: float) -> list:
    """Run the whole pipeline and return a list of computed-system dicts
    (empty list means no qualifying system). Rendering is done by the caller
    from session_state so results survive download-button reruns."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out_dir = WEB_OUTPUTS_DIR / f"fn{target_fn:g}_hfov{target_hfov:g}_{timestamp}"
    base_out_dir.mkdir(parents=True, exist_ok=True)

    test_csv = base_out_dir / "generated_test_input.csv"
    with st.spinner("Generating test set..."):
        generate_arrangement_csv(target_fn, target_hfov, test_csv)

    with st.spinner("Running stage-1 (Edmund) test and filtering..."):
        metrics_csv = run_stage1_test(test_csv, base_out_dir / "stage1_test")
        glass_csv = run_glass_matching(metrics_csv)
        top_systems = select_top_systems(metrics_csv, NUM_TOP_SYSTEMS)

    if not top_systems:
        return []

    systems = []
    with st.spinner("Exporting ZMX and generating plots..."):
        for rank, sys_info in enumerate(top_systems, start=1):
            systems.append(
                _compute_one_system(
                    rank=rank,
                    sys_info=sys_info,
                    metrics_csv=metrics_csv,
                    glass_csv=glass_csv,
                    base_out_dir=base_out_dir,
                )
            )
    return systems


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.5rem; padding-bottom: 3rem; max-width: 1280px;}
        h1 {font-weight: 700; letter-spacing: -0.5px;}
        .stButton > button {border-radius: 8px; font-weight: 600;}
        .stDownloadButton > button {border-radius: 8px; font-weight: 600;}
        div[data-testid="stImage"] img {border: 1px solid #e8e8e8; border-radius: 10px; background: #ffffff; padding: 4px;}
        div[data-testid="stImage"] figcaption {font-weight: 600; color: #555; text-align: center;}
        div[data-testid="stMetricValue"] {font-size: 1.4rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_app() -> None:
    st.set_page_config(page_title="ScanLens Lens System Generator (Edmund Library)", layout="wide")
    _inject_css()
    st.title("ScanLens Lens System Generator (Edmund Library)")

    if not STAGE1_CKPT.exists():
        st.error(f"Stage-1 model checkpoint not found: {STAGE1_CKPT}")
        return

    st.subheader("Target Parameters")
    st.caption("Suggested range: HFOV 8-10 deg, F# 9-17.5")
    col1, col2 = st.columns(2)
    target_fn = col1.number_input(
        "Target F#", value=9.75, format="%.6f", help="Suggested range: 9-17.5"
    )
    target_hfov = col2.number_input(
        "Target HFOV (deg)", value=8.5, format="%.6f", help="Suggested range: 8-10 deg"
    )

    if st.button("Generate & Filter Systems", type="primary"):
        # Reset previous results so a failed run does not show stale output.
        st.session_state["results"] = None
        try:
            st.session_state["results"] = _run_full_pipeline(
                float(target_fn), float(target_hfov)
            )
        except Exception as exc:
            st.exception(exc)

    # Render from session_state on every run so that clicking a download
    # button (which triggers a Streamlit rerun) does not clear the results.
    results = st.session_state.get("results")
    if results is not None:
        if not results:
            st.warning("No qualifying system found.")
        else:
            for data in results:
                _display_one_system(data)


if __name__ == "__main__":
    run_app()
