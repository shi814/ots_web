#!/usr/bin/env python3
"""Match predicted test rows to nearby GT systems and write a Markdown report."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from exports.scanlens_export_from_csv import (
    _infer_pred_nsurf_and_offset,
    _infer_glass_name_from_nd,
    _read_pred_row_from_vals,
    read_orig_row,
)
from USL_Loss import USL_Loss
from utils import LensBatch


DEFAULT_SURF10_CSV = "data/scan_lens_dataset_surf10_reorder.csv"
DEFAULT_SURF12_CSV = "data/scan_lens_dataset_surf12_reorder.csv"


@dataclass(frozen=True)
class LensRow:
    source: str
    row_idx: int
    fn: float
    hfov: float
    n_surf: int
    materials: np.ndarray
    ct: np.ndarray
    rms: float | None = None
    dist: float | None = None
    tele: float | None = None
    efl_est: float | None = None
    efl_ideal: float | None = None
    loss_efl: float | None = None

    @property
    def total_surface_count(self) -> int:
        return self.n_surf + 1


@dataclass(frozen=True)
class MatchResult:
    pred: LensRow
    gt: LensRow
    score: float
    material_score: float
    structure_score: float
    system_score: float
    metric_score: float
    rms_floor_penalty: float


@dataclass(frozen=True)
class ReportSelections:
    matches: list[MatchResult]
    best_rms_rows: list[LensRow]

    @property
    def pred_row_indices(self) -> list[int]:
        seen: set[int] = set()
        out: list[int] = []
        for match in self.matches:
            idx = int(match.pred.row_idx)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        for row in self.best_rms_rows:
            idx = int(row.row_idx)
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
        return out

    @property
    def gt_rows(self) -> list[LensRow]:
        seen: set[tuple[str, int, int]] = set()
        out: list[LensRow] = []
        for match in self.matches:
            key = (match.gt.source, int(match.gt.row_idx), int(match.gt.n_surf))
            if key not in seen:
                seen.add(key)
                out.append(match.gt)
        return out


def _as_project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), path)


def _fmt(value: float | None, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if not np.isfinite(value):
        return "N/A"
    return f"{float(value):.{digits}g}"


def _fmt_compact_list(values: Sequence[str | float | int], max_items: int | None = None) -> str:
    items = list(values)
    if max_items is not None and len(items) > max_items:
        items = items[:max_items] + ["..."]
    return ", ".join(str(x) for x in items)


def _glass_name_triplet(n_triplet: Sequence[float]) -> str:
    if all(abs(float(x) - 1.0) <= 1e-4 for x in n_triplet):
        return "Air"
    name = _infer_glass_name_from_nd(float(n_triplet[1]))
    if name:
        return name
    return "/".join(_fmt(float(x), digits=5) for x in n_triplet)


def _display_curvature_as_radius(curvature: float) -> str:
    if abs(float(curvature)) <= 1e-12:
        return "Plane"
    return _fmt(1.0 / float(curvature), digits=5)


def _load_pred_rows(pred_csv: str) -> list[LensRow]:
    df = pd.read_csv(pred_csv, header=None)
    rows: list[LensRow] = []
    for row_idx, row in df.iterrows():
        vals = row.to_numpy(dtype=float)
        n_surf, offset_ct = _infer_pred_nsurf_and_offset(vals)
        (fn, hfov), materials, ct_pairs = _read_pred_row_from_vals(
            vals, n_surf=n_surf, offset_ct=offset_ct
        )
        rms = float(vals[-7]) if len(vals) >= 7 else None
        dist = float(vals[-6]) if len(vals) >= 6 else None
        tele = float(vals[-5]) if len(vals) >= 5 else None
        efl_est = float(vals[-2]) if len(vals) >= 2 else None
        efl_ideal = float(vals[-1]) if len(vals) >= 1 else None
        loss_efl = None
        if efl_est is not None and efl_ideal is not None:
            loss_efl = abs(efl_est - efl_ideal) / (abs(efl_ideal) + 1e-10)
        rows.append(
            LensRow(
                source="pred",
                row_idx=int(row_idx),
                fn=float(fn),
                hfov=float(hfov),
                n_surf=int(n_surf),
                materials=np.asarray(materials, dtype=float),
                ct=np.asarray(ct_pairs, dtype=float),
                rms=rms,
                dist=dist,
                tele=tele,
                efl_est=efl_est,
                efl_ideal=efl_ideal,
                loss_efl=loss_efl,
            )
        )
    return rows


def _radius_to_curvature(ct_pairs: np.ndarray) -> np.ndarray:
    out = np.asarray(ct_pairs, dtype=float).copy()
    radius = out[:, 0]
    curvature = np.zeros_like(radius, dtype=float)
    valid = np.abs(radius) > 1e-12
    curvature[valid] = 1.0 / radius[valid]
    out[:, 0] = curvature
    return out


def _load_gt_rows(csv_path: str, n_surf: int, radius_to_curvature: bool = True) -> list[LensRow]:
    df = pd.read_csv(csv_path, header=None)
    rows: list[LensRow] = []
    for row_idx in range(len(df)):
        (fn, hfov), materials, ct_pairs = read_orig_row(csv_path, n_surf=n_surf, row=row_idx)
        ct_array = np.asarray(ct_pairs, dtype=float)
        if radius_to_curvature:
            ct_array = _radius_to_curvature(ct_array)
        rows.append(
            LensRow(
                source=os.path.basename(csv_path),
                row_idx=int(row_idx),
                fn=float(fn),
                hfov=float(hfov),
                n_surf=int(n_surf),
                materials=np.asarray(materials, dtype=float),
                ct=ct_array,
            )
        )
    return rows


def _metric_vector(row: LensRow) -> np.ndarray:
    values = [row.rms, row.dist, row.tele, row.loss_efl]
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def _default_usl_opt():
    return SimpleNamespace(
        nWL=3,
        nField=3,
        nRayDensity=11,
        EPD=4.0,
        max_seq_length=11,
        output_size=3,
        enable_rms_filter=False,
        enable_efl_filter=False,
        distortion_mode="zemax_ftan",
        distortion_ref_angle_deg=0.01,
        usl_loss_variant="stage1_geometric_v1",
        efl_loss_tolerance=0.1,
    )


def _compute_usl_metrics_for_rows(
    rows: Sequence[LensRow],
    batch_size: int = 64,
) -> list[LensRow]:
    if not rows:
        return []

    opt = _default_usl_opt()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optics = USL_Loss(opt).to(device)
    optics.eval()

    out_rows: list[LensRow] = []
    max_surf_lens = 11
    n_surf = rows[0].n_surf

    for start in range(0, len(rows), batch_size):
        chunk = list(rows[start:start + batch_size])
        x = torch.as_tensor(
            [[row.fn, row.hfov] for row in chunk],
            dtype=torch.float64,
            device=device,
        )
        n = torch.as_tensor(
            np.stack([row.materials for row in chunk], axis=0),
            dtype=torch.float64,
            device=device,
        )
        ct = torch.as_tensor(
            np.stack([row.ct for row in chunk], axis=0),
            dtype=torch.float64,
            device=device,
        )
        bsz = x.shape[0]
        surf_lens = torch.full((bsz,), n_surf, device=device, dtype=torch.long)
        mask = torch.arange(max_surf_lens, device=device)[None, :].expand(bsz, max_surf_lens) < n_surf
        lens = LensBatch(x, n, ct)

        with torch.no_grad():
            _, _, metrics = optics(
                lens,
                max_surf_lens,
                surf_lens,
                mask,
                epoch=5000,
                save=1,
                apply_hard_filter=False,
            )

        metric_values = {
            key: metrics[key].detach().cpu().numpy()
            for key in ("rms", "dist", "tele", "EFL_est", "EFL_ideal", "loss_EFL")
        }
        for i, row in enumerate(chunk):
            efl_est = float(metric_values["EFL_est"][i])
            efl_ideal = float(metric_values["EFL_ideal"][i])
            efl_rel_error = abs(efl_est - efl_ideal) / (abs(efl_ideal) + 1e-10)
            out_rows.append(
                LensRow(
                    source=row.source,
                    row_idx=row.row_idx,
                    fn=row.fn,
                    hfov=row.hfov,
                    n_surf=row.n_surf,
                    materials=row.materials,
                    ct=row.ct,
                    rms=float(metric_values["rms"][i]),
                    dist=float(metric_values["dist"][i]),
                    tele=float(metric_values["tele"][i]),
                    efl_est=efl_est,
                    efl_ideal=efl_ideal,
                    loss_efl=float(efl_rel_error),
                )
            )

    return out_rows


def _feature_scales(rows: Sequence[LensRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    material_stack = np.stack([row.materials.reshape(-1) for row in rows], axis=0)
    structure_stack = np.stack([row.ct.reshape(-1) for row in rows], axis=0)
    metric_stack = np.stack([_metric_vector(row) for row in rows], axis=0)
    material_scale = np.nanstd(material_stack, axis=0)
    structure_scale = np.nanstd(structure_stack, axis=0)
    metric_scale = np.nanstd(metric_stack, axis=0)
    material_scale = np.where(material_scale < 1e-8, 1.0, material_scale)
    structure_scale = np.where(structure_scale < 1e-8, 1.0, structure_scale)
    metric_scale = np.where(metric_scale < 1e-8, 1.0, metric_scale)
    return material_scale, structure_scale, metric_scale


def _score_match(
    pred: LensRow,
    gt: LensRow,
    material_scale: np.ndarray,
    structure_scale: np.ndarray,
    metric_scale: np.ndarray,
    material_weight: float,
    system_weight: float,
    metric_weight: float,
    rms_floor_tolerance: float,
    rms_floor_penalty_weight: float,
) -> tuple[float, float, float, float, float, float]:
    pred_material = pred.materials.reshape(-1)
    gt_material = gt.materials.reshape(-1)
    pred_structure = pred.ct.reshape(-1)
    gt_structure = gt.ct.reshape(-1)
    pred_metrics = _metric_vector(pred)
    gt_metrics = _metric_vector(gt)

    material_score = float(np.mean(np.abs(pred_material - gt_material) / material_scale))
    structure_score = float(np.mean(np.abs(pred_structure - gt_structure) / structure_scale))
    fn_score = abs(float(pred.fn) - float(gt.fn)) / (abs(float(pred.fn)) + 1e-10)
    hfov_score = abs(float(pred.hfov) - float(gt.hfov)) / (abs(float(pred.hfov)) + 1e-10)
    system_score = float(0.5 * (fn_score + hfov_score))
    metric_score = float(np.nanmean(np.abs(pred_metrics - gt_metrics) / metric_scale))
    if not np.isfinite(metric_score):
        metric_score = 0.0

    rms_floor_penalty = 0.0
    if pred.rms is not None and gt.rms is not None and np.isfinite(pred.rms) and np.isfinite(gt.rms):
        allowed_floor = float(pred.rms) * (1.0 - float(rms_floor_tolerance))
        if float(gt.rms) < allowed_floor:
            rms_floor_penalty = (allowed_floor - float(gt.rms)) / (abs(float(pred.rms)) + 1e-10)

    structure_weight = max(0.0, 1.0 - material_weight - system_weight - metric_weight)
    total = (
        material_weight * material_score
        + structure_weight * structure_score
        + float(system_weight) * system_score
        + metric_weight * metric_score
        + float(rms_floor_penalty_weight) * rms_floor_penalty
    )
    return total, material_score, structure_score, system_score, metric_score, float(rms_floor_penalty)


def _same_fn_hfov(pred: LensRow, gt: LensRow, tol: float) -> bool:
    return (
        abs(float(pred.fn) - float(gt.fn)) <= float(tol)
        and abs(float(pred.hfov) - float(gt.hfov)) <= float(tol)
    )


def _gt_rms_strictly_higher(pred: LensRow, gt: LensRow) -> bool:
    if pred.rms is None or gt.rms is None:
        return False
    if not np.isfinite(float(pred.rms)) or not np.isfinite(float(gt.rms)):
        return False
    return float(gt.rms) > float(pred.rms)


def _find_best_match(
    pred: LensRow,
    gt_rows: Sequence[LensRow],
    material_scale: np.ndarray,
    structure_scale: np.ndarray,
    metric_scale: np.ndarray,
    material_weight: float,
    system_weight: float,
    metric_weight: float,
    rms_floor_tolerance: float,
    rms_floor_penalty_weight: float,
    fn_hfov_tol: float,
) -> MatchResult | None:
    best: MatchResult | None = None
    for gt in gt_rows:
        # Hard rule 1: only same (F#, HFOV) systems.
        if not _same_fn_hfov(pred, gt, tol=fn_hfov_tol):
            continue
        # Hard rule 2: GT RMS must be strictly higher than Pred RMS.
        if not _gt_rms_strictly_higher(pred, gt):
            continue
        score, material_score, structure_score, system_score, metric_score, rms_floor_penalty = _score_match(
            pred,
            gt,
            material_scale,
            structure_scale,
            metric_scale,
            material_weight,
            system_weight,
            metric_weight,
            rms_floor_tolerance,
            rms_floor_penalty_weight,
        )
        result = MatchResult(
            pred,
            gt,
            score,
            material_score,
            structure_score,
            system_score,
            metric_score,
            rms_floor_penalty,
        )
        if best is None or result.score < best.score:
            best = result
    return best


def _material_structure_rank_score(match: MatchResult) -> float:
    return float(match.material_score + match.structure_score + match.system_score)


def _system_group_key(row: LensRow, digits: int = 6) -> tuple[float, float]:
    return (round(float(row.fn), digits), round(float(row.hfov), digits))


def _pred_design_signature(row: LensRow) -> tuple:
    material_sig = tuple(np.round(row.materials.reshape(-1), 5).tolist())
    structure_sig = tuple(np.round(row.ct.reshape(-1), 4).tolist())
    return (int(row.n_surf), material_sig, structure_sig)


def _select_diverse_best_rms_per_fn_hfov(
    rows: Sequence[LensRow],
    top_n_per_group: int = 3,
    max_per_nsurf: int = 2,
) -> list[LensRow]:
    grouped: dict[tuple[float, float], list[LensRow]] = {}
    for row in rows:
        grouped.setdefault(_system_group_key(row), []).append(row)

    selected: list[LensRow] = []
    for key in sorted(grouped.keys()):
        group_rows = grouped[key]
        ranked_rows = sorted(
            group_rows,
            key=lambda r: (
                float("inf") if r.rms is None or not np.isfinite(float(r.rms)) else float(r.rms),
                int(r.row_idx),
            ),
        )
        seen_signatures: set[tuple] = set()
        chosen_count = 0
        chosen_by_nsurf: dict[int, int] = {}
        for row in ranked_rows:
            signature = _pred_design_signature(row)
            if signature in seen_signatures:
                continue
            n_surf = int(row.n_surf)
            if int(max_per_nsurf) > 0 and chosen_by_nsurf.get(n_surf, 0) >= int(max_per_nsurf):
                continue
            seen_signatures.add(signature)
            selected.append(row)
            chosen_count += 1
            chosen_by_nsurf[n_surf] = chosen_by_nsurf.get(n_surf, 0) + 1
            if chosen_count >= int(top_n_per_group):
                break
    return selected


def _select_top_k_matches(
    matches: Sequence[MatchResult],
    top_k: int,
    diversify_by_system: bool = True,
) -> list[MatchResult]:
    ranked = sorted(matches, key=_material_structure_rank_score)
    if top_k <= 0:
        return ranked
    if not diversify_by_system:
        return ranked[:top_k]

    selected: list[MatchResult] = []
    used_groups: set[tuple[float, float]] = set()
    remaining: list[MatchResult] = []

    # Pass 1: prefer one best match per (F#, HFOV) group.
    for match in ranked:
        key = _system_group_key(match.pred)
        if key in used_groups:
            remaining.append(match)
            continue
        selected.append(match)
        used_groups.add(key)
        if len(selected) >= top_k:
            return selected

    # Pass 2: if top_k still not full, fill by remaining global best scores.
    for match in remaining:
        selected.append(match)
        if len(selected) >= top_k:
            break
    return selected


def _select_best_rms_per_fn_hfov(rows: Sequence[LensRow]) -> list[LensRow]:
    best_by_group: dict[tuple[float, float], LensRow] = {}
    for row in rows:
        key = (round(float(row.fn), 6), round(float(row.hfov), 6))
        current = best_by_group.get(key)
        row_rms = float("inf") if row.rms is None else float(row.rms)
        current_rms = float("inf") if current is None or current.rms is None else float(current.rms)
        if current is None or row_rms < current_rms:
            best_by_group[key] = row
    return sorted(best_by_group.values(), key=lambda r: (float(r.fn), float(r.hfov), float("inf") if r.rms is None else float(r.rms)))


def _format_compact_row(label: str, row: LensRow) -> str:
    materials = [_glass_name_triplet(row.materials[i]) for i in range(row.n_surf)]
    curvatures = [_display_curvature_as_radius(float(row.ct[i, 0])) for i in range(row.n_surf)]
    thicknesses = [_fmt(float(row.ct[i, 1]), digits=5) for i in range(row.n_surf)]
    return (
        f"| {label} | {row.total_surface_count}面 | {_fmt(row.fn)} | {_fmt(row.hfov)} | "
        f"{_fmt(row.rms)} | {_fmt(row.dist)} | {_fmt(row.tele)} | {_fmt(row.loss_efl)} | "
        f"{_fmt_compact_list(materials)} | {_fmt_compact_list(curvatures)} | "
        f"{_fmt_compact_list(thicknesses)} |"
    )


def _write_compact_table_header(lines: list[str]) -> None:
    lines.extend(
        [
            "| 类型 | 序列 | F# | HFOV | RMS | dist | tele | EFL error | 材料 | C | t |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )


def _write_compact_comparison_table(lines: list[str], match: MatchResult) -> None:
    pred = match.pred
    gt = match.gt
    rows = [("GT", gt), ("pred", pred)]
    _write_compact_table_header(lines)
    for label, row in rows:
        lines.append(_format_compact_row(label, row))
    lines.append("")


def _write_best_rms_sections(lines: list[str], rows: Sequence[LensRow]) -> None:
    lines.extend(
        [
            "## Best RMS Per F#/HFOV",
            "",
            "For each `(F#, HFOV)` group in the prediction CSV, the system with the smallest RMS is selected.",
            "",
        ]
    )
    for idx, row in enumerate(rows, start=1):
        lines.extend(
            [
                f"### Best RMS {idx}",
                "",
                f"- Pred row: `{row.row_idx}`",
                "",
            ]
        )
        _write_compact_table_header(lines)
        lines.append(_format_compact_row("pred", row))
        lines.append("")


def _write_selected_pred_sections(
    lines: list[str],
    rows: Sequence[LensRow],
    top_n_per_group: int,
    max_per_nsurf: int,
) -> None:
    grouped: dict[tuple[float, float], list[LensRow]] = {}
    for row in rows:
        grouped.setdefault(_system_group_key(row), []).append(row)

    lines.extend(
        [
            "## Selected Prediction Systems",
            "",
            (
                "For each `(F#, HFOV)` group in the prediction CSV, select up to "
                f"`{top_n_per_group}` systems with the smallest RMS, while requiring "
                "different material / lens-combination signatures."
            ),
            (
                f"Additional constraint: keep at most `{max_per_nsurf}` designs for each "
                "`n_surf` group, so the final selection stays balanced across 9-surface "
                "and 11-surface systems whenever candidates exist."
            ),
            "",
        ]
    )
    for idx, key in enumerate(sorted(grouped.keys()), start=1):
        fn, hfov = key
        group_rows = sorted(
            grouped[key],
            key=lambda r: (
                float("inf") if r.rms is None or not np.isfinite(float(r.rms)) else float(r.rms),
                int(r.row_idx),
            ),
        )
        lines.extend(
            [
                f"### Group {idx}",
                "",
                f"- F#: `{_fmt(fn)}`",
                f"- HFOV: `{_fmt(hfov)}`",
                f"- Selected rows: `{len(group_rows)}`",
                f"- Pred row indices: `{', '.join(str(int(row.row_idx)) for row in group_rows)}`",
                "",
            ]
        )
        _write_compact_table_header(lines)
        for row in group_rows:
            lines.append(_format_compact_row("pred", row))
        lines.append("")


def write_markdown_report(
    matches: Sequence[MatchResult],
    out_md: str,
    pred_csv: str,
    best_rms_rows: Sequence[LensRow] | None = None,
    pred_only_selection: bool = False,
    pred_top_n_per_group: int = 4,
    pred_max_per_nsurf: int = 2,
) -> None:
    if pred_only_selection:
        lines: list[str] = [
            "# Prediction Selection Report",
            "",
            f"Prediction CSV: `{pred_csv}`",
            "",
        ]
    else:
        lines = [
            "# GT Match Report",
            "",
            f"Prediction CSV: `{pred_csv}`",
            "",
            "GT optical metrics are computed with `USL_Loss` before matching.",
            "Matching rules (hard constraints first): same `F#` and `HFOV`, and `GT RMS > Pred RMS`.",
            "Among valid candidates, ranking is still based on material/structure/system/metric score.",
            "",
            "## Top GT Matches",
            "",
        ]
        for idx, match in enumerate(matches, start=1):
            lines.extend(
                [
                    f"## Match {idx}",
                    "",
                    f"- Pred row: `{match.pred.row_idx}`",
                    f"- GT file: `{match.gt.source}`",
                    f"- GT row: `{match.gt.row_idx}`",
                    f"- Match score: `{_fmt(match.score)}`",
                    f"- Material score: `{_fmt(match.material_score)}`",
                    f"- Structure score: `{_fmt(match.structure_score)}`",
                    f"- F#/HFOV score: `{_fmt(match.system_score)}`",
                    f"- Material + structure + system rank score: `{_fmt(_material_structure_rank_score(match))}`",
                    f"- Metrics score: `{_fmt(match.metric_score)}`",
                    f"- RMS floor penalty: `{_fmt(match.rms_floor_penalty)}`",
                    "",
                ]
            )
            _write_compact_comparison_table(lines, match)

    if not pred_only_selection:
        if not matches:
            lines.extend(
                [
                    "_No GT matches found under the current matching rules._",
                    "",
                ]
            )
        if best_rms_rows:
            _write_selected_pred_sections(
                lines,
                best_rms_rows,
                top_n_per_group=pred_top_n_per_group,
                max_per_nsurf=pred_max_per_nsurf,
            )
    else:
        if best_rms_rows:
            _write_selected_pred_sections(
                lines,
                best_rms_rows,
                top_n_per_group=pred_top_n_per_group,
                max_per_nsurf=pred_max_per_nsurf,
            )

    os.makedirs(os.path.dirname(os.path.abspath(out_md)), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_selected_rows_csv(
    selections: ReportSelections,
    out_csv: str,
    gt_csv_by_nsurf: dict[int, str],
) -> None:
    rows = [
        {
            "kind": "pred",
            "pred_row_idx": idx,
            "gt_csv": "",
            "gt_row_idx": "",
            "n_surf": "",
        }
        for idx in selections.pred_row_indices
    ]
    for gt in selections.gt_rows:
        rows.append(
            {
                "kind": "gt",
                "pred_row_idx": "",
                "gt_csv": gt_csv_by_nsurf.get(int(gt.n_surf), ""),
                "gt_row_idx": int(gt.row_idx),
                "n_surf": int(gt.n_surf),
            }
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[GT Match] Saved selected export rows: {out_csv}")


def generate_gt_match_report(
    pred_csv: str,
    out_md: str,
    surf10_csv: str = DEFAULT_SURF10_CSV,
    surf12_csv: str = DEFAULT_SURF12_CSV,
    max_rows: int = 0,
    material_weight: float = 0.4,
    system_weight: float = 0.2,
    metric_weight: float = 0.2,
    gt_metric_batch_size: int = 64,
    gt_radius_to_curvature: bool = True,
    top_k_matches: int = 10,
    include_best_rms_per_group: bool = True,
    selected_rows_csv: str | None = None,
    rms_floor_tolerance: float = 0.1,
    rms_floor_penalty_weight: float = 2.0,
    fn_hfov_tol: float = 1e-6,
    diversify_top_k_by_system: bool = True,
    pred_only_selection: bool = False,
    pred_top_n_per_group: int = 4,
    pred_max_per_nsurf: int = 2,
) -> str:
    pred_csv = _as_project_path(pred_csv)
    out_md = _as_project_path(out_md)
    surf10_csv = _as_project_path(surf10_csv)
    surf12_csv = _as_project_path(surf12_csv)

    pred_rows = _load_pred_rows(pred_csv)
    if max_rows and max_rows > 0:
        pred_rows = pred_rows[: int(max_rows)]
    selected_pred_rows = _select_diverse_best_rms_per_fn_hfov(
        pred_rows,
        top_n_per_group=int(pred_top_n_per_group),
        max_per_nsurf=int(pred_max_per_nsurf),
    )

    if pred_only_selection:
        selections = ReportSelections(matches=[], best_rms_rows=list(selected_pred_rows))
        write_markdown_report(
            [],
            out_md,
            pred_csv=pred_csv,
            best_rms_rows=selected_pred_rows,
            pred_only_selection=True,
            pred_top_n_per_group=int(pred_top_n_per_group),
            pred_max_per_nsurf=int(pred_max_per_nsurf),
        )
        if selected_rows_csv:
            write_selected_rows_csv(
                selections,
                _as_project_path(selected_rows_csv),
                gt_csv_by_nsurf={},
            )
        print(f"[Pred Select] Saved report: {out_md}")
        return out_md

    gt_by_nsurf = {
        9: _load_gt_rows(surf10_csv, n_surf=9, radius_to_curvature=gt_radius_to_curvature),
        11: _load_gt_rows(surf12_csv, n_surf=11, radius_to_curvature=gt_radius_to_curvature),
    }
    gt_by_nsurf = {
        n_surf: _compute_usl_metrics_for_rows(rows, batch_size=gt_metric_batch_size)
        for n_surf, rows in gt_by_nsurf.items()
    }
    scales_by_nsurf = {
        n_surf: _feature_scales(rows) for n_surf, rows in gt_by_nsurf.items()
    }

    matches: list[MatchResult] = []
    for pred in pred_rows:
        if pred.n_surf not in gt_by_nsurf:
            print(f"[GT Match] Skip pred row {pred.row_idx}: unsupported n_surf={pred.n_surf}")
            continue
        material_scale, structure_scale, metric_scale = scales_by_nsurf[pred.n_surf]
        match = _find_best_match(
            pred,
            gt_by_nsurf[pred.n_surf],
            material_scale,
            structure_scale,
            metric_scale,
            material_weight=float(material_weight),
            system_weight=float(system_weight),
            metric_weight=float(metric_weight),
            rms_floor_tolerance=float(rms_floor_tolerance),
            rms_floor_penalty_weight=float(rms_floor_penalty_weight),
            fn_hfov_tol=float(fn_hfov_tol),
        )
        if match is None:
            print(
                "[GT Match] Skip pred row "
                f"{pred.row_idx}: no GT candidate satisfies "
                f"same(F#,HFOV, tol={fn_hfov_tol}) and GT_RMS > Pred_RMS."
            )
            continue
        matches.append(match)

    if top_k_matches and top_k_matches > 0:
        matches = _select_top_k_matches(
            matches,
            top_k=int(top_k_matches),
            diversify_by_system=bool(diversify_top_k_by_system),
        )

    best_rms_rows = _select_best_rms_per_fn_hfov(pred_rows) if include_best_rms_per_group else []
    report_pred_rows = selected_pred_rows if selected_pred_rows else list(best_rms_rows)
    selections = ReportSelections(matches=matches, best_rms_rows=list(report_pred_rows))

    write_markdown_report(
        matches,
        out_md,
        pred_csv=pred_csv,
        best_rms_rows=report_pred_rows,
        pred_only_selection=False,
        pred_top_n_per_group=int(pred_top_n_per_group),
        pred_max_per_nsurf=int(pred_max_per_nsurf),
    )
    if selected_rows_csv:
        write_selected_rows_csv(
            selections,
            _as_project_path(selected_rows_csv),
            gt_csv_by_nsurf={9: surf10_csv, 11: surf12_csv},
        )
    print(f"[GT Match] Saved report: {out_md}")
    return out_md


def main() -> None:
    parser = argparse.ArgumentParser(description="Match prediction CSV rows to GT CSV rows.")
    parser.add_argument("--pred_csv", required=True, help="test_output_metrics_pred*.csv")
    parser.add_argument("--out_md", required=True, help="Output Markdown path.")
    parser.add_argument("--surf10_csv", default=DEFAULT_SURF10_CSV)
    parser.add_argument("--surf12_csv", default=DEFAULT_SURF12_CSV)
    parser.add_argument("--max_rows", type=int, default=0, help="Use 0 to report all rows.")
    parser.add_argument("--material_weight", type=float, default=0.4)
    parser.add_argument(
        "--system_weight",
        type=float,
        default=0.2,
        help="F-number/HFOV closeness weight for GT matching score.",
    )
    parser.add_argument("--metric_weight", type=float, default=0.2)
    parser.add_argument(
        "--rms_floor_tolerance",
        type=float,
        default=0.1,
        help="Allow matched GT RMS to be this fraction below pred RMS without penalty.",
    )
    parser.add_argument(
        "--rms_floor_penalty_weight",
        type=float,
        default=2.0,
        help="Penalty weight when matched GT RMS is much smaller than pred RMS.",
    )
    parser.add_argument(
        "--fn_hfov_tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when checking Pred/GT F# and HFOV equality.",
    )
    parser.add_argument("--gt_metric_batch_size", type=int, default=64)
    parser.add_argument("--top_k_matches", type=int, default=10)
    parser.add_argument(
        "--diversify_top_k_by_system",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When selecting top-k matches, prefer different (F#, HFOV) groups first.",
    )
    parser.add_argument(
        "--pred_only_selection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only select diverse prediction systems per (F#, HFOV), without GT matching.",
    )
    parser.add_argument(
        "--pred_top_n_per_group",
        type=int,
        default=4,
        help="When pred_only_selection is enabled, keep up to N diverse low-RMS predictions per (F#, HFOV).",
    )
    parser.add_argument(
        "--pred_max_per_nsurf",
        type=int,
        default=2,
        help="When pred_only_selection is enabled, keep at most N predictions for each n_surf within one (F#, HFOV) group.",
    )
    parser.add_argument("--selected_rows_csv", default="")
    parser.add_argument(
        "--include_best_rms_per_group",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--gt_radius_to_curvature",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert GT CSV radius columns to curvature before USL_Loss and matching.",
    )
    args = parser.parse_args()

    generate_gt_match_report(
        pred_csv=args.pred_csv,
        out_md=args.out_md,
        surf10_csv=args.surf10_csv,
        surf12_csv=args.surf12_csv,
        max_rows=args.max_rows,
        material_weight=args.material_weight,
        system_weight=args.system_weight,
        metric_weight=args.metric_weight,
        gt_metric_batch_size=args.gt_metric_batch_size,
        gt_radius_to_curvature=args.gt_radius_to_curvature,
        top_k_matches=args.top_k_matches,
        include_best_rms_per_group=args.include_best_rms_per_group,
        selected_rows_csv=args.selected_rows_csv or None,
        rms_floor_tolerance=args.rms_floor_tolerance,
        rms_floor_penalty_weight=args.rms_floor_penalty_weight,
        fn_hfov_tol=args.fn_hfov_tol,
        diversify_top_k_by_system=args.diversify_top_k_by_system,
        pred_only_selection=args.pred_only_selection,
        pred_top_n_per_group=args.pred_top_n_per_group,
        pred_max_per_nsurf=args.pred_max_per_nsurf,
    )


if __name__ == "__main__":
    main()
