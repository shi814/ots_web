"""
扫描镜头导出工具 - 将CSV中的单行数据导出为Zemax兼容的ZMX和JSON文件

功能：
  - 从测试结果CSV中提取指定行的镜头设计参数
  - 自动识别9面或11面镜头系统
  - 生成Zemax (.zmx) 和 JSON (.json) 格式的镜头文件
  - 支持玻璃匹配结果注入（可选）

支持的CSV格式：
  - pred: 测试预测结果 (test_output_metrics_pred_*.csv)
    第一列：F数(FN)，第二列：半视场角(HFOV)
  - orig: 原始数据集 (scan_lens_dataset_surfXX_reorder.csv)
    第一列：F数(FN)，第二列：半视场角(HFOV)，第三列：表面数量

使用方法：

1. 直接运行（推荐）：
   - 修改下面的 DEFAULT_EXPORT_CONFIG 中的参数
   - 运行：python scanlens_export_from_csv.py

2. 命令行参数：
   python exports/scanlens_export_from_csv.py --format pred --csv log/251223/rmsfilter_off/test_output_metrics_pred_rmsfilter_off.csv --row 191 --glass_matching_csv log/251223/rmsfilter_off/glass_matching_results_test_output_metrics_pred_rmsfilter_off.csv

输出文件：
  - ZMX文件：用于Zemax光学设计软件
  - JSON文件：包含完整的镜头参数信息
"""

import argparse
import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

try:
    # When imported as a package module: `from exports.scanlens_export_from_csv import ...`
    from .lens_export_utils import LensExporter
except Exception:  # pragma: no cover
    # When run as a script from repo root: `python exports/scanlens_export_from_csv.py`
    from lens_export_utils import LensExporter


DEFAULT_EXPORT_CONFIG = {
    # Choose: "pred" or "orig"
    "format": "pred",
    # Which CSV and which row (0-based)
    # 修改这里：设置为你想导出的 CSV 路径和行号
    "csv": r"log/260120/stage1/stage_1.0/train_output_metrics_pred_rmsfilter_on.csv",
    "row": 2,  # 0-based row index, 第3行 = row 2
    # Sequence length:
    # - 0 / None: auto infer from row content (recommended for mixed 9/11 CSV)
    # - 9 or 11 : force
    "n_surf": 0,
    # pred-only: CT column offset.
    # IMPORTANT: in this repo's pred CSV, it's typically:
    #   - n_surf=9  -> offset_ct=6
    #   - n_surf=11 -> offset_ct=0  (the first CT pair is usually STOP: (0, ~39mm))
    # Set to None to auto-infer from n_surf.
    "offset_ct": None,
    # orig-only: treat CT[:,0] as radius and convert to curvature
    "ct_is_radius": False,
    # Optical/system constants
    "epd": 4.0,          # Entrance pupil diameter (mm)
    # Aperture mode for ZMX header:
    # - "fnum": write `FNUM {FN} 0` (Image Space F/# style used by common Zemax imaging samples)
    # - "enpd": write `ENPD {epd}`
    "aperture_mode": "fnum",
    # HFOV interpretation (FIXED):
    # - CSV col 1 is HALF field angle HFOV in degrees.
    "semi_diam": 50.0,   # Zemax semi-diameter for non-stop surfaces (large avoids clipping)
    "semi_diam_stop": None,  # Stop semi-diameter; default = epd/2 when stop is detected
    "r_sensor": 20.0,    # Sensor semi-diameter used in JSON/ZMX image surface
    # Output
    "out_dir": "exports",
    "basename": "",      # optional, without extension; default auto from csv+row
    # Optional: inject matched glass names into JSON/ZMX using this file
    "glass_matching_csv": r"log/260120/stage1/stage_1.0/glass_matching_results_train_output_metrics_pred_rmsfilter_on.csv",
}


def _read_csv_row_as_floats(path: str, row_idx: int) -> List[float]:
    if row_idx < 0:
        raise ValueError(f"row must be >= 0, got {row_idx}")
    with open(path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == row_idx:
                try:
                    return [float(x) for x in row]
                except Exception as e:
                    raise ValueError(f"Failed parsing row {row_idx} in {path}") from e
    raise IndexError(f"Row {row_idx} out of range for {path}")


def _read_pred_row_from_vals(
    vals: Sequence[float], n_surf: int, offset_ct: int
) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    """
    Parse one 'pred' row from an already-loaded float array.

    Layout (see USL_Loss.read_scan_lens_csv):
      col 0..1: X = [FN, HFOV]
      col 2 .. 2+3*n_surf-1: Y (N_bgr)
      col 2+3*n_surf+offset_ct .. 2+5*n_surf+offset_ct-1: CT (curv, thick)
    """
    if n_surf <= 0:
        raise ValueError(f"n_surf must be positive, got {n_surf}")
    if offset_ct < 0:
        raise ValueError(f"offset_ct must be >=0, got {offset_ct}")
    if len(vals) < 2 + 5 * n_surf + offset_ct:
        raise ValueError(f"Row has {len(vals)} cols, but need at least {2 + 5*n_surf + offset_ct}.")

    fn, hfov = float(vals[0]), float(vals[1])

    y0 = 2
    y1 = 2 + 3 * n_surf
    y = vals[y0:y1]
    n_bgr = [tuple(float(x) for x in y[i:i + 3]) for i in range(0, len(y), 3)]

    ct0 = 2 + 3 * n_surf + offset_ct
    ct1 = 2 + 5 * n_surf + offset_ct
    ct = vals[ct0:ct1]
    ct_pairs = [tuple(float(x) for x in ct[i:i + 2]) for i in range(0, len(ct), 2)]
    return (fn, hfov), n_bgr, ct_pairs


def _infer_pred_nsurf_and_offset(vals: Sequence[float], tol: float = 1e-10) -> Tuple[int, int]:
    """
    Auto-infer whether a mixed pred CSV row is a 9-surf or 11-surf system.

    Heuristic used in this repo:
    - Mixed pred CSV often stores max-11 N_bgr and max-11 CT, with 9-surf samples padded by zeros
      on the last 2 surfaces (both N_bgr and CT).
    - If both the last-2-surface N_bgr and the last-2-surface CT are (near) zero -> treat as 9.

    Returns: (n_surf, offset_ct) where offset_ct is 6 for 9-surf, 0 for 11-surf.
    """
    # Indices for a max-11 layout
    max_n = 11
    y0 = 2
    y1 = y0 + 3 * max_n  # 2 + 33 = 35
    ct0 = y1             # 35
    ct1 = ct0 + 2 * max_n  # 35 + 22 = 57

    # If row is too short for the max-11 blocks, fall back to a safe guess:
    # try 9-surf first (it needs fewer columns), otherwise 11-surf.
    if len(vals) < ct1:
        if len(vals) >= 2 + 5 * 9 + 6:  # minimal for (9, offset=6)
            return 9, 6
        if len(vals) >= 2 + 5 * 9:      # minimal for (9, offset=0)
            return 9, 0
        return 11, 0

    # Last 2 surfaces in N_bgr (surface idx 9,10; 0-based)
    y_pad = vals[y0 + 3 * 9: y1]  # 6 numbers
    # Last 2 surfaces in CT (4 numbers)
    ct_pad = vals[ct0 + 2 * 9: ct1]

    def _all_near_zero(seq: Sequence[float]) -> bool:
        return all(abs(float(x)) <= tol for x in seq)

    is_9 = _all_near_zero(y_pad) and _all_near_zero(ct_pad)
    return (9, 6) if is_9 else (11, 0)


def _read_glass_names_for_row(glass_matching_csv: str, original_row_idx: int) -> dict:
    """
    Read glass names from the matching results CSV produced by verify_glass_name_matching.py.
    Returns: {layer_index(int, 1-based): glass_name(str)}
    Example keys in CSV: Layer_2_Name, Layer_4_Name, ...
    """
    if not glass_matching_csv:
        return {}
    if not os.path.exists(glass_matching_csv):
        raise FileNotFoundError(f"glass_matching_csv not found: {glass_matching_csv}")

    with open(glass_matching_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                idx = int(float(row.get("original_row_idx", -1)))
            except Exception:
                continue
            if idx != original_row_idx:
                continue

            out: dict[int, str] = {}
            for k, v in row.items():
                if not k or not k.endswith("_Name"):
                    continue
                name = (v or "").strip()
                if not name:
                    continue
                # Layer_{n}_Name
                try:
                    n_str = k.split("_", 2)[1]
                    layer_n = int(n_str)
                except Exception:
                    continue
                out[layer_n] = name
            return out

    # not found is OK (some rows may not have matches)
    return {}


def read_pred_row(path: str, n_surf: int, row: int, offset_ct: int) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    """
    Prediction CSV layout (see USL_Loss.read_scan_lens_csv):
      col 0..1: X = [FN, HFOV]
      col 2 .. 2+3*n_surf-1: Y (N_bgr)
      col 2+3*n_surf+offset_ct .. 2+5*n_surf+offset_ct-1: CT (curv, thick)
    """
    vals = _read_csv_row_as_floats(path, row)
    return _read_pred_row_from_vals(vals, n_surf=n_surf, offset_ct=offset_ct)


def read_orig_row(path: str, n_surf: int, row: int) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]], List[Tuple[float, float]]]:
    """
    Ground-truth CSV layout (see USL_Loss.read_orig_scan_lens_csv):
      col 0..1: X = [FN, HFOV]
      col 2   : (usually) nSurf_total in the file, ignored here
      col 3 .. 3+3*n_surf-1: Y (N_bgr)
      col (3+3*n_surf)+3 .. + 2*n_surf-1 : CT (radius, thick) in many datasets
    """
    vals = _read_csv_row_as_floats(path, row)
    if len(vals) < 3 + 3 * n_surf + 3 + 2 * n_surf:
        raise ValueError(f"Row has {len(vals)} cols, but need at least {3 + 3*n_surf + 3 + 2*n_surf}.")
    fn, hfov = vals[0], vals[1]

    y0 = 3
    y1 = 3 + 3 * n_surf
    y = vals[y0:y1]
    n_bgr = [tuple(y[i:i + 3]) for i in range(0, len(y), 3)]

    ct0 = y1 + 3
    ct1 = ct0 + 2 * n_surf
    ct = vals[ct0:ct1]
    ct_pairs = [tuple(ct[i:i + 2]) for i in range(0, len(ct), 2)]
    return (fn, hfov), n_bgr, ct_pairs


def _abbe_vd(n_f: float, n_d: float, n_c: float) -> float:
    denom = (n_f - n_c)
    if abs(denom) < 1e-12:
        return 0.0
    return (n_d - 1.0) / denom


def _infer_glass_name_from_nd(n_d: float) -> Optional[str]:
    """Infer the discrete OTS material name used by this scan-lens dataset."""
    known = {
        1.51712: "N-BK7",
        1.78590: "N-SF11",
        1.67352: "N-SF5",
        1.45874: "C79-80",
        1.43403: "CAF2",
    }
    nd = float(n_d)
    ref_nd, name = min(known.items(), key=lambda item: abs(item[0] - nd))
    return name if abs(ref_nd - nd) <= 0.003 else None


@dataclass
class ScanLensSurface:
    """One sequential surface in Zemax terms (space after surface has material n_bgr)."""
    d: float                 # absolute z position of the surface
    curv: float              # curvature c = 1/R
    n_bgr: Tuple[float, float, float]  # (n_F, n_d, n_C) aligned with WAVL in lens_export_utils
    semi_diam: float = 50.0
    is_stop: bool = False
    glass_name: Optional[str] = None

    def surf_dict(self) -> dict:
        n_f, n_d, n_c = self.n_bgr
        is_air = abs(n_d - 1.0) < 1e-6 and abs(n_f - 1.0) < 1e-6 and abs(n_c - 1.0) < 1e-6
        vd = _abbe_vd(n_f, n_d, n_c) if not is_air else None
        return {
            "d": self.d,
            "curv": self.curv,
            "n_bgr": [float(self.n_bgr[0]), float(self.n_bgr[1]), float(self.n_bgr[2])],
            "semi_diam": self.semi_diam,
            "is_stop": bool(self.is_stop),
            "material": {
                "type": "air" if is_air else "glass",
                "name": None if is_air else (self.glass_name or _infer_glass_name_from_nd(n_d)),
                "nd": None if is_air else float(n_d),
                "vd": None if is_air else float(vd),
            },
        }

    def zmx_str(self, surf_idx: int, d_next) -> str:
        # d_next can be torch scalar / numpy scalar / float; str() keeps precision reasonably.
        disz = float(d_next.item()) if hasattr(d_next, "item") else float(d_next)

        n_f, n_d, n_c = self.n_bgr
        is_air = abs(n_d - 1.0) < 1e-6 and abs(n_f - 1.0) < 1e-6 and abs(n_c - 1.0) < 1e-6

        # Use a simple user-defined glass if it's not air.
        # Zemax accepts ___BLANK with (nd, Vd) style fields in many exports; keep it minimal.
        glass_line = ""
        if not is_air:
            vd = _abbe_vd(n_f, n_d, n_c)
            # If we have a matched glass name (e.g. N-BK7 / N-SF11 / CAF2 / C79-80), write it.
            # Otherwise fall back to ___BLANK but still provide nd/vd.
            glass_name = self.glass_name or _infer_glass_name_from_nd(n_d) or "___BLANK"
            glass_line = f"GLAS {glass_name} 0 0 {n_d:.6f} {vd:.6f} 0 0 0 0 0 0\n"

        stop_line = "STOP\n" if self.is_stop else ""
        return (
            f"SURF {surf_idx}\n"
            f"{stop_line}"
            f"TYPE STANDARD\n"
            f"CURV {self.curv:.16E}\n"
            f"DISZ {disz:.16g}\n"
            f"{glass_line}"
            f"DIAM {self.semi_diam}\n"
        )


@dataclass
class ScanLens:
    """Minimal lens object compatible with LensExporter."""
    foclen: float
    fnum: float
    enpd: float
    float_enpd: bool
    aperture_mode: str
    rfov: float  # radians (LensExporter converts to degrees internally)
    r_sensor: float
    d_sensor: float
    sensor_size: Sequence[float]
    surfaces: List[ScanLensSurface]
    lens_info: str = "scanlens_from_csv"


def build_scanlens_from_row(
    fn: float,
    hfov_deg: float,
    n_bgr: List[Tuple[float, float, float]],
    ct: List[Tuple[float, float]],
    *,
    epd: float,
    ct_is_radius: bool,
    semi_diam: float,
    semi_diam_stop: Optional[float],
    r_sensor: float,
    aperture_mode: str,
    glass_names_by_layer: Optional[dict],
) -> ScanLens:
    if len(n_bgr) != len(ct):
        raise ValueError(f"n_bgr len {len(n_bgr)} must equal CT len {len(ct)}")

    # Convert HFOV degrees to radians so LensExporter head_str prints sensible degrees.
    rfov_rad = hfov_deg / 57.3

    # Build surfaces with absolute positions.
    d = 0.0
    surfaces: List[ScanLensSurface] = []
    # Heuristic: detect STOP as the first surface when it is a plane (curv==0) and medium is air.
    # This matches the 11-seq pred CSV in this repo where CT[0] is often (0, ~39mm).
    stop_detected = (
        len(ct) > 0
        and abs(float(ct[0][0])) < 1e-12
        and len(n_bgr) > 0
        and all(abs(float(x) - 1.0) < 1e-6 for x in n_bgr[0])
    )
    stop_sd = (epd / 2.0) if (semi_diam_stop is None) else float(semi_diam_stop)
    for (nF, nd, nC), (c_or_r, thick) in zip(n_bgr, ct):
        if ct_is_radius:
            r = float(c_or_r)
            curv = 0.0 if abs(r) < 1e-12 else 1.0 / r
        else:
            curv = float(c_or_r)

        layer_idx = len(surfaces) + 1  # 1-based layer index matching "Layer_{k}_Name"
        glass_name = None
        if glass_names_by_layer:
            glass_name = glass_names_by_layer.get(layer_idx)

        surfaces.append(
            ScanLensSurface(
                d=d,
                curv=curv,
                n_bgr=(float(nF), float(nd), float(nC)),
                semi_diam=(stop_sd if (stop_detected and len(surfaces) == 0) else semi_diam),
                is_stop=bool(stop_detected and len(surfaces) == 0),
                glass_name=glass_name,
            )
        )
        d += float(thick)

    # Focal length: for this project EFL_ideal is often FN * EPD.
    foclen = float(fn) * float(epd)

    return ScanLens(
        foclen=foclen,
        fnum=float(fn),
        enpd=float(epd),
        float_enpd=False,
        aperture_mode=str(aperture_mode),
        rfov=float(rfov_rad),
        r_sensor=float(r_sensor),
        d_sensor=float(d),
        sensor_size=(2 * r_sensor, 2 * r_sensor),
        surfaces=surfaces,
        lens_info="scanlens_from_csv_row",
    )


def export_from_config(cfg: dict) -> Tuple[str, str]:
    """Export using a python dict config (used by DEFAULT_EXPORT_CONFIG and CLI). Returns (json_path, zmx_path)."""
    fmt = cfg["format"]
    csv_path = cfg["csv"]
    row = int(cfg["row"])
    n_surf_raw = cfg.get("n_surf", 0)
    n_surf = 0 if (n_surf_raw is None) else int(n_surf_raw)
    offset_ct_raw = cfg.get("offset_ct", None)
    ct_is_radius = bool(cfg.get("ct_is_radius", False))
    epd = float(cfg.get("epd", 4.0))
    aperture_mode = str(cfg.get("aperture_mode", "fnum")).strip().lower()
    semi_diam = float(cfg.get("semi_diam", 50.0))
    semi_diam_stop = cfg.get("semi_diam_stop", None)
    r_sensor = float(cfg.get("r_sensor", 20.0))
    out_dir = cfg.get("out_dir", "exports")
    basename = cfg.get("basename", "")
    glass_matching_csv = cfg.get("glass_matching_csv", "")

    os.makedirs(out_dir, exist_ok=True)

    if fmt == "pred":
        # Load row once; enables auto-infer without re-reading CSV multiple times.
        vals = _read_csv_row_as_floats(csv_path, row)

        # Auto infer n_surf (9 vs 11) when n_surf <= 0.
        inferred = False
        if n_surf <= 0:
            n_surf, inferred_offset = _infer_pred_nsurf_and_offset(vals)
            offset_ct = inferred_offset if (offset_ct_raw is None) else int(offset_ct_raw)
            inferred = True
        else:
            # pred: allow None -> auto infer (important for 11-seq where offset_ct must be 0)
            if offset_ct_raw is None:
                if n_surf == 9:
                    offset_ct = 6
                elif n_surf == 11:
                    offset_ct = 0
                else:
                    raise ValueError(
                        f"pred format: cannot auto-infer offset_ct for n_surf={n_surf}; set offset_ct explicitly."
                    )
            else:
                offset_ct = int(offset_ct_raw)

        if inferred:
            print(f"[Export] pred auto-infer: n_surf={n_surf}, offset_ct={offset_ct}")
        (fn, hfov), n_bgr, ct = _read_pred_row_from_vals(vals, n_surf=n_surf, offset_ct=offset_ct)
    else:
        if n_surf <= 0:
            raise ValueError("orig format requires n_surf to be explicitly set (e.g. 9 or 11).")
        (fn, hfov), n_bgr, ct = read_orig_row(csv_path, n_surf, row)

    # FIXED: CSV col 1 is HALF field angle in degrees
    hfov_deg = float(hfov)

    glass_names_by_layer = {}
    if glass_matching_csv:
        glass_names_by_layer = _read_glass_names_for_row(glass_matching_csv, original_row_idx=row)

    lens = build_scanlens_from_row(
        fn=fn,
        hfov_deg=hfov_deg,
        n_bgr=n_bgr,
        ct=ct,
        epd=epd,
        ct_is_radius=ct_is_radius,
        semi_diam=semi_diam,
        semi_diam_stop=semi_diam_stop,
        r_sensor=r_sensor,
        aperture_mode=aperture_mode,
        glass_names_by_layer=glass_names_by_layer,
    )

    if basename:
        base = basename
    else:
        csv_name = os.path.splitext(os.path.basename(csv_path))[0]
        base = f"{csv_name}_row{row}_nsurf{n_surf}"

    json_path = os.path.join(out_dir, base + ".json")
    zmx_path = os.path.join(out_dir, base + ".zmx")

    exporter = LensExporter(lens)
    exporter.write_lens_json(json_path)
    exporter.write_lens_zmx(zmx_path)
    return json_path, zmx_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", choices=["pred", "orig"], required=True, help="CSV format kind")
    ap.add_argument("--csv", required=True, help="CSV file path")
    ap.add_argument("--row", type=int, required=True, help="0-based row index")
    ap.add_argument("--n_surf", type=int, default=0, help="sequence length (9/11). Use 0 to auto-infer (pred only)")
    ap.add_argument("--offset_ct", type=int, default=-1, help="pred-only: CT column offset. Use -1 to auto (recommended)")
    ap.add_argument("--ct_is_radius", action="store_true", help="treat CT[:,0] as radius and convert to curvature (orig datasets often need this)")
    ap.add_argument("--epd", type=float, default=4.0, help="Entrance pupil diameter (mm)")
    ap.add_argument("--aperture_mode", choices=["fnum", "enpd"], default="fnum", help="ZMX aperture header mode")
    ap.add_argument("--semi_diam", type=float, default=50.0, help="Zemax semi-diameter for all surfaces (large avoids clipping)")
    ap.add_argument("--semi_diam_stop", type=float, default=None, help="Zemax semi-diameter for STOP surface (default: epd/2 when stop is detected)")
    ap.add_argument("--r_sensor", type=float, default=20.0, help="Sensor semi-diameter used in JSON/ZMX image surface")
    ap.add_argument("--glass_matching_csv", type=str, default="", help="Optional: glass matching CSV (with Layer_k_Name columns) to inject glass names into ZMX/JSON")
    ap.add_argument("--out_dir", type=str, default="exports", help="Output directory")
    ap.add_argument("--basename", type=str, default="", help="Output base name (without extension). Default auto from csv+row")
    args = ap.parse_args()

    cfg = {
        "format": args.format,
        "csv": args.csv,
        "row": args.row,
        "n_surf": args.n_surf,
        "offset_ct": None if args.offset_ct < 0 else args.offset_ct,
        "ct_is_radius": args.ct_is_radius,
        "epd": args.epd,
        "aperture_mode": args.aperture_mode,
        "semi_diam": args.semi_diam,
        "semi_diam_stop": args.semi_diam_stop,
        "r_sensor": args.r_sensor,
        "glass_matching_csv": args.glass_matching_csv,
        "out_dir": args.out_dir,
        "basename": args.basename,
    }
    export_from_config(cfg)


if __name__ == "__main__":
    # If user doesn't want CLI: just edit DEFAULT_EXPORT_CONFIG and run this file.
    if len(sys.argv) == 1:
        export_from_config(DEFAULT_EXPORT_CONFIG)
    else:
        main()
