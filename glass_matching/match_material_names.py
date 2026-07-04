#!/usr/bin/env python3
"""
给 glass/Material_C_T_Data.csv 补齐玻璃牌号（来自 schott_glass_with_n.xlsx）

Material_C_T_Data.csv 格式（每行6列）：
  [n1, n2, n3, R1, R2, Thickness]

schott_glass_with_n.xlsx：
  - 第一列：玻璃牌号（name）
  - 折射率列：列名以 "n_" 开头（例如 n_486.1nm / n_587.6nm / n_656.3nm）

输出：
  默认写到 glass/Material_C_T_Data_with_name.csv
  列为：name, n1, n2, n3, R1, R2, Thickness
"""

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def _find_ri_cols(df: pd.DataFrame) -> List[str]:
    ri_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("n_")]

    def _wave_key(col: str) -> float:
        # 尝试解析 n_587.6nm / n_587nm / n_587.6 这类
        try:
            s = col.split("_", 1)[1]
            s = s.replace("nm", "")
            return float(s)
        except Exception:
            return 0.0

    ri_cols = sorted(ri_cols, key=_wave_key)
    return ri_cols


def load_schott_table(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schott excel not found: {path}")

    df = pd.read_excel(path)
    if df.shape[1] < 4:
        raise ValueError("Schott 表列数不足：至少需要 name 列 + 3 个折射率列")

    ri_cols = _find_ri_cols(df)
    if len(ri_cols) < 3:
        raise ValueError("Schott 表缺少折射率列（期望至少 3 个以 n_ 开头的列）")

    names = df.iloc[:, 0].astype(str).tolist()
    ri = df[ri_cols[:3]].to_numpy(dtype=np.float64)
    return ri, names, ri_cols[:3]


def match_names(
    material_csv: str,
    schott_xlsx: str,
    out_csv: str,
    ri_tol: float = 1e-4,
) -> None:
    if not os.path.exists(material_csv):
        raise FileNotFoundError(f"Material file not found: {material_csv}")

    mat = np.loadtxt(material_csv, delimiter=",", dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 6:
        raise ValueError(f"Material_C_T_Data.csv 格式不对，期望至少 6 列，实际: {mat.shape}")

    mat_ri = mat[:, :3]  # (N,3)
    schott_ri, schott_names, ri_cols = load_schott_table(schott_xlsx)

    # 计算每个 material 到所有 schott 的距离，取最近邻
    # (N,1,3) - (1,M,3) -> (N,M,3) -> (N,M)
    diff = mat_ri[:, None, :] - schott_ri[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    min_idx = np.argmin(dists, axis=1)
    min_dist = dists[np.arange(dists.shape[0]), min_idx]

    names_out = [
        schott_names[i] if min_dist[r] <= ri_tol else f"Unknown_{r+1}"
        for r, i in enumerate(min_idx)
    ]

    out = pd.DataFrame(
        {
            "name": names_out,
            "n1": mat[:, 0],
            "n2": mat[:, 1],
            "n3": mat[:, 2],
            "R1": mat[:, 3],
            "R2": mat[:, 4],
            "Thickness": mat[:, 5],
        }
    )

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")

    # 打印摘要
    n_unknown = int((out["name"].str.startswith("Unknown_")).sum())
    print("=== Material -> Schott name matching ===")
    print(f"- Material rows: {len(out)}")
    print(f"- Schott RI columns used: {ri_cols}")
    print(f"- ri_tol: {ri_tol}")
    print(f"- Unknown rows: {n_unknown}")
    print(f"- Output: {out_csv}")


def ensure_material_library_exists(
    material_csv: str = "glass/Material_C_T_Data.csv",
    schott_xlsx: str = "glass/schott_glass_with_n.xlsx",
    out_csv: str = "glass/Material_C_T_Data_with_name.csv",
    ri_tol: float = 1e-4,
    force_update: bool = False,
) -> str:
    """
    确保带牌号的材料库存在，如果不存在则生成

    Args:
        material_csv: 原始材料数据文件路径
        schott_xlsx: Schott玻璃表文件路径
        out_csv: 输出文件路径
        ri_tol: 折射率匹配容忍度
        force_update: 是否强制重新生成

    Returns:
        带牌号的材料库文件路径
    """
    if not force_update and os.path.exists(out_csv):
        print(f"Named material library already exists: {out_csv}")
        return out_csv

    print(f"Generating named material library: {out_csv}")
    match_names(material_csv, schott_xlsx, out_csv, ri_tol)
    return out_csv


