#!/usr/bin/env python3
"""
验证折射率到玻璃牌号的反向匹配功能
并处理预测结果 CSV 文件，匹配每一片玻璃的牌号
"""

import torch
import numpy as np
import pandas as pd
import os
import sys

# 复用导出脚本里的"9/11 面 + offset_ct"自动识别逻辑，避免在 glass matching 里写一套会漂的规则
try:
    # 尝试多种导入方式
    try:
        from exports.scanlens_export_from_csv import _infer_pred_nsurf_and_offset  # type: ignore
    except ImportError:
        # 如果作为包导入失败，尝试直接导入
        import sys
        import os
        # 添加项目根目录到路径
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from exports.scanlens_export_from_csv import _infer_pred_nsurf_and_offset  # type: ignore
except Exception as e:
    print(f"Warning: 无法导入 _infer_pred_nsurf_and_offset: {e}")
    _infer_pred_nsurf_and_offset = None

# 尝试从 utils 导入，如果失败则忽略（兼容性）
try:
    from utils import get_OTS_CT
except ImportError:
    get_OTS_CT = None

# === 数据源 ===
# 只使用带牌号的材料库：name,n1,n2,n3,R1,R2,Thickness
MATERIAL_WITH_NAME_PATH = 'glass/Material_C_T_Data_with_name.csv'

class GlassLibrary:
    def __init__(self,
                 material_with_name_path=MATERIAL_WITH_NAME_PATH,
                 ri_tol=1e-4):
        """
        material_path: OTS 库的 RI+R1+R2+厚度（无名称）
        schott_path  : Schott 玻璃表，包含牌号与对应 RI
        ri_tol       : RI 匹配容忍度
        """
        self.data = []  # List[dict]: {name, ri, r1, r2, th, c1, c2, index}
        self._load_library(material_with_name_path, ri_tol)
    
    def _load_library(self, material_with_name_path, ri_tol):
        """
        只读取带 name 的材料库 CSV：name,n1,n2,n3,R1,R2,Thickness
        """
        if not os.path.exists(material_with_name_path):
            raise FileNotFoundError(
                f"Named material library not found: {material_with_name_path}\n"
                f"请先运行 match_material_names.py 生成该文件。"
            )

        print(f"Loading named materials from {material_with_name_path} ...")
        df = pd.read_csv(material_with_name_path)
        required_cols = {"name", "n1", "n2", "n3", "R1", "R2", "Thickness"}
        if not required_cols.issubset(set(df.columns)):
            raise ValueError(
                f"材料库缺少必要列：{required_cols}，实际列：{list(df.columns)}"
            )

        # ri_tol 仍保留：用于后续 match_glass_detailed 的误差权重/阈值策略扩展（目前 name 已固定）
        _ = ri_tol

        for i, row in df.iterrows():
            name = str(row["name"])
            ri_tensor = torch.tensor([row["n1"], row["n2"], row["n3"]], dtype=torch.float32)
            r1 = float(row["R1"])
            r2 = float(row["R2"])
            th = float(row["Thickness"])
            self.data.append({
                "index": i + 1,  # 行号 1-based（与该 CSV 顺序一致）
                "name": name,
                "ri": ri_tensor,
                "r1": r1,
                "r2": r2,
                "th": th,
                "c1": 1.0 / r1 if abs(r1) > 1e-9 else 0.0,
                "c2": 1.0 / r2 if abs(r2) > 1e-9 else 0.0,
            })

        print(f"Loaded {len(self.data)} named materials from CSV.")

    def match_glass_detailed(self, query_ri, query_c1, query_th, query_c2=None):
        """
        匹配最接近的玻璃条目，考虑 RI, C1, Th, C2
        query_ri: Tensor (3,)
        query_c1: float (curvature 1)
        query_th: float (thickness)
        query_c2: float (curvature 2, optional)
        
        Returns: best_entry (dict), min_error
        """
        if not self.data:
            return None, float('inf')
            
        best_entry = None
        min_error = float('inf')
        
        # Weights for distance metric
        # RI is small (~1.5), C is small (~0.02), Th is mid (~5)
        # Normalize: RI*100, C*1000, Th*1
        W_RI = 100.0
        W_C = 1000.0
        W_TH = 1.0
        
        for entry in self.data:
            # 1. RI Error
            ri_err = torch.norm(query_ri - entry['ri']).item()
            
            # 2. Geometry Error (Direct)
            # Match: (c1, th, c2) vs (lib_c1, lib_th, lib_c2)
            c1_err = abs(query_c1 - entry['c1'])
            th_err = abs(query_th - entry['th'])
            
            c2_err = 0.0
            if query_c2 is not None:
                c2_err = abs(query_c2 - entry['c2'])
            
            total_err = (ri_err * W_RI) + (c1_err * W_C) + (th_err * W_TH) + (c2_err * W_C)
            
            if total_err < min_error:
                min_error = total_err
                best_entry = entry
                
            # 3. Geometry Error (Flipped Lens)
            # Match: (c1, th, c2) vs (-lib_c2, lib_th, -lib_c1)
            # Note: Signs depend on convention. Assuming c = 1/R. 
            # If lens is flipped, R1_new = -R2_old, R2_new = -R1_old.
            
            c1_err_f = abs(query_c1 - (-entry['c2']))
            th_err_f = abs(query_th - entry['th'])
            
            c2_err_f = 0.0
            if query_c2 is not None:
                c2_err_f = abs(query_c2 - (-entry['c1']))
            
            total_err_f = (ri_err * W_RI) + (c1_err_f * W_C) + (th_err_f * W_TH) + (c2_err_f * W_C)

            if total_err_f < min_error:
                min_error = total_err_f
                # Make a copy to indicate flip if needed, or just return entry
                # For now just return the entry, user just wants to know "which piece"
                best_entry = entry
                
        return best_entry, min_error

    def match_glass(self, query_ris):
        """
        保留旧接口兼容性，只匹配 RI
        """
        if isinstance(query_ris, list):
            query_ris = torch.tensor(query_ris, dtype=torch.float32)
            
        if query_ris.dim() == 1:
            query_ris = query_ris.unsqueeze(0)
            
        names = []
        dists = []
        
        for i in range(query_ris.size(0)):
            ri = query_ris[i]
            # Find best match based on RI only
            best_entry = None
            min_dist = float('inf')
            
            for entry in self.data:
                dist = torch.norm(ri - entry['ri']).item()
                if dist < min_dist:
                    min_dist = dist
                    best_entry = entry
            
            if best_entry:
                names.append(best_entry['name'])
                dists.append(torch.tensor(min_dist)) # keep as tensor for consistency
            else:
                names.append("Unknown")
                dists.append(torch.tensor(float('inf')))
                
        return names, torch.tensor(dists)

def process_prediction_csv(input_file, output_file, glass_lib):
    """
    读取预测 CSV，识别折射率列和CT列，匹配玻璃，保存结果
    """
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return

    print(f"Processing {input_file}...")
    df = pd.read_csv(input_file, header=None)
    
    # 输入 CSV 是 mixed 9/11 面系统（padding 到 11 面），不能用“固定 11 面”去读每一行。
    # 必须对每一行先判断是 9 还是 11（以及 offset_ct），再用正确的 CT block 去取 (curv, thickness)。
    if _infer_pred_nsurf_and_offset is None:
        raise RuntimeError(
            "无法导入 exports.scanlens_export_from_csv._infer_pred_nsurf_and_offset，"
            "请确认从仓库根目录运行，并且 exports/scanlens_export_from_csv.py 存在。"
        )
    
    new_data = []
    
    for idx, row in df.iterrows():
        row_data = {'original_row_idx': idx}

        # -------- per-row infer 9/11 + offset --------
        vals = row.values.astype(float)
        n_surfaces, offset_ct = _infer_pred_nsurf_and_offset(vals)
        # pred CSV layout: X(2) + Y(3*n_surfaces) + offset_ct + CT(2*n_surfaces)
        ct_start_col = 2 + 3 * n_surfaces + offset_ct
        ri_col_indices = [2 + i * 3 for i in range(n_surfaces)]
        
        # Iterate over surfaces (only valid surfaces for this row)
        for k in range(n_surfaces):
            # RI
            ri_col = ri_col_indices[k]
            vals_ri = row[ri_col : ri_col+3].values.astype(float)
            ri_tensor = torch.tensor(vals_ri, dtype=torch.float32)
            
            # Skip if Air (RI ~ 1.0)
            if torch.mean(ri_tensor) < 1.1:
                continue
                
            # CT
            # Curvature is at ct_start_col + 2*k
            # Thickness is at ct_start_col + 2*k + 1
            ct_idx = ct_start_col + 2*k
            
            if ct_idx + 1 >= len(row):
                break
                
            c1_val = float(row[ct_idx])
            th_val = float(row[ct_idx+1])
            
            # Try to get next surface curvature (c2)
            c2_val = None
            if k + 1 < n_surfaces:
                ct_next_idx = ct_start_col + 2*(k+1)
                if ct_next_idx < len(row):
                    c2_val = float(row[ct_next_idx])
            
            # Match
            best_entry, err = glass_lib.match_glass_detailed(ri_tensor, c1_val, th_val, c2_val)
            
            if best_entry:
                prefix = f'Layer_{k+1}'
                row_data[f'{prefix}_Name'] = best_entry['name']
                row_data[f'{prefix}_LibIdx'] = best_entry['index']
                row_data[f'{prefix}_MatchErr'] = f"{err:.4f}"
                row_data[f'{prefix}_Pred_R1'] = "Plane" if abs(c1_val) <= 1e-6 else f"{1.0/c1_val:.2f}"
                row_data[f'{prefix}_Pred_Th'] = f"{th_val:.3f}"
                row_data[f'{prefix}_Lib_R1'] = "Plane" if abs(best_entry['r1']) <= 1e-9 else f"{best_entry['r1']:.2f}"
                row_data[f'{prefix}_Lib_Th'] = f"{best_entry['th']:.3f}"
                row_data[f'{prefix}_Lib_R2'] = "Plane" if abs(best_entry['r2']) <= 1e-9 else f"{best_entry['r2']:.2f}"

        new_data.append(row_data)

    res_df = pd.DataFrame(new_data)
    res_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def process_test_output_csv(
    input_csv: str,
    output_csv: str = None,
    material_lib_path: str = MATERIAL_WITH_NAME_PATH,
    ri_tol: float = 1e-4,
) -> str:
    """
    处理测试输出CSV文件，为预测结果匹配玻璃牌号

    Args:
        input_csv: 输入的预测结果CSV文件路径
        output_csv: 输出的玻璃匹配结果CSV文件路径，如果为None则自动生成
        material_lib_path: 带牌号的材料库文件路径
        ri_tol: 折射率匹配容忍度

    Returns:
        输出文件路径
    """
    if output_csv is None:
        # 自动生成输出路径
        base_name = os.path.basename(input_csv)
        dir_name = os.path.dirname(input_csv)
        name_without_ext = os.path.splitext(base_name)[0]
        output_csv = os.path.join(dir_name, f"glass_matching_results_{name_without_ext}.csv")

    # 初始化玻璃库
    glass_lib = GlassLibrary(material_lib_path, ri_tol)

    # 执行处理
    if os.path.exists(input_csv):
        process_prediction_csv(input_csv, output_csv, glass_lib)
        return output_csv
    else:
        print(f"Input CSV not found: {input_csv}")
        return None


if __name__ == "__main__":
    # 默认处理 train_output_metrics_pred_rmsfilter_on.csv
    import argparse
    
    parser = argparse.ArgumentParser(description="匹配预测结果CSV中的玻璃牌号")
    parser.add_argument(
        "--input_csv",
        type=str,
        default=r"log/260120/stage1/stage_1.0/train_output_metrics_pred_rmsfilter_on.csv",
        help="输入的预测结果CSV文件路径"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出的玻璃匹配结果CSV文件路径（默认自动生成）"
    )
    parser.add_argument(
        "--material_lib",
        type=str,
        default=MATERIAL_WITH_NAME_PATH,
        help="带牌号的材料库文件路径"
    )
    parser.add_argument(
        "--ri_tol",
        type=float,
        default=1e-4,
        help="折射率匹配容忍度"
    )
    
    args = parser.parse_args()
    
    output_csv = process_test_output_csv(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        material_lib_path=args.material_lib,
        ri_tol=args.ri_tol
    )
    
    if output_csv:
        print(f"\n玻璃匹配完成！输出文件: {output_csv}")
    else:
        print("\n玻璃匹配失败！")
