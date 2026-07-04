"""
点列图可视化模块 - Zemax风格
只使用绿色波段，分别显示三个视场的点列图
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import torch
from typing import Optional, Union


def _infer_pred_nsurf_and_offset(vals, tol=1e-10):
    """Infer 9/11-surface prediction CSV layout and CT offset."""
    max_n = 11
    y0 = 2
    y1 = y0 + 3 * max_n
    ct0 = y1
    ct1 = ct0 + 2 * max_n

    if len(vals) < ct1:
        if len(vals) >= 2 + 5 * 9 + 6:
            return 9, 6
        if len(vals) >= 2 + 5 * 9:
            return 9, 0
        return 11, 0

    y_pad = vals[y0 + 3 * 9:y1]
    ct_pad = vals[ct0 + 2 * 9:ct1]

    def _all_near_zero(seq):
        return all(abs(float(x)) <= tol for x in seq)

    return (9, 6) if _all_near_zero(y_pad) and _all_near_zero(ct_pad) else (11, 0)


def _read_pred_row(csv_path, row_idx):
    """Read one mixed prediction CSV row into FN/HFOV, N_bgr, CT arrays."""
    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if row_idx < 0 or row_idx >= data.shape[0]:
        raise IndexError(f"row_idx {row_idx} out of range for {data.shape[0]} rows")

    vals = data[row_idx]
    n_surf, offset_ct = _infer_pred_nsurf_and_offset(vals)
    needed = 2 + 5 * n_surf + offset_ct
    if vals.shape[0] < needed:
        raise ValueError(f"CSV row has {vals.shape[0]} columns, but needs at least {needed}")

    fn = float(vals[0])
    hfov = float(vals[1])
    n_bgr = vals[2:2 + 3 * n_surf].reshape(1, n_surf, 3)
    ct_start = 2 + 3 * n_surf + offset_ct
    ct = vals[ct_start:ct_start + 2 * n_surf].reshape(1, n_surf, 2)
    return fn, hfov, n_surf, n_bgr, ct


class SpotDiagramCSVLensSystem:
    """Minimal CSV lens wrapper needed for spot diagram ray tracing."""

    def __init__(self, csv_path: str, row_idx: int = 0, device: Optional[str] = None):
        from USL_Loss import USL_Loss

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.dtype = torch.float64
        self.row_idx = row_idx

        self.fn, self.hfov, self.n_surf, n_bgr, ct = _read_pred_row(csv_path, row_idx)
        self.N_bgr = torch.tensor(n_bgr, dtype=self.dtype, device=self.device, requires_grad=False)
        self.CT = torch.tensor(ct, dtype=self.dtype, device=self.device, requires_grad=False)
        self.X = torch.tensor([[self.fn, self.hfov]], dtype=self.dtype, device=self.device, requires_grad=False)

        class Opt:
            def __init__(self, n_surf):
                self.N = 11
                self.M = 3
                self.nRayDensity = 11
                self.nField = 3
                self.nWL = 3
                self.EPD = 4.0
                self.nSurf = n_surf
                self.max_surf = n_surf
                self.w_rms = 1.0
                self.w_distortion = 1.0
                self.w_tele = 1.0
                self.enable_rms_filter = False
                self.enable_efl_filter = False

        self.raytracer = USL_Loss(Opt(self.n_surf), dtype=self.dtype).to(self.device)
        self.air_gap_params_list = self._build_air_gap_params()

    def _build_air_gap_params(self):
        air_gap_params = []
        n_bgr_np = self.N_bgr.detach().cpu().numpy()[0]
        for idx, n_bgr in enumerate(n_bgr_np):
            if np.allclose(n_bgr, 1.0, atol=1e-3):
                air_gap_params.append({
                    "surface_idx": idx,
                    "param": self.CT[0, idx, 1].clone().detach(),
                })
        return air_gap_params


def extract_green_spot_data(outRays, X_data, n_field=3):
    """
    从光线追踪结果中提取绿色波段的点列图数据
    
    Args:
        outRays: 光线追踪结果，形状 (nTotal, 6, nW)
            - nTotal = nLens * nSample * nField
            - 6: (x, y, z, vx, vy, vz)
            - nW = 3: (B, G, R)
        X_data: 系统参数，形状 (nLens, 2) [FN, HFOV]
        n_field: 视场数量，默认3
    
    Returns:
        imax_green: 绿色波段x坐标，形状 (nLens, nField, nSample)
        imay_green: 绿色波段y坐标（相对于绿色主光线），形状 (nLens, nField, nSample)
    """
    nW = 3  # B, G, R
    nL = X_data.shape[0]
    nTotal = outRays.shape[0]
    nS = nTotal // (nL * n_field)
    
    # 提取x和y坐标
    imx = outRays[:, 0, :]  # (nTotal, nW)
    imy = outRays[:, 1, :]  # (nTotal, nW)
    
    # 按照USL_Loss._spot_rms的方式重新组织：view(nL, nS, nT, nW) 然后 permute
    # 注意：outRays的排列顺序是 (lens, sample, field)，所以view(nL, nS, n_field, nW)是正确的
    imx4d = imx.view(nL, nS, n_field, nW)  # (nLens, nSample, nField, nW)
    imy4d = imy.view(nL, nS, n_field, nW)  # (nLens, nSample, nField, nW)
    
    # 转置为 (nW, nLens, nField, nSample)，与USL_Loss保持一致
    imx4d = imx4d.permute(3, 0, 2, 1)  # (nW, nLens, nField, nSample)
    imy4d = imy4d.permute(3, 0, 2, 1)  # (nW, nLens, nField, nSample)
    
    # 提取绿色波段（索引1）
    imx_green = imx4d[1, :, :, :]  # (nLens, nField, nSample)
    imy_green = imy4d[1, :, :, :]  # (nLens, nField, nSample)
    
    # 找到绿色主光线（中心采样点）
    cent = nS // 2
    cimy_green = imy_green[:, :, cent:cent+1]  # (nLens, nField, 1) - 主光线y坐标
    
    # 计算相对于主光线的偏移（与USL_Loss._spot_rms一致）
    imax_green = imx_green  # x方向相对于主光线（主光线x=0）
    imay_green = imy_green - cimy_green  # y方向相对于主光线，广播到 (nLens, nField, nSample)
    
    return imax_green, imay_green


def draw_spot_diagram_green(
    imax_green: Union[torch.Tensor, np.ndarray],
    imay_green: Union[torch.Tensor, np.ndarray],
    scale: Optional[float] = None,
    field_labels: Optional[list] = None,
    figsize: tuple = (13.5, 5.0),
    dpi: int = 150,
    save_path: Optional[str] = None,
    title: str = "Spot Diagram (Green Light)"
):
    """
    绘制Zemax风格的点列图（只使用绿色波段）
    
    Args:
        imax_green: x坐标（相对于主光线），形状 (nLens, nField, nSample) 或 (nField, nSample)
        imay_green: y坐标（相对于主光线），形状 (nLens, nField, nSample) 或 (nField, nSample)
        scale: 显示范围（微米），如果为None则自动计算
        field_labels: 视场标签列表，如 ['0°', 'HFOV/2', 'HFOV']
        figsize: 图形大小
        dpi: 分辨率
        save_path: 保存路径，如果为None则不保存
        title: 图形标题
    
    Returns:
        fig, axes: matplotlib图形和坐标轴对象
    """
    # 转换为numpy数组
    if isinstance(imax_green, torch.Tensor):
        imax_green = imax_green.detach().cpu().numpy()
    if isinstance(imay_green, torch.Tensor):
        imay_green = imay_green.detach().cpu().numpy()
    
    # 处理维度：如果是3维，取第一个镜头
    if imax_green.ndim == 3:
        imax_green = imax_green[0]  # (nField, nSample)
        imay_green = imay_green[0]  # (nField, nSample)
    
    # 转换为微米
    imax_green = imax_green * 1000  # mm -> um
    imay_green = imay_green * 1000  # mm -> um
    
    nField, nSample = imax_green.shape
    
    # 自动计算scale（如果未提供）
    if scale is None:
        # 计算每个视场的RMS，取最大值
        rms_per_field = np.sqrt(np.mean(imax_green**2 + imay_green**2, axis=1))
        max_rms = np.max(rms_per_field)
        scale = max_rms * 1.5  # 显示范围是最大RMS的1.5倍（只比最大RMS大一些）
        scale = max(scale, 10)  # 至少10微米
    
    # 默认视场标签
    if field_labels is None:
        field_labels = [f'Field {i+1}' for i in range(nField)]
    
    # 创建子图：一行三列
    fig, axes = plt.subplots(1, nField, figsize=figsize, dpi=dpi)
    if nField == 1:
        axes = [axes]  # 确保axes是列表
    
    # 定义三个视场的颜色和标记形状：center蓝色圆点，mid绿色+号，edge红色三角形
    field_colors = ['blue', 'lime', 'red']  # center, mid, edge
    field_markers = ['o', '+', '^']  # center圆点，mid+号，edge三角形
    field_names = ['Center', 'Mid', 'Edge']  # 用于图例
    
    # 用于收集图例信息
    legend_handles = []
    
    # 绘制每个视场
    for i in range(nField):
        ax = axes[i]
        
        # 提取该视场的数据
        X = imax_green[i, :]
        Y = imay_green[i, :]
        
        # 使用对应视场的颜色和标记形状
        color = field_colors[i] if i < len(field_colors) else 'lime'
        marker = field_markers[i] if i < len(field_markers) else '+'
        field_name = field_names[i] if i < len(field_names) else f'Field {i+1}'
        
        # 绘制散点图，添加label用于图例
        # 对于圆点和三角形，调整大小和样式
        if marker == 'o':
            scatter = ax.scatter(X, Y, c=color, marker=marker, s=15, alpha=0.7, edgecolors='none', label=field_name)
        elif marker == '^':
            scatter = ax.scatter(X, Y, c=color, marker=marker, s=20, alpha=0.7, edgecolors='none', label=field_name)
        else:
            scatter = ax.scatter(X, Y, c=color, marker=marker, s=20, alpha=0.7, linewidths=0.5, label=field_name)
        
        # 收集图例句柄（只收集一次，避免重复）
        if i < len(field_colors):
            # 创建图例句柄，使用对应的标记形状
            legend_handles.append(plt.Line2D([0], [0], marker=marker, color=color, 
                                            markeredgecolor=color, markersize=10, 
                                            markeredgewidth=1.5 if marker != 'o' else 0,
                                            markerfacecolor=color if marker != '+' else 'none',
                                            label=field_name, 
                                            linestyle='None', alpha=0.7))
        
        # 设置等比例
        ax.set_aspect('equal', 'box')
        
        # 设置坐标范围（与总长度一致，不留边距，这样中括号能与边界对齐）
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        
        # 网格设置
        ax.grid(True, color='gainsboro', linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # 刻度设置
        major_locator = scale / 1
        minor_locator = scale / 5
        ax.xaxis.set_major_locator(plt.MultipleLocator(major_locator))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(minor_locator))
        ax.yaxis.set_major_locator(plt.MultipleLocator(major_locator))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(minor_locator))
        
        # Zemax风格：只显示纵坐标轴的总长度，不显示刻度标签
        ax.set_xticks([-scale, 0, scale])
        ax.set_xticklabels([])  # 横坐标不显示标签
        ax.set_yticks([-scale, 0, scale])
        ax.set_yticklabels([])  # 纵坐标也不显示刻度标签
        
        # 添加RMS信息
        rms = np.sqrt(np.mean(X**2 + Y**2))
        ax.text(0.02, 0.98, f'RMS: {rms:.2f} μm', 
                transform=ax.transAxes, 
                fontsize=15, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 设置标题
        ax.set_title(field_labels[i], fontsize=17, fontweight='bold')
        
        # Zemax风格：只在第一个子图显示纵坐标轴的总长度，带中括号标记
        if i == 0:
            total_length = scale * 2  # 总长度 = 2 * scale
            
            # 在纵坐标轴左侧绘制中括号标记，表示总长度
            bracket_width = scale * 0.03  # 中括号水平宽度（微米）
            bracket_x = -scale * 1.05  # 中括号x位置（在坐标轴左侧，稍微超出边界）
            bracket_y_top = scale  # 与图的上边界对齐
            bracket_y_bottom = -scale  # 与图的下边界对齐
            
            # 绘制中括号：竖线 + 上横线 + 下横线
            # 竖线
            ax.plot([bracket_x, bracket_x], [bracket_y_bottom, bracket_y_top], 
                   'k-', linewidth=1.5, clip_on=False)
            # 上横线（向右）
            ax.plot([bracket_x, bracket_x + bracket_width], 
                   [bracket_y_top, bracket_y_top], 
                   'k-', linewidth=1.5, clip_on=False)
            # 下横线（向右）
            ax.plot([bracket_x, bracket_x + bracket_width], 
                   [bracket_y_bottom, bracket_y_bottom], 
                   'k-', linewidth=1.5, clip_on=False)
            
            # 在中括号中间显示总长度（垂直显示）
            ax.text(bracket_x - bracket_width * 0.5, 0, 
                   f'{total_length:.0f} μm', 
                   fontsize=13, 
                   verticalalignment='center',
                   horizontalalignment='right',
                   rotation=90,
                   clip_on=False)
    
    # 整体标题（增加与图的距离）
    fig.suptitle(title, fontsize=17, fontweight='bold', y=1.05)
    
    # 在整个图形上添加统一的图例（右上角，垂直排列，白色背景框）
    if legend_handles:
        legend = fig.legend(handles=legend_handles, loc='upper right', 
                           bbox_to_anchor=(0.98, 1.05), ncol=1,  # 往上挪，y从0.98改为1.0
                           fontsize=14, framealpha=1.0, 
                           fancybox=True, shadow=False,  # 取消阴影
                           facecolor='white',  # 白色背景
                           edgecolor='black',  # 黑色边框
                           frameon=True)  # 显示边框
    
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # 为图例和标题留出更多空间
    
    # 保存
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Spot diagram saved to: {save_path}")
    
    return fig, axes


def plot_spot_diagram_from_lens_system(
    lens_system,
    scale: Optional[float] = None,
    save_path: Optional[str] = None,
    title: Optional[str] = None
):
    """
    从CSV lens system对象生成点列图
    
    Args:
        lens_system: CSV lens system对象
        scale: 显示范围（微米），如果为None则自动计算
        save_path: 保存路径
        title: 图形标题
    
    Returns:
        fig, axes: matplotlib图形和坐标轴对象
    """
    from utils import LensBatch
    
    # 构建CT（使用当前优化后的参数）
    CT_optimizable = lens_system.CT.clone()
    for param_dict in lens_system.air_gap_params_list:
        idx = param_dict['surface_idx']
        CT_optimizable[0, idx, 1] = param_dict['param']
    
    # 创建LensBatch
    lens_batch = LensBatch(lens_system.X, lens_system.N_bgr, CT_optimizable)
    
    # 光线追踪
    max_surf_lens = torch.tensor([lens_system.n_surf], dtype=torch.long, device=lens_system.device)
    surf_lens = torch.tensor([lens_system.n_surf], dtype=torch.long, device=lens_system.device)
    mask = torch.ones(1, lens_system.n_surf, dtype=torch.bool, device=lens_system.device)
    
    raytrace_out = lens_system.raytracer.raytrace_all(
        lens_batch, max_surf_lens, surf_lens, mask
    )
    outRays = raytrace_out[0]
    
    # 提取绿色波段数据
    imax_green, imay_green = extract_green_spot_data(
        outRays, lens_system.X, n_field=lens_system.raytracer.n_field
    )
    
    # 生成视场标签
    hfov_deg = lens_system.hfov
    field_labels = [
        '0° (Center)',
        f'{hfov_deg/2:.1f}° (Mid)',
        f'{hfov_deg:.1f}° (Edge)'
    ]
    
    # 生成标题
    if title is None:
        title = f"Spot Diagram - Green Light (FN={lens_system.fn:.2f}, HFOV={hfov_deg:.1f}°)"
    
    # 绘制
    fig, axes = draw_spot_diagram_green(
        imax_green, imay_green,
        scale=scale,
        field_labels=field_labels,
        save_path=save_path,
        title=title
    )
    
    return fig, axes


def plot_spot_diagram_from_csv(
    csv_path: str,
    row_idx: int = 0,
    scale: Optional[float] = None,
    save_path: Optional[str] = None
):
    """
    从CSV文件生成点列图
    
    Args:
        csv_path: CSV文件路径
        row_idx: 行索引（从0开始）
        scale: 显示范围（微米），如果为None则自动计算
        save_path: 保存路径
    
    Returns:
        fig, axes: matplotlib图形和坐标轴对象
    """
    # 创建镜头系统
    lens_system = SpotDiagramCSVLensSystem(csv_path=csv_path, row_idx=row_idx)
    
    # 生成点列图
    fig, axes = plot_spot_diagram_from_lens_system(
        lens_system,
        scale=scale,
        save_path=save_path
    )
    
    return fig, axes


if __name__ == "__main__":
    import argparse
    import sys
    import os
    
    # 添加项目根目录到路径
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    parser = argparse.ArgumentParser(description="生成Zemax风格的点列图（绿色波段）")
    parser.add_argument("csv_path", type=str, help="CSV文件路径")
    parser.add_argument("--row_idx", type=int, default=0, help="要可视化的行索引（从0开始，默认0）")
    parser.add_argument("--scale", type=float, default=None, help="显示范围（微米），如果未指定则自动计算")
    parser.add_argument("--output", "-o", type=str, default=None, help="输出文件路径（默认：spot_diagram_row{row_idx}.png）")
    parser.add_argument("--show", action="store_true", help="显示图形（使用matplotlib GUI）")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # 生成输出文件名
    if args.output is None:
        csv_basename = os.path.splitext(os.path.basename(args.csv_path))[0]
        args.output = f"{csv_basename}_row{args.row_idx}_spot_diagram.png"
    
    # 生成点列图
    try:
        print(f"Generating spot diagram from CSV...")
        print(f"  CSV file: {args.csv_path}")
        print(f"  Row index: {args.row_idx}")
        print(f"  Output file: {args.output}")
        if args.scale:
            print(f"  Display range: +/-{args.scale} um")
        
        fig, axes = plot_spot_diagram_from_csv(
            csv_path=args.csv_path,
            row_idx=args.row_idx,
            scale=args.scale,
            save_path=args.output
        )
        
        print(f"\n[OK] Spot diagram generated: {args.output}")
        
        if args.show:
            plt.show()
        else:
            plt.close(fig)  # 不显示时关闭图形以释放内存
            
    except Exception as e:
        print(f"Error: failed to generate spot diagram: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

