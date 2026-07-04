#!/usr/bin/env python3
"""
使用示例：如何使用镜头可视化包

使用方法：
1. 在当前项目根目录运行：python lens_visualization/example.py
2. 或者修改下面的 lens_file 路径为你的JSON文件路径
"""

import sys
import os
from pathlib import Path

# 添加包路径（如果不在当前目录）
# 如果 lens_visualization 在当前目录的父目录，取消下面的注释
# sys.path.insert(0, str(Path(__file__).parent.parent))

from lens_visualization import LensVisualizer

def main():
    # 1. 初始化可视化器（加载镜头JSON文件）
    # 修改下面的路径为你的JSON文件路径
    lens_file = "datasets/lenses/camera/converted_lens.json"
    
    # 如果文件不存在，提示用户
    if not os.path.exists(lens_file):
        print(f"错误: 文件 {lens_file} 不存在")
        print("请修改 example.py 中的 lens_file 变量为你的JSON文件路径")
        return
    
    visualizer = LensVisualizer(lens_file)
    
    # 打印镜头信息
    info = visualizer.get_lens_info()
    print("镜头信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 2. 绘制2D布局图
    print("\n正在绘制2D布局图...")
    visualizer.draw_2d_layout(
        filename="output_2d_layout.png",
        depth=float("inf"),  # 无穷远物距
        zmx_format=True,     # 使用Zemax格式
        show=False
    )
    print("2D布局图已保存: output_2d_layout.png")
    
    # 3. 绘制畸变网格图
    print("\n正在绘制畸变网格图...")
    visualizer.draw_distortion(
        save_name="output_distortion_grid.png",
        num_grid=16,         # 16x16网格
        depth=float("inf"), # 无穷远物距
        wvln=0.587,         # 绿色光波长
        show=False
    )
    print("畸变网格图已保存: output_distortion_grid.png")
    
    # 4. 绘制径向畸变曲线（子午面）
    print("\n正在绘制子午面径向畸变曲线...")
    visualizer.draw_distortion_radial(
        rfov=None,          # 使用镜头默认rfov
        save_name="output_distortion_radial_meridional.png",
        num_points=21,
        wvln=0.587,
        plane="meridional", # 子午面
        ray_aiming=True,
        show=False
    )
    print("子午面径向畸变曲线已保存: output_distortion_radial_meridional.png")
    
    # 5. 绘制径向畸变曲线（弧矢面）
    print("\n正在绘制弧矢面径向畸变曲线...")
    visualizer.draw_distortion_radial(
        rfov=None,
        save_name="output_distortion_radial_sagittal.png",
        num_points=21,
        wvln=0.587,
        plane="sagittal",    # 弧矢面
        ray_aiming=True,
        show=False
    )
    print("弧矢面径向畸变曲线已保存: output_distortion_radial_sagittal.png")
    
    print("\n所有图表绘制完成！")

if __name__ == "__main__":
    main()

