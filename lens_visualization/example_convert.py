#!/usr/bin/env python3
"""
JSON格式转换示例

使用方法：
1. 在当前项目根目录运行：python lens_visualization/example_convert.py
2. 或者修改下面的 input_file 路径为你的JSON文件路径
"""

import sys
import os
from pathlib import Path

# 添加包路径（如果不在当前目录）
# 如果 lens_visualization 在当前目录的父目录，取消下面的注释
# sys.path.insert(0, str(Path(__file__).parent.parent))

from lens_visualization import LensVisualizer, convert_and_save

def main():
    # 修改下面的路径为你的JSON文件路径
    input_file = "datasets/lenses/camera/test_output_metrics_pred_min_rms_selected_row0_nsurf9.json"
    
    if not os.path.exists(input_file):
        print(f"错误: 文件 {input_file} 不存在")
        print("请修改 example_convert.py 中的 input_file 变量为你的JSON文件路径")
        return
    
    # 示例1：使用 LensVisualizer 的静态方法转换
    print("=" * 60)
    print("示例1：使用 LensVisualizer.convert_json_format()")
    print("=" * 60)
    
    LensVisualizer.convert_json_format(
        input_file=input_file,
        output_file="converted_lens_example1.json",
        use_ray_tracing=False  # 不使用光线追迹
    )
    
    # 示例2：使用光线追迹自动计算有效口径
    print("\n" + "=" * 60)
    print("示例2：使用光线追迹自动计算有效口径")
    print("=" * 60)
    
    LensVisualizer.convert_json_format(
        input_file=input_file,
        output_file="converted_lens_example2.json",
        use_ray_tracing=True  # 使用光线追迹
    )
    
    # 示例3：直接使用转换函数
    print("\n" + "=" * 60)
    print("示例3：直接使用 convert_and_save() 函数")
    print("=" * 60)
    
    convert_and_save(
        input_file=input_file,
        output_file="converted_lens_example3.json",
        use_ray_tracing=False
    )
    
    # 示例4：转换后立即使用
    print("\n" + "=" * 60)
    print("示例4：转换后立即绘制2D图")
    print("=" * 60)
    
    # 先转换
    LensVisualizer.convert_json_format(
        input_file=input_file,
        output_file="converted_lens_example4.json",
        use_ray_tracing=False
    )
    
    # 然后使用转换后的文件
    visualizer = LensVisualizer("converted_lens_example4.json")
    info = visualizer.get_lens_info()
    print("\n转换后的镜头信息:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n转换完成！")

if __name__ == "__main__":
    main()

