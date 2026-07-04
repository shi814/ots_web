#!/usr/bin/env python3
"""
将OTS数据库的.mat文件转换为CSV格式
"""

import hdf5storage as hdf5
import pandas as pd
import numpy as np

def convert_mat_to_csv(mat_file, csv_file):
    """将.mat文件转换为CSV"""

    print(f"正在处理: {mat_file}")

    # 读取.mat文件
    try:
        mat_data = hdf5.loadmat(mat_file)
        print(f"成功读取.mat文件")
    except Exception as e:
        print(f"读取失败: {e}")
        return

    # 查看文件结构
    print("文件包含的键:")
    for key in mat_data.keys():
        if not key.startswith('__'):
            data = mat_data[key]
            if hasattr(data, 'shape'):
                print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
            else:
                print(f"  {key}: {type(data)}")

    # 提取主要数据
    try:
        glass_data = mat_data['Glass'].squeeze()
        radius1 = mat_data['Radius1'].squeeze()
        radius2 = mat_data['Radius2'].squeeze()
        thickness = mat_data['Thickness'].squeeze()

        print("数据提取成功:")
        print(f"  Glass: {len(glass_data)} 项")
        print(f"  Radius1: {len(radius1)} 项")
        print(f"  Radius2: {len(radius2)} 项")
        print(f"  Thickness: {len(thickness)} 项")

        # 处理玻璃名称
        glass_names = []
        for g in glass_data:
            if isinstance(g, (bytes, bytearray)):
                name = g.decode('utf-8')
            else:
                name = str(g)
            # 清理格式
            name = name.replace("[[", "").replace("]]", "").replace("'", "").strip()
            glass_names.append(name)

        # 转换为DataFrame
        df = pd.DataFrame({
            'Glass': glass_names,
            'Radius1': radius1.astype(float),
            'Radius2': radius2.astype(float),
            'Thickness': thickness.astype(float)
        })

        # 保存为CSV
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"✅ 成功保存到: {csv_file}")
        print(f"   数据行数: {len(df)}")
        print(f"   列数: {len(df.columns)}")

        # 显示前几行
        print("前5行数据:")
        print(df.head())

        return df

    except KeyError as e:
        print(f"❌ 数据提取失败，缺少键: {e}")
        return None
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None

def main():
    print("=" * 60)
    print("OTS数据库.mat文件转换为CSV")
    print("=" * 60)

    # 处理两个文件
    files_to_convert = [
        ('data/ots_lens_catalaogue.mat', 'data/ots_lens_catalaogue.csv'),
        ('data/ots_lens_catalogue_clean.mat', 'data/ots_lens_catalogue_clean.csv')
    ]

    for mat_file, csv_file in files_to_convert:
        print(f"\n{'='*40}")
        convert_mat_to_csv(mat_file, csv_file)

    print(f"\n{'='*60}")
    print("转换完成！")
    print("生成的文件:")
    for _, csv_file in files_to_convert:
        print(f"  {csv_file}")

if __name__ == "__main__":
    main()
