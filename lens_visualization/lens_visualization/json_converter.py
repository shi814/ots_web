"""
JSON格式转换工具
将新格式的JSON转换为标准DeepLens格式
"""

import json
import numpy as np
import torch
import tempfile
import os
import sys

# 处理相对导入：如果直接运行此文件，需要添加路径
try:
    from .geolens import GeoLens
except ImportError:
    # 直接运行时，添加路径
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from lens_visualization.geolens import GeoLens


def calculate_effective_aperture_radii(
    std_data,
    margin_factor=1,
    num_fov=7,
    num_rays_per_fov=200,
    temp_aperture_scale=5.0,
    temp_aperture_min=50.0,
):
    """
    通过光线追迹自动计算每个表面的有效孔径半径（Zemax 风格：以 Stop 约束为准）

    核心：使用"通过 Stop 的光线"作为有效光线集合，而不是"到达传感器"的光线。
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(std_data, f, indent=2)
        temp_file = f.name

    try:
        # Aperture estimation is a lightweight display helper. Use CPU here to
        # avoid CUDA-side ray-tracing failures from falling back to oversized
        # default apertures.
        lens = GeoLens(filename=temp_file, device="cpu")
        
        # 确保入瞳已经计算（GeoLens初始化时会自动调用calc_pupil，但为了安全起见，再次确认）
        if not hasattr(lens, 'entr_pupilr') or lens.entr_pupilr is None:
            print("[Aperture] 入瞳未计算，手动调用calc_pupil()")
            lens.calc_pupil()

        # -------- 1) 明确 Stop surface index（优先用 std_data 标注）--------
        stop_idx = None
        for i, s in enumerate(std_data["surfaces"]):
            if s.get("type") == "Stop":
                stop_idx = i
                break
        if stop_idx is None:
            raise RuntimeError("std_data 中未标注 Stop 面（type == 'Stop'），无法按 Zemax 逻辑计算净口径。")

        # -------- 2) 入瞳（用于给出一个合理的临时大口径）--------
        try:
            pupilz, pupilr = lens.get_entrance_pupil()
            print(f"[Aperture] 入瞳位置: z={pupilz:.4f}mm, r={pupilr:.4f}mm")
        except Exception as e:
            print(f"[Aperture] 警告: get_entrance_pupil失败: {e}")
            pupilz, pupilr = 0.0, float(lens.surfaces[0].r)
            print(f"[Aperture] 使用第一个表面的r作为fallback: {pupilr:.4f}mm")

        # Zemax 常用做法：保持 Stop 半径不变，其他面给足够大口径，避免非 Stop 处的人为截光
        temp_aperture = max(pupilr * float(temp_aperture_scale), float(temp_aperture_min))

        original_radii = []
        for i, surf in enumerate(lens.surfaces):
            original_radii.append(float(surf.r))
            if i != stop_idx and float(surf.r) < temp_aperture:
                surf.r = temp_aperture

        # -------- 3) 视场角采样 --------
        max_fov = 10.0
        fov_angles = np.linspace(0.0, max_fov, num_fov)
        fov_directions = [(0.0, 1.0)]  # 只沿 y 方向

        max_radii = [0.0] * len(lens.surfaces)

        total_rays = 0
        total_used = 0
        
        print(f"[Aperture] 开始光线追迹:")
        print(f"  - Stop表面索引: {stop_idx}, r={lens.surfaces[stop_idx].r:.4f}mm")
        print(f"  - 临时大口径: {temp_aperture:.4f}mm")
        print(f"  - 视场角范围: 0-{max_fov}度, {num_fov}个采样点")
        print(f"  - 每个视场角光线数: {num_rays_per_fov}")

        # -------- 4) 核心追迹 + 以 Stop 为准的 mask --------
        for fov in fov_angles:
            for dx, dy in fov_directions:
                fov_x = 0.0
                fov_y = float(fov)

                try:
                    rays = lens.sample_parallel(
                        fov_x=[fov_x],
                        fov_y=[fov_y],
                        num_rays=num_rays_per_fov,
                        entrance_pupil=True,
                        scale_pupil=1.0,
                    )
                except Exception as e:
                    print(f"[Aperture] 警告: sample_parallel失败 (fov={fov:.2f}): {e}")
                    continue

                # record=True：需要每个面交点
                try:
                    ray, ray_o_record = lens.trace2sensor(rays, record=True)
                except Exception as e:
                    print(f"[Aperture] 警告: trace2sensor失败 (fov={fov:.2f}): {e}")
                    continue
                    
                if not ray_o_record or len(ray_o_record) < (len(lens.surfaces) + 1):
                    print(f"[Aperture] 警告: ray_o_record无效 (fov={fov:.2f}), len={len(ray_o_record) if ray_o_record else 0}, expected={len(lens.surfaces) + 1}")
                    continue

                # 取 Stop 面交点：record_idx = stop_idx + 1（跳过 origin）
                stop_rec = ray_o_record[stop_idx + 1]
                if stop_rec is None or not isinstance(stop_rec, torch.Tensor):
                    continue

                # flatten 到 [Nrays, 3]
                stop_pts = stop_rec.reshape(-1, stop_rec.shape[-1])[..., :3]
                # 通过 Stop：Stop 交点非 NaN
                stop_pass = ~(
                    torch.isnan(stop_pts[:, 0]) |
                    torch.isnan(stop_pts[:, 1]) |
                    torch.isnan(stop_pts[:, 2])
                )

                n_pass = int(stop_pass.sum().item())
                total_rays += int(stop_pts.shape[0])
                if n_pass == 0:
                    if fov == fov_angles[0] or fov == fov_angles[-1]:  # 只在第一个和最后一个视场角打印
                        print(f"[Aperture] 警告: fov={fov:.2f}度时，没有光线通过Stop表面 (总光线数={stop_pts.shape[0]})")
                    continue

                # 对每个面计算包络：只用 stop_pass 且该面交点非 NaN
                for surf_idx in range(len(lens.surfaces)):
                    rec = ray_o_record[surf_idx + 1]
                    if rec is None or not isinstance(rec, torch.Tensor):
                        continue

                    pts = rec.reshape(-1, rec.shape[-1])[..., :3]
                    if pts.shape[0] != stop_pass.shape[0]:
                        continue

                    ok = stop_pass & ~(
                        torch.isnan(pts[:, 0]) |
                        torch.isnan(pts[:, 1]) |
                        torch.isnan(pts[:, 2])
                    )
                    if int(ok.sum().item()) == 0:
                        continue

                    v = pts[ok]
                    r = torch.sqrt(v[:, 0] ** 2 + v[:, 1] ** 2)
                    mr = float(r.max().item())
                    if np.isfinite(mr) and mr > max_radii[surf_idx]:
                        max_radii[surf_idx] = mr

                total_rays += int(stop_pts.shape[0])
                total_used += n_pass

        # -------- 5) 恢复原始口径 --------
        for i, surf in enumerate(lens.surfaces):
            surf.r = original_radii[i]

        # -------- 6) 应用余量，并强制 Stop 半径不变 --------
        effective_radii = [float(r) * float(margin_factor) for r in max_radii]

        # 检查是否所有max_radii都是0（表示光线追迹失败或没有找到有效光线）
        all_zero = all(r == 0.0 for r in max_radii)
        if all_zero:
            print(f"[Aperture] 警告: 所有表面的max_radii都是0，可能光线追迹失败或没有找到通过Stop的光线")
            print(f"[Aperture] traced_rays={total_rays}, stop_pass={total_used}")
            print(f"[Aperture] 将使用原始semi_diam值")
            # 如果所有都是0，说明光线追迹失败，应该返回原始值
            return [surf["r"] for surf in std_data["surfaces"]]

        # 没算到的面：回退用原值（std_data 的 r）
        # 但要注意：如果原始值是50.0（可能是默认值），不应该盲目使用
        min_radius = 1.0
        for i, r in enumerate(effective_radii):
            if not np.isfinite(r) or r < min_radius:
                fallback = float(std_data["surfaces"][i].get("r", min_radius))
                # 如果fallback是50.0且max_radii是0，说明这个面没有计算成功
                # 应该保持原始值，但给出警告
                if abs(fallback - 50.0) < 0.1 and max_radii[i] == 0.0:
                    print(f"[Aperture] 警告: 表面{i}使用fallback值{fallback:.2f}mm（可能是默认值50.0）")
                effective_radii[i] = max(fallback, min_radius)

        # Stop 半径保持不变
        effective_radii[stop_idx] = float(std_data["surfaces"][stop_idx]["r"])

        print(f"[Aperture] max_fov={max_fov:.3f} deg, traced_rays={total_rays}, stop_pass={total_used}")
        print("[Aperture] max_r (before margin):")
        for i, r in enumerate(max_radii):
            print(f"  surf {i}: {r:.4f} mm")
        print("[Aperture] effective_r (after margin):")
        for i, r in enumerate(effective_radii):
            t = std_data["surfaces"][i].get("type", "Unknown")
            print(f"  surf {i} ({t}): {r:.4f} mm")

        return effective_radii

    except Exception as e:
        print(f"[Aperture] ray-tracing failed: {e}")
        return [surf["r"] for surf in std_data["surfaces"]]
    finally:
        try:
            os.remove(temp_file)
        except Exception:
            pass


def convert_new_format_to_standard(new_data, use_ray_tracing=True):
    """
    将新格式的json转换为标准DeepLens格式
    
    Args:
        new_data (dict): 新格式的JSON数据
        use_ray_tracing (bool): 是否使用光线追迹自动计算有效口径
    
    Returns:
        dict: 标准格式的JSON数据
    """
    # 复制基本信息
    std_data = {
        "info": new_data.get("info", "Converted from new format"),
        "foclen": new_data["foclen"],
        "fnum": new_data["fnum"],
        "r_sensor": new_data["r_sensor"],
        "d_sensor": new_data.get("d_sensor", new_data.get("d_sensor")),
        "sensor_size": new_data["sensor_size"],
    }

    # 如果有enpd，添加它
    if "enpd" in new_data:
        std_data["enpd"] = new_data["enpd"]
    
    # 如果有rfov，添加它（用于光线追迹时的视场角）
    if "rfov" in new_data:
        std_data["rfov"] = new_data["rfov"]

    std_surfaces = []
    prev_mat2 = "air"  # 第一个表面前的材料默认为空气

    for i, surf in enumerate(new_data["surfaces"]):
        std_surf = {
            "idx": i,
            "d": surf["d"],
            "d_next": surf["d_next"]
        }

        # 判断表面类型
        if surf.get("is_stop", False):
            # 光阑表面
            std_surf["type"] = "Stop"
            std_surf["r"] = surf["semi_diam"]
        elif surf["curv"] == 0.0:
            # 曲率为0 - 无论是空气还是玻璃，都是平面
            std_surf["type"] = "Plane"
            std_surf["r"] = surf["semi_diam"]
        else:
            # 曲率不为0 - 球面（无论是空气还是玻璃）
            std_surf["type"] = "Spheric"
            std_surf["r"] = surf["semi_diam"]

        # 设置曲率
        std_surf["c"] = surf["curv"]
        std_surf["roc"] = 1.0 / surf["curv"] if surf["curv"] != 0 else float('inf')

        # 设置材料 - 考虑连续性
        # mat1应该是前一个表面的mat2
        std_surf["mat1"] = prev_mat2

        if surf["material"]["type"] == "air":
            # 空气表面，mat2也是空气
            std_surf["mat2"] = "air"
            prev_mat2 = "air"
        else:
            # 玻璃表面，从n_bgr数组中使用绿色波长的折射率
            n_green = surf["n_bgr"][1] if len(surf["n_bgr"]) > 1 else surf["n_bgr"][0]
            vd = surf["material"].get("vd")
            if vd:
                std_surf["mat2"] = f"{n_green:.5f}/{vd:.2f}"
            else:
                std_surf["mat2"] = f"{n_green:.5f}/50.0"  # 默认vd值
            prev_mat2 = std_surf["mat2"]  # 更新prev_mat2为当前mat2

        if surf.get("material", {}).get("type") != "air":
            material_name = surf.get("material", {}).get("name")
            if material_name:
                std_surf["material_display_name"] = material_name

        std_surfaces.append(std_surf)

    std_data["surfaces"] = std_surfaces
    
    # 如果启用光线追迹，计算每个表面的有效孔径半径
    if use_ray_tracing:
        print("\n使用光线追迹计算有效口径...")
        effective_radii = calculate_effective_aperture_radii(std_data)
        
        # 检查是否所有半径都是50.0（可能是fallback值）
        all_50 = all(abs(r - 50.0) < 0.1 for r in effective_radii)
        if all_50:
            print(f"[警告] 所有表面的有效半径都是50.0，可能光线追迹失败")
            print(f"[警告] 将保持原始semi_diam值，不更新为50.0")
            # 不更新，保持原始的semi_diam值
        else:
            # 更新每个表面的r值
            for i, r in enumerate(effective_radii):
                std_data["surfaces"][i]["r"] = round(r, 4)
    
    return std_data


def convert_and_save(input_file, output_file, use_ray_tracing=True):
    """
    转换并保存json文件
    
    Args:
        input_file (str): 输入文件路径
        output_file (str): 输出文件路径
        use_ray_tracing (bool): 是否使用光线追迹自动计算有效口径
    
    Returns:
        dict: 转换后的标准格式数据
    """
    # 读取新格式文件
    with open(input_file, 'r') as f:
        new_data = json.load(f)

    # 转换为标准格式
    std_data = convert_new_format_to_standard(new_data, use_ray_tracing=use_ray_tracing)

    # 保存为标准格式
    with open(output_file, 'w') as f:
        json.dump(std_data, f, indent=2)

    print(f"\n转换完成: {input_file} -> {output_file}")
    return std_data


if __name__ == "__main__":
    """
    直接运行此文件进行测试
    
    使用方法:
        python lens_visualization/json_converter.py [input_file] [output_file] [--use_ray_tracing]
    
    示例:
        python lens_visualization/json_converter.py exports/test_output_metrics_pred_min_rms_selected_row0_nsurf9.json test_output.json
        python lens_visualization/json_converter.py exports/test_output_metrics_pred_min_rms_selected_row0_nsurf9.json test_output.json --use_ray_tracing
    """
    import sys
    import os
    
    # 添加项目根目录到路径
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # 添加lens_visualization到路径
    lens_viz_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, lens_viz_dir)
    
    # 默认测试文件
    default_input = "exports/test_output_metrics_pred_min_rms_selected_row0_nsurf9.json"
    default_output = "test_converted_output.json"
    
    if len(sys.argv) < 2:
        print("=" * 60)
        print("JSON格式转换工具 - 测试模式")
        print("=" * 60)
        print(f"\n使用方法:")
        print(f"  python {sys.argv[0]} <input_file> [output_file] [--use_ray_tracing]")
        print(f"\n示例:")
        print(f"  python {sys.argv[0]} {default_input} {default_output}")
        print(f"  python {sys.argv[0]} {default_input} {default_output} --use_ray_tracing")
        print(f"\n使用默认文件进行测试...")
        input_file = default_input
        output_file = default_output
        use_ray_tracing = False
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(input_file)[0] + "_converted.json"
        use_ray_tracing = "--use_ray_tracing" in sys.argv
    
    if not os.path.exists(input_file):
        print(f"[错误] 输入文件不存在: {input_file}")
        print(f"[提示] 请提供正确的文件路径")
        sys.exit(1)
    
    print("=" * 60)
    print("JSON格式转换测试")
    print("=" * 60)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"使用光线追迹: {use_ray_tracing}")
    print("=" * 60)
    print()
    
    try:
        result = convert_and_save(
            input_file=input_file,
            output_file=output_file,
            use_ray_tracing=use_ray_tracing
        )
        print("\n" + "=" * 60)
        print("转换成功！")
        print("=" * 60)
        print(f"输出文件: {output_file}")
        print(f"表面数量: {len(result['surfaces'])}")
        if use_ray_tracing:
            print("\n有效口径计算结果:")
            for i, surf in enumerate(result['surfaces']):
                print(f"  表面{i} ({surf.get('type', 'Unknown')}): r={surf['r']:.4f}mm")
    except Exception as e:
        print(f"\n[错误] 转换失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

