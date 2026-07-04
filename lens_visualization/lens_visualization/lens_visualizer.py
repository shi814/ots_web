"""
简化的镜头可视化接口
"""

import sys
from pathlib import Path

# 添加当前包路径到sys.path
package_dir = Path(__file__).parent
sys.path.insert(0, str(package_dir))

# 导入必要的模块（使用相对导入）
from .geolens import GeoLens
from .json_converter import convert_new_format_to_standard, convert_and_save


class LensVisualizer:
    """镜头可视化器 - 简化的接口类"""
    
    def __init__(self, lens_file=None, **kwargs):
        """
        初始化镜头可视化器
        
        Args:
            lens_file (str): 镜头JSON文件路径
            **kwargs: 传递给GeoLens的其他参数
        """
        self.lens = GeoLens(filename=lens_file, **kwargs)
    
    def draw_2d_layout(
        self,
        filename,
        depth=float("inf"),
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
        show=False,
    ):
        """
        绘制2D布局图
        
        Args:
            filename (str): 保存文件名
            depth (float): 物距，默认无穷远
            zmx_format (bool): 是否使用Zemax格式
            multi_plot (bool): 是否创建多个子图（每个波长一个）
            lens_title (str): 图表标题
            show (bool): 是否显示图表
        """
        self.lens.draw_layout(
            filename=filename,
            depth=depth,
            zmx_format=zmx_format,
            multi_plot=multi_plot,
            lens_title=lens_title,
            show=show,
        )
    
    def draw_distortion(
        self,
        save_name=None,
        num_grid=16,
        depth=-20000.0,
        wvln=0.587,
        show=False,
    ):
        """
        绘制畸变网格图
        
        Args:
            save_name (str): 保存文件名，默认None会自动生成
            num_grid (int): 网格点数，默认16
            depth (float): 物距，默认-20000.0mm
            wvln (float): 波长（微米），默认0.587
            show (bool): 是否显示图表
        """
        self.lens.draw_distortion(
            save_name=save_name,
            num_grid=num_grid,
            depth=depth,
            wvln=wvln,
            show=show,
        )
    
    def draw_distortion_radial(
        self,
        rfov=None,
        save_name=None,
        num_points=21,
        wvln=0.587,
        plane="meridional",
        ray_aiming=True,
        show=False,
    ):
        """
        绘制径向畸变曲线（Zemax风格）
        
        Args:
            rfov (float): 半对角线视场角（度），默认使用镜头rfov
            save_name (str): 保存文件名，默认None会自动生成
            num_points (int): 采样点数，默认21
            wvln (float): 波长（微米），默认0.587
            plane (str): 平面类型，"meridional"或"sagittal"
            ray_aiming (bool): 是否使用光线瞄准
            show (bool): 是否显示图表
        """
        if rfov is None:
            # 将弧度转换为度
            rfov = float(self.lens.rfov * 180 / 3.14159)
        
        self.lens.draw_distortion_radial(
            rfov=rfov,
            save_name=save_name,
            num_points=num_points,
            wvln=wvln,
            plane=plane,
            ray_aiming=ray_aiming,
            show=show,
        )
    
    def get_lens_info(self):
        """获取镜头信息"""
        return {
            "foclen": float(self.lens.foclen),
            "fnum": float(self.lens.fnum),
            "rfov_rad": float(self.lens.rfov),
            "rfov_deg": float(self.lens.rfov * 180 / 3.14159),
            "sensor_size": self.lens.sensor_size,
            "num_surfaces": len(self.lens.surfaces),
        }
    
    @staticmethod
    def convert_json_format(input_file, output_file, use_ray_tracing=True):
        """
        将新格式的JSON转换为标准DeepLens格式
        
        Args:
            input_file (str): 输入文件路径（新格式JSON）
            output_file (str): 输出文件路径（标准格式JSON）
            use_ray_tracing (bool): 是否使用光线追迹自动计算有效口径，默认True
        
        Returns:
            dict: 转换后的标准格式数据
        
        Example:
            >>> LensVisualizer.convert_json_format(
            ...     "new_format.json",
            ...     "standard_format.json",
            ...     use_ray_tracing=True  # 默认已经开启光线追迹
            ... )
        """
        return convert_and_save(input_file, output_file, use_ray_tracing=use_ray_tracing)

