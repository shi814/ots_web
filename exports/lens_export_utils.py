"""
Lens Export Utilities - 从 DeepLens 项目移植的代码
用于将镜头设计导出为 JSON 和 ZMX 格式的文件
"""

import json


def _to_float(x):
    """Convert torch scalar / numpy scalar / python number to float."""
    # torch scalar
    if hasattr(x, "item") and callable(getattr(x, "item")):
        try:
            return float(x.item())
        except Exception:
            pass
    # numpy scalar / python number
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"Cannot convert to float: {type(x)}") from e


class LensExporter:
    """
    通用的镜头导出器类
    从 DeepLens 项目移植的核心导出功能
    """

    def __init__(self, lens_object):
        """
        初始化导出器

        Args:
            lens_object: 镜头对象，需要包含以下属性：
                - lens_info (可选)
                - foclen: 焦距
                - fnum: F数
                - enpd: 入瞳直径 (如果float_enpd为False)
                - float_enpd: 是否浮动入瞳直径
                - r_sensor: 传感器半径
                - d_sensor: 传感器距离
                - sensor_size: 传感器尺寸
                - sensor_res: 传感器分辨率 (可选)
                - surfaces: 表面列表，每个表面需要有surf_dict()和zmx_str()方法
        """
        self.lens = lens_object

    def write_lens_json(self, filename="./lens.json"):
        """
        将镜头导出为 JSON 格式

        Args:
            filename: 输出文件名
        """
        data = {}

        # 基本镜头信息
        data["info"] = getattr(self.lens, "lens_info", "None")
        data["foclen"] = round(self.lens.foclen, 4)
        data["fnum"] = round(self.lens.fnum, 4)

        # 视场角信息（HFOV：半视场角，度）- 直接使用CSV中的原始值
        if hasattr(self.lens, 'hfov'):
            data["hfov"] = round(_to_float(self.lens.hfov), 4)
        
        # 视场角信息（RFOV：半视场角，弧度）
        # 对于dataclass，直接访问字段，hasattr可能不准确
        rfov_value = getattr(self.lens, 'rfov', None)
        if rfov_value is not None:
            data["rfov"] = round(_to_float(rfov_value), 6)  # 弧度值保留更多小数位

        # 入瞳直径信息
        if hasattr(self.lens, 'float_enpd') and not self.lens.float_enpd:
            data["enpd"] = round(self.lens.enpd, 4)

        # 传感器信息
        data["r_sensor"] = self.lens.r_sensor
        data["d_sensor"] = round(_to_float(self.lens.d_sensor), 4)
        data["sensor_size"] = [round(i, 4) for i in self.lens.sensor_size]

        if hasattr(self.lens, 'sensor_res'):
            data["sensor_res"] = self.lens.sensor_res

        # 表面信息
        data["surfaces"] = []
        for i, s in enumerate(self.lens.surfaces):
            surf_dict = {"idx": i}
            surf_dict.update(s.surf_dict())

            # 计算下一个表面的距离
            if i < len(self.lens.surfaces) - 1:
                surf_dict["d_next"] = round(
                    _to_float(self.lens.surfaces[i + 1].d) - _to_float(self.lens.surfaces[i].d), 4
                )
            else:
                surf_dict["d_next"] = round(
                    _to_float(self.lens.d_sensor) - _to_float(self.lens.surfaces[i].d), 4
                )

            data["surfaces"].append(surf_dict)

        # 写入文件
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"镜头已导出到 {filename}")

    def write_lens_zmx(self, filename="./lens.zmx"):
        """
        将镜头导出为 ZMX 格式 (Zemax)

        Args:
            filename: 输出文件名
        """
        lens_zmx_str = ""

        # Determine the aperture header string.
        aperture_mode = str(getattr(self.lens, "aperture_mode", "enpd")).strip().lower()
        if aperture_mode == "fnum":
            aperture_str = f"FNUM {self.lens.fnum} 0"
        elif hasattr(self.lens, 'float_enpd') and self.lens.float_enpd:
            aperture_str = "FLOA"
        else:
            aperture_str = f"ENPD {self.lens.enpd}"

        # ZMX 文件头
        # ScanLens convention (default):
        # - HFOV is HALF field angle in degrees (stored in lens.rfov in radians).
        # - YFLN uses mid-field = 0.5*HFOV, max-field = 1.0*HFOV (to match existing hand-made .zmx in this repo).
        # - Default glass catalog: SCHOTT CORNING INFRARED
        # - Default wavelengths (micron): 0.45 / 0.58 / 0.75
        #
        # Optional overrides on lens object:
        #   - lens.zemax_gcat: str
        #   - lens.zemax_wavl: list/tuple of 3 floats (micron)
        #   - lens.zemax_yfln_mid_factor: float (default 0.5)
        #   - lens.zemax_yfln_max_factor: float (default 1.0)
        gcat = getattr(self.lens, "zemax_gcat", "SCHOTT CORNING INFRARED")
        wavl = getattr(self.lens, "zemax_wavl", (0.45, 0.58, 0.75))
        try:
            w0, w1, w2 = float(wavl[0]), float(wavl[1]), float(wavl[2])
        except Exception:
            w0, w1, w2 = 0.45, 0.58, 0.75
        y_mid = float(getattr(self.lens, "zemax_yfln_mid_factor", 0.5))
        y_max = float(getattr(self.lens, "zemax_yfln_max_factor", 1.0))

        head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
{aperture_str}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT {gcat}
XFLN 0. 0. 0.
YFLN 0.0 {y_mid * getattr(self.lens, 'rfov', 1.0) * 57.3} {y_max * getattr(self.lens, 'rfov', 1.0) * 57.3}
WAVL {w0} {w1} {w2}
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
TYPE STANDARD
CURV 0.0
DISZ INFINITY
"""
        lens_zmx_str += head_str

        # 表面字符串
        for i, s in enumerate(self.lens.surfaces):
            d_next = (
                self.lens.surfaces[i + 1].d - self.lens.surfaces[i].d
                if i < len(self.lens.surfaces) - 1
                else self.lens.d_sensor - self.lens.surfaces[i].d
            )
            surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
            lens_zmx_str += surf_str

        # 传感器字符串
        sensor_str = f"""SURF {len(self.lens.surfaces) + 1}
TYPE STANDARD
CURV 0.
DISZ 0.0
DIAM {self.lens.r_sensor}
"""

        lens_zmx_str += sensor_str

        # 写入文件
        with open(filename, "w", encoding='utf-8') as f:
            f.write(lens_zmx_str)

        print(f"镜头已导出到 {filename}")


class HybridLensExporter(LensExporter):
    """
    混合镜头专用导出器
    处理包含几何光学和衍射光学表面的混合镜头
    """

    def write_lens_json(self, filename="./hybrid_lens.json"):
        """
        将混合镜头导出为 JSON 格式

        Args:
            filename: 输出文件名
        """
        geolens = self.lens.geolens
        data = {}

        # 基本镜头信息
        data["info"] = getattr(geolens, "lens_info", "None")
        data["foclen"] = round(geolens.foclen, 4)
        data["fnum"] = round(geolens.fnum, 4)
        data["r_sensor"] = round(geolens.r_sensor, 4)
        data["d_sensor"] = round(_to_float(geolens.d_sensor), 4)
        data["sensor_size"] = [round(i, 4) for i in geolens.sensor_size]

        # 视场角信息（HFOV：半视场角，度）
        if hasattr(geolens, 'hfov'):
            data["hfov"] = round(_to_float(geolens.hfov), 4)
        
        # 视场角信息（RFOV：半视场角，弧度）
        rfov_value = getattr(geolens, 'rfov', None)
        if rfov_value is not None:
            data["rfov"] = round(_to_float(rfov_value), 6)  # 弧度值保留更多小数位

        if hasattr(geolens, 'sensor_res'):
            data["sensor_res"] = geolens.sensor_res

        # 几何光学表面
        data["surfaces"] = []
        for i, s in enumerate(geolens.surfaces[:-1]):  # 排除最后一个表面 (DOE)
            surf_dict = s.surf_dict()

            if i < len(geolens.surfaces) - 2:
                surf_dict["d_next"] = round(
                    _to_float(geolens.surfaces[i + 1].d) - _to_float(geolens.surfaces[i].d), 3
                )
            else:
                surf_dict["d_next"] = round(
                    _to_float(geolens.d_sensor) - _to_float(geolens.surfaces[i].d), 3
                )

            data["surfaces"].append(surf_dict)

        # DOE 信息
        data["DOE"] = self.lens.doe.surf_dict()

        # 写入文件
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        print(f"混合镜头已导出到 {filename}")


# 使用示例
def example_usage():
    """
    使用示例函数
    """
    # 假设你有一个镜头对象 lens
    # from your_lens_library import YourLensClass
    # lens = YourLensClass(...)

    # # 对于普通几何镜头:
    # exporter = LensExporter(lens)
    # exporter.write_lens_json("my_lens.json")
    # exporter.write_lens_zmx("my_lens.zmx")

    # # 对于混合镜头:
    # hybrid_exporter = HybridLensExporter(hybrid_lens)
    # hybrid_exporter.write_lens_json("my_hybrid_lens.json")

    print("请参考上述注释中的使用方法")


if __name__ == "__main__":
    example_usage()
