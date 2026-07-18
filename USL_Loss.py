"""
Description: Optical layer for scan lens design: multiple variation outputs
Reference  : scanlens_surf10_line1.zmx; scanlens_surf10_line17.zmx; scanlens_surf12_line1.zmx
             scan_lens_dataset_surf10.csv; scan_lens_dataset_surf12.csv // 这些文件作为输入都ok！
Author     : Yunfeng Nie, Runmu Su
Debug Date : 2025-8-25, YN created the file.
             8-26，YN加入loss计算.
             8.27，YN调试loss_spot, loss_ray, loss_ovlp, loss_dist, loss_chrom, loss_tele.
             8.28，Su 调试变序列，即不同面数的光线追迹
             8.29，Su 将此USL_loss与transformer联动
             12.01,Su 在光线追击模型_surf3Draytrace，添加了光线的惩罚措施
             12.01 Su 增加了mask筛选机制
             12.02 YN 代码命名和传参的精简！！
             12.22 SJQ 增加EFL筛选机制
Status     : ok!
"""

import torch.nn as nn
from timeit import default_timer as tc
from utils import *
from typing import Tuple

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 统一定义所有 metrics key
USL_METRIC_KEYS = [
    "X",            # filtered X
    "N",            # filtered N_bgr
    "CT",           # filtered CT
    "composite",    # composite loss
    "rms",          # rms spot
    "rms_g",        # g-spot only
    "loss_ray",     # ray miss penalty
    "loss_ovlp",    # overlap penalty
    "dist",         # distortion
    "loss_dist",    # distortion penalty
    "dist_signed",  # signed distortion
    "dist_target",  # target image-height error
    "tele",         # tele angle
    "loss_tele",    # tele penalty
    "EFL_est",      # estimated EFL from ray tracing
    "EFL_ideal",    # ideal EFL from F# * EPD
    "EFL_first_order",   # first-order EFL from ABCD matrix
    "loss_EFL",     # EFL loss
    "loss_EFL_trace",    # primary EFL loss selected by efl_loss_mode
    "loss_EFL_control",  # trace-vs-first-order control loss
    # 如果以后需要 chromatic，继续往下加
    # "chrom",
]


USL_LOSS_CONFIGS = {
    "stage1_geometric_v1": {
        "loss_function": "USL_Loss.stage1_geometric_v1",
        "aggregation": "geometric_mean",
        "multiplicative_terms": (
            ("loss_ray", 1000.0),
            ("loss_dist", 100.0),
            ("loss_EFL", 10),
            ("loss_ovlp", 1000.0),
            ("loss_tele", 1.0),
        ),
        "loss_formula": (
            "geomean(rms*(1+1000*loss_ray)*(1+100*loss_dist)"
            "*(1+10*loss_EFL)*(1+1000*loss_ovlp)*(1+1*loss_tele))"
        ),
    },
    "default_additive_v1": {
        "loss_function": "USL_Loss.default_additive_v1",
        "aggregation": "mean",
        "additive_terms": (
            ("rms", 500.0),
            ("loss_ray", 1000.0),
            ("loss_dist", 200.0),
            ("loss_EFL", 200.0),
            ("loss_ovlp", 1000.0),
            ("loss_tele", 20.0),
        ),
        "loss_formula": (
            "mean(500*rms + 1000*loss_ray + 200*loss_dist + 200*loss_EFL "
            "+ 1000*loss_ovlp + 20*loss_tele)"
        ),
    },
}


# =============================================================================
# CSV readers
# =============================================================================
def read_scan_lens_csv(
    path: str,
    n_surf: int,
    row: int,
    offset_ct: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one sample from prediction CSV (mixed 9/11 surfaces).

    Parameters
    ----------
    path : str
        CSV path.
    n_surf : int
        Number of surfaces for this lens (9 or 11).
    row : int
        Row index to read.
    offset_ct : int
        Column offset for CT block (for 9/11 seq alignment).

    Returns
    -------
    X_data : (1, 2)  [FN, HFOV]
    Y_bgr  : (1, n_surf, 3)
    CT     : (1, n_surf, 2)
    """
    data = np.loadtxt(path, delimiter=",", dtype=float)

    X = data[row:row + 1, :2]  # [FN, HFOV]
    Y = data[row:row + 1, 2: 2 + n_surf * 3]          # 3 * n_surf
    CT = data[row:row + 1, 2 + n_surf * 3 + offset_ct: 2 + n_surf * 5 + offset_ct]

    return X, Y.reshape(1, n_surf, 3), CT.reshape(1, n_surf, 2)


def read_orig_scan_lens_csv(
    path: str,
    n_surf: int,
    row: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read one sample from a ground-truth lens-parameter CSV file.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    n_surf : int
        Number of surfaces (e.g., 9 or 11).
    row : int, optional
        Which row to read. Default = 0.

    Returns
    -------
    X_data : ndarray, shape (1, 2)
        Lens-level parameters [FN, HFOV].
    Y_bgr  : ndarray, shape (1, n_surf, 3)
        Per-surface background parameters.
    CT     : ndarray, shape (1, n_surf, 2)
        Per-surface curvature / thickness parameters.
    """
    data = np.loadtxt(path, delimiter=",", dtype=float)

    # Lens-level features (FN, HFOV)
    X = data[row:row + 1, :2]

    # Y block: 3 values per surface, starting from col 3
    y_start = 3
    y_end = y_start + 3 * n_surf
    Y = data[row:row + 1, y_start:y_end].reshape(1, n_surf, 3)

    # CT block: 2 values per surface, starting after Y block + 3 columns
    # (original format: 3 columns before Y, 3 columns after Y)
    ct_start = y_end + 3
    ct_end = ct_start + 2 * n_surf
    CT = data[row:row + 1, ct_start:ct_end].reshape(1, n_surf, 2)

    return X, Y, CT


def create_pupil_grid(N_pupil: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """
    生成单位圆 pupil 网格，只跟 N_pupil 和 dtype 有关。
    Notes: 观察到torch.linspace和np.linspace会导致 2-3%的 USL loss计算差异。
    返回:
        pupil_xyz : (nPts, 3)，单位圆内的 [x, y, z]，z 恒为 0
    """
    lin = torch.linspace(-1.0, 1.0, N_pupil, dtype=dtype)
    px0, py0 = torch.meshgrid(lin, lin, indexing="xy")
    mask = px0**2 + py0**2 <= 1.0

    px_unit = px0[mask]
    py_unit = py0[mask]
    pz_unit = torch.zeros_like(px_unit)

    pupil_xyz = torch.stack([px_unit, py_unit, pz_unit], dim=1)  # (nPts, 3)
    return pupil_xyz

def create_pupil_grid_np(N_pupil: int) -> np.ndarray:
    """
    使用 NumPy 生成单位圆 pupil 网格，只跟 N_pupil 有关。
    返回:
        pupil_xyz : (nPts, 3)，单位圆内的 [x, y, z]，z 恒为 0
    """
    lin = np.linspace(-1.0, 1.0, N_pupil, dtype=np.float64)
    px0, py0 = np.meshgrid(lin, lin, indexing="xy")
    mask = px0**2 + py0**2 <= 1.0

    px_unit = px0[mask]          # (nPts,)
    py_unit = py0[mask]
    pz_unit = np.zeros_like(px_unit)

    pupil_xyz = np.stack([px_unit, py_unit, pz_unit], axis=1)  # (nPts, 3)
    return pupil_xyz

# -----------------------------------------------------------------------------
# Core optical module
# -----------------------------------------------------------------------------
class USL_Loss(nn.Module):
    """Unified scan-lens loss module.

    Required opt fields
    -------------------
    - N, M (pupil grid and field counts)
    - nSurf or max_surf (maximum surface count)
    """

    def __init__(self, opt, dtype=torch.float64):
        super().__init__()
        self.opt = opt
        self.dtype = dtype

        # Optical system parameters #
        self.n_wavelength= opt.nWL
        self.n_field = opt.nField
        self.n_pupil = opt.nRayDensity
        self.EPD = opt.EPD # mm

        # === 新增：预计算 pupil 网格，减少重复生成的耗时 ===
        self.pupil_xyz_np = create_pupil_grid_np(self.n_pupil)
        self.nSample = self.pupil_xyz_np.shape[0]
        pupil_xy2 = np.sum(self.pupil_xyz_np[:, :2] ** 2, axis=1)
        self.chief_sample_idx = int(np.argmin(pupil_xy2))
        axis_x_abs = np.abs(self.pupil_xyz_np[:, 0])
        axis_candidates = np.where(axis_x_abs == axis_x_abs.min())[0]
        self.marginal_y_sample_idx = int(axis_candidates[np.argmax(self.pupil_xyz_np[axis_candidates, 1])])

        ## 性能要求
        self.maxDist = 0.02  # 光学畸变需要小于0.02 = 2%
        self.maxTele = 2  # 光学远心角度需要小于2deg, in degrees

    def _loss_config(self):
        variant = str(getattr(self.opt, "usl_loss_variant", "stage1_geometric_v1"))
        if variant not in USL_LOSS_CONFIGS:
            variant = "stage1_geometric_v1"
        return variant, USL_LOSS_CONFIGS[variant]

    def describe_loss(self):
        variant, cfg = self._loss_config()
        efl_loss_mode = str(getattr(self.opt, "efl_loss_mode", "trace")).lower()
        if efl_loss_mode not in {"trace", "abcd"}:
            efl_loss_mode = "trace"
        efl_ctrl_enabled = bool(getattr(self.opt, "enable_efl_first_order_control", True))
        if efl_loss_mode == "trace":
            efl_control_mode = "trace_efl_to_ideal"
        else:
            efl_control_mode = (
                "first_order_abcd_to_ideal+trace_consistency"
                if efl_ctrl_enabled
                else "first_order_abcd_to_ideal"
            )
        metadata = {
            "loss_function": cfg["loss_function"],
            "loss_formula": cfg["loss_formula"],
            "usl_loss_variant": variant,
            "distortion_mode": str(getattr(self.opt, "distortion_mode", "zemax_ftan")),
            "distortion_ref_angle_deg": str(getattr(self.opt, "distortion_ref_angle_deg", 0.01)),
            "efl_loss_mode": efl_loss_mode,
            "efl_control_mode": efl_control_mode,
            "efl_first_order_weight": str(getattr(self.opt, "efl_first_order_weight", 0.5)),
            "efl_first_order_tolerance": str(getattr(self.opt, "efl_first_order_tolerance", 0.1)),
            "loss_filtering": (
                f"hard_rms_filter={'on' if getattr(self.opt, 'enable_rms_filter', True) else 'off'}, "
                f"hard_efl_filter={'on' if getattr(self.opt, 'enable_efl_filter', False) else 'off'}"
            ),
        }
        return metadata

    def attach_loss_metadata(self):
        metadata = self.describe_loss()
        for key, value in metadata.items():
            setattr(self.opt, key, value)
        return metadata

    def print_loss_metadata(self, prefix="[Loss]"):
        metadata = self.describe_loss()
        print(f"{prefix} active loss: {metadata['loss_function']}")
        print(f"{prefix} formula: {metadata['loss_formula']}")
        print(f"{prefix} distortion_mode: {metadata['distortion_mode']}")
        print(f"{prefix} filtering: {metadata['loss_filtering']}")
        return metadata

    # -------------------------- helpers -------------------------- #
    @staticmethod
    def _insert_last_n(Y_bgr, surf_idx):
        """Insert an extra surface for refractive index triplets at `surf_lens`.
        Y_bgr : (B, L, 3)  ->  (B, L+1, 3) with ones at insert pos
        """
        B, L, C = Y_bgr.shape
        out = torch.zeros(B, L + 1, C, device=Y_bgr.device, dtype=Y_bgr.dtype)
        out[:, :L, :] = Y_bgr
        out[torch.arange(B, device=Y_bgr.device), surf_idx, :] = 1.0
        return out

    @staticmethod
    def _insert_last_ct(Y_ct, surf_idx):
        """Insert an extra surface for [curv, thick] at `surf_lens` (zeros)."""
        B, L, C = Y_ct.shape
        out = torch.zeros(B, L + 1, C, device=Y_ct.device, dtype=Y_ct.dtype)
        out[:, :L, :] = Y_ct
        return out

    def _generate_in_rays(self, X_data: torch.Tensor):
        """
        使用当前 USL_Loss 配置生成入射光线集合。

        X_data : (nLens, 2)  [FN, HFOV]

        返回:
          inRays   : (nLens*nPts*M, 6) [px,py,pz,vx,vy,vz]
          ideal_y  : (nLens*nPts*M, 1)
          lens_id  : (nLens*nPts*M,)
          theta_vals : (nLens*nPts*M,)  # 如需像差分析可以用
          nPts     : int, pupil 采样点数
        """
        device = X_data.device
        dtype = self.dtype

        # ---- 把 X_data 搬到 numpy，保持与原 generate_in_rays 完全一致 ----
        X_np = X_data.detach().cpu().numpy()
        nLens = X_np.shape[0]

        FN   = X_np[:, 0].astype(np.float64)  # F-number
        HFOV = X_np[:, 1].astype(np.float64)  # 视场角（度）
        ENPD = float(self.EPD)               # 出瞳直径
        EFL  = FN * ENPD                     # 有效焦距

        # HFOV 转弧度（保持原实现）
        HFOV_rad = np.deg2rad(HFOV)

        # 1) pupil 上的点：使用 __init__ 里预计算的 self.pupil_xyz_np
        P_unit = self.pupil_xyz_np.astype(np.float64)   # (nPts, 3)
        nPts = P_unit.shape[0]

        # 对每个镜头缩放 ENPD/2：得到 (nLens, nPts, 3)
        P = np.broadcast_to(P_unit, (nLens, nPts, 3)) * ENPD / 2.0

        # 2) 方向向量：每镜头 M 个角度 (nLens, M)
        M = self.n_field
        factors = np.linspace(0.0, 1.0, M, dtype=np.float64)[None, :]   # (1,M)
        thetas = HFOV_rad[:, None] * factors                            # (nLens,M)

        vx = np.zeros_like(thetas)                                      # (nLens,M)
        vy = np.sin(thetas)
        vz = np.sqrt(np.clip(1.0 - vy**2, 0.0, 1.0))
        V = np.stack([vx, vy, vz], axis=2)                              # (nLens,M,3)

        # 3) 做笛卡尔积：每镜头 nPts 个点 × M 个方向
        P4 = P[:, :, None, :]                     # (nLens, nPts, 1, 3)
        V4 = V[:, None, :, :]                     # (nLens, 1, M, 3)
        P_all = np.broadcast_to(P4, (nLens, nPts, M, 3))
        V_all = np.broadcast_to(V4, (nLens, nPts, M, 3))

        # 合并位置与方向，展平为 (nLens*nPts*M, 6)
        inRays = np.concatenate([P_all, V_all], axis=3).reshape(-1, 6)

        # 4) ideal_y = EFL * tan(theta) 与射线对齐
        theta_grid = np.broadcast_to(thetas[:, None, :], (nLens, nPts, M))
        EFL_grid   = np.broadcast_to(EFL[:, None, None], (nLens, nPts, M))
        ideal_y = (EFL_grid * np.tan(theta_grid)).transpose(0, 2, 1).reshape(-1, 1)

        # 5) lens_id 和 theta_vals
        lens_id = np.repeat(np.arange(nLens, dtype=np.int64), nPts * M)
        theta_vals = theta_grid.reshape(-1)

        # ---- 转回 torch，保持 dtype / device 一致 ----
        inRays_t   = torch.as_tensor(inRays,   dtype=dtype, device=device)
        ideal_y_t  = torch.as_tensor(ideal_y,  dtype=dtype, device=device)
        lens_id_t  = torch.as_tensor(lens_id,  dtype=torch.long, device=device)
        theta_vals_t = torch.as_tensor(theta_vals, dtype=dtype, device=device)

        return inRays_t, ideal_y_t, lens_id_t, theta_vals_t, nPts

    def _field_angles_rad(self, X_data: torch.Tensor) -> torch.Tensor:
        """Return the field-angle samples used by the current USL trace."""
        hfov_rad = X_data[:, 1] * (torch.pi / 180.0)
        factors = torch.linspace(0.0, 1.0, self.n_field, dtype=self.dtype, device=X_data.device)
        return hfov_rad[:, None] * factors[None, :]

    def _generate_green_chief_rays(self, field_angles_rad: torch.Tensor):
        """Generate chief rays at the pupil center for arbitrary field angles."""
        device = field_angles_rad.device
        nLens, nField = field_angles_rad.shape

        pos = torch.zeros((nLens, nField, 3), dtype=self.dtype, device=device)
        vx = torch.zeros_like(field_angles_rad)
        vy = torch.sin(field_angles_rad)
        vz = torch.sqrt(torch.clamp(1.0 - vy ** 2, min=0.0))
        dirs = torch.stack([vx, vy, vz], dim=2)

        in_rays = torch.cat([pos, dirs], dim=2).reshape(-1, 6)
        lens_id = torch.arange(nLens, device=device, dtype=torch.long).repeat_interleave(nField)
        return in_rays, lens_id

    def _trace_green_chief_heights(
        self,
        lens: LensBatch,
        max_surf_lens,
        surf_lens,
        mask,
        field_angles_rad: torch.Tensor,
    ) -> torch.Tensor:
        """Trace green chief rays and return image-surface radial heights."""
        N_bgr = self._insert_last_n(lens.N_bgr, surf_lens)
        CT = self._insert_last_ct(lens.CT, surf_lens)

        n_g = N_bgr[..., 1]
        curv, thick = CT[..., 0], CT[..., 1]

        in_rays, lens_id = self._generate_green_chief_rays(field_angles_rad)
        nLens, nField = field_angles_rad.shape
        out_g = in_rays.clone()

        for i in range(max_surf_lens):
            valid = mask[lens_id, i]
            if not torch.any(valid):
                continue

            ng_in = n_g[:, i].index_select(0, lens_id[valid]).view(-1, 1)
            ng_out = n_g[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            ci_i = curv[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            ti_i = thick[:, i].index_select(0, lens_id[valid]).view(-1, 1)

            out_gv, _ = self._surf3Draytrace(out_g[valid], ng_in, ng_out, ti_i, ci_i)
            out_g[valid] = out_gv

        chief_xy = out_g.view(nLens, nField, 6)[:, :, :2]
        eps = torch.finfo(chief_xy.dtype).eps
        return torch.sqrt(torch.clamp(chief_xy[..., 0] ** 2 + chief_xy[..., 1] ** 2, min=eps))

    def _first_order_efl_abcd(self, lens: LensBatch, surf_lens, mask) -> torch.Tensor:
        """Compute first-order EFL using paraxial ABCD matrices on green index."""
        N_bgr = self._insert_last_n(lens.N_bgr, surf_lens)
        CT = self._insert_last_ct(lens.CT, surf_lens)

        n_g = N_bgr[..., 1]
        curv, thick = CT[..., 0], CT[..., 1]
        nLens = lens.X.shape[0]
        device_local = lens.X.device
        dtype_local = self.dtype
        eps = torch.finfo(dtype_local).eps

        # System matrix M = [A B; C D] for each lens
        A = torch.ones(nLens, dtype=dtype_local, device=device_local)
        B = torch.zeros(nLens, dtype=dtype_local, device=device_local)
        C = torch.zeros(nLens, dtype=dtype_local, device=device_local)
        D = torch.ones(nLens, dtype=dtype_local, device=device_local)

        max_surf_lens = int(mask.shape[1])
        for i in range(max_surf_lens):
            valid = mask[:, i]
            if not torch.any(valid):
                continue

            n1 = n_g[:, i].clamp_min(eps)
            n2 = n_g[:, i + 1].clamp_min(eps)
            c = curv[:, i + 1]
            t = thick[:, i]

            # Step matrix (translation then refraction): M_step = R @ T
            # T = [[1, t], [0, 1]]
            # R = [[1, 0], [-(n2-n1)/n2 * c, n1/n2]]
            p = -((n2 - n1) / n2) * c
            q = n1 / n2
            sA = torch.ones_like(A)
            sB = t
            sC = p
            sD = p * t + q

            newA = sA * A + sB * C
            newB = sA * B + sB * D
            newC = sC * A + sD * C
            newD = sC * B + sD * D

            A = torch.where(valid, newA, A)
            B = torch.where(valid, newB, B)
            C = torch.where(valid, newC, C)
            D = torch.where(valid, newD, D)

        # For n_in ~= n_out ~= 1, EFL ≈ -1 / C
        C_safe = torch.where(torch.abs(C) < 1e-10, torch.full_like(C, 1e-10), C)
        return torch.abs(-1.0 / C_safe)


    def _surf3Draytrace(self, inRays, n1, n2, t, c):
        """ calculate the ray positions after a spherical surface.
            inRays: [nRay, 6]
            n1: refractive index before surface, same tensor size as inRays or a scalar
            n2: refractive index after surface, same tensor size as inRays or a scalar
            t: thickness before surface, same tensor size as inRays or a scalar
            c: curvature of the surface, starting from pupil (not include), so index+1.
            Returns:
                outRays - 6D of ray vectors, px, py, pz and vx, vy, vz. [nRay, 6]
                lossRays - a value for missing ray loss.
        """
        eps = 1e-6
        x1, y1, z1, X1, Y1, Z1 = torch.chunk(inRays, 6, dim=1)
        e = t * Z1 - (x1 * X1 + y1 * Y1 + z1 * Z1)  # Intermediate
        M1z = z1 + e * Z1 - t  # Intermediate
        M12 = x1 * x1 + y1 * y1 + z1 * z1 - e * e + t * t - 2 * t * z1  # Intermediate
        E12 = Z1 * Z1 - c * (c * M12 - 2 * M1z)  # square of E1, must in [0, 1]
        E12_c = torch.clamp(E12, min=0 + eps, max=1 - eps)  # 1xnRay, make sure 0 < E12 < 1
        E1 = torch.sqrt(E12_c)  # sin of incident angles, E1 = sin_t1, [0, 1]

        # --- denom 检查 ---
        denom = Z1 + E1
        # print("denom", torch.isnan(denom).any(), denom.min().item(), denom.max().item())
        # denom_mask = (denom > 1e-4) & (torch.abs(Z1) > 1e-3)
        denom_mask = denom > 1e-6
        denom_safe = torch.where(denom_mask, denom, torch.ones_like(denom))

        # --- 交点 ---
        #这时我们不应该“伪造”交点，所以直接让光线保持在原位置(x1, y1, z1)，并且会在 valid_mask 里把这条光线标记为无效。

        L = e + (c * M12 - 2 * M1z) / denom_safe
        x2 = torch.where(denom_mask, x1 + L * X1, x1)
        y2 = torch.where(denom_mask, y1 + L * Y1, y1)
        z2 = torch.where(denom_mask, z1 + L * Z1 - t, z1)

        E22 = 1 - ((n1 / n2) ** 2) * (1 - E1 ** 2)  # square of E2, must in [0, 1]
        # TIR_mask = E22 < 0
        # valid_mask = denom_mask & (~TIR_mask) #有效光线

        E22_c = torch.clamp(E22, min=0 + eps, max=1 - eps)  # 1xnRay, make sure 0 < E22 < 1
        E2 = torch.sqrt(E22_c)  # sin of outgoing angles, E2 = sin_t2, (0, 1)
        g1 = E2 - (n1 / n2) * E1  # Intermediate
        Z2 = (n1 / n2) * Z1 - g1 * c * z2 + g1
        Y2 = (n1 / n2) * Y1 - g1 * c * y2
        X2 = (n1 / n2) * X1 - g1 * c * x2

        dirs = torch.cat([X2, Y2, Z2], dim=1)
        norm2 = torch.sum(dirs ** 2, dim=1, keepdim=True)
        #=== 方向归一化 ===
        zero_mask = norm2 < 1e-6
        dirs_normalized = dirs / torch.sqrt(norm2.clamp(min=1e-6))
        dummy_vec = torch.tensor([0.0, 0.0, 1.0],device=dirs.device,dtype=dirs.dtype).expand_as(dirs)
        dirs = torch.where(zero_mask, dummy_vec, dirs_normalized)
        X2, Y2, Z2 = torch.chunk(dirs, 3, dim=1)
        outRays = torch.cat((x2, y2, z2, X2, Y2, Z2), 1)  # dim=1

        # valid_ratio = denom_mask.float().mean().item()
        # print(f"Valid rays ratio: {valid_ratio:.3f}")

        # 对每根光线单独计算损失。不允许TIR或者missing！如果不符合则有损失项，大小与偏离程度有关系。
        lossRays = torch.maximum(E12 - 1, torch.full_like(E12, 0)) + \
                   torch.maximum(E22 - 1, torch.full_like(E22, 0))  # penalty for E12&E22 >1
        lossRays += torch.maximum(-1 * E12, torch.full_like(E12, 0)) + \
                    torch.maximum(-1 * E22, torch.full_like(E22, 0))  # penalty for E12&E22 <0

        # 加入 valid_mask 惩罚（无效光线→大 penalty）
        lossRays = lossRays + (~denom_mask).float()
        lossRays = lossRays + zero_mask.float()

        return outRays, lossRays


    # -------------------------- main trace -------------------------- #
    def raytrace_all(self, lens: LensBatch, max_surf_lens, surf_lens, mask):
        """Ray tracing through all surfaces.

        X_data : (B, 2)  [FN, HFOV]
        N_bgr  : (B, L, 3)
        CT     : (B, L, 2)  [curv, thick]
        surf_lens : (B,) the insert position for the final surface
        mask   : (B, L)  True for valid surfaces (before insert)
        """
        X_data = lens.X
        N_bgr = lens.N_bgr
        CT = lens.CT
        device_local = X_data.device
        nLens = X_data.shape[0]  # 不是镜片数目，是数据集里的镜头数目

        # Insert last surfaces to align (L -> L+1)
        N_bgr = self._insert_last_n(N_bgr, surf_lens)
        CT = self._insert_last_ct(CT, surf_lens)

        n_b, n_g, n_r = N_bgr[..., 0], N_bgr[..., 1], N_bgr[..., 2]
        curv, thick = CT[..., 0], CT[..., 1]
        # curv = torch.where(curv != 0, 1.0 / curv, torch.zeros_like(curv))

        inRays, ideal_y, lens_id, _, nSample = self._generate_in_rays(X_data)

        nThetas = self.n_field
        nTotal = inRays.shape[0]  # nLens * nSample * nField
        nL, nS, nT = nLens, nSample, nThetas

        out_b = inRays.clone()
        out_g = inRays.clone()
        out_r = inRays.clone()

        # 逐射线累计的惩罚项（列向量）
        loss_ray_total = torch.zeros(nTotal, 1, device=device_local, dtype=self.dtype)
        ovlp_total = torch.zeros(nTotal, 1, device=device_local, dtype=self.dtype)
        # global_mask = torch.ones(nTotal, dtype=torch.bool, device=device)\
        # valid_mask_all = torch.ones(nTotal, dtype=torch.bool, device=device)
        prev_zg = None

        # Loop over surfaces (use up to max_surf_lens surfaces)
        for i in range(max_surf_lens):
            valid = mask[lens_id, i]
            if not torch.any(valid):
                continue

            # 取出有效射线的参数
            nb_in = n_b[:, i].index_select(0, lens_id[valid]).view(-1, 1)
            ng_in = n_g[:, i].index_select(0, lens_id[valid]).view(-1, 1)
            nr_in = n_r[:, i].index_select(0, lens_id[valid]).view(-1, 1)
            nb_out = n_b[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            ng_out = n_g[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            nr_out = n_r[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            ci_i = curv[:, i + 1].index_select(0, lens_id[valid]).view(-1, 1)
            ti_i = thick[:, i].index_select(0, lens_id[valid]).view(-1, 1)

            out_bv, lb = self._surf3Draytrace(out_b[valid], nb_in, nb_out, ti_i, ci_i)
            out_gv, lg = self._surf3Draytrace(out_g[valid], ng_in, ng_out, ti_i, ci_i)
            out_rv, lr= self._surf3Draytrace(out_r[valid], nr_in, nr_out, ti_i, ci_i)

            out_b[valid], out_g[valid], out_r[valid] = out_bv, out_gv, out_rv
            loss_ray_total[valid] += (lb + lg + lr) / 3.0

            # # 累积有效性
            # valid_mask_all[valid] &= mask_b & mask_g & mask_r
            # global_mask[valid] &= local_mask

            # 厚度 overlap 检查 (仅有效射线参与)
            zg = out_g[valid, 2:3]  # 当前 green 波段 z
            if prev_zg is not None:
                delta = prev_zg[valid] - ti_i - zg
                ovlp_total[valid] += torch.maximum(delta, torch.zeros_like(delta))
            if prev_zg is None:
                prev_zg = torch.zeros_like(out_g[:, 2:3])
            prev_zg[valid] = zg
        ## TODO: 直接通过最后一面的WL=g数据获得实际EFL
        edge = self.marginal_y_sample_idx
        # out_g: (nTotal, 6)
        VXg = out_g[:, 3].view(nL, nS, nT)
        VYg = out_g[:, 4].view(nL, nS, nT)
        VZg = out_g[:, 5].view(nL, nS, nT)

        VX_edge = VXg[:, edge, 0]  # theta=0, 最后一个 sample
        VY_edge = VYg[:, edge, 0]
        VZ_edge = VZg[:, edge, 0]

        # === 有效光线过滤 ===
        # valid_mask_edge = valid_mask_all.view(nL, nS, nT)[:, edge, 0]  # 每个镜头的边缘 chief ray 是否有效

        eps = torch.finfo(out_g.dtype).eps
        vnorm = torch.sqrt(torch.clamp(VX_edge ** 2 + VY_edge ** 2 + VZ_edge ** 2, min=eps))
        cos_th = torch.clamp(VZ_edge / vnorm, -1.0, 1.0)
        theta = torch.acos(cos_th)  # 弧度

        EFL_est_g = self.EPD / 2 / torch.tan(theta + eps)  # (nLens,)
        EFL_first_order = self._first_order_efl_abcd(lens, surf_lens, mask)
        EFL_ideal = X_data[:, 0] * self.EPD  # 40mm
        # Primary EFL source is selected by efl_loss_mode: legacy trace EFL or ABCD EFL.
        efl_loss_mode = str(getattr(self.opt, "efl_loss_mode", "trace")).lower()
        if efl_loss_mode not in {"trace", "abcd"}:
            efl_loss_mode = "trace"
        efl_primary = EFL_first_order if efl_loss_mode == "abcd" else EFL_est_g
        efl_error_ratio = torch.abs(efl_primary - EFL_ideal) / (EFL_ideal + eps)
        efl_loss_tolerance = float(getattr(self.opt, "efl_loss_tolerance", 0.1))
        loss_EFL_trace = torch.clamp(efl_error_ratio - efl_loss_tolerance, min=0.0)

        # 控制项：trace EFL 与一阶 ABCD EFL 需要一致
        enable_efl_ctrl = bool(getattr(self.opt, "enable_efl_first_order_control", True)) and efl_loss_mode == "abcd"
        ctrl_tol = float(getattr(self.opt, "efl_first_order_tolerance", efl_loss_tolerance))
        ctrl_w = float(getattr(self.opt, "efl_first_order_weight", 0.5))
        efl_ctrl_ratio = torch.abs(EFL_est_g - EFL_first_order) / (torch.abs(EFL_first_order) + eps)
        loss_EFL_control = torch.clamp(efl_ctrl_ratio - ctrl_tol, min=0.0)
        if not enable_efl_ctrl:
            loss_EFL_control = torch.zeros_like(loss_EFL_control)
        loss_EFL = loss_EFL_trace + ctrl_w * loss_EFL_control

        # 拼接最终三波段
        outRays = torch.stack([out_b, out_g, out_r], dim=2)  # (nTotal, 6, 3)

        # 按 lens 统计平均
        loss_ray = loss_ray_total.view(nL, nT, nS).mean(dim=(1, 2))  # [nLens]
        loss_ovlp = ovlp_total.view(nL, nT, nS).mean(dim=(1, 2))  # [nLens]

        return (
            outRays,
            ideal_y,
            loss_ray,
            loss_ovlp,
            loss_EFL,
            EFL_est_g,
            EFL_ideal,
            EFL_first_order,
            loss_EFL_trace,
            loss_EFL_control,
        )


    # -------------------------- metrics -------------------------- #
    def _spot_rms(self, outRays, X_data):
        """RMS spot using green chief ray as reference."""

        # nL, nS, nT, nW = nL, nS, self.nThetas, self.nWavelength
        nT = self.n_field
        nW = self.n_wavelength
        nL = X_data.shape[0]
        nTotal = outRays.shape[0]
        nS = nTotal // (nL * nT)

        cent = self.chief_sample_idx
        # outRays [nTotal, 6, nW] => [nW, nTotal] => [nW, nL, nT, nS]
        imx = outRays[:, 0, :]  # (nTotal, nW)
        imy = outRays[:, 1, :]

        # mask 展开成 [nL,nS,nT,nW]
        # mask4d = valid_mask_all.view(nL, nS, nT, 1).expand(-1, -1, -1, nW)
        imx4d = imx.view(nL, nS, nT, nW).permute(3, 0, 2, 1)  # (nL, nS, nT, nW) -> (nW, nL, nT, nS)
        imy4d = imy.view(nL, nS, nT, nW).permute(3, 0, 2, 1)  # (nL, nS, nT, nW) -> (nW, nL, nT, nS)
        # mask4d = mask4d.permute(3, 0, 2, 1)

        cimy_b = imy4d[1:2, :, :, cent:cent + 1]
        cimy_rgb = imy4d[:, :, :, cent:cent + 1]  # [nWL, nLens, nT, 1]
        imax = imx4d
        imay = imy4d - cimy_b
        # num_valid = mask4d.sum(dim=3).clamp(min=1.0)  # 每个视场有效光线数
        spot2 = imax ** 2 + imay ** 2
        # spot2_sum = (spot2 * mask4d).sum(dim=3)
        # rms = torch.sqrt(spot2_sum / num_valid)
        rms = torch.sqrt(spot2.mean(dim=3))  # [nWL,nLens,nThetas]
        # print(rms.isnan().any())
        rms = rms.permute(1, 0, 2)  # [nLens, nWL, nThetas]
        rms_g = rms[:,1:2,:]
        rms_g_mean = rms_g.mean(dim=(1,2))
        rms_mean = torch.sqrt((rms_g ** 2).mean(dim=(1, 2)))  # [nLens]  # checkpoint for spot size among WL and fields.
        return rms_mean, rms_g_mean, cimy_rgb

    def _chrom_loss(self, cimy_rgb):
        """Chromatic shift between R/G and B/G chief rays."""
        # 只考虑rgb主光线与g主光线之间的误差值！！
        # cimy_rgb: [nWL,nLens,nThetas,1]
        assert cimy_rgb.shape[0] == 3, f"We need three wavelengths (B,G,R)，but only get {cimy_rgb.shape[0]}"
        imax_R = cimy_rgb[2, :, :, :] - cimy_rgb[1, :, :, :]
        imax_B = cimy_rgb[0, :, :, :] - cimy_rgb[1, :, :, :]
        loss_chrom = torch.sqrt(((imax_R ** 2 + imax_B ** 2)).mean(dim=1)).squeeze(-1)  # [nLens, nW]
        return loss_chrom

    def _distortion(self, cimy_rgb, ideal_y, X_data):
        """Radial distortion penalty relative to ideal image height.
         Notes: Only g channel is calculated!
        Parameters
        ----------
        cimy_rgb : (3, nL, nT, 1)
            Chief ray image heights for B, G, R.
        ideal_y : (nTotal, 1)
            Ideal image heights for all rays (flattened).
        X_data : (nL, 2)
            Used to infer nL, nS, nT.
        """
        # 1) 通过 X_data 和 ideal_y 推导 nL, nT, nS
        nL = X_data.shape[0]  # 镜头数量
        nT = self.n_field  # 视场数
        nTotal = ideal_y.shape[0]  # 所有镜头 × 所有视场 × pupil sample 数
        nS = nTotal // (nL * nT)  # 每个镜头的 pupil sample 数

        # 2) reshape ideal_y → (nL, nT, nS)，取中心 pupil 点
        cent = self.chief_sample_idx
        ideal_y_reshaped = ideal_y.view(nL, nT, nS)[:, :, cent:cent + 1]  # [nL, nT, 1]

        # 3) 只用 G 波段的 chief ray 位置与理想像高比较
        ideal_r = ideal_y_reshaped ** 2
        delta_r = (cimy_rgb[1, :, :, :] - ideal_y_reshaped) ** 2  # [nL, nT, 1]

        # 4) 避免 ideal_r 为 0 的点
        mask = ideal_r != 0
        ideal_r_nz = ideal_r[mask].view(nL, -1)
        delta_r_nz = delta_r[mask].view(nL, -1)

        dist = torch.sqrt(delta_r_nz / ideal_r_nz)  # [nL, nT']
        penalty_dist = torch.maximum(dist - self.maxDist,
                                     torch.full_like(dist, 0))
        loss_dist = penalty_dist.amax(dim=1)  # [nL]   # only green
        dist_max = dist.amax(dim=1)  # [nL]  # only green

        return loss_dist, dist_max

    def _distortion_zemax_ftan(self, lens: LensBatch, outRays, ideal_y, max_surf_lens, surf_lens, mask):
        """Zemax-like F-Tan(theta) distortion using a small-angle chief-ray reference."""
        X_data = lens.X
        nL = X_data.shape[0]
        nT = self.n_field
        nW = self.n_wavelength
        nTotal = outRays.shape[0]
        nS = nTotal // (nL * nT)
        eps = torch.finfo(outRays.dtype).eps

        cent = self.chief_sample_idx
        chief_x = outRays[:, 0, :].view(nL, nS, nT, nW)[:, cent, :, 1]
        chief_y = outRays[:, 1, :].view(nL, nS, nT, nW)[:, cent, :, 1]
        chief_radial = torch.sqrt(torch.clamp(chief_x ** 2 + chief_y ** 2, min=eps))

        field_angles = self._field_angles_rad(X_data)
        ref_angle_deg = float(getattr(self.opt, "distortion_ref_angle_deg", 0.01))
        ref_angle_rad = max(ref_angle_deg, 1e-6) * torch.pi / 180.0
        ref_angles = torch.full((nL, 1), ref_angle_rad, dtype=self.dtype, device=X_data.device)
        ref_height_small = self._trace_green_chief_heights(
            lens,
            max_surf_lens,
            surf_lens,
            mask,
            ref_angles,
        )

        tan_ref = torch.tan(ref_angles).clamp_min(eps)
        ref_height = ref_height_small * torch.tan(field_angles) / tan_ref

        nonzero_field = field_angles.abs() > 1e-12
        dist_signed_fields = torch.zeros_like(chief_radial)
        dist_abs_fields = torch.zeros_like(chief_radial)
        if torch.any(nonzero_field):
            denom = ref_height.abs().clamp_min(eps)
            dist_signed_fields[nonzero_field] = (
                chief_radial[nonzero_field] - ref_height[nonzero_field]
            ) / denom[nonzero_field]
            dist_abs_fields[nonzero_field] = torch.abs(dist_signed_fields[nonzero_field])

        penalty_dist = torch.maximum(dist_abs_fields - self.maxDist, torch.zeros_like(dist_abs_fields))
        loss_dist = penalty_dist.amax(dim=1)
        dist_max = dist_abs_fields.amax(dim=1)

        dist_argmax = dist_abs_fields.argmax(dim=1)
        dist_signed = dist_signed_fields[torch.arange(nL, device=X_data.device), dist_argmax]

        ideal_height = torch.abs(ideal_y.view(nL, nT, nS)[:, :, cent])
        dist_target_fields = torch.zeros_like(chief_radial)
        target_mask = ideal_height > eps
        if torch.any(target_mask):
            dist_target_fields[target_mask] = (
                torch.abs(chief_radial[target_mask] - ideal_height[target_mask])
                / ideal_height[target_mask].clamp_min(eps)
            )
        dist_target = dist_target_fields.amax(dim=1)

        return loss_dist, dist_max, dist_signed, dist_target

    def _telecentricity(self, outRays, X_data):
        """Image-space telecentricity penalty (degrees over threshold)."""
        # nL, nS, nT, nW = nL, nS, self.nThetas, self.nWavelength
        nT = self.n_field
        nW = self.n_wavelength
        nL = X_data.shape[0]
        nTotal = outRays.shape[0]
        nS = nTotal // (nL * nT)

        eps = torch.finfo(outRays.dtype).eps
        cent = self.chief_sample_idx
        VX = outRays[:, 3, :].view(nL, nS, nT, nW)[:, cent, :, :]  # [nTotal, 6, 3] -> [nTotal, 3] -> [nL, nT, nW]
        VY = outRays[:, 4, :].view(nL, nS, nT, nW)[:, cent, :, :]
        VZ = outRays[:, 5, :].view(nL, nS, nT, nW)[:, cent, :, :]

        vnorm = torch.sqrt(torch.clamp(VX ** 2 + VY ** 2 + VZ ** 2, min=eps))  # >0
        cosang = torch.clamp(VZ / vnorm, -1.0 + 1e-6, 1.0 - 1e-6)
        tele_deg = torch.acos(cosang) * (180.0 / torch.pi)  # [nL, nT, nW]
        tele_deg_g = tele_deg[:,:,1:2]

        # 把无效光线设置为 0 惩罚（不参与 penalty）
        # tele_deg = torch.where(mask4d, tele_deg, torch.zeros_like(tele_deg))

        # 超限惩罚：max(θ - self.maxTele, 0)
        penalty_tele=torch.maximum(tele_deg_g - self.maxTele, torch.full_like(tele_deg_g, 0))
        # penalty_tele = torch.clamp(tele_deg_g - self.maxTele, min=0.0)  # [nL, nT, nW], clamp把负数赋值为0，因为非负才做惩罚
        # 每个镜头的最差（跨视场&波段）惩罚值
        loss_tele = penalty_tele.amax(dim=(1, 2))  # [nLens],
        tele = tele_deg_g.amax(dim=(1, 2))
        return loss_tele, tele

    # ========================= 对外接口================== #
    def compute_metrics(
        self,
        lens: LensBatch,
        out_all,
        ideal_y,
        loss_ray,
        loss_ovlp,
        loss_EFL,
        EFL_est,
        EFL_ideal,
        EFL_first_order,
        loss_EFL_trace,
        loss_EFL_control,
        max_surf_lens,
        surf_lens,
        mask,
        epoch,
        save,
        apply_hard_filter=True,
    ):
        """Aggregate all loss terms into a unified metric (geometric mean for robustness)."""
        # 从Class LensBatch里传参
        X_data = lens.X
        N_data = lens.N_bgr
        CT_data = lens.CT

        # 通过 shape 推导维度信息，不再从外面传 nL / nS
        nL = X_data.shape[0]
        nT = self.n_field
        nTotal = out_all.shape[0]  # 所有镜头 * 所有视场 * 所有 pupil sample
        nS = nTotal // (nL * nT)

        rms_mean,rms_g_mean, cimy_rgb = self._spot_rms(out_all, X_data)
        # loss_chrom = self._chrom_loss(cimy_rgb)
        distortion_mode = getattr(self.opt, "distortion_mode", "zemax_ftan")
        if distortion_mode == "target_height":
            loss_dist, dist = self._distortion(cimy_rgb, ideal_y, X_data)
            dist_signed = dist
            dist_target = dist
        else:
            loss_dist,dist,dist_signed,dist_target = self._distortion_zemax_ftan(
                lens, out_all, ideal_y, max_surf_lens, surf_lens, mask
            )
        loss_tele,tele = self._telecentricity(out_all, X_data)

        # 2. 计算训练进度，动态调整 drop_ratio（RMS筛选）
        progress = epoch / 5000  # 当前训练进度（0 ~ 1）
        if progress < 0.2:  # 阶段1：快速下降。 0-1000
            # exp/log插值，τ: 1.0 -> 0.1
            drop_ratio_rms = 5 - (5 - 1) * (progress / 0.2)
        elif progress < 0.8:  # 阶段2：缓慢下降
            # drop_ratio=0.1 * (0.05 / 0.1) ** ((progress - 0.2) / 0.6)
            drop_ratio_rms = 1 - (1 - 0.05) * ((progress - 0.2) / 0.6)
        else:  # 阶段3：保持低温，接近hard
            drop_ratio_rms = 0.05

        # 2.1 计算EFL筛选的动态阈值（从宽松到严格）
        # EFL比值阈值：允许的最大偏离度 (从50%逐渐降到30%)
        if progress < 0.2:  # 阶段1：快速下降
            efl_ratio_threshold = 0.8 - (0.8 - 0.6) * (progress / 0.2)  # 0.8 -> 0.6
        elif progress < 0.8:  # 阶段2：缓慢下降
            efl_ratio_threshold = 0.6 - (0.6 - 0.3) * ((progress - 0.2) / 0.6)  # 0.6 -> 0.3
        else:  # 阶段3：保持严格
            efl_ratio_threshold = 0.3

        # 3. 按需应用硬筛选
        enable_rms_filter = getattr(self.opt, "enable_rms_filter", True)
        enable_efl_filter = getattr(self.opt, "enable_efl_filter", False)

        eps = torch.finfo(EFL_ideal.dtype).eps
        efl_loss_mode = str(getattr(self.opt, "efl_loss_mode", "trace")).lower()
        if efl_loss_mode not in {"trace", "abcd"}:
            efl_loss_mode = "trace"
        efl_filter_value = EFL_first_order if efl_loss_mode == "abcd" else EFL_est
        efl_ratio = efl_filter_value / (EFL_ideal + eps)
        efl_deviation = torch.abs(efl_ratio - 1.0)

        with torch.no_grad():
            if apply_hard_filter:
                if enable_rms_filter:
                    rms_keep_mask = (rms_mean <= drop_ratio_rms)
                else:
                    rms_keep_mask = torch.ones_like(rms_mean, dtype=torch.bool)

                if enable_efl_filter:
                    fixed_efl_loss_threshold = getattr(self.opt, "efl_filter_max_loss", None)
                    if fixed_efl_loss_threshold is not None:
                        efl_keep_mask = (loss_EFL < float(fixed_efl_loss_threshold))
                    else:
                        efl_keep_mask = (efl_deviation <= efl_ratio_threshold)
                else:
                    efl_keep_mask = torch.ones_like(rms_mean, dtype=torch.bool)

                keep_mask = rms_keep_mask & efl_keep_mask
            else:
                keep_mask = torch.ones_like(rms_mean, dtype=torch.bool)

        rms_sel = rms_mean[keep_mask]  # [nKeep]
        rms_g_sel = rms_g_mean[keep_mask]  # [nKeep]
        loss_ray_sel = loss_ray[keep_mask]
        loss_dist_sel = loss_dist[keep_mask]
        dist_sel = dist[keep_mask]
        dist_signed_sel = dist_signed[keep_mask]
        dist_target_sel = dist_target[keep_mask]
        loss_EFL_sel = loss_EFL[keep_mask]
        EFL_est_sel = EFL_est[keep_mask]
        EFL_ideal_sel = EFL_ideal[keep_mask]
        EFL_first_order_sel = EFL_first_order[keep_mask]
        loss_EFL_trace_sel = loss_EFL_trace[keep_mask]
        loss_EFL_control_sel = loss_EFL_control[keep_mask]
        loss_ovlp_sel = loss_ovlp[keep_mask]
        loss_tele_sel = loss_tele[keep_mask]
        tele_sel = tele[keep_mask]

        # 4. 先对全 batch 计算 composite，再按 keep_mask 聚合
        loss_variant, loss_cfg = self._loss_config()
        term_values = {
            "rms": rms_mean,
            "loss_ray": loss_ray,
            "loss_dist": loss_dist,
            "loss_EFL": loss_EFL,
            "loss_ovlp": loss_ovlp,
            "loss_tele": loss_tele,
        }
        if loss_cfg["aggregation"] == "geometric_mean":
            composite_all = rms_mean
            for term_name, weight in loss_cfg["multiplicative_terms"]:
                composite_all = composite_all * (1 + float(weight) * term_values[term_name])
        else:
            composite_all = rms_mean * 0.0
            for term_name, weight in loss_cfg["additive_terms"]:
                composite_all = composite_all + float(weight) * term_values[term_name]

        composite_sel = composite_all[keep_mask]

        if composite_sel.numel() == 0:
            loss = rms_mean.sum() * 0.0
            loss_spot = loss
        elif loss_variant == "stage1_geometric_v1":
            loss = torch.exp(torch.mean(torch.log(composite_sel + eps)))
            loss_spot = torch.mean(rms_sel)
        else:
            loss = torch.mean(composite_sel)
            loss_spot = torch.mean(rms_sel)

        if save == 1:
            metrics = {
                "X": X_data[keep_mask, :],
                "N": N_data[keep_mask, :],
                "CT": CT_data[keep_mask, :],
                "composite": composite_sel,
                "rms": rms_sel,
                "rms_g": rms_g_sel,
                "loss_ray": loss_ray_sel,
                "loss_ovlp": loss_ovlp_sel,
                "dist": dist_sel,
                "loss_dist": loss_dist_sel,
                "dist_signed": dist_signed_sel,
                "dist_target": dist_target_sel,
                "tele": tele_sel,
                "loss_tele": loss_tele_sel,
                "EFL_est": EFL_est_sel,
                "EFL_ideal": EFL_ideal_sel,
                "EFL_first_order": EFL_first_order_sel,
                "loss_EFL": loss_EFL_sel,
                "loss_EFL_trace": loss_EFL_trace_sel,
                "loss_EFL_control": loss_EFL_control_sel,
            }

            # 可选：检查 keys 是否与 USL_METRIC_KEYS 对齐，避免遗漏
            for k in metrics.keys():
                assert k in USL_METRIC_KEYS, f"Unknown metric: {k}"

            return loss, loss_spot, metrics

        return loss, loss_spot


    def forward(self, lens: LensBatch, max_surf_lens, surf_lens, mask, epoch, save=0, apply_hard_filter=True):

        (
            outRays,
            ideal_y,
            loss_ray,
            loss_ovlp,
            loss_EFL,
            EFL_est,
            EFL_ideal,
            EFL_first_order,
            loss_EFL_trace,
            loss_EFL_control,
        ) = self.raytrace_all(
            lens, max_surf_lens, surf_lens, mask
        )

        return self.compute_metrics(
            lens,
            outRays, ideal_y,
            loss_ray, loss_ovlp,
            loss_EFL, EFL_est, EFL_ideal, EFL_first_order, loss_EFL_trace, loss_EFL_control,
            max_surf_lens, surf_lens, mask,
            epoch,
            save,
            apply_hard_filter
        )

# -----------------------------------------------------------------------------
# Minimal demo
# -----------------------------------------------------------------------------
def run_usl_demo(
    opt,
    X_data: torch.Tensor,
    Y_data: torch.Tensor,
    CT_data: torch.Tensor,
    nSurf_used: int,
    tag: str = "",
):
    """
    通用 demo：给定已经是 torch.Tensor 的 X/Y/CT 和使用的面数 nSurf_used，
    自动构造 LensBatch 和 mask，跑一次 USL_Loss 并打印时间。

    X_data : (B, 2)
    Y_data : (B, nSurf_used, 3)
    CT_data: (B, nSurf_used, 2)
    nSurf_used : 实际参与追迹的表面数，例如 9 或 11
    """
    B = X_data.shape[0]
    max_surf_lens = 11  # 你的模型目前就是对齐到 11 面

    # mask: (B, max_surf_lens)，前 nSurf_used 个为 True
    mask = torch.arange(max_surf_lens, device=device)[None, :].expand(B, max_surf_lens) < nSurf_used
    surf_lens = torch.full((B,), nSurf_used, device=device, dtype=torch.long)

    lens = LensBatch(X_data, Y_data, CT_data)
    optics = USL_Loss(opt).to(device)

    t0 = tc()
    loss, rms_g = optics(lens, max_surf_lens, surf_lens, mask, epoch=0, save=0)
    t1 = tc()

    prefix = f"[{tag}] " if tag else ""
    print(f"{prefix}USL loss:", loss.detach().cpu().numpy())
    print(f"{prefix}RMS G spot:", rms_g.detach().cpu().numpy())
    print(f"{prefix}Time cost: {t1 - t0:.4f} s")


if __name__ == "__main__":
    opt = set_parser()

    # ==========================
    # Demo 1: prediction, 9 surfaces
    # 对应文件: ./log/251125/test_output_metrics_pred.csv
    # Zemax: 1125_Transformer_test_output_nohard_line1_pred_10.zmx
    # ==========================
    dataset_path = r"./log/251125/test_output_metrics_pred.csv"
    nSurf_pred = 9
    row_pred = 66
    offset_ct_pred = 6

    # # ==========================
    # # Demo 2: prediction, 11 surfaces
    # # 对应文件: ./log/251125/test_output_metrics_pred.csv
    # # Zemax: scanlens_surf12_line31_pred_251125.zmx
    # # ==========================
    # dataset_path = r"./log/251125/test_output_metrics_pred.csv"
    # nSurf_pred = 11
    # row_pred = 30
    # offset_ct_pred = 0

    # ========= Commom code for Demo 1 and 2================
    X_np, Y_np, CT_np = read_scan_lens_csv(
        dataset_path,
        n_surf=nSurf_pred,
        row=row_pred,
        offset_ct=offset_ct_pred,
    )
    # 原函数已经给了 shape (1, nSurf, 3)/(1, nSurf, 2)，这里直接转 tensor 即可
    X_t = torch.tensor(X_np, dtype=torch.float64, device=device)
    Y_t = torch.tensor(Y_np, dtype=torch.float64, device=device)
    CT_t = torch.tensor(CT_np, dtype=torch.float64, device=device)

    run_usl_demo(opt, X_t, Y_t, CT_t, nSurf_used=nSurf_pred)

    # # # ==========================
    # # # Demo 3: orig ground truth, 11 surfaces
    # # # 对应文件: ./data/scan_lens_dataset_surf12_reorder.csv
    # # # Zemax: scanlens_surf12_line1.zmx  (示例)
    # # # ==========================
    # # dataset_path_gt = r"./data/scan_lens_dataset_surf12_reorder.csv"
    # # nSurf = 12          # 文件里总表面数
    #
    # # ==========================
    # # Demo 4: orig ground truth, 9 surfaces
    # # 对应文件: ./data/scan_lens_dataset_surf10_reorder.csv
    # # Zemax: scanlens_surf10_line1.zmx  (示例)
    # # ==========================
    # dataset_path_gt = r"./data/scan_lens_dataset_surf10_reorder.csv"
    # nSurf = 10          # 文件里总表面数
    #
    # # ===========common code for Demo 3 and 4 ============
    # X_gt, Y_gt, CT_gt = read_orig_scan_lens_csv(
    #     dataset_path_gt,
    #     nSurf-1 )
    # Y_gt = Y_gt.reshape(1, nSurf-1, 3)  # 将数据reshape成 [镜头数,序列长度，3通道]
    # CT_gt = CT_gt.reshape(1, nSurf-1, 2)  # 将数据reshape成 [镜头数,序列长度，2]
    # X_gt_t = torch.tensor(X_gt, dtype=torch.float64, device=device)
    # Y_gt_t = torch.tensor(Y_gt, dtype=torch.float64, device=device)
    # CT_gt_t = torch.tensor(CT_gt, dtype=torch.float64, device=device)
    # CT_gt_t[...,0] = torch.where(CT_gt_t[...,0] != 0, 1.0 / CT_gt_t[...,0], torch.zeros_like(CT_gt_t[...,0]))
    # run_usl_demo(opt, X_gt_t, Y_gt_t, CT_gt_t, nSurf-1)
