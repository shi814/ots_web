'''
Description: Scan lens测试功能和展示
Debug Date:  2025-12-2，目前具备把训练好的网络，生成test数据集。
Status     : ok!
TODO: 增加对test数据集的结果进行可视化，包括
(1) RMS spot size - 直方图， Spot size vs F-number/FOV - 热力图， 散点图。
(2) spot diagram - 后续。
Debug Date:  2025-12-4，增加Spot和Telecentricity的散点图直方图和热力图，增加单独打印efl_est_g，efl_ideal，loss_efl的代码。
TODO:改代码测试不加mask的模型，看看效果。
Debug Date：2025-12-15，增加输出zmx和json功能
'''

from torch.utils.data import DataLoader
from utils import *
import os
import pandas as pd

from dataset_norm import (
    load_test_data,
    convert2real_dataSys,
    convert2real_dataBGR,
)
from USL_Loss import USL_Loss
from data_process.load_dataset import ValDataset
from glass_matching import ensure_material_library_exists, process_test_output_csv
from match_gt_report import generate_gt_match_report
from plot_result_summary import plot_result_summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _rms_filter_tag(opt) -> str:
    """
    根据 RMS 筛选开关生成输出文件后缀，避免不同模式互相覆盖。
    """
    enabled = getattr(opt, "enable_rms_filter", True)
    return "rmsfilter_on" if enabled else "rmsfilter_off"

def _output_dir(opt) -> str:
    """
    输出目录：
    - 开启筛选：用 opt.save_path
    - 关闭筛选：用 opt.save_path/rmsfilter_off（如果 save_path 本身已是 rmsfilter_off 则不再套娃）
    """
    enabled = getattr(opt, "enable_rms_filter", True)
    if enabled:
        out_dir = opt.save_path
    else:
        out_dir = opt.save_path if os.path.basename(opt.save_path) == "rmsfilter_off" else os.path.join(opt.save_path, "rmsfilter_off")

    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def _with_tag(filename: str, tag: str) -> str:
    """
    将 tag 插入到文件名（扩展名前）。
    e.g. test.csv + tag -> test_rmsfilter_on.csv
    """
    base, ext = os.path.splitext(filename)
    return f"{base}_{tag}{ext}"


def _infer_loss_csv_path(opt, out_dir: str):
    """
    Try common run directories for log_loss.csv. Returning None is allowed:
    plot_result_summary() will still generate metrics plots and threshold reports.
    """
    candidates = [
        os.path.join(out_dir, "log_loss.csv"),
        os.path.join(opt.save_path, "log_loss.csv"),
    ]

    load_name = getattr(opt, "load_name", "")
    if load_name:
        ckpt_path = _resolve_existing_path(load_name)
        ckpt_dir = os.path.dirname(ckpt_path)
        run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir
        candidates.append(os.path.join(run_dir, "log_loss.csv"))

    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return None


def _threshold_tag(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _short_export_tag(tag: str) -> str:
    if "eflerror_lt_" in tag:
        return "efl_lt_" + tag.rsplit("eflerror_lt_", 1)[1]
    if tag:
        return tag
    return "all"


def _filter_csv_by_efl_error(metrics_csv: str, loss_csv: str, threshold: float):
    """
    Keep only rows whose relative EFL error is below threshold.

    metrics_csv tail layout stores EFL_est / EFL_ideal in the last two columns.
    The matching loss CSV has the same row order, so the same mask is applied.
    """
    metrics_df = pd.read_csv(metrics_csv, header=None)
    if metrics_df.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in metrics CSV, got {metrics_df.shape[1]}")

    efl_est = pd.to_numeric(metrics_df.iloc[:, -2], errors="coerce")
    efl_ideal = pd.to_numeric(metrics_df.iloc[:, -1], errors="coerce")
    efl_error = (efl_est - efl_ideal).abs() / (efl_ideal.abs() + 1e-10)
    keep_mask = efl_error < float(threshold)

    tag = f"eflerror_lt_{_threshold_tag(threshold)}"
    metrics_root, metrics_ext = os.path.splitext(metrics_csv)
    filtered_metrics_csv = f"{metrics_root}_{tag}{metrics_ext}"
    metrics_df.loc[keep_mask].to_csv(filtered_metrics_csv, header=False, index=False, encoding="utf-8")

    filtered_loss_csv = None
    if loss_csv and os.path.exists(loss_csv):
        loss_df = pd.read_csv(loss_csv, header=None)
        if len(loss_df) != len(metrics_df):
            print(
                f"[Warning] loss CSV row count mismatch: metrics={len(metrics_df)}, "
                f"loss={len(loss_df)}. Skipping filtered loss CSV."
            )
        else:
            loss_root, loss_ext = os.path.splitext(loss_csv)
            filtered_loss_csv = f"{loss_root}_{tag}{loss_ext}"
            loss_df.loc[keep_mask].to_csv(filtered_loss_csv, header=False, index=False, encoding="utf-8")

    print(
        f"[Filter] EFL relative error < {threshold:g}: "
        f"kept {int(keep_mask.sum())}/{len(metrics_df)} rows"
    )
    print(f"[Filter] metrics CSV: {filtered_metrics_csv}")
    if filtered_loss_csv:
        print(f"[Filter] loss CSV: {filtered_loss_csv}")

    return filtered_metrics_csv, filtered_loss_csv, int(keep_mask.sum()), len(metrics_df)


def _read_selected_export_rows(selected_rows_csv: str):
    if not selected_rows_csv or not os.path.exists(selected_rows_csv):
        return None
    df = pd.read_csv(selected_rows_csv)
    if "kind" in df.columns:
        df = df[df["kind"].fillna("").astype(str).str.lower() == "pred"]
    if "pred_row_idx" not in df.columns:
        raise ValueError(f"selected rows CSV must contain pred_row_idx column: {selected_rows_csv}")
    row_indices = []
    for value in df["pred_row_idx"].dropna().tolist():
        if str(value).strip() == "":
            continue
        idx = int(value)
        if idx not in row_indices:
            row_indices.append(idx)
    return row_indices


def _read_selected_gt_export_rows(selected_rows_csv: str):
    if not selected_rows_csv or not os.path.exists(selected_rows_csv):
        return []
    df = pd.read_csv(selected_rows_csv)
    if "kind" not in df.columns:
        return []
    required = {"gt_csv", "gt_row_idx", "n_surf"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"selected rows CSV missing columns {sorted(missing)}: {selected_rows_csv}")

    gt_df = df[df["kind"].fillna("").astype(str).str.lower() == "gt"]
    rows = []
    seen = set()
    for _, row in gt_df.iterrows():
        gt_csv = str(row["gt_csv"]).strip()
        if not gt_csv:
            continue
        gt_row_idx = int(row["gt_row_idx"])
        n_surf = int(row["n_surf"])
        key = (gt_csv, gt_row_idx, n_surf)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"gt_csv": gt_csv, "gt_row_idx": gt_row_idx, "n_surf": n_surf})
    return rows


def _selected_export_out_dir(opt, tag: str) -> str:
    export_out_dir = getattr(opt, "export_out_dir", "") or "exports"
    export_row_arg = int(getattr(opt, "export_row", -1))
    if export_row_arg < 0:
        export_out_dir = os.path.join(export_out_dir, f"zmx_json_{_short_export_tag(tag)}")
    os.makedirs(export_out_dir, exist_ok=True)
    return export_out_dir


def _export_zmx_json_rows(opt, export_csv: str, glass_matching_csv: str, tag: str, row_indices=None):
    """
    Export ZMX/JSON files from the test metrics CSV.

    Behavior:
    - export_row >= 0: export that single 0-based row.
    - export_row < 0 : export all rows, optionally capped by export_max_rows.
    """
    from exports.scanlens_export_from_csv import export_from_config

    df = pd.read_csv(export_csv, header=None)
    n_rows = len(df)
    export_row_arg = int(getattr(opt, "export_row", -1))
    if row_indices is not None:
        row_indices = [int(idx) for idx in row_indices]
        bad_rows = [idx for idx in row_indices if idx < 0 or idx >= n_rows]
        if bad_rows:
            raise IndexError(f"selected export rows out of range for {export_csv}: {bad_rows[:10]}")
    elif export_row_arg >= 0:
        if export_row_arg >= n_rows:
            raise IndexError(f"export_row {export_row_arg} out of range for {export_csv} with {n_rows} rows")
        row_indices = [export_row_arg]
    else:
        row_indices = list(range(n_rows))
        max_rows = int(getattr(opt, "export_max_rows", 0) or 0)
        if max_rows > 0:
            row_indices = row_indices[:max_rows]

    export_format = getattr(opt, "export_format", "pred")
    export_n_surf = int(getattr(opt, "export_n_surf", 0))
    export_epd = float(getattr(opt, "export_epd", 4.0))
    export_out_dir = getattr(opt, "export_out_dir", "exports")
    export_offset_ct_arg = int(getattr(opt, "export_offset_ct", -1))
    export_offset_ct = None if export_offset_ct_arg < 0 else export_offset_ct_arg

    export_out_dir = _selected_export_out_dir(opt, tag)

    manifest_rows = []
    for i, row_idx in enumerate(row_indices, start=1):
        basename = f"row{row_idx:05d}"
        cfg = {
            "format": export_format,
            "csv": export_csv,
            "row": int(row_idx),
            "n_surf": export_n_surf,
            "offset_ct": export_offset_ct,
            "ct_is_radius": False,
            "epd": export_epd,
            "semi_diam": 50.0,
            "semi_diam_stop": None,
            "r_sensor": 20.0,
            "out_dir": export_out_dir,
            "basename": basename,
            "glass_matching_csv": glass_matching_csv,
        }
        record = {"row_idx": row_idx, "status": "failed", "json_path": "", "zmx_path": "", "error": ""}
        try:
            json_path, zmx_path = export_from_config(cfg)
            record.update({"status": "exported", "json_path": json_path, "zmx_path": zmx_path})
            print(f"[Export] [{i}/{len(row_indices)}] row={row_idx} ZMX: {zmx_path}")
        except Exception as e:
            record["error"] = str(e)
            print(f"[Warning] export row {row_idx} failed: {e}")
        manifest_rows.append(record)

    manifest_path = os.path.join(export_out_dir, "export_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")
    success_count = sum(1 for row in manifest_rows if row["status"] == "exported")
    print(f"[Export] Finished {success_count}/{len(manifest_rows)} rows. Manifest: {manifest_path}")
    return manifest_path


def _export_gt_zmx_json_rows(opt, selected_gt_rows, tag: str):
    from exports.scanlens_export_from_csv import export_from_config

    if not selected_gt_rows:
        print("[Export] No matched GT rows selected for ZMX/JSON export.")
        return ""

    export_epd = float(getattr(opt, "export_epd", 4.0))
    export_out_dir = _selected_export_out_dir(opt, tag)

    manifest_rows = []
    for i, row in enumerate(selected_gt_rows, start=1):
        gt_csv = row["gt_csv"]
        gt_row_idx = int(row["gt_row_idx"])
        n_surf = int(row["n_surf"])
        surf_tag = "surf10" if n_surf == 9 else "surf12" if n_surf == 11 else f"nsurf{n_surf}"
        basename = f"gt_{surf_tag}_row{gt_row_idx:05d}"
        cfg = {
            "format": "orig",
            "csv": gt_csv,
            "row": gt_row_idx,
            "n_surf": n_surf,
            "offset_ct": None,
            "ct_is_radius": True,
            "epd": export_epd,
            "semi_diam": 50.0,
            "semi_diam_stop": None,
            "r_sensor": 20.0,
            "out_dir": export_out_dir,
            "basename": basename,
            "glass_matching_csv": "",
        }
        record = {
            "kind": "gt",
            "gt_csv": gt_csv,
            "gt_row_idx": gt_row_idx,
            "n_surf": n_surf,
            "status": "failed",
            "json_path": "",
            "zmx_path": "",
            "error": "",
        }
        try:
            json_path, zmx_path = export_from_config(cfg)
            record.update({"status": "exported", "json_path": json_path, "zmx_path": zmx_path})
            print(f"[Export] GT [{i}/{len(selected_gt_rows)}] row={gt_row_idx} ZMX: {zmx_path}")
        except Exception as e:
            record["error"] = str(e)
            print(f"[Warning] export GT row {gt_row_idx} failed: {e}")
        manifest_rows.append(record)

    manifest_path = os.path.join(export_out_dir, "gt_export_manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")
    success_count = sum(1 for row in manifest_rows if row["status"] == "exported")
    print(f"[Export] Finished GT {success_count}/{len(manifest_rows)} rows. Manifest: {manifest_path}")
    return manifest_path


# Air-gap unsupervised residual test configuration.
# Stage-1 checkpoints are tested directly. Stage-2 AirGapUnsupervised
# checkpoints are tested as residual air-gap refinements on top of their
# recorded stage-1 base checkpoint.
USE_AIRGAP_DELTA_RESIDUAL = True
AIRGAP_BASE_CKPT = os.path.join(
    "log",
    "260521_1013",
    "stage_1",
    "checkpoints",
    "SLT_rmsfilter_on_epoch5000_bs512.pth",
)
AIRGAP_DELTA_SCALE_MM = 10.0


def _read_parameters_file(save_path: str) -> dict:
    params = {}
    param_candidates = [
        os.path.join(save_path, "parameters.txt"),
        os.path.join(save_path, "parameters_airgap_unsupervised.txt"),
    ]
    param_path = next((p for p in param_candidates if os.path.exists(p)), None)
    if param_path is None:
        return params
    with open(param_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            params[key.strip()] = value.strip()
    return params


def _read_checkpoint_run_parameters(opt) -> dict:
    load_name = getattr(opt, "load_name", "")
    if not load_name:
        return {}

    ckpt_path = _resolve_existing_path(load_name)
    ckpt_dir = os.path.dirname(ckpt_path)
    run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "checkpoints" else ckpt_dir
    return _read_parameters_file(run_dir)


def _read_test_parameters(opt) -> dict:
    params = _read_parameters_file(opt.save_path)
    if params:
        return params
    return _read_checkpoint_run_parameters(opt)


def _resolve_existing_path(path_value: str) -> str:
    if not path_value:
        return path_value
    project_root = os.path.dirname(os.path.abspath(__file__))
    candidates = [path_value, os.path.normpath(path_value), os.path.abspath(path_value)]

    normalized = path_value.replace("\\", "/")
    marker = "/scanlens_v1/"
    if marker in normalized:
        rel = normalized.split(marker, 1)[1]
        candidates.append(os.path.join(project_root, *rel.split("/")))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return path_value


def _resolve_test_checkpoint(opt) -> str:
    if getattr(opt, "load_name", ""):
        return _resolve_existing_path(opt.load_name)
    return os.path.join(
        "log",
        "260521_1013",
        "stage_1",
        "checkpoints",
        "SLT_rmsfilter_on_epoch5000_bs512.pth",
    )


def _resolve_airgap_base_checkpoint(opt) -> str:
    params = _read_test_parameters(opt)
    base_ckpt = params.get("load_name", "")
    if base_ckpt:
        return _resolve_existing_path(base_ckpt)
    return _resolve_existing_path(AIRGAP_BASE_CKPT)


def _resolve_airgap_delta_scale(opt) -> float:
    params = _read_test_parameters(opt)
    try:
        return float(params.get("air_delta_scale_mm", AIRGAP_DELTA_SCALE_MM))
    except (TypeError, ValueError):
        return float(AIRGAP_DELTA_SCALE_MM)


def _uses_airgap_delta_residual(checkpoint_path: str) -> bool:
    """Use residual-air test path only for stage-2 unsupervised checkpoints."""
    if not USE_AIRGAP_DELTA_RESIDUAL:
        return False
    base = os.path.basename(checkpoint_path)
    return "AirGapUnsupervised" in base


def _forward_test_model(opt, model, base_model, X_sys, X_bgr_seq, X_type, X_seq_length):
    if getattr(opt, "use_airgap_delta_residual", False) and base_model is not None:
        CT_base, _ = base_model(X_sys, X_bgr_seq, X_type, X_seq_length, opt.epochs, hard=True)
        CT_data, mask_net = model(
            X_sys,
            X_bgr_seq,
            X_type,
            X_seq_length,
            opt.epochs,
            hard=True,
            air_base_ct=CT_base.detach(),
            air_delta_scale_mm=getattr(opt, "airgap_delta_scale_mm", AIRGAP_DELTA_SCALE_MM),
        )
        return CT_data, mask_net

    return model(X_sys, X_bgr_seq, X_type, X_seq_length, opt.epochs, hard=True)


def test(opt):
    # ================== 功能开关配置 ==================
    # 只需要改这里的 True / False，即可控制对应功能是否执行
    ENABLE_FEATURES = {
        "save_csv": True,               # 保存测试结果到 CSV（metrics_pred / loss_pred）
        "efl_result_filter": True,      # 后处理只保留 EFL 相对误差低于阈值的系统
        "result_summary": True,         # 使用 plot_result_summary.py 生成完整测试结果汇总
        "gt_match_report": True,        # 匹配预测结果到原始GT库，并生成Markdown报告
        "glass_matching": True,         # 玻璃匹配（依赖 metrics_pred.csv）
        "export_zmx_json": True,        # ZMX/JSON 导出（只导出报告中选中的 pred/GT 系统）
    }

    if not getattr(opt, "export_zmx_json", True):
        ENABLE_FEATURES["export_zmx_json"] = False
    if os.environ.get("SCANLENS_TEST_LIGHTWEIGHT", "0") == "1":
        ENABLE_FEATURES.update({
            "efl_result_filter": False,
            "result_summary": False,
            "gt_match_report": False,
            "glass_matching": False,
            "export_zmx_json": False,
        })

    print("=" * 80)
    print("TEST CONFIGURATION (ENABLE_FEATURES):")
    print("=" * 80)
    for k, v in ENABLE_FEATURES.items():
        print(f"  {k}: {'ENABLED' if v else 'DISABLED'}")
    print("=" * 80)
    
    # 显示样本筛选配置
    print("\n" + "=" * 80)
    print("样本筛选配置:")
    print("=" * 80)
    rms_status = "开启" if getattr(opt, "enable_rms_filter", True) else "关闭"
    efl_status = "开启" if getattr(opt, "enable_efl_filter", True) else "关闭"
    print(f"  RMS 筛选: {rms_status}")
    print(f"  EFL 筛选: {efl_status}")
    print("=" * 80)
    
    print()

    # 如果禁用导出，则强制关闭 opt.export_zmx_json，避免 save_csv() 内触发导出逻辑
    if not ENABLE_FEATURES.get("export_zmx_json", True):
        try:
            setattr(opt, "export_zmx_json", False)
        except Exception:
            pass

    # ================== 准备数据 ==================
    # Read the test CSV path at call time so long-running in-process callers
    # (e.g. the Streamlit web app) can switch test sets between runs without
    # being pinned to the value captured at dataset_norm import time.
    _test_csv = os.environ.get("SCANLENS_TEST_CSV")
    if _test_csv:
        X_val_sys, X_val_bgr, X_val_type = load_test_data(_test_csv)
    else:
        X_val_sys, X_val_bgr, X_val_type = load_test_data()
    seq_lengths_val = X_val_sys[:, 2].astype(int)
    X_val_sys = X_val_sys[:, :2]

    seed = int(getattr(opt, "seed", DEFAULT_SEED))
    val_dataset = ValDataset(X_val_sys, X_val_bgr, X_val_type, seq_lengths_val)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        worker_init_fn=seed_worker,
        generator=make_torch_generator(seed, offset=20),
    )

    # ================== 加载模型 ==================
    # Stage 1 checkpoints are tested directly. Stage 2 AirGapUnsupervised
    # checkpoints are tested with residual air-gap refinement.
    save_name = _resolve_test_checkpoint(opt)
    if not os.path.exists(save_name):
        raise FileNotFoundError(f"Checkpoint not found: {save_name}")
    print(f"[Test] Using checkpoint: {save_name}")
    model = create_transformer_val(opt, save_name).to(device)
    base_model = None
    opt.airgap_delta_scale_mm = _resolve_airgap_delta_scale(opt)
    opt.use_airgap_delta_residual = _uses_airgap_delta_residual(save_name)
    if opt.use_airgap_delta_residual:
        base_ckpt = _resolve_airgap_base_checkpoint(opt)
        if not os.path.exists(base_ckpt):
            raise FileNotFoundError(f"Base checkpoint not found for air-gap residual test: {base_ckpt}")
        print("[Test] Using stage-2 unsupervised air-gap residual mode.")
        print(f"[Test] Base stage-1 checkpoint: {base_ckpt}")
        print(f"[Test] Air-gap delta scale: {opt.airgap_delta_scale_mm}")
        base_model = create_transformer_val(opt, base_ckpt).to(device)
        base_model.eval()
    else:
        print("[Test] Using direct stage-1 checkpoint mode.")

    USL_loss = USL_Loss(opt).to(device)
    USL_loss.eval()
    model.eval()

    # ================== 收集容器 ==================
    X_sys_params = []
    X_bgr_list = []   # 收集导出用的 (N, max_seq*nWL) 特征
    X_ct = []

    loss_all = []
    spot_all = []
    dist_all = []
    tele_all = []
    ovlp_all = []
    rays_all = []
    loss_dist_all = []
    loss_tele_all = []
    efl_est_all = []
    efl_ideal_all = []
    loss_efl_all = []
    # 如需 chrom / EFL，将来可以从 metrics 里再加

    Total_loss = 0.0
    spot_total_loss = 0.0
    count = 0
    lens_batch = LensBatch(None, None, None)  # 只创建一次

    # ================== forward ==================
    for X_sys, X_bgr, X_type, X_seq_length in val_dataloader:
        X_sys = X_sys.to(device)
        X_bgr = X_bgr.to(device)
        X_seq_length = X_seq_length.to(device)
        X_type = X_type.to(device)

        B, n_fea = X_bgr.shape
        # (B, max_seq * nWL) -> (B, max_seq, nWL)
        X_bgr_seq = X_bgr.view(B, opt.max_seq_length, opt.nWL)

        # Transformer 预测 CT & mask (no grad; we never update network here)
        with torch.no_grad():
            CT_data, mask_net = _forward_test_model(
                opt, model, base_model, X_sys, X_bgr_seq, X_type, X_seq_length
            )

            # 反归一化到真实物理量
            X_sys_real = convert2real_dataSys(X_sys)
            X_bgr_flat = X_bgr_seq.view(B, n_fea)
            X_bgr_real = convert2real_dataBGR(X_bgr_flat)
            X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)

            # 注意：原代码是 ~mask，这里保持一致
            mask = ~mask_net

            # 赋值给 LensBatch
            lens_batch.X = X_sys_real
            lens_batch.N_bgr = X_bgr_real
            lens_batch.CT = CT_data

        # ================== USL_Loss 前向（导出 metrics 用 no_grad） ==================
        with torch.no_grad():
            # 新接口：当 save=1 时，返回 (loss, loss_spot, metrics)
            apply_test_hard_filter = bool(
                getattr(opt, "enable_rms_filter", True)
                or getattr(opt, "enable_efl_filter", True)
            )
            loss, loss_spot, metrics = USL_loss(
                lens_batch,
                opt.max_seq_length,
                X_seq_length,
                mask,
                opt.epochs,
                save=1,
                apply_hard_filter=apply_test_hard_filter,
            )

            # metrics 是一个 dict，按键取出
            X_datasel = metrics["X"]        # (nKeep, 2)
            N_datasel = metrics["N"]        # (nKeep, max_seq, nWL)
            CT_datasel = metrics["CT"]      # (nKeep, max_seq, output_size-1)

            # 展平成和原来一致的格式
            N_datasel = N_datasel.view(N_datasel.shape[0], opt.max_seq_length * opt.nWL)
            CT_datasel = CT_datasel.view(
                CT_datasel.shape[0], opt.max_seq_length * (opt.output_size - 1)
            )

            # 追加到列表
            X_sys_params.append(X_datasel)
            X_bgr_list.append(N_datasel)
            X_ct.append(CT_datasel)

            loss_all.append(metrics["composite"])
            spot_all.append(metrics["rms"])
            dist_all.append(metrics["dist"])
            tele_all.append(metrics["tele"])
            ovlp_all.append(metrics["loss_ovlp"])
            rays_all.append(metrics["loss_ray"])
            loss_dist_all.append(metrics["loss_dist"])
            loss_tele_all.append(metrics["loss_tele"])
            efl_est_all.append(metrics["EFL_est"])
            efl_ideal_all.append(metrics["EFL_ideal"])
            loss_efl_all.append(metrics["loss_EFL"])

            count += 1
            Total_loss += loss.item()
            spot_total_loss += loss_spot.item()

    # ================== 汇总所有 batch ==================
    X_sys_params = torch.cat(X_sys_params, dim=0)
    X_bgr_seq = torch.cat(X_bgr_list, dim=0)
    X_ct = torch.cat(X_ct, dim=0)

    loss_all = torch.cat(loss_all, dim=0).view(-1, 1)
    spot_all = torch.cat(spot_all, dim=0).view(-1, 1)
    dist_all = torch.cat(dist_all, dim=0).view(-1, 1)
    tele_all = torch.cat(tele_all, dim=0).view(-1, 1)
    ovlp_all = torch.cat(ovlp_all, dim=0).view(-1, 1)
    rays_all = torch.cat(rays_all, dim=0).view(-1, 1)
    loss_dist_all = torch.cat(loss_dist_all, dim=0).view(-1, 1)
    loss_tele_all = torch.cat(loss_tele_all, dim=0).view(-1, 1)
    efl_est_all = torch.cat(efl_est_all, dim=0).view(-1, 1)
    efl_ideal_all = torch.cat(efl_ideal_all, dim=0).view(-1, 1)
    loss_efl_all = torch.cat(loss_efl_all, dim=0).view(-1, 1)

    # ================== 写 CSV ==================
    if ENABLE_FEATURES.get("save_csv", True):
        save_csv(
            opt,
            X_sys_params,
            X_bgr_seq,
            X_ct,
            loss_all,
            spot_all,
            dist_all,
            tele_all,
            ovlp_all,
            rays_all,
            loss_dist_all,
            loss_tele_all,
            efl_est_all,
            efl_ideal_all,
        )
    else:
        print("[Info] save_csv disabled: skipping CSV saving.")

    # 计算常用路径（无论是否 save_csv，都可能基于已有历史 CSV 做分析）
    tag = _rms_filter_tag(opt)
    out_dir = _output_dir(opt)
    test_csv = os.path.join(out_dir, _with_tag("test_output_metrics_pred.csv", tag))
    test_loss_csv = os.path.join(out_dir, _with_tag("test_output_loss_pred.csv", tag))
    analysis_csv = test_csv
    analysis_loss_csv = test_loss_csv
    analysis_tag = tag

    # ================== EFL结果筛选 ==================
    if ENABLE_FEATURES.get("efl_result_filter", True):
        print("\n" + "="*60)
        print("Filtering results by EFL error...")
        print("="*60)

        if os.path.exists(test_csv):
            try:
                efl_threshold = float(getattr(opt, "test_efl_error_threshold", 0.1))
                analysis_csv, filtered_loss_csv, kept_rows, total_rows = _filter_csv_by_efl_error(
                    test_csv,
                    test_loss_csv,
                    threshold=efl_threshold,
                )
                analysis_loss_csv = filtered_loss_csv or test_loss_csv
                analysis_tag = f"{tag}_eflerror_lt_{_threshold_tag(efl_threshold)}"
                if kept_rows == 0:
                    print("[Warning] EFL filter kept 0 rows; downstream summary/export may be skipped or fail.")
            except Exception as e:
                print(f"[Warning] efl_result_filter failed: {e}")
                print("          Falling back to unfiltered test CSV.")
                analysis_csv = test_csv
                analysis_loss_csv = test_loss_csv
                analysis_tag = tag
        else:
            print(f"[Warning] Test CSV not found for EFL filtering: {test_csv}")

    # ================== 生成测试结果汇总 ==================
    if ENABLE_FEATURES.get("result_summary", True):
        print("\n" + "="*60)
        print("Generating result summary...")
        print("="*60)

        if os.path.exists(analysis_csv):
            try:
                summary_dir = os.path.join(out_dir, f"test_result_summary_{analysis_tag}")
                loss_csv = analysis_loss_csv if os.path.exists(analysis_loss_csv) else _infer_loss_csv_path(opt, out_dir)
                plot_result_summary(
                    loss_csv_path=loss_csv,
                    metrics_csv_path=analysis_csv,
                    output_dir=summary_dir,
                )
            except Exception as e:
                print(f"[Warning] result_summary failed: {e}")
        else:
            print(f"[Warning] Test CSV not found for result summary: {analysis_csv}")
            if not ENABLE_FEATURES.get("save_csv", True):
                print("         Hint: enable ENABLE_FEATURES['save_csv']=True to generate the CSV first.")
    else:
        print("[Info] result_summary disabled: skipping summary generation.")

    # ================== 匹配GT数据并生成Markdown报告 ==================
    gt_match_selected_rows_csv = ""
    if ENABLE_FEATURES.get("gt_match_report", True) and getattr(opt, "gt_match_report", True):
        print("\n" + "="*60)
        print("Generating GT match report...")
        print("="*60)

        if os.path.exists(analysis_csv):
            try:
                gt_report_path = os.path.join(out_dir, f"gt_match_report_{analysis_tag}.md")
                gt_match_selected_rows_csv = os.path.join(out_dir, f"gt_match_selected_rows_{analysis_tag}.csv")
                generate_gt_match_report(
                    pred_csv=analysis_csv,
                    out_md=gt_report_path,
                    surf10_csv=getattr(opt, "gt_surf10_csv", "data/scan_lens_dataset_surf10_reorder.csv"),
                    surf12_csv=getattr(opt, "gt_surf12_csv", "data/scan_lens_dataset_surf12_reorder.csv"),
                    max_rows=int(getattr(opt, "gt_match_max_rows", 0) or 0),
                    material_weight=float(getattr(opt, "gt_match_material_weight", 0.4)),
                    system_weight=float(getattr(opt, "gt_match_system_weight", 0.2)),
                    metric_weight=float(getattr(opt, "gt_match_metric_weight", 0.2)),
                    gt_radius_to_curvature=bool(getattr(opt, "gt_radius_to_curvature", True)),
                    top_k_matches=int(getattr(opt, "gt_match_top_k", 10) or 0),
                    include_best_rms_per_group=bool(getattr(opt, "include_best_rms_per_group", True)),
                    selected_rows_csv=gt_match_selected_rows_csv,
                    rms_floor_tolerance=float(getattr(opt, "gt_match_rms_floor_tolerance", 0.1)),
                    rms_floor_penalty_weight=float(getattr(opt, "gt_match_rms_floor_penalty_weight", 2.0)),
                )
            except Exception as e:
                print(f"[Warning] gt_match_report failed: {e}")
                gt_match_selected_rows_csv = ""
        else:
            print(f"[Warning] Test CSV not found for GT match report: {analysis_csv}")
            if not ENABLE_FEATURES.get("save_csv", True):
                print("         Hint: enable ENABLE_FEATURES['save_csv']=True to generate the CSV first.")
    else:
        print("[Info] gt_match_report disabled: skipping GT match report.")

    # ================== 玻璃匹配 ==================
    glass_matching_csv = ""
    if ENABLE_FEATURES.get("glass_matching", True):
        print("\n" + "="*60)
        print("Performing glass matching...")
        print("="*60)

        if os.path.exists(analysis_csv):
            try:
                # 确保材料库存在
                _ = ensure_material_library_exists()
                glass_matching_csv = process_test_output_csv(analysis_csv)
                print(f"Glass matching results saved to: {glass_matching_csv}")
                # 保存玻璃匹配结果路径，供 ZMX 导出使用
                setattr(opt, "export_glass_matching_csv", glass_matching_csv)
            except Exception as e:
                print(f"[Warning] glass_matching failed: {e}")
        else:
            print(f"[Warning] Test CSV not found for glass matching: {analysis_csv}")
            if not ENABLE_FEATURES.get("save_csv", True):
                print("         Hint: enable ENABLE_FEATURES['save_csv']=True to generate the CSV first.")
    else:
        print("[Info] glass_matching disabled: skipping glass matching.")

    # ================== ZMX/JSON 导出（最后一步）==================
    if ENABLE_FEATURES.get("export_zmx_json", True):
        print("\n" + "="*60)
        print("Exporting ZMX/JSON...")
        print("="*60)

        if os.path.exists(analysis_csv):
            try:
                export_csv = getattr(opt, "export_csv", "") or analysis_csv
                export_glass_matching_csv = glass_matching_csv or getattr(opt, "export_glass_matching_csv", "")
                selected_export_rows = _read_selected_export_rows(gt_match_selected_rows_csv)
                selected_gt_rows = _read_selected_gt_export_rows(gt_match_selected_rows_csv)
                if selected_export_rows is None and int(getattr(opt, "export_row", -1)) < 0:
                    print("[Warning] No GT report selected rows found; skipping batch ZMX/JSON export.")
                    return Total_loss / count, spot_total_loss / count
                if (not getattr(opt, "export_out_dir", "") or getattr(opt, "export_out_dir", "") == "exports"):
                    setattr(opt, "export_out_dir", out_dir)
                manifest_path = _export_zmx_json_rows(
                    opt,
                    export_csv=export_csv,
                    glass_matching_csv=export_glass_matching_csv,
                    tag=analysis_tag,
                    row_indices=selected_export_rows,
                )
                print(f"[Export] Manifest: {manifest_path}")
                gt_manifest_path = _export_gt_zmx_json_rows(
                    opt,
                    selected_gt_rows=selected_gt_rows,
                    tag=analysis_tag,
                )
                if gt_manifest_path:
                    print(f"[Export] GT manifest: {gt_manifest_path}")
            except Exception as e:
                print(f"[Warning] export_zmx_json failed: {e}")
        else:
            print(f"[Warning] Test CSV not found for ZMX/JSON export: {analysis_csv}")
            if not ENABLE_FEATURES.get("save_csv", True):
                print("         Hint: enable ENABLE_FEATURES['save_csv']=True to generate the CSV first.")
    else:
        print("[Info] export_zmx_json disabled: skipping ZMX/JSON export.")

    return Total_loss / count, spot_total_loss / count


def save_csv(
    opt,
    X_sys,
    X_bgr,
    X_ct,
    loss_all,
    spot_all,
    dist_all,
    tele_all,
    ovlp_all,
    rays_all,
    loss_dist_all,
    loss_tele_all,
    efl_est,
    efl_ideal,
):
    """
    保存两份 CSV：
      1) test_output_metrics_pred.csv  : 带 dist/tele 等指标，最后两列是 EFL_est 和 EFL_ideal
      2) test_output_loss_pred.csv    : 带 loss_dist / loss_tele 等 loss 项（不包含 EFL_est / EFL_ideal）
    """
    x_data = torch.cat(
        (X_sys, X_bgr, X_ct, loss_all, spot_all, dist_all, tele_all, ovlp_all, rays_all, efl_est, efl_ideal),
        dim=1,
    )
    x1_data = torch.cat(
        (X_sys, X_bgr, X_ct, loss_all, spot_all, loss_dist_all, loss_tele_all, ovlp_all, rays_all),
        dim=1,
    )

    test_output = pd.DataFrame(x_data.cpu().numpy())
    test_output1 = pd.DataFrame(x1_data.cpu().numpy())

    tag = _rms_filter_tag(opt)
    out_dir = _output_dir(opt)
    save_name = os.path.join(out_dir, _with_tag("test_output_metrics_pred.csv", tag))
    save_name1 = os.path.join(out_dir, _with_tag("test_output_loss_pred.csv", tag))

    test_output.to_csv(save_name, header=None, index=False, encoding="utf-8")
    test_output1.to_csv(save_name1, header=None, index=False, encoding="utf-8")


def save_efl_csv(opt, efl_est, efl_ideal, loss_efl):
    """
    保存 EFL 相关数据到单独的文件
    efl_est   : 从光线追迹估计的 EFL
    efl_ideal : 理想的 EFL (F# * EPD)
    loss_efl  : EFL 损失
    """
    efl_data = torch.cat(
        (efl_est, efl_ideal, loss_efl),
        dim=1,
    )
    
    efl_df = pd.DataFrame(efl_data.cpu().numpy())
    efl_df.columns = ['EFL_est', 'EFL_ideal', 'loss_EFL']
    
    tag = _rms_filter_tag(opt)
    out_dir = _output_dir(opt)
    save_name = os.path.join(out_dir, _with_tag("test_output_efl.csv", tag))
    efl_df.to_csv(save_name, header=True, index=False, encoding="utf-8")
    print(f"EFL data saved to: {save_name}")


if __name__ == "__main__":
    opt = set_parser()
    set_random_seed(opt.seed)
    os.makedirs(opt.save_path, exist_ok=True)
    test_loss, test_spot_loss = test(opt)

    print("test_loss: %.9f" % test_loss)       # output test loss
    print("test_spot: %.9f" % test_spot_loss)  # output test spot loss
