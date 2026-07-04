'''
Description: Scan lens下在不同代码之间复用的一些功能函数和所有数据输入的整体接口。
Debug Date:
Status     : ok!
TODO：
'''

import os
import random
import torch
from models import TransformerClass_Model
import argparse
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DEFAULT_SEED = 2026


def set_random_seed(seed=DEFAULT_SEED, deterministic=True):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "use_deterministic_algorithms"):
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                pass

    print(f"[Seed] random seed fixed to {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_torch_generator(seed=DEFAULT_SEED, offset=0):
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator

# ----------------------------------------
#                 Hyperparameters
# ----------------------------------------
def set_parser():
    # ----------------------------------------
    #        Initialize the parameters
    # ----------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model_path', type=str, default='checkpoints', help='saving model')
    parser.add_argument('--save_path', type=str, default='./log/260521_1013', help='saving log ')
    parser.add_argument('--save_by_epoch', type=int, default=1000, help='save checkpoints by epoch')
    parser.add_argument('--load_name', type=str, default='', help='load the pre-trained model')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED, help='random seed')
    # ./log/0729USL/checkpoints/BP_epoch20000_bs2048.pth

    # Training parameters
    parser.add_argument('--epochs', type=int, default=5000, help='size of the batches')
    parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
    parser.add_argument('--input_size', type=int, default=256, help='input features for RNN')
    parser.add_argument('--hidden_size', type=int, default=512, help='output features for RNN')
    parser.add_argument('--output_size', type=int, default=3, help='final output feature of RNN')
    parser.add_argument('--num_layers', type=int, default=6, help='layers of Transformer')
    parser.add_argument('--num_heads', type=int, default=8, help='heads of Transformer')
    parser.add_argument('--sys_dim', type=int, default=2, help='input features of systems in the embedding layer')
    parser.add_argument('--lr', type=float, default=1e-4, help='Adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.9, help='Adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='Adam: decay of second order momentum of gradient')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay for optimizer')
    parser.add_argument('--lr_decrease_epoch', type=int, default=500,
                        help='lr decrease at certain epoch and its multiple')
    parser.add_argument('--lr_decrease_factor', type=float, default=0.5, help='lr decrease factor')
    parser.add_argument('--train_stage', type=float, default=1, help='training stage: 1 (base only), 1.5 (joint), 2 (fine-tune)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of cpu threads to use during batch generation')
    parser.add_argument('--gammna', type=float, default=0.5)
    parser.add_argument('--lr_steps', type=list, default=[(x + 1) * 1000 for x in range(5000 // 1000)])


    # Lens parameters
    parser.add_argument('--nWL', type=int, default=3, help='input features of 折射率 in the embedding layer')
    parser.add_argument('--max_seq_length', type=int, default=11, help='max sequence length')
    parser.add_argument('--seq1', type=int, default=9, help='10面系统的序列数')
    parser.add_argument('--seq2', type=int, default=11, help='12面系统的序列数')
    # Pupil 采样参数
    parser.add_argument("--nRayDensity", type=int, default=11, help="pupil 采样网格边长，单位圆内有效")
    parser.add_argument("--nField", type=int, default=3, help="视场采样个数，thetas 等分 0..HFOV")
    parser.add_argument("--EPD", type=float, default=4.0, help="入瞳直径,单位mm")
    # loss 权重
    parser.add_argument("--w_rms", type=float, default=1.0)
    parser.add_argument("--w_distortion", type=float, default=10.0)
    parser.add_argument("--w_tele", type=float, default=1.0)
    parser.add_argument(
        "--distortion_mode",
        choices=["zemax_ftan", "target_height"],
        default="zemax_ftan",
        help="Distortion metric used inside USL_Loss.",
    )
    parser.add_argument(
        "--distortion_ref_angle_deg",
        type=float,
        default=0.01,
        help="Small chief-ray field angle used to build the Zemax-like F-Tan(theta) distortion reference.",
    )

    # add loss_spot Pars
    # === RMS 筛选开关（默认保持原行为：开启）===
    # 用法：
    # - 默认开启：不加任何参数
    # - 显式开启：--enable_rms_filter
    # - 显式关闭：--disable_rms_filter
    rms_filter_group = parser.add_mutually_exclusive_group()
    rms_filter_group.add_argument(
        "--enable_rms_filter",
        dest="enable_rms_filter",
        action="store_true",
        help="Enable RMS filtering in USL_Loss (default: enabled)",
    )
    rms_filter_group.add_argument(
        "--disable_rms_filter",
        dest="enable_rms_filter",
        action="store_false",
        help="Disable RMS filtering in USL_Loss",
    )
    parser.set_defaults(enable_rms_filter=True)
    parser.add_argument('--max_rmsSpotR', default=0.04, type=float, help='max rms spot Radius')  # 0.04 or 0.03
    parser.add_argument('--spot_weight', default=0, type=float, help='weight on loss_spot')  # default 0.1
    parser.add_argument(
        "--test_efl_error_threshold",
        default=0.1,
        type=float,
        help="Post-test filter threshold for relative EFL error before summary/glass matching/ZMX export.",
    )

    # === EFL 筛选开关（默认保持原行为：开启）===
    # 用法：
    # - 默认开启：不加任何参数
    # - 显式开启：--enable_efl_filter
    # - 显式关闭：--disable_efl_filter
    efl_filter_group = parser.add_mutually_exclusive_group()
    efl_filter_group.add_argument(
        "--enable_efl_filter",
        dest="enable_efl_filter",
        action="store_true",
        help="Enable EFL filtering in USL_Loss (default: enabled)",
    )
    efl_filter_group.add_argument(
        "--disable_efl_filter",
        dest="enable_efl_filter",
        action="store_false",
        help="Disable EFL filtering in USL_Loss",
    )
    parser.set_defaults(enable_efl_filter=True)
    efl_fo_group = parser.add_mutually_exclusive_group()
    efl_fo_group.add_argument(
        "--enable_efl_first_order_control",
        dest="enable_efl_first_order_control",
        action="store_true",
        help="Enable first-order (ABCD) EFL consistency control in USL_Loss (default: enabled).",
    )
    efl_fo_group.add_argument(
        "--disable_efl_first_order_control",
        dest="enable_efl_first_order_control",
        action="store_false",
        help="Disable first-order (ABCD) EFL consistency control and use trace-only EFL loss.",
    )
    parser.set_defaults(enable_efl_first_order_control=True)
    parser.add_argument(
        "--efl_first_order_weight",
        type=float,
        default=0.5,
        help="Weight for first-order EFL control loss in total loss_EFL.",
    )
    parser.add_argument(
        "--efl_first_order_tolerance",
        type=float,
        default=0.1,
        help="Tolerance for |EFL_trace-EFL_first_order|/|EFL_first_order| before penalty.",
    )

    # -------------------------
    # Optional: export design rows to ZMX/JSON for Zemax verification
    # -------------------------
    export_group = parser.add_mutually_exclusive_group()
    export_group.add_argument(
        "--export_zmx_json",
        dest="export_zmx_json",
        action="store_true",
        help="Enable exporting GT-report-selected CSV rows to ZMX/JSON after test() (default: enabled)",
    )
    export_group.add_argument(
        "--no_export_zmx_json",
        dest="export_zmx_json",
        action="store_false",
        help="Disable exporting ZMX/JSON after test()",
    )
    parser.set_defaults(export_zmx_json=True)

    parser.add_argument(
        "--export_row",
        type=int,
        default=-1,
        help="0-based row index in export_csv. Use -1 to export GT-report-selected rows (default)",
    )
    parser.add_argument(
        "--export_max_rows",
        type=int,
        default=0,
        help="When export_row=-1, limit selected-row export to the first N rows. Use 0 for all selected rows.",
    )
    parser.add_argument(
        "--export_csv",
        type=str,
        default="",
        help="CSV path to export from. Default: the test_output_metrics_pred_*.csv just generated",
    )
    parser.add_argument("--export_format", choices=["pred", "orig"], default="pred", help="CSV format kind for exporter")
    parser.add_argument("--export_n_surf", type=int, default=0, help="Sequence length to export (9 or 11). Use 0 to auto-infer from CSV (recommended)")
    parser.add_argument(
        "--export_offset_ct",
        type=int,
        default=-1,
        help="pred-only: CT offset. Use -1 to auto (9->6, 11->0)",
    )
    parser.add_argument(
        "--export_epd",
        type=float,
        default=4.5,
        help="ENPD/EPD used in ZMX (set explicitly for EFL validation)",
    )
    parser.add_argument(
        "--export_out_dir",
        type=str,
        default="",
        help="Output directory for exported .zmx/.json. Default: current test save_path.",
    )
    parser.add_argument(
        "--export_glass_matching_csv",
        type=str,
        default="log/251202/glass_matching_results_rmsfilter_on.csv",
        help="Optional: glass matching CSV (Layer_k_Name columns) to inject glass names",
    )
    gt_match_group = parser.add_mutually_exclusive_group()
    gt_match_group.add_argument(
        "--enable_gt_match_report",
        dest="gt_match_report",
        action="store_true",
        help="Generate a Markdown report matching test predictions to GT CSV rows.",
    )
    gt_match_group.add_argument(
        "--no_gt_match_report",
        dest="gt_match_report",
        action="store_false",
        help="Disable GT match Markdown report generation after testing.",
    )
    parser.set_defaults(gt_match_report=True)
    parser.add_argument(
        "--gt_surf10_csv",
        type=str,
        default="data/scan_lens_dataset_surf10_reorder.csv",
        help="GT CSV for 10-surface systems.",
    )
    parser.add_argument(
        "--gt_surf12_csv",
        type=str,
        default="data/scan_lens_dataset_surf12_reorder.csv",
        help="GT CSV for 12-surface systems.",
    )
    parser.add_argument(
        "--gt_match_max_rows",
        type=int,
        default=0,
        help="Maximum prediction rows to include in GT match report. Use 0 for all rows.",
    )
    parser.add_argument(
        "--gt_match_top_k",
        type=int,
        default=10,
        help="Keep only the top-K GT matches ranked by material + structure distance.",
    )
    parser.add_argument(
        "--gt_match_material_weight",
        type=float,
        default=0.4,
        help="Material-vs-structure weight for GT matching score.",
    )
    parser.add_argument(
        "--gt_match_metric_weight",
        type=float,
        default=0.2,
        help="Metrics weight for GT matching score. Structure uses the remaining weight.",
    )
    parser.add_argument(
        "--gt_match_system_weight",
        type=float,
        default=0.2,
        help="F-number/HFOV closeness weight for GT matching score.",
    )
    parser.add_argument(
        "--gt_match_rms_floor_tolerance",
        type=float,
        default=0.1,
        help="Allow matched GT RMS to be this fraction below pred RMS without penalty.",
    )
    parser.add_argument(
        "--gt_match_rms_floor_penalty_weight",
        type=float,
        default=2.0,
        help="Penalty weight when matched GT RMS is much smaller than pred RMS.",
    )
    best_rms_group = parser.add_mutually_exclusive_group()
    best_rms_group.add_argument(
        "--include_best_rms_per_group",
        dest="include_best_rms_per_group",
        action="store_true",
        help="Append one minimum-RMS prediction per (F#, HFOV) group to the GT match report.",
    )
    best_rms_group.add_argument(
        "--no_best_rms_per_group",
        dest="include_best_rms_per_group",
        action="store_false",
        help="Do not append minimum-RMS prediction rows per (F#, HFOV) group.",
    )
    parser.set_defaults(include_best_rms_per_group=True)
    gt_radius_group = parser.add_mutually_exclusive_group()
    gt_radius_group.add_argument(
        "--gt_radius_to_curvature",
        dest="gt_radius_to_curvature",
        action="store_true",
        help="Convert GT radius columns to curvature before GT matching (default).",
    )
    gt_radius_group.add_argument(
        "--no_gt_radius_to_curvature",
        dest="gt_radius_to_curvature",
        action="store_false",
        help="Use GT CT columns as already-curvature values.",
    )
    parser.set_defaults(gt_radius_to_curvature=True)
    opt = parser.parse_args()
    return opt

# ----------------------------------------
#                 Network
# ----------------------------------------

def create_transformer_model(opt):
    # Initialize the network
    group_mask, group_c, group_t, uniq_keys = get_OTS_CT()
    model = TransformerClass_Model.LensTransformer(opt, group_mask, group_c, group_t, uniq_keys)
    if opt.load_name:
        trained_dict = torch.load(opt.load_name, map_location="cpu", weights_only=False)
        load_dict(model, trained_dict)
        print('Generator is loaded!')
    else:
        TransformerClass_Model.weights_init(model)
        print('Generator is created!')
    return model

def create_transformer_val(opt, load_name):
    # Initialize the network
    group_mask, group_c, group_t, uniq_keys = get_OTS_CT()
    model = TransformerClass_Model.LensTransformer(opt,  group_mask, group_c, group_t, uniq_keys)
    trained_dict = torch.load(load_name, map_location="cpu", weights_only=False)
    load_dict(model, trained_dict)
    print('Generator is loaded!')
    return model

def load_dict(process_net, pretrained_net):
    # Get the dict from pre-trained network
    pretrained_dict = pretrained_net
    # Get the dict from processing network
    process_dict = process_net.state_dict()
    # Delete the extra keys of pretrained_dict that do not belong to process_dict
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in process_dict}
    # Update process_dict using pretrained_dict
    process_dict.update(pretrained_dict)
    # Load the updated dict to processing network
    process_net.load_state_dict(process_dict)
    return process_net


# ----------------------------------------
#             PATH processing
# ----------------------------------------
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# ----------------------------------------
#            RECORD processing
# ----------------------------------------

# def save_loss_path(opt):
#     loss_csv = open(os.path.join(opt.save_path, 'log_loss.csv'), 'a+')
#     return loss_csv
#
# def save_time_path(opt):
#     time_csv = open(os.path.join(opt.save_path, 'log_time.csv'), 'a+')
#     return time_csv

#Record loss
def record_loss(
    loss_csv,
    epoch,
    train_loss,
    val_loss,
    train_spot_loss,
    val_spot_loss,
    train_mse_loss=None,
    val_mse_loss=None,
    val_filtered_loss=None,
    val_filtered_spot_loss=None,
    val_pass_rate=None,
    val_kept=None,
    val_total=None,
):
    """ Record many results.
    
    Args:
        loss_csv: CSV file handle
        epoch: epoch number
        train_loss: training total loss
        val_loss: validation total loss
        train_spot_loss: training spot loss
        val_spot_loss: validation spot loss
        train_mse_loss: training MSE loss (optional, for air gap supervision)
        val_mse_loss: validation MSE loss (optional, for air gap supervision)
    """
    if val_filtered_loss is not None:
        loss_csv.write('{},{},{},{},{},{},{},{},{},{}\n'.format(
            epoch,
            train_loss,
            val_loss,
            val_filtered_loss,
            train_spot_loss,
            val_spot_loss,
            val_filtered_spot_loss,
            val_pass_rate,
            val_kept,
            val_total,
        ))
    elif train_mse_loss is not None and val_mse_loss is not None:
        # 包含MSE loss的格式
        loss_csv.write('{},{},{},{},{},{},{}\n'.format(
            epoch, train_loss, val_loss, train_spot_loss, val_spot_loss, train_mse_loss, val_mse_loss))
    else:
        # 原始格式（向后兼容）
        loss_csv.write('{},{},{},{},{}\n'.format(epoch, train_loss, val_loss, train_spot_loss, val_spot_loss))
    loss_csv.flush()
    # loss_csv.close

#Record time
def record_time(time_csv, time):
    time_csv.write('{} \n'.format(time))
    time_csv.flush()
    # time_csv.close

#Record parameters
def infer_loss_metadata(opt):
    keys = (
        "loss_function",
        "loss_formula",
        "usl_loss_variant",
        "distortion_mode",
        "distortion_ref_angle_deg",
        "efl_control_mode",
        "efl_first_order_weight",
        "efl_first_order_tolerance",
        "loss_filtering",
        "metric_source",
    )
    return {key: getattr(opt, key) for key in keys if hasattr(opt, key)}


def record_parameters(opt):
    argsDict = opt.__dict__
    with open(os.path.join(opt.save_path, 'parameters.txt'), 'w') as f:
         f.writelines('------------------ start ------------------' + '\n')
         for eachArg, value in argsDict.items():
              f.writelines(eachArg + ' : ' + str(value) + '\n')
         f.writelines('------------------- end -------------------')


# 取OTS中，某一种玻璃下的曲率与厚度
def get_OTS_CT():
    ri_atol = 1e-6
    Material_C_data = np.loadtxt('./glass/Material_C_T_Data.csv',delimiter=',', dtype=float)
    material_array = torch.as_tensor(Material_C_data,dtype=torch.float32, device =device)
    RIs = material_array[:, :3]

    R_pairs = material_array[:, 3:5]  # [N,2] 半径，0=∞

    # 统一 -0.0 → 0.0
    R_pairs = torch.where(torch.isclose(R_pairs, torch.zeros_like(R_pairs), atol=1e-12),
                          torch.zeros_like(R_pairs), R_pairs)

    # 半径 -> 曲率 c=1/R，平面保持 0
    c_pairs = torch.where(R_pairs != 0, 1.0 / R_pairs, torch.zeros_like(R_pairs))  # [N,2]
    # 厚度
    thick = material_array[:, 5:6]

    # === 按 RI 分组（量化成整数key，避免浮点微差） ===
    # 每个唯一 key 表示一种玻璃；同组内可有多个曲率候选
    keys = torch.round(RIs / ri_atol).to(torch.long)  # [N,3]
    uniq_keys, inv = torch.unique(keys, dim=0, return_inverse=True)  # inv: N -> G
    G = uniq_keys.size(0)
    counts = torch.bincount(inv, minlength=G)
    maxK = int(counts.max().item())

    group_RIs = torch.zeros(G, 3, device=device)
    group_c = torch.zeros(G, maxK, 2, device=device)
    group_t = torch.zeros(G, maxK, 1, device=device)
    group_mask = torch.zeros(G, maxK, dtype=torch.bool, device=device)

    # 预处理阶段允许 for 循环；运行期全部并行索引
    for g in range(G):
        idx = (inv == g).nonzero(as_tuple=True)[0]
        k = idx.numel()
        group_RIs[g] = RIs[idx[0]]
        group_c[g, :k] = c_pairs[idx]
        group_t[g, :k] = thick[idx]
        group_mask[g, :k] = True
    return group_mask,group_c,group_t, uniq_keys

# ----------------------------------------
#            Lens Data & Ray Config 封装成一个Class，减少传参个数！
# ----------------------------------------
class LensBatch:
    """
    封装一批镜头参数：
      X      : (B, 2)       镜头级特征 [FN, HFOV]
      N_bgr  : (B, L, 3)    每面折射率 [n_b, n_g, n_r]
      CT     : (B, L, 2)    每面几何参数 [curv, thick]
    """
    __slots__ = ("X", "N_bgr", "CT")  # 禁止动态增加属性
    def __init__(self, X, N_bgr, CT):
        self.X = X          # torch.Tensor
        self.N_bgr = N_bgr  # torch.Tensor
        self.CT = CT        # torch.Tensor
