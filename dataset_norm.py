'''
Description: train下面的数据处理函数，主要针对数据集的归一化和反归一化，
Debug Date : 2025-12-1，新的分类模型，只需要输入做归一化。输出不需要处理。
Status: ok!
TODO：清空部分冗余代码，命名清晰。统一代码逻辑！

'''


import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold

_ORIGIN_DATA_CACHE = None

ORIGIN_DATA_PATH = os.environ.get("SCANLENS_ORIGIN_CSV", "./data/surf10_12_ul_1104.csv")
TRAIN_DATA_PATH = os.environ.get("SCANLENS_TRAIN_CSV", "./data/scan_lens_train_ul_20260512.csv")
VAL_DATA_PATH = os.environ.get("SCANLENS_VAL_CSV", "./data/scan_lens_val_ul_20260512.csv")
TEST_DATA_PATH = os.environ.get("SCANLENS_TEST_CSV", "./data/scan_lens_test_ul_20260512.csv")


def load_origin_data():
    global _ORIGIN_DATA_CACHE
    if _ORIGIN_DATA_CACHE is None:
        _ORIGIN_DATA_CACHE = np.loadtxt(ORIGIN_DATA_PATH, delimiter=',', dtype=float)
    return _ORIGIN_DATA_CACHE

def _load_split_data(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    X_sys = data[:, :3]
    X_bgr = data[:, 3:3*11+3]
    X_type = data[:, 3*11+3:]
    X_sys = normalize_dataSys(X_sys)
    X_bgr = normalize_dataBGR(X_bgr)
    return X_sys, X_bgr, X_type


def load_train_data(csv_path=TRAIN_DATA_PATH):
    return _load_split_data(csv_path)


def load_validate_data(csv_path=VAL_DATA_PATH):
    return _load_split_data(csv_path)


def load_test_data(csv_path=TEST_DATA_PATH):
    return _load_split_data(csv_path)

def normalize_dataSys(data, LB=0, UB=1):
    """" Normalize all the data to be within range [LB,UB]
    Args:
        data - input M Row x N Col in tensor,  M is the sample Num
    Return:
        data_norm - normalized data withint desired range.
    """
    origin_data = load_origin_data()  # TODO:应在全部数据下归一化
    # origin_data = torch.tensor(origin_data[:,:11], dtype=torch.float32).cuda()
    origin_data = origin_data[:, :2]
    data_min= np.min(origin_data ,axis=0)
    data_max= np.max(origin_data, axis=0)
    denom = (data_max - data_min).reshape(1,-1)
    denom[denom==0] = 1
    data_norm = data.copy()
    data_norm[:,:2] = (UB-LB) * (data[:,:2] - data_min.reshape(1,-1))/(denom) + LB

    return data_norm

def normalize_dataBGR(data,LB=0, UB=1):
    """" Normalize all the data to be within range [LB,UB]
    Args:
        data - input M Row x N Col in tensor,  M is the sample Num
    Return:
        data_norm - normalized data withint desired range.
    """
    origin_data = load_origin_data()  # TODO:应在全部数据下归一化
    # origin_data = torch.tensor(origin_data[:,:11], dtype=torch.float32).cuda()
    origin_data = origin_data[:, 3:3*11+3]
    data_min= np.min(origin_data, axis=0)
    data_max= np.max(origin_data, axis=0)
    denom = (data_max - data_min).reshape(1,-1)
    denom[denom==0] = 1
    data_norm = (UB-LB) * (data - data_min.reshape(1,-1))/(denom) + LB
    # 处理特殊列
    is_all_zero = (data_min == 0) & (data_max == 0)
    is_all_one = (data_min == 1) & (data_max == 1)

    data_norm[:, is_all_zero] = -1
    data_norm[:, is_all_one] =0

    return data_norm

def convert2real_dataSys(X_norm,  LB=0, UB=1):
    """" Convert X_norm within [LB,UB] back to X_real
    Args:
        X_norm - input M Row x N Col in tensor,  M is the sample Num
    Return:
        X_real - real data of X, can be used for Zemax or Loss calculations.
    """
    origin_data = load_origin_data()  # TODO:应在全部数据下归一化
    origin_data = torch.as_tensor(origin_data, dtype=torch.float32, device=X_norm.device)
    origin_data = origin_data[:, :2]
    X_real_min, index = torch.min(origin_data, dim=0)
    X_real_max, index = torch.max(origin_data, dim=0)

    X_real = (X_norm - LB) * (X_real_max - X_real_min) / (UB-LB) + X_real_min

    return X_real

def convert2real_dataBGR(X_norm, LB=0, UB=1):
    """" Convert X_norm within [LB,UB] back to X_real
    Args:
        X_norm - input M Row x N Col in tensor,  M is the sample Num
    Return:
        X_real - real data of X, can be used for Zemax or Loss calculations.
    """
    origin_data = load_origin_data()  # TODO:应在全部数据下归一化
    origin_data = torch.as_tensor(origin_data, dtype=torch.float32, device=X_norm.device)
    origin_data = origin_data[:, 3:3*11+3]
    X_real_min, index = torch.min(origin_data, dim=0)
    X_real_max, index = torch.max(origin_data, dim=0)
    # 全常数列掩码
    const_mask = torch.isclose(X_real_min, X_real_max)
    X_real = (X_norm - LB) * (X_real_max - X_real_min) / (UB-LB) + X_real_min
    # # 常数列直接赋值
    X_real[:, const_mask] = X_real_min[const_mask]
    return X_real

# def _softplus(x, beta=8):
#     soft_sigmoid = torch.log(1 + torch.exp(beta * x)) / beta
#     return soft_sigmoid
#
# def _convert2real_t(t_pred, t_min, t_range, beta=0.1):
#     t_real = t_min + _softplus(t_pred - t_min, beta) - _softplus(t_pred - (t_range + t_min), beta)
#     return t_real
#
# def convert2real_dataY(opt,Y_norm,seq_lens,LB=-1, UB=1):
#     """" Convert Y_norm within [LB,UB] back to Y_real
#     Args:
#         Y_norm - input M Row x N Col in tensor,  M is the sample Num
#     Return:
#         Y_real - real data of Y, can be used for Zemax or Loss calculations.
#     """
#
#     device, dtype = Y_norm.device, Y_norm.dtype
#     B,surf_lens,n_fea = Y_norm.shape
#     # Y_real_max = torch.tensor([0, 43.9, 0.05, 9, 0.05, 12.8, 0.05, 10.05, 0.05, 8, 0.05, 8, 0.05, 19.5, 0.05, 11.9, 0.05, 9, 0.05,9,0.05, 57,0,0], dtype=torch.float32).cuda()
#     # Y_real_min = torch.tensor([0,15.5, -0.05, 2, -0.05, 0.24, -0.05, 2, -0.05, 0.1, -0.05, 2, -0.05, 0.24, -0.05, 2, -0.05, 0.24, -0.05,8,-0.05,49,0,0], dtype=torch.float32).cuda()
#     Y_real_max_9 = torch.tensor(
#         [[0, 41], [0.1, 13], [0.1, 13], [0.1, 13], [0.1, 7], [0.1, 13], [0.1, 20], [0.1, 13], [0.1, 72]],
#         dtype=torch.float32).cuda()
#     Y_real_min_9 = torch.tensor(
#         [[0, 15], [-0.1, 1], [-0.1, 0.2], [-0.1, 1], [-0.1, 0.1], [-0.1, 1], [-0.1, 0.2], [-0.1, 1],
#          [-0.1, 19]], dtype=torch.float32).cuda()
#     Y_real_max_11 = torch.tensor(
#         [[0, 44], [0.1, 13], [0.1, 8], [0.1, 13], [0.1, 9], [0.1, 13], [0.1, 5], [0.1, 13], [0.1, 11], [0.1, 13],
#          [0.1, 57]], dtype=torch.float32).cuda()
#     Y_real_min_11 = torch.tensor(
#         [[0, 39], [-0.1, 1], [-0.1, 0.2], [-0.1, 1], [-0.1, 0.2], [-0.1, 1], [-0.1, 2], [-0.1, 1], [-0.1, 0.2],
#          [-0.1, 1], [-0.1, 49]], dtype=torch.float32).cuda()
#     # Y_real_max_9 = torch.tensor([
#     #     [0.00, 40.0],  # 面1：物距
#     #     [0.10, 8.0],  # 面2：第一透镜厚度
#     #     [0.10, 12.0],  # 面3：第一空气间距
#     #     [0.10, 6.0],  # 面4：第二透镜厚度
#     #     [0.10, 8.0],  # 面5：第二空气间距
#     #     [0.10, 10.0],  # 面6：第三透镜厚度
#     #     [0.10, 20.0],  # 面7：第三空气间距 / 扫描模块距离
#     #     [0.10, 12.0],  # 面8：第四透镜厚度
#     #     [0.10, 80.0],  # 面9：像距
#     # ], dtype=torch.float32).cuda()
#     #
#     # Y_real_min_9 = torch.tensor([
#     #     [0.00, 10.0],  # 面1：物距下限
#     #     [-0.10, 2.0],  # 面2：透镜厚度下限
#     #     [-0.10, 0.5],  # 面3：空气间距下限
#     #     [-0.10, 2.0],  # 面4：透镜厚度下限
#     #     [-0.10, 0.5],  # 面5：空气间距下限
#     #     [-0.10, 2.0],  # 面6：透镜厚度下限
#     #     [-0.10, 1.0],  # 面7：空气间距下限
#     #     [-0.10, 2.0],  # 面8：透镜厚度下限
#     #     [-0.10, 40.0],  # 面9：像距下限
#     # ], dtype=torch.float32).cuda()
#     #
#     # Y_real_max_11 = torch.tensor([
#     #     [0.00, 45.0],  # 面1：物距
#     #     [0.10, 6.0],  # 面2：第一透镜厚度
#     #     [0.10, 8.0],  # 面3：第一空气间距
#     #     [0.10, 6.0],  # 面4：第二透镜厚度
#     #     [0.10, 8.0],  # 面5：第二空气间距
#     #     [0.10, 5.0],  # 面6：第三透镜厚度
#     #     [0.10, 8.0],  # 面7：第三空气间距
#     #     [0.10, 5.0],  # 面8：第四透镜厚度
#     #     [0.10, 8.0],  # 面9：第四空气间距
#     #     [0.10, 5.0],  # 面10：第五透镜厚度
#     #     [0.10, 90.0],  # 面11：像距
#     # ], dtype=torch.float32).cuda()
#     #
#     # Y_real_min_11 = torch.tensor([
#     #     [0.00, 15.0],  # 面1：物距下限
#     #     [-0.10, 2.0],  # 面2：透镜厚度下限
#     #     [-0.10, 0.5],  # 面3：空气间距下限
#     #     [-0.10, 2.0],  # 面4：透镜厚度下限
#     #     [-0.10, 0.5],  # 面5：空气间距下限
#     #     [-0.10, 2.0],  # 面6：透镜厚度下限
#     #     [-0.10, 0.5],  # 面7：空气间距下限
#     #     [-0.10, 2.0],  # 面8：透镜厚度下限
#     #     [-0.10, 0.5],  # 面9：空气间距下限
#     #     [-0.10, 2.0],  # 面10：透镜厚度下限
#     #     [-0.10, 50.0],  # 面11：像距下限
#     # ], dtype=torch.float32).cuda()
#
#     # Y_real_max_9 = torch.tensor(
#     #     [[0, 41], [0.1, 12], [0.1,13], [0.1, 12], [0.1, 7], [0.1, 12], [0.1, 20], [0.1, 12], [0.1, 72]],
#     #     dtype=torch.float32).cuda()
#     # Y_real_min_9 = torch.tensor(
#     #     [[0, 15], [-0.1, 2], [-0.1,0.2], [-0.1,2], [-0.1, 0.1], [-0.1, 2], [-0.1, 0.2], [-0.1, 2],
#     #      [-0.1, 19]], dtype=torch.float32).cuda()
#     # Y_real_max_11 = torch.tensor(
#     #     [[0, 41], [0.1, 12], [0.1,8], [0.1, 12], [0.1, 9], [0.1, 12], [0.1, 5], [0.1, 12], [0.1, 11], [0.1, 12],
#     #      [0.1, 72]], dtype=torch.float32).cuda()
#     # Y_real_min_11 = torch.tensor(
#     #     [[0, 15], [-0.1, 2], [-0.1, 0.2], [-0.1, 2], [-0.1, 0.2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 0.2],
#     #      [-0.1, 2], [-0.1, 19]], dtype=torch.float32).cuda()
#
#     # Y_real_max_9 = torch.tensor(
#     #     [[0, 41], [0.1, 12], [0.1,12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 72]],
#     #     dtype=torch.float32).cuda()
#     # Y_real_min_9 = torch.tensor(
#     #     [[0, 15], [-0.1, 2], [-0.1,2], [-0.1,2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2],
#     #      [-0.1, 19]], dtype=torch.float32).cuda()
#     # Y_real_max_11 = torch.tensor(
#     #     [[0, 44], [0.1, 12], [0.1,12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12], [0.1, 12],
#     #      [0.1, 57]], dtype=torch.float32).cuda()
#     # Y_real_min_11 = torch.tensor(
#     #     [[0, 39], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2], [-0.1, 2],
#     #      [-0.1, 2], [-0.1, 49]], dtype=torch.float32).cuda()
#     # 输出初始化
#     Y_real = torch.zeros_like(Y_norm)
#
#     # === 并行处理 10面 ===
#     # idx9 = (seq_lens == 9)
#     # if idx9.any():
#     #     n9 = idx9.sum()
#     #     min9 = Y_real_min_9.unsqueeze(0).expand(n9, -1, -1)  # [n10, 10, D]
#     #     max9 = Y_real_max_9.unsqueeze(0).expand(n9, -1, -1)
#     #     Y_real[idx9, :9, :] = (Y_norm[idx9, :9, :] - LB) * (max9 - min9) / (UB - LB) + min9
#     #
#     # # === 并行处理 12面 ===
#     # idx11 = (seq_lens == 11)
#     # if idx11.any():
#     #     n11 = idx11.sum()
#     #     min11 = Y_real_min_11.unsqueeze(0).expand(n11, -1, -1)  # [n12, 12, D]
#     #     max11 = Y_real_max_11.unsqueeze(0).expand(n11, -1, -1)
#     #     Y_real[idx11, :11, :] = (Y_norm[idx11, :11, :] - LB) * (max11 - min11) / (UB - LB) + min11
#
#     # === 并行处理 9 长度 (原 10 面) ===
#     idx9 = (seq_lens == 9)
#     if idx9.any():
#         n9 = idx9.sum()
#         min9 = Y_real_min_9.unsqueeze(0).expand(n9, -1, -1)  # [n9, 9, 2]
#         max9 = Y_real_max_9.unsqueeze(0).expand(n9, -1, -1)
#
#         # 曲率: 直接复制
#         Y_real[idx9, :9, 0] = (Y_norm[idx9, :9, 0] - LB) * (max9[:, :, 0] - min9[:, :, 0]) / (UB - LB) + min9[:, :, 0]
#
#         # 距离: 反归一化
#         t_range9 = max9[:, :, 1] - min9[:, :, 1]
#         Y_real[idx9, :9, 1] = _convert2real_t(Y_norm[idx9, :9, 1],min9[:, :, 1],t_range9)
#         # Y_real[idx9, :9, 1] = (Y_norm[idx9, :9, 1] - LB) * (max9[:, :, 1] - min9[:, :, 1]) / (UB - LB) + min9[:, :, 1]
#
#     # === 并行处理 11 长度 (原 12 面) ===
#     idx11 = (seq_lens == 11)
#     if idx11.any():
#         n11 = idx11.sum()
#         min11 = Y_real_min_11.unsqueeze(0).expand(n11, -1, -1)  # [n11, 11, 2]
#         max11 = Y_real_max_11.unsqueeze(0).expand(n11, -1, -1)
#
#         # 曲率: 直接复制
#         # Y_real[idx11, :11, 0] = Y_norm[idx11, :11, 0]
#         Y_real[idx11, :11, 0] = (Y_norm[idx11, :11, 0] - LB) * (max11[:, :, 0] - min11[:, :, 0]) / (UB - LB) + min11[:, :, 0]
#
#         # 距离: 归一化
#         t_range11 = max11[:, :, 1] - min11[:, :, 1]
#         Y_real[idx11, :11, 1] = _convert2real_t(Y_norm[idx11, :11, 1], min11[:, :, 1], t_range11)
#         # Y_real[idx11, :11, 1] = (Y_norm[idx11, :11, 1] - LB) * (max11[:, :, 1] - min11[:, :, 1]) / (UB - LB) + min11[:,:, 1]
#     #
#     Y_real[:, 0, 0] = 0.0
#
#     return Y_real

if __name__ == '__main__':

    # #——————————————————————————————————数据split处理————————————————————————————————————————————#
    # 取数据
    orig_data = np.loadtxt('./data/surf10_12_ul_1129.csv', delimiter=',', dtype=None)
    # 划分数据集
    train_set, val_set = train_test_split(orig_data, train_size=0.8, random_state=2026, shuffle=True)
    # val_set, test_set = train_test_split(Test_set, test_size=0.1, random_state=None, shuffle=True)
    print(len(train_set), "train +", len(val_set), "val")
    # print(len(val_set), "train +", len(test_set), "test")
    Train_data = pd.DataFrame(train_set)  # 数据有三列，列名分别为one,two,three
    # Test_data = pd.DataFrame(test_set)
    Val_data = pd.DataFrame(val_set)
    Train_data.to_csv('./data/scan_lens_train_ul_1129.csv', header=None, index=False, encoding="utf-8")
    # Test_data.to_csv('./data/Triplet_test_0802.csv', header=None, index=False, encoding="utf-8")
    Val_data.to_csv('./data/scan_lens_val_ul_1129.csv', header=None, index=False, encoding="utf-8")
    # —————————————————————————————————————————————————————————————————————————————————————————————#


    # X_norm, Y_norm = load_train_data()
    # X_norm = X_norm.cuda()
    # Y_norm = Y_norm.cuda()
    # data_norm = torch.cat((X_norm, Y_norm), dim=1)
    # data_norm = data_norm.cpu().numpy()
    # X_real = convert2real_dataX(X_norm)
    # Y_real = convert2real_dataY(Y_norm)
    # Real_data = torch.cat((X_real, Y_real), dim=1)
    # Real_data = Real_data.cpu().numpy()
    # fig, axes = plt.subplots(1, 2)
    # plt.subplots_adjust(left=0.1, bottom=None, right=0.9, top=None, wspace=0.6, hspace=0.6)
    # sns.distplot(data_norm, bins=10, kde_kws={'color': 'seagreen', 'lw': 3}, hist_kws={'color': 'b'}, ax=axes[0],
    #              norm_hist=True)
    # sns.distplot(Real_data, bins=10, kde_kws={'color': 'seagreen', 'lw': 3}, hist_kws={'color': 'b'}, ax=axes[1],
    #              norm_hist=True)
    # #   plt.hist(data, bins=12, rwidth=0.9, density=True)
    # #   plt.title('Length distribution')
    # #   plt.xlabel('Length')
    # #   plt.ylabel('Probability') # 输出正态分布曲线和直方图
    # plt.show()
