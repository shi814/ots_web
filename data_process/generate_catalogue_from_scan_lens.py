'''
Description: 生成OTS的catalogue目录，然后做成分类用的 classifier
Debug Date :
Status: ok!
TODO: 把这个整理好 放到 Data_preprocess.py. 不要的代码全部清理。
      要有一个示例，支持直接从data folder读数据！
'''

import numpy as np
import pandas as pd
# import scipy.io as sio

# glassData = np.loadtxt('.\glass\schott_glass_pars.csv', delimiter=',', dtype=None)
# K1, K2, K3, L1, L2, L3 = np.hsplit(glassData, 6)  # as input, 3 col
# print(K1.shape)
# # wgl = np.array([0.58756, 0.54607])   # to check with Schott, nd = 587.56nm, ne = 546.07nm
# wgl = np.array([0.45, 0.58, 0.75])  # to get the RIs for RGB channels [0.486, 0.588, 0.656]
# RIs = np.sqrt(K1*wgl**2/(wgl**2 - L1) + K2*wgl**2/(wgl**2 - L2) + K3*wgl**2/(wgl**2 - L3) + 1)

# ots_data = hdf5.loadmat('./data/ots_lens_catalogue_clean.mat')['Thickness']
# print(ots_data)
# a=0

surf10_data = np.loadtxt('./data/scan_lens_dataset_surf10_reorder.csv', delimiter=',', dtype=None)
surf12_data = np.loadtxt('./data/scan_lens_dataset_surf12_reorder.csv', delimiter=',', dtype=None)
ots_glass_c_t_data = np.loadtxt('./glass/ots_glass_c_t.csv', delimiter=',')
n_10 = len(surf10_data)
n_12 = len(surf12_data)

material_data10 = np.array(surf10_data[:,3:3 + 10*3],dtype=np.float64).reshape(n_10,10,3)  # 12 面每面 3 个数
material_data12 = np.array(surf12_data[:,3:3 + 12*3],dtype=np.float64).reshape(n_12,12,3)  # 10 面每面 3 个数
material_data10 = material_data10[:,~np.all(material_data10 == 1, axis=(0,2)),:]
material_data12 = material_data12[:,~np.all(material_data12 == 1, axis=(0,2)),:]

R_data10 = surf10_data[:,3 + 10*3:3 + 10*3 + 2*10].reshape(n_10,10,2)
R_data12 = surf12_data[:,3 + 12*3:3 + 12*3 + 2*12].reshape(n_12,12,2)
R_data10 = R_data10[:,1:-1,:].reshape(n_10,4,2,2)
R_data12 = R_data12[:,1:-1,:].reshape(n_12,5,2,2)
# T_data10 = R_data10[:,:,0,1]
R_T_data10 = np.stack([R_data10[:,:,0,0], R_data10[:,:,1,0],R_data10[:,:,0,1]], axis=2)
R_T_data12 = np.stack([R_data12[:,:,0,0], R_data12[:,:,1,0],R_data12[:,:,0,1]], axis=2)

# T_data10 =

R_T_reverse_data10 = np.stack([-R_T_data10[:,:,1],-R_T_data10[:,:,0],R_T_data10[:,:,2]], axis=2)#-R_data10[..., ::-1]
R_T_reverse_data12 = np.stack([-R_T_data12[:,:,1],-R_T_data12[:,:,0],R_T_data12[:,:,2]], axis=2)#-R_data12[..., ::-1]

BGR_C_T_data10 = np.concatenate((material_data10,R_T_data10),axis=2).reshape(-1,6)
BGR_C_T_data12 = np.concatenate((material_data12,R_T_data12),axis=2).reshape(-1,6)

BGR_C_T_reverse_data10 = np.concatenate((material_data10,R_T_reverse_data10),axis=2).reshape(-1,6)
BGR_C_T_reverse_data12 = np.concatenate((material_data12,R_T_reverse_data12),axis=2).reshape(-1,6)

BGR_C_T_data = np.concatenate((BGR_C_T_data10,BGR_C_T_data12,BGR_C_T_reverse_data10,BGR_C_T_reverse_data12),axis=0)
A = unique_bgr = np.unique(BGR_C_T_data, axis=0)
BGR_C_T_data = np.concatenate((BGR_C_T_data,ots_glass_c_t_data),axis=0)

# 提取 BGR 部分
bgr =BGR_C_T_data[:, :3]

# 找到唯一的 BGR 组合
unique_bgr = np.unique(bgr, axis=0)

# 组装结果数组
result_list = []
for ub in unique_bgr:
    mask = np.all(bgr == ub, axis=1)      # 筛选条件
    curves = BGR_C_T_data[mask, 3:]         # 对应 R1, R2
    # 把每个曲率组合拼到 BGR 后面
    for r1, r2, t in curves:
        result_list.append([*ub, r1, r2, t])

# 转为 numpy 数组
result_array = np.array(result_list)
unique_result = np.unique(result_array, axis=0)

material_c_data = pd.DataFrame(unique_result)
material_c_data.to_csv('./glass/Material_C_Data.csv', header=None, index=False, encoding="utf-8")

# ots_glass_c_data = np.loadtxt('./glass/ots_glass_c.csv', delimiter=',')
# q=0
