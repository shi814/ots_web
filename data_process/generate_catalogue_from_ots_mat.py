
'''
Description: 生成OTS的catalogue目录，然后做成分类用的 classifier
Debug Date :
Status: ok!
TODO: 把这个整理好 放到 Data_preprocess.py. 不要的代码全部清理。
      要有一个示例，支持直接从data folder读数据！
'''


import pandas as pd
import numpy as np
import hdf5storage as hdf5

# === Step 1. 读取 .mat 文件 ===
mat_data = hdf5.loadmat('../data/ots_lens_catalogue_clean.mat')
glass_data = mat_data['Glass'].squeeze()
Radius1 = np.nan_to_num(mat_data['Radius1'].squeeze().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
Radius2 = np.nan_to_num(mat_data['Radius2'].squeeze().astype(float), nan=0.0, posinf=0.0, neginf=0.0)
Thick_data = mat_data['Thickness'].squeeze().astype(float)

# 转成字符串
glass_names = [g.decode() if isinstance(g, (bytes, bytearray)) else str(g) for g in glass_data]
glass_names = [g.replace("[[", "").replace("]]", "").replace("'", "") for g in glass_names]

# === Step 2. 读取 Excel 系数表 ===
df = pd.read_excel("./glass/schott_glass_pars.xlsx", header=None)

# 波长 (μm)
wgl = np.array([0.45, 0.58, 0.75])  # RGB 三个波长

# === Step 3. 保存 Sellmeier 系数 和 折射率+半径 ===
with open("../glass/ots_lens.txt", "w") as f1, open("../glass/ots_lens_glass_RIs.txt", "w") as f2:
    for i, name in enumerate(glass_names):
        row = df[df[0] == name]
        if not row.empty:
            coeffs = row.iloc[0, 1:].astype(float).tolist()
            coeffs_fmt = "   ".join([f"{v:.8E}" for v in coeffs])
            f1.write(f"{name}   {coeffs_fmt}\n")

            K1, K2, K3, L1, L2, L3 = coeffs
            RIs = np.sqrt(
                (K1*wgl**2)/(wgl**2 - L1) +
                (K2*wgl**2)/(wgl**2 - L2) +
                (K3*wgl**2)/(wgl**2 - L3) + 1
            )
            RIs_fmt = "   ".join([f"{v:.5f}" for v in RIs])
            f2.write(f"{name}   {RIs_fmt}   {Radius1[i]}   {Radius2[i]}\n")
        else:
            f1.write(f"{name}   NOT_FOUND\n")
            f2.write(f"{name}   NOT_FOUND\n")

# === Step 4. 读回折射率数据 ===
ots_lens_data = pd.read_csv("../glass/ots_lens_glass_RIs.txt", delim_whitespace=True, header=None)
lens_data = np.array(ots_lens_data.iloc[:, 1:], dtype=float)  # [nB,nG,nR,R1,R2]

# === Step 5. 翻转曲率并取负号 ===
lens_reverse_data = lens_data.copy()
# lens_reverse_data[:, -3:-1] = -lens_data[:, -2:-4:-1]

# 拼接
BGR_C_data = lens_data#np.concatenate((lens_data, lens_reverse_data), axis=0)

# === Step 6. 提取唯一的 BGR，并按组匹配曲率 ===
bgr = BGR_C_data[:, :3]   # 提取 BGR 折射率
unique_bgr = np.unique(bgr, axis=0)

result_list = []
for ub in unique_bgr:
    mask = np.all(bgr == ub, axis=1)
    curves = BGR_C_data[mask, 3:]  # 取对应 R1,R2
    for r1, r2, t in curves:
        result_list.append([*ub, r1, r2])

# === Step 7. 去重得到最终结果 ===
result_array = np.array(result_list)
unique_result = np.unique(result_array, axis=0)

ots_glass_c_data = pd.DataFrame(unique_result)
ots_glass_c_data.to_csv('./glass/ots_glass_c_new.csv', header=None, index=False, encoding="utf-8")

print("最终结果维度:", unique_result.shape)
print(unique_result[:5])  # 查看前 5 行
