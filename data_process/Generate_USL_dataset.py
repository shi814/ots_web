'''
Description: 生成无监督的训练数据集。
Debug Date :
Status: ok!
TODO: 把这个整理好 放到 Data_preprocess.py. 不要的代码全部清理。
      要有一个示例，支持直接从data folder读数据！
'''


import numpy as np
import pandas as pd
import itertools

# ============================
# 1️⃣ 工具函数
# ============================

def unique_triplets_from_batch(data):
    """按 3 列为一组，返回唯一组合"""
    arr = np.array(data, dtype=np.float64).reshape(-1, 3)
    return np.unique(arr, axis=0)

def merge_unique_triplets(data1, data2):
    """合并两批数据的唯一组合，返回每批唯一组合和全局唯一组合"""
    uniq1 = unique_triplets_from_batch(data1)
    uniq2 = unique_triplets_from_batch(data2)
    all_uniq = np.vstack([uniq1, uniq2])
    global_unique = np.unique(all_uniq, axis=0)
    return uniq1, uniq2, global_unique

def unique_10faces(data):
    """返回不重复的 10 面组合（每面 3 个数）"""
    uniq, idx = np.unique(data, axis=0, return_index=True)
    return uniq, idx

# ============================
# 2️⃣ 生成随机 10 面/12 面组合
# ============================

def generate_random_10surf_combinations(uniq_all, num_samples=93, seed=42):
    """
    生成随机 10 面组合：
    - pattern: 空气(A)/玻璃(G) = ["A","G","A","G","A","G","A","G","A","A"]
    - 保证不重复已有组合
    """
    np.random.seed(seed)
    air = np.array([1,1,1], dtype=np.float64)
    glass = [np.array(row, dtype=np.float64).reshape(3,) for row in uniq_all if not np.allclose(row, air)]
    pattern = ["A","G","A","G","A","G","A","G","A","A"]
    glass_positions = [i for i,p in enumerate(pattern) if p=="G"]
    n_glass = len(glass_positions)

    rng = np.random.default_rng(seed)
    new_combos = []
    # existing_set = set(map(tuple, existing_combos))

    while len(new_combos) < num_samples:
        glass_choice_idx = rng.integers(low=0, high=len(glass), size=n_glass)
        seq = []
        g_iter = iter(glass_choice_idx)
        for p in pattern:
            seq.append(air if p=="A" else glass[next(g_iter)])
        seq_arr = np.stack(seq, axis=0).reshape(-1)
        new_combos.append(seq_arr)
        # if tuple(seq_arr) not in existing_set:
        #     new_combos.append(seq_arr)
        #     existing_set.add(tuple(seq_arr))

    all_combos = np.vstack([new_combos])
    return all_combos

def generate_random_12surf_combinations(uniq_all, num_samples=93, seed=42):
    """
    生成随机 12 面组合：
    - pattern: ["A","G","A","G","A","G","A","G","A","G","A","A"]
    - 保证不重复已有组合
    """
    np.random.seed(seed)
    air = np.array([1,1,1], dtype=np.float64)
    glass = [np.array(row, dtype=np.float64).reshape(3,) for row in uniq_all if not np.allclose(row, air)]
    pattern = ["A","G","A","G","A","G","A","G","A","G","A","A"]
    glass_positions = [i for i,p in enumerate(pattern) if p=="G"]
    n_glass = len(glass_positions)

    rng = np.random.default_rng(seed)
    new_combos = []
    # existing_set = set(map(tuple, existing_combos))

    while len(new_combos) < num_samples:
        glass_choice_idx = rng.integers(low=0, high=len(glass), size=n_glass)
        seq = []
        g_iter = iter(glass_choice_idx)
        for p in pattern:
            seq.append(air if p=="A" else glass[next(g_iter)])
        seq_arr = np.stack(seq, axis=0).reshape(-1)
        new_combos.append(seq_arr)
        # if tuple(seq_arr) not in existing_set:
        #     new_combos.append(seq_arr)
        #     existing_set.add(tuple(seq_arr))

    all_combos = np.vstack([new_combos])
    return all_combos

# ============================
# 3️⃣ 读取数据，提取材料信息
# ============================

data1 = np.loadtxt('./data/scan_lens_dataset_surf12_reorder.csv', delimiter=',')
data2 = np.loadtxt('./data/scan_lens_dataset_surf10_reorder.csv', delimiter=',')
ots_glass_c_data = np.loadtxt('./glass/ots_glass_c.csv', delimiter=',')

material_data1 = data1[:,3:3 + 12*3]  # 12 面每面 3 个数
material_data2 = data2[:,3:3 + 10*3]  # 10 面每面 3 个数

uniq1, uniq2, global_uniq = merge_unique_triplets(material_data1, material_data2)
global_uniq = np.delete(global_uniq, 3, axis=0)
uniq_10faces, idx1 = unique_10faces(material_data2)
uniq_12faces, idx2 = unique_10faces(material_data1)
ots_glass = np.array(ots_glass_c_data[:,:3],dtype=np.float64)
global_data = np.concatenate((global_uniq,ots_glass),axis=0)
global_uniq_data = np.unique(global_data, axis=0)
print(global_uniq_data)
# ============================
# 4️⃣ 生成随机组合 + 已有组合，得到最终 100 种组合
# ============================

all_100_10surf_combos = generate_random_10surf_combinations(global_uniq_data, num_samples=100)
all_100_12surf_combos = generate_random_12surf_combinations(global_uniq_data, num_samples=100)
all_100_10surf_combos = all_100_10surf_combos[:,:10*3-3]
all_100_12surf_combos = all_100_12surf_combos[:,:12*3-3]

# # ============================
# # 5️⃣ FN/FOV 网格
# # ============================
#
fn_values = np.linspace(5,9.75,6)
# values = np.arange(start=4, stop=9.75, step=1.5)
fov_values = np.linspace(8.5,10,4)
a=0
fn_fov_combos = np.array(list(itertools.product(fn_values, fov_values)))  # shape (15,2)
#
# # ============================
# # 6️⃣ 生成训练数组
# # ============================
#
data_10surf_rows = []
for lens_seq in all_100_10surf_combos:
    for fn_fov in fn_fov_combos:
        row = np.concatenate([fn_fov, lens_seq.flatten()])
        data_10surf_rows.append(row)
data_10surf_array = np.stack(data_10surf_rows, axis=0)

data_12surf_rows = []
for lens_seq in all_100_12surf_combos:
    for fn_fov in fn_fov_combos:
        row = np.concatenate([fn_fov, lens_seq.flatten()])
        data_12surf_rows.append(row)
data_12surf_array = np.stack(data_12surf_rows, axis=0)

print("10 面训练数组 shape:", data_10surf_array.shape)
print("12 面训练数组 shape:", data_12surf_array.shape)

# ============================
# 7️⃣ 保存 CSV
# ============================
# 设置随机种子保证可复现
seed = 42
rng = np.random.default_rng(seed)

# 打乱 10 面数据
rng.shuffle(data_10surf_array)
# 打乱 12 面数据
rng.shuffle(data_12surf_array)

pd.DataFrame(data_10surf_array).to_csv('../data/surf10_ul_1129.csv', header=None, index=False)
pd.DataFrame(data_12surf_array).to_csv('./data/surf12_ul_1129.csv', header=None, index=False)
print("CSV 文件已保存完成")

surf10_data = np.loadtxt('./data/surf10_ul_1129.csv', delimiter=',')
surf12_data= np.loadtxt('./data/surf12_ul_1129.csv', delimiter=',')
sys_10surf = surf10_data[:,:2]
sys_12surf = surf12_data[:,:2]

material_10surf = surf10_data[:,2:]  # 12 面每面 3 个数
material_12surf = surf12_data[:,2:]  # 10 面每面 3 个数

# 序列长度
seq_lengths_10 = np.full(len(surf10_data), 9)
seq_lengths_12 = np.full(len(surf12_data), 11)

feat_per_face = 3
max_seq_len = max(seq_lengths_10.max(), seq_lengths_12.max())
# ---------------------------
# Padding 折射率
# ---------------------------
def pad_surf_data(surf_data, seq_len, feat_per_face, max_seq_len):
    padded_list = []
    for arr, L in zip(surf_data, seq_len):
        seq_arr = arr.reshape(L, feat_per_face)
        if L < max_seq_len:
            pad = np.zeros((max_seq_len - L, feat_per_face))
            seq_arr = np.vstack([seq_arr, pad])
        padded_list.append(seq_arr)
    return np.stack(padded_list, axis=0)  # shape: [B, max_seq_len, feat_per_face]

padded_surf_10 = pad_surf_data(material_10surf, seq_lengths_10, feat_per_face, max_seq_len)
padded_surf_12 = pad_surf_data(material_12surf, seq_lengths_12, feat_per_face, max_seq_len)

# ---------------------------
# 连接系统参数 + 折射率 + 序列长度
# ---------------------------
X_all = np.vstack([sys_10surf,sys_12surf])
surf_all = np.vstack([padded_surf_10, padded_surf_12])
seq_len_all = np.concatenate([seq_lengths_10, seq_lengths_12])

B = surf_all.shape[0]
surf_flat = surf_all.reshape(B, max_seq_len * feat_per_face)

# 合并系统参数 + 折射率 + 序列长度
all_dataset = np.hstack([X_all, seq_len_all[:, None], surf_flat])

# 保存到 csv
df = pd.DataFrame(all_dataset)
df.to_csv('./data/surf10_12_ul_1129.csv', index=False, header=False)

print("保存完成，形状:", all_dataset.shape)