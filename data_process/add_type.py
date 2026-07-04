'''
Description: 编制序列，比如说，AGAG - 0101
Debug Date :
Status: ok!
TODO: 把这个整理好 放到 Data_preprocess.py. 不要的代码全部清理。
      要有一个示例，支持直接从data folder读数据！
'''


import numpy as np

def generate_type_seq_from_padded(surf_seq, seq_len, nWL=3, nSurf=11, use_pattern=True):
    """
    Function: 根据折射率数据生成lens-sequence.
    surf_seq: [max_face*nWL]
    seq_len: 实际面数
    """
    surf_seq = surf_seq.reshape(nSurf, nWL)

    if use_pattern:
        # 严格交替 A/G: A=0, G=1
        type_seq = [i % 2 for i in range(seq_len)]
    else:
        # 用折射率判断
        n_values = surf_seq[:, 0]
        type_seq = (n_values[:seq_len] > 1.0005).astype(int)

    # padding
    pad_len = nSurf - seq_len
    type_seq = type_seq + [0] * pad_len

    return np.array(type_seq, dtype=int)


def save_dataset_with_types(X_sys, seq_lengths, X_bgr, csv_path, glss_dim=3, max_face=11, use_pattern=True):
    rows = []
    N = X_sys.shape[0]

    for i in range(N):
        sys_params = X_sys[i]              # [sys_dim]
        seq_len = int(seq_lengths[i])
        surf_seq_flat = X_bgr[i]           # [max_face*glss_dim]

        # 生成类型序列
        type_seq = generate_type_seq_from_padded(surf_seq_flat, seq_len, glss_dim, max_face, use_pattern)

        # 一行: 系统参数 + seq_len + surf_seq(已padding) + type_seq
        row = list(sys_params) + [seq_len] \
              + surf_seq_flat.flatten().tolist() \
              + type_seq.tolist()
        rows.append(row)

    # 保存
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False, header=False)
    print(f"✅ 保存完成: {csv_path}, 共 {N} 条序列, 每条固定 {max_face} 面 (已padding)")


if __name__ == "__main__":
    # 测试代码，只在直接运行此文件时执行
    try:
        data = np.loadtxt('./data/surf10_12_ul_1129.csv', delimiter=',', dtype=None)
        X_sys, X_bgr = data[:, :3], data[:, 3:]

        train_data = np.loadtxt('./data/scan_lens_train_ul_1129.csv', delimiter=',', dtype=None)
        X_train_sys, X_train_bgr = train_data[:, :3], train_data[:, 3:]

        validate_data = np.loadtxt('./data/scan_lens_val_ul_1129.csv', delimiter=',', dtype=None)
        # validate_data = torch.tensor(validate_data, dtype=torch.float32).cuda()
        X_val_sys, X_val_bgr = validate_data[:, :3], validate_data[:, 3:]
        seq_lengths = (X_sys[:, 2] ).astype(int)
        seq_lengths_train = (X_train_sys[:, 2] ).astype(int)
        seq_lengths_val = (X_val_sys[:, 2]).astype(int)
        save_dataset_with_types(X_sys[:, :2], seq_lengths, X_bgr, "./data/surf10_12_ul_1129.csv")
        save_dataset_with_types(X_train_sys[:, :2], seq_lengths_train, X_train_bgr, "./data/scan_lens_train_ul_1129.csv")
        save_dataset_with_types(X_val_sys[:, :2], seq_lengths_val, X_val_bgr, "./data/scan_lens_val_ul_1129.csv")
    except FileNotFoundError as e:
        print(f"测试数据文件不存在，跳过测试代码: {e}")


import torch
import pandas as pd
import numpy as np

# def build_vocab(max_pairs=11):
#     vocab = {"PAD": 0}
#     idx = 1
#     for i in range(1, max_pairs+1):
#         vocab[f"t{i}"] = idx; idx += 1
#         vocab[f"c{i}_1"] = idx; idx += 1
#         vocab[f"c{i}_2"] = idx; idx += 1
#     return vocab
#
#
# import pandas as pd
#
#
# import numpy as np
# import pandas as pd
#
# def expand_face_sequence_random(face_seq, true_len=None, max_len=30):
#     """
#     每行扩展序列 → 在 [-1,1] 内随机采样
#     """
#     tokens = []
#     counter = 1
#
#     if true_len is not None:
#         face_seq = face_seq[:true_len]
#
#     for f in face_seq:
#         if f == 0:   # A 面
#             tokens.append("t")
#             counter += 1
#         elif f == 1: # G 面
#             tokens.append("c1")
#             tokens.append("t")
#             tokens.append("c2")
#             counter += 1
#
#     length = len(tokens)
#
#     if length > 0:
#         values = np.random.uniform(-1, 1, length)
#     else:
#         values = np.array([])
#
#     # === PAD 用 -1 填充 ===
#     if length < max_len:
#         values = np.concatenate([values, -1*np.ones(max_len - length)])
#     else:
#         values = values[:max_len]
#         length = max_len
#
#     return values, length
#
#
# def batch_expand_and_save(X_sys, _train_bgr,batch_face_types, seq_lengths, max_len=22, save_path="expanded_sequences.csv", mode="random"):
#     expanded_data = []
#     lengths = []
#
#     for i in range(len(batch_face_types)):
#         if mode == "random":
#             values, length = expand_face_sequence_random(batch_face_types[i], seq_lengths[i], max_len=max_len)
#         else:
#             values, length = expand_face_sequence_unique(batch_face_types[i], seq_lengths[i], max_len=max_len)
#
#         expanded_data.append(values)
#         lengths.append(length)
#
#     expanded_data = np.array(expanded_data)
#
#     df = pd.DataFrame(
#         np.hstack([X_sys,np.array(lengths)[:, None],np.array(lengths)[:, None] // 2 + 1, X_train_bgr,expanded_data])
#     )
#
#     df.to_csv(save_path, index=False, header=False, encoding="utf-8")
#     print(f"已保存到 {save_path} （mode={mode}）")
#     return df
#
# # ===================
# # 示例流程
# # ===================
# if __name__ == "__main__":
#     # 构建词表
#     vocab = build_vocab(max_pairs=7)
#
#     train_data = np.loadtxt('./data/surf10_12_ul_0908.csv', delimiter=',', dtype=None)
#     # train_data = torch.tensor(train_data, dtype=torch.float32)
#     X_train_sys, X_train_bgr, X_train_type = train_data[:, :3], train_data[:, 3:3*11+3], train_data[:,3*11+3:3*11+14]
#     seq_lengths_train = (X_train_sys[:, 2]).astype(int)
#     # 生成扩展序列
#     # token_ids, lengths = expand_face_types_batch(X_train_type, vocab, max_len=21)
#
#     # ===== 保存为 CSV =====
#     df = batch_expand_and_save(X_train_sys[:,:2], X_train_bgr, X_train_type, seq_lengths_train, max_len=21, save_path="./data/surf10_12_ul_0922.csv")
#     print(df)



