"""
Deprecated model module.

Status:
- Current in-repo training/testing uses `models/TransformerClass_Model.py`
  through `utils.create_transformer_model()` and `utils.create_transformer_val()`.
- This file is kept temporarily for historical experiments and rollback.

If external scripts/notebooks still import this module, migrate those callers
before archival/removal.
"""

import torch
import torch.nn as nn
import argparse
import torch
import utils
import math
import os
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
# from Data_processing import load_train_data, load_validate_data, normalize_dataX, normalize_dataY
def weights_init(net, init_type = '', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    """
    def init_func(m):
        classname = m.__class__.__name__
        # for every Linear layer in a model
        # m.weight.data shoud be taken from a normal distribution
        # m.bias.data should be 0
        if classname.find('Linear') != -1:
            m.weight.data.normal_(0, 0.05)
            # nn.init.xavier_uniform_(m.weight)
            #torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            # torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            #torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            #torch.nn.init.constant_(m.bias.data, 0.0)
            m.bias.data.fill_(0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

# # -----------------------------
# # Embedding 系统参数 -> 序列
# # -----------------------------
class EmbeddingSeq(nn.Module):
    def __init__(self, sys_dim, glss_dim, hidden_dim, max_seq_len):
        """
        sys_dim: 系统参数维度
        glss_dim: 每个面折射率参数维度
        hidden_dim: Transformer d_model
        max_seq_len: 最大面数 (比如 11)
        """
        super().__init__()
        # self.token_fc = nn.Linear(glss_dim, hidden_dim)  # 面折射率 -> embedding
        # self.token_fc = nn.Sequential(
        #     nn.Linear(glss_dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, hidden_dim)
        # )
        # # 面特征 -> embedding
        self.token_fc_list = nn.ModuleList([
            nn.Linear(glss_dim, hidden_dim) for _ in range(max_seq_len)
        ])
        self.sys_proj = nn.Linear(sys_dim, hidden_dim)   # 系统参数 -> embedding
        self.type_emb = nn.Embedding(2, hidden_dim)      # 面类型 (A=0, G=1)
        self.max_seq_len = max_seq_len

    def forward(self, sys_params, surf_seq,type_seq):
        """
        sys_params: [B, sys_dim]
        surf_seq:   [B, n_face, glss_dim]   每个面的折射率特征
        type_seq:   [B, n_face]             面类型 (0=A, 1=G)
        """
        B, n_face, _ = surf_seq.shape
        x = []
        # === 1) 面折射率 embedding ===
        # x = self.token_fc(surf_seq)  # [B, n_face, H]

        for i in range(self.max_seq_len):
            out = self.token_fc_list[i](surf_seq[:, i, :])  # [B, H]
            x.append(out.unsqueeze(1))
        x = torch.cat(x, dim=1)  # [B, L, H]

        # === 2) 系统参数 embedding，加到每个面上 ===
        sys_emb = self.sys_proj(sys_params).unsqueeze(1).expand(-1, n_face, -1)
        x = x + sys_emb

        # === 3) 类型 embedding (A/G) ===
        x = x + self.type_emb(type_seq)  # [B, n_face, H]

        return x  # [B, n_face, H]

# -----------------------------
# Positional Embedding
# -----------------------------

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, seq_len):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model),requires_grad=True)
        self.repe = nn.Parameter(torch.randn(1, seq_len, d_model),requires_grad=True)
        self.proj = nn.Linear(2 * d_model, d_model)
        # nn.init.normal_(self.pe, 0, 0.05)
        # nn.init.normal_(self.repe, 0, 0.05)

    def forward(self, x):
        seq_len = x.size(1)
        forward_pe = self.pe[:, :seq_len, :]
        backward_pe = torch.flip(self.repe[:, :seq_len, :], dims=[1])#self.repe[:, -seq_len:, :]
        pe = torch.cat([forward_pe, backward_pe], dim=-1)
        pe = self.proj(pe)
        return x + pe

# class PositionalEmbedding(nn.Module):
#     def __init__(self, d_model, seq_len):
#         super().__init__()
#         self.d_model = d_model
#         self.seq_len = seq_len
#
#         # 正向编码
#         self.pe_fwd = self._build_pe(seq_len, d_model)
#         # 反向编码
#         self.pe_bwd = torch.flip(self.pe_fwd, dims=[1])  # 在 seq_len 维度翻转
#
#         # 拼接后再投影回 d_model
#         self.proj = nn.Linear(2*d_model, d_model)
#
#     def _build_pe(self, seq_len, d_model):
#         pe = torch.zeros(seq_len, d_model)
#         position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         return pe.unsqueeze(0)  # [1, seq_len, d_model]
#
#     def forward(self, x):
#         # [B,L,d_model]
#         L = x.size(1)
#         fwd = self.pe_fwd[:, :L, :].to(x.device)
#         bwd = self.pe_bwd[:, -L:, :].to(x.device)
#         pe_cat = torch.cat([fwd, bwd], dim=-1)  # [1,L,2*d_model]
#         return x + self.proj(pe_cat)


# -----------------------------
# Transformer + TFCBlock
# -----------------------------
class TFCBlock(nn.Module):
    def __init__(self, input_size, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.Tanh(),
            nn.Linear(input_size // 2, input_size // 4),
            nn.Tanh(),
            nn.Linear(input_size // 4, out_dim)
        )

    def forward(self, x):
        return self.block(x)

class LensTransformer(nn.Module):
    def __init__(self, opt,nhead=8, num_layers=6):
        super().__init__()
        self.embedding = EmbeddingSeq(opt.sys_dim,opt.glss_dim, opt.input_size, opt.seq_len)
        self.pos_emb = PositionalEmbedding(opt.input_size, opt.seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=opt.input_size, nhead=nhead,dim_feedforward=opt.hidden_size,dropout=0.1, batch_first=True,norm_first=True,activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.heads_A_9 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(9)])
        self.heads_G_9 = nn.ModuleList([TFCBlock(opt.input_size, 3) for _ in range(9)])
        # self.heads_G_9_c1 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(9)])
        # self.heads_G_9_c2 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(9)])
        # self.heads_G_9_t = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(9)])

        self.heads_A_11 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(11)])
        self.heads_G_11 = nn.ModuleList([TFCBlock(opt.input_size, 3) for _ in range(11)])
        # self.heads_G_11_c1 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(11)])
        # self.heads_G_11_c2 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(11)])
        # self.heads_G_11_t = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(11)])

        self.max_seq_len = opt.seq_len
        self.output_size = opt.output_size

    def forward(self, sys_params, surf_seq,type_seq,seq_lengths):
        B, _ = sys_params.shape

        # === Embedding ===
        x = self.embedding(sys_params, surf_seq, type_seq)
        x = self.pos_emb(x)

        # === Transformer ===
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=x.device)[None, :] >= seq_lengths[:, None]
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B,L,H]
        B,L,H = x.shape

        outputs = torch.zeros(B, self.max_seq_len, self.output_size-1, device=x.device)

        # === 9 面样本 ===
        idx9 = (seq_lengths == 9)
        if idx9.any():
            x9 = x[idx9, :9, :]  # [N9, 9, H]
            type9 = type_seq[idx9, :9]  # [N9, 9]

            outs9 = []
            for i in range(9):
                feat = x9[:, i, :]
                mask_A = type9[:, i] == 0
                mask_G = type9[:, i] == 1
                out_i = []#torch.zeros(feat.size(0), self.output_size-1, device=x.device)
                if mask_A.any():
                    pred_A = self.heads_A_9[i](feat[mask_A])
                    # pred_A_soft = F.softplus(pred_A)
                    out_i.append(pred_A)
                    # out_i[mask_A, 1:2] = pred_A
                if mask_G.any():
                    # pred_c1 = torch.tanh(self.heads_G_9_c1[i](feat[mask_G]))
                    # pred_c2 = torch.tanh(self.heads_G_9_c2[i](feat[mask_G]))
                    # pred_t = self.heads_G_9_t[i](feat[mask_G])  # 厚度一般非负
                    # pred_G = self.heads_G_9[i](feat[mask_G])
                    # pred_G_soft = torch.cat((pred_c1, pred_t, pred_c2), dim=1)
                    pred_G = self.heads_G_9[i](feat[mask_G])
                    pred_G_soft = torch.cat((torch.tanh(pred_G[:, 0:1]), pred_G[:, 1:2], torch.tanh(pred_G[:, 2:3])),dim=1)
                    out_i.append(pred_G_soft)
                    # out_i.append(pred_G_soft)
                    # out_i[mask_G, :] = pred_G
                outs9.append(out_i)
            outs9 = [t for sub in outs9 for t in sub ]

            outs9 = torch.cat(outs9, dim=1)  # [N9, 9, out_dim]
            curv_zero = torch.zeros_like(outs9[:,0:1])
            outs9 = torch.cat((curv_zero,outs9), dim=1)  # [N9, 9, out_dim]
            outs9 = outs9.view(outs9.shape[0], 9, self.output_size-1)  # [256,9,2]
            outputs[idx9, :9, :] = outs9

        # === 11 面样本 ===
        idx11 = (seq_lengths == 11)
        if idx11.any():
            x11 = x[idx11, :11, :]  # [N11, 11, H]
            type11 = type_seq[idx11, :11]  # [N11, 11]

            outs11 = []
            for i in range(11):
                feat = x11[:, i, :]
                mask_A = type11[:, i] == 0
                mask_G = type11[:, i] == 1
                out_i = []#torch.zeros(feat.size(0), self.output_size, device=x.device)
                if mask_A.any():
                    pred_A = self.heads_A_11[i](feat[mask_A])
                    # pred_A_soft = F.softplus(pred_A)
                    out_i.append(pred_A)
                    # out_i[mask_A, 1:2] = pred_A
                if mask_G.any():
                    # pred_G = self.heads_G_11[i](feat[mask_G])
                    # pred_c1 = torch.tanh(self.heads_G_11_c1[i](feat[mask_G]))
                    # pred_c2 = torch.tanh(self.heads_G_11_c2[i](feat[mask_G]))
                    # pred_t = self.heads_G_11_t[i](feat[mask_G])
                    # pred_G_soft = torch.cat((pred_c1, pred_t, pred_c2),dim=1)
                    pred_G = self.heads_G_11[i](feat[mask_G])
                    pred_G_soft = torch.cat((torch.tanh(pred_G[:, 0:1]), pred_G[:, 1:2], torch.tanh(pred_G[:, 2:3])),dim=1)
                    out_i.append(pred_G_soft)
                    # out_i[mask_G, :] = pred_G
                outs11.append(out_i)
            outs11 = [t for sub in outs11 for t in sub]
            outs11 = torch.cat(outs11, dim=1)  # [N9, 9, out_dim]
            curv_zero = torch.zeros_like(outs11[:, 0:1])
            outs11 = torch.cat((curv_zero, outs11), dim=1)  # [N9, 9, out_dim]
            outs11 = outs11.view(outs11.shape[0], 11, self.output_size - 1)  # [256,9,2]
            outputs[idx11, :11, :] = outs11
        return outputs, padding_mask

