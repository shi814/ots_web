import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from dataset_norm import convert2real_dataBGR
import utils
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

# -----------------------------
# Transformer + TFCBlock
# -----------------------------
class TFCBlock(nn.Module):
    def __init__(self, input_size, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_size, out_dim))

    def forward(self, x):
        return self.block(x)


def _softplus(x, beta=8):
    soft_sigmoid = torch.log(1 + torch.exp(beta * x)) / beta
    return soft_sigmoid
    #return F.softplus(x, beta=beta)
def _convert2real_t(t_pred, t_min, t_range, beta=0.1):
    t_real = t_min + _softplus(t_pred - t_min, beta) - _softplus(t_pred - (t_range + t_min), beta)
    return t_real
# ===============================================================
# 动态玻璃分类头（替换掉原 head_G）
# ===============================================================
class DynamicGroupClassifier(nn.Module):
    def __init__(self, d_model, maxK):
        super().__init__()
        self.fc_logits = nn.Linear(d_model,maxK)#TFCBlock(d_model,maxK)

    def forward(self, feat, group_idx, group_mask, group_pairs, group_t, tau=0.2,hard=False):
        """
        feat: [N, d_model]
        group_idx: [N]
        group_mask: [G, maxK]
        group_pairs: [G, maxK, 2]
        group_t: [G, maxK, 1]
        """
        logits = self.fc_logits(feat)  # [N, maxK]

        valid_mask = group_mask[group_idx]  # [N, maxK]
        logits = logits.masked_fill(~valid_mask, float('-inf'))

        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)
        curv_sel = torch.einsum("nk,nkm->nm", probs, group_pairs[group_idx])  # [N,2]
        thick_sel = torch.einsum("nk,nkm->nm", probs, group_t[group_idx])  # [N,1]

        return curv_sel, thick_sel

# import

# ===============================================================
# LensTransformer 替换分类头版
# ===============================================================
class LensTransformer(nn.Module):
    def __init__(self, opt,group_mask, group_c, group_t, uniq_keys):
        super().__init__()
        self.embedding = EmbeddingSeq(opt.sys_dim,opt.nWL,opt.input_size,opt.max_seq_length)
        self.pos_emb = PositionalEmbedding(opt.input_size,opt.max_seq_length)
        nhead,num_layers = opt.num_heads, opt.num_layers

        enc_layer = nn.TransformerEncoderLayer(
            d_model=opt.input_size, nhead=nhead,
            dim_feedforward=opt.hidden_size, dropout=0.1,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # === Air 面预测保持不变 ===
        self.heads_A_9 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(opt.seq1)])
        self.heads_A_11 = nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(opt.seq2)])

        # === lens替换为分类头 ===
        maxK = group_mask.size(1)
        self.class_heads_G_9 = nn.ModuleList([
            DynamicGroupClassifier(opt.input_size, maxK=maxK) for _ in range(opt.seq1)
        ])
        self.class_heads_G_11 = nn.ModuleList([
            DynamicGroupClassifier(opt.input_size, maxK=maxK) for _ in range(opt.seq2)
        ])

        self.group_mask, self.group_c, self.group_t, self.uniq_keys=group_mask, group_c, group_t, uniq_keys
        self.max_seq_len = opt.max_seq_length
        self.output_size = opt.output_size
        self.max_epoch = opt.epochs
        self.tau_start = 1
        self.tau_end = 0.01
        self.opt = opt
        self.ri_atol=1e-6
        self.Y_real_max_9 = torch.tensor([43,  60,  35,  48, 46],dtype=torch.float32).cuda()
        self.Y_real_min_9 = torch.tensor([10,   0.2,  0.1, 0.2, 1], dtype=torch.float32).cuda()
        self.Y_real_max_11 = torch.tensor([85,   50,   33,   48, 32, 32], dtype=torch.float32).cuda()
        self.Y_real_min_11 = torch.tensor([10,  0.2,  0.2,   2,  0.2,1], dtype=torch.float32).cuda()
        #现在zemax里优化，确定距离限制（物理意义）

    # ===== 根据 RI 找库组索引 =====
    def get_group_idx(self, ri_seq):
        ri_keys = torch.round(ri_seq / self.ri_atol).to(torch.long)
        match_g = (ri_keys.unsqueeze(1) == self.uniq_keys.unsqueeze(0)).all(dim=-1)
        return match_g.float().argmax(dim=1)

    def _get_tau(self, epoch):
        """
        三阶段退火:
        - 前20%: τ 从 1.0 快速降到 0.1
        - 中间60%: τ 从 0.1 缓慢降到 0.05
        - 后20%: τ 固定到 tau_end (如0.01)，建议切 hard
        """
        # decay = (self.tau_end / self.tau_start) ** (1.0 / self.max_epoch)
        # return max(self.tau_end, self.tau_start * (decay ** epoch))
        #
        progress = epoch / self.max_epoch

        if progress < 0.2:  # 阶段1：快速下降
            # exp/log插值，τ: 1.0 -> 0.1
            return self.tau_start * (0.1 / self.tau_start) ** (progress / 0.2)

        elif progress < 0.8:  # 阶段2：缓慢下降
            # τ: 0.1 -> 0.05
            return 0.1 * (0.05 / 0.1) ** ((progress - 0.2) / 0.6)

        else:  # 阶段3：保持低温，接近hard
            return self.tau_end  # 通常设成 0.01

    # ==========================================================
    # forward
    # ==========================================================
    def forward(
        self,
        sys_params,
        bgr_seq,
        type_seq,
        seq_length,
        epoch,
        hard=False,
        air_base_ct=None,
        air_delta_scale_mm=10.0,
    ):
        B, _ = sys_params.shape
        device = sys_params.device

        # === Embedding + Transformer ===
        x = self.embedding(sys_params, bgr_seq, type_seq)
        x = self.pos_emb(x)
        max_len = x.size(1)
        padding_mask = torch.arange(max_len, device=device)[None, :] >= seq_length[:, None]
        x = self.transformer(x, src_key_padding_mask=padding_mask)  # [B,L,H]

        outputs = torch.zeros(B, self.max_seq_len, self.output_size - 1, device=device)
        B, seg_len, nWL = bgr_seq.shape
        bgr_seq = bgr_seq.view(B, seg_len*nWL)
        bgr_seq = convert2real_dataBGR(bgr_seq)
        bgr_seq = bgr_seq.view(B, seg_len, nWL)

        # 曲率+厚度分类 logits & group_idx（用于 CT 多样性正则）
        maxK = self.group_mask.size(1)
        tau = self._get_tau(epoch)

        # === 9 面系统 ===
        idx9 = (seq_length == 9)
        if idx9.any():
            x9 = x[idx9, :9, :]
            type9 = type_seq[idx9, :9]
            surf9 = bgr_seq[idx9, :9]
            b_idx = idx9.nonzero(as_tuple=True)[0]
            outs9 = []
            a_idx =0
            for i in range(9):
                feat = x9[:, i, :]
                mask_A = type9[:, i] == 0
                mask_G = type9[:, i] == 1
                out_i = []

                if mask_A.any():
                    pred_A = self.heads_A_9[i](feat[mask_A])
                    # ✅ 根据当前空气间隔编号选取范围
                    if a_idx < len(self.Y_real_max_9):
                        t_min = self.Y_real_min_9[a_idx]
                        t_max = self.Y_real_max_9[a_idx]
                    else:
                        # 若超出范围，默认最后一个
                        t_min = self.Y_real_min_9[-1]
                        t_max = self.Y_real_max_9[-1]

                    t_range9 = t_max - t_min
                    if air_base_ct is not None:
                        base_t = air_base_ct[idx9, i, 1:2][mask_A]
                        delta_mm = air_delta_scale_mm * torch.tanh(pred_A / air_delta_scale_mm)
                        pred_A = torch.clamp(base_t + delta_mm, min=t_min, max=t_max)
                    else:
                        # pred_A = F.softplus(pred_A)
                        # pred_A = t_min + t_range9 * torch.sigmoid(pred_A)
                        pred_A = _convert2real_t(pred_A, t_min, t_range9)

                    out_i.append(pred_A)
                    a_idx += 1  # ✅ 下一个空气面再取下一组范

                if mask_G.any():
                    feat_G = feat[mask_G]
                    ri_G = surf9[:, i, :][mask_G]
                    group_idx = self.get_group_idx(ri_G)
                    curv_sel, thick_sel = self.class_heads_G_9[i](
                        feat_G, group_idx,
                        self.group_mask, self.group_c, self.group_t, tau,hard
                    )
                    pred_G = torch.cat([curv_sel[:, 0:1], thick_sel, curv_sel[:, 1:2]], dim=1)
                    out_i.append(pred_G)

                outs9.append(out_i)
            outs9 = [t for sub in outs9 for t in sub]
            outs9 = torch.cat(outs9, dim=1)  # [N9, 9, out_dim]
            curv_zero = torch.zeros_like(outs9[:, 0:1])
            outs9 = torch.cat((curv_zero, outs9), dim=1)  # [N9, 9, out_dim]
            outs9 = outs9.view(outs9.shape[0], 9, self.output_size - 1)  # [256,9,2]
            outputs[idx9, :9, :] = outs9

        # === 11 面系统 ===
        idx11 = (seq_length == 11)
        if idx11.any():
            x11 = x[idx11, :11, :]
            type11 = type_seq[idx11, :11]
            surf11 = bgr_seq[idx11, :11]
            b_idx = idx11.nonzero(as_tuple=True)[0]
            outs11 = []
            a_idx = 0
            for i in range(11):
                feat = x11[:, i, :]
                mask_A = type11[:, i] == 0
                mask_G = type11[:, i] == 1
                out_i = []

                if mask_A.any():
                    pred_A = self.heads_A_11[i](feat[mask_A])
                    # ✅ 根据当前空气间隔编号选取范围
                    if a_idx < len(self.Y_real_max_11):
                        t_min = self.Y_real_min_11[a_idx]
                        t_max = self.Y_real_max_11[a_idx]
                    else:
                        # 若超出范围，默认最后一个
                        t_min = self.Y_real_min_11[-1]
                        t_max = self.Y_real_max_11[-1]

                    t_range11 = t_max - t_min
                    if air_base_ct is not None:
                        base_t = air_base_ct[idx11, i, 1:2][mask_A]
                        delta_mm = air_delta_scale_mm * torch.tanh(pred_A / air_delta_scale_mm)
                        pred_A = torch.clamp(base_t + delta_mm, min=t_min, max=t_max)
                    else:
                        # pred_A = F.softplus(pred_A)
                        # pred_A = t_min + t_range11 * torch.sigmoid(pred_A)
                        pred_A = _convert2real_t(pred_A, t_min, t_range11)
                    out_i.append(pred_A)
                    a_idx += 1  # ✅ 下一个空气面再取下一组范围

                if mask_G.any():
                    feat_G = feat[mask_G]
                    ri_G = surf11[:, i, :][mask_G]
                    group_idx = self.get_group_idx(ri_G)
                    curv_sel, thick_sel = self.class_heads_G_11[i](
                        feat_G, group_idx,
                        self.group_mask, self.group_c, self.group_t, tau,hard
                    )
                    pred_G = torch.cat([curv_sel[:, 0:1], thick_sel, curv_sel[:, 1:2]], dim=1)

                    out_i.append(pred_G)
                    # outputs[idx11, :11, :][mask_G] = pred_G
                outs11.append(out_i)
            outs11 = [t for sub in outs11 for t in sub]
            outs11 = torch.cat(outs11, dim=1)  # [N9, 9, out_dim]
            curv_zero = torch.zeros_like(outs11[:, 0:1])
            outs11 = torch.cat((curv_zero, outs11), dim=1)  # [N9, 9, out_dim]
            outs11 = outs11.view(outs11.shape[0], 11, self.output_size - 1)  # [256,9,2]
            outputs[idx11, :11, :] = outs11


        return outputs, padding_mask

if __name__ == "__main__":
    # 构造混合 batch（2个系统：一个9面，一个11面）
    B = 2
    max_seq_len = 11
    seq_lengths = torch.tensor([9, 11]).cuda()

    sys_params = torch.randn(B, 3).cuda()
    surf_seq = torch.randn(B, max_seq_len, 3).cuda()
    type_seq = torch.randint(0, 2, (B, max_seq_len)).cuda()

    opt = argparse.Namespace(sys_dim=3, nWL=3, input_size=128, hidden_size=256, max_seq_length=11, output_size=4,num_heads=8,num_layers=6)

    group_mask, group_c, group_t= utils.get_OTS_CT()
    model = LensTransformer(opt, group_mask,group_c,group_t).cuda()
    out, mask = model(sys_params, surf_seq, type_seq, seq_lengths)
    print(out.shape)  # [2, 11, 3]


