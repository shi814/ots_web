import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
from dataset_norm import convert2real_dataBGR
import utils

# Use CUDA when available, otherwise fall back to CPU (e.g. Streamlit Cloud).
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        max_seq_len: 最大面数 (比如 13)
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


class DynamicGroupRegressor(nn.Module):
    """Continuously regress glass geometry inside each RI group's library range."""

    def __init__(self, d_model, maxK):
        super().__init__()
        self.fc_params = nn.Linear(d_model, 3)

    def forward(self, feat, group_idx, group_mask, group_pairs, group_t, tau=0.2, hard=False):
        del tau, hard
        raw = torch.sigmoid(self.fc_params(feat))
        valid = group_mask[group_idx]
        curv = group_pairs[group_idx]
        thick = group_t[group_idx]

        inf = torch.full_like(curv, float("inf"))
        neg_inf = torch.full_like(curv, float("-inf"))
        curv_min = torch.where(valid.unsqueeze(-1), curv, inf).amin(dim=1)
        curv_max = torch.where(valid.unsqueeze(-1), curv, neg_inf).amax(dim=1)

        thick_inf = torch.full_like(thick, float("inf"))
        thick_neg_inf = torch.full_like(thick, float("-inf"))
        thick_min = torch.where(valid.unsqueeze(-1), thick, thick_inf).amin(dim=1)
        thick_max = torch.where(valid.unsqueeze(-1), thick, thick_neg_inf).amax(dim=1)

        curv_pred = curv_min + raw[:, :2] * (curv_max - curv_min)
        thick_pred = thick_min + raw[:, 2:3] * (thick_max - thick_min)
        return curv_pred, thick_pred

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
        seq_lengths_raw = getattr(opt, "seq_lengths", "7,9,11,13")
        self.supported_seq_lengths = tuple(
            int(x) for x in str(seq_lengths_raw).split(",") if str(x).strip()
        )
        self.heads_A_by_len = nn.ModuleDict({
            str(seq_len): nn.ModuleList([TFCBlock(opt.input_size, 1) for _ in range(seq_len)])
            for seq_len in self.supported_seq_lengths
        })
        self.heads_A_7 = self.heads_A_by_len["7"]
        self.heads_A_13 = self.heads_A_by_len["13"]

        # Glass geometry uses either the main classification head or the
        # bounded continuous-regression ablation head.
        maxK = group_mask.size(1)
        self.glass_head_mode = str(
            getattr(opt, "glass_head_mode", "classification")
        ).lower()
        glass_head_cls = (
            DynamicGroupRegressor
            if self.glass_head_mode == "regression"
            else DynamicGroupClassifier
        )
        self.class_heads_G_by_len = nn.ModuleDict({
            str(seq_len): nn.ModuleList([
                glass_head_cls(opt.input_size, maxK=maxK) for _ in range(seq_len)
            ])
            for seq_len in self.supported_seq_lengths
        })
        self.class_heads_G_7 = self.class_heads_G_by_len["7"]
        self.class_heads_G_13 = self.class_heads_G_by_len["13"]

        self.group_mask, self.group_c, self.group_t, self.uniq_keys=group_mask, group_c, group_t, uniq_keys
        self.max_seq_len = opt.max_seq_length
        self.output_size = opt.output_size
        self.max_epoch = opt.epochs
        self.tau_start = 1
        self.tau_end = 0.01
        self.opt = opt
        self.ri_atol = float(os.environ.get("SCANLENS_RI_ATOL", "1e-5"))
        self.Y_real_max_by_len = {
            7: torch.tensor([43, 60, 35, 48], dtype=torch.float32).to(_DEVICE),
            9: torch.tensor([41, 13, 13, 13, 7], dtype=torch.float32).to(_DEVICE),
            11: torch.tensor([44, 13, 8, 13, 9, 57], dtype=torch.float32).to(_DEVICE),
            13: torch.tensor([85, 50, 33, 48, 32, 32, 32], dtype=torch.float32).to(_DEVICE),
        }
        self.Y_real_min_by_len = {
            7: torch.tensor([10, 0.2, 0.1, 0.2], dtype=torch.float32).to(_DEVICE),
            9: torch.tensor([15, 1, 0.2, 1, 0.1], dtype=torch.float32).to(_DEVICE),
            11: torch.tensor([39, 1, 0.2, 1, 0.2, 49], dtype=torch.float32).to(_DEVICE),
            13: torch.tensor([10, 0.2, 0.2, 2, 0.2, 1, 0.2], dtype=torch.float32).to(_DEVICE),
        }
        self.Y_real_max_7 = self.Y_real_max_by_len[7]
        self.Y_real_min_7 = self.Y_real_min_by_len[7]
        self.Y_real_max_13 = self.Y_real_max_by_len[13]
        self.Y_real_min_13 = self.Y_real_min_by_len[13]
        #现在zemax里优化，确定距离限制（物理意义）

    # ===== 根据 RI 找库组索引 =====
    def get_group_idx(self, ri_seq):
        ri_keys = torch.round(ri_seq / self.ri_atol).to(torch.long)
        match_g = (ri_keys.unsqueeze(1) == self.uniq_keys.unsqueeze(0)).all(dim=-1)
        has_match = match_g.any(dim=1)
        if not bool(has_match.all()):
            missing = ri_seq[~has_match].detach().cpu().numpy()
            raise ValueError(f"RI group not found in material library. Missing RI examples: {missing[:5]}")
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

    def _forward_one_length(
        self,
        x,
        bgr_seq,
        type_seq,
        seq_length,
        outputs,
        cur_len,
        tau,
        hard,
        air_base_ct=None,
        air_delta_scale_mm=10.0,
    ):
        idx = seq_length == cur_len
        if not idx.any():
            return

        x_cur = x[idx, :cur_len, :]
        type_cur = type_seq[idx, :cur_len]
        surf_cur = bgr_seq[idx, :cur_len]
        heads_A = self.heads_A_by_len[str(cur_len)]
        class_heads_G = self.class_heads_G_by_len[str(cur_len)]
        y_real_max = self.Y_real_max_by_len[cur_len]
        y_real_min = self.Y_real_min_by_len[cur_len]
        outs = []
        a_idx = 0

        for i in range(cur_len):
            feat = x_cur[:, i, :]
            mask_A = type_cur[:, i] == 0
            mask_G = type_cur[:, i] == 1
            out_i = []

            if mask_A.any():
                pred_A = heads_A[i](feat[mask_A])
                if a_idx < len(y_real_max):
                    t_min = y_real_min[a_idx]
                    t_max = y_real_max[a_idx]
                else:
                    t_min = y_real_min[-1]
                    t_max = y_real_max[-1]

                t_range = t_max - t_min
                if air_base_ct is not None:
                    base_t = air_base_ct[idx, i, 1:2][mask_A]
                    delta_mm = air_delta_scale_mm * torch.tanh(pred_A / air_delta_scale_mm)
                    pred_A = torch.clamp(base_t + delta_mm, min=t_min, max=t_max)
                else:
                    pred_A = _convert2real_t(pred_A, t_min, t_range)

                out_i.append(pred_A)
                a_idx += 1

            if mask_G.any():
                feat_G = feat[mask_G]
                ri_G = surf_cur[:, i, :][mask_G]
                group_idx = self.get_group_idx(ri_G)
                curv_sel, thick_sel = class_heads_G[i](
                    feat_G, group_idx,
                    self.group_mask, self.group_c, self.group_t, tau, hard
                )
                pred_G = torch.cat([curv_sel[:, 0:1], thick_sel, curv_sel[:, 1:2]], dim=1)
                out_i.append(pred_G)

            outs.append(out_i)

        outs = [t for sub in outs for t in sub]
        outs = torch.cat(outs, dim=1)
        curv_zero = torch.zeros_like(outs[:, 0:1])
        outs = torch.cat((curv_zero, outs), dim=1)
        outs = outs.view(outs.shape[0], cur_len, self.output_size - 1)
        outputs[idx, :cur_len, :] = outs

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

        for cur_len in (9, 11):
            self._forward_one_length(
                x,
                bgr_seq,
                type_seq,
                seq_length,
                outputs,
                cur_len,
                tau,
                hard,
                air_base_ct=air_base_ct,
                air_delta_scale_mm=air_delta_scale_mm,
            )

        # === 7 / 13 面系统 ===
        idx7 = (seq_length == 7)
        if idx7.any():
            x7 = x[idx7, :7, :]
            type7 = type_seq[idx7, :7]
            surf7 = bgr_seq[idx7, :7]
            outs7 = []
            a_idx = 0
            for i in range(7):
                feat = x7[:, i, :]
                mask_A = type7[:, i] == 0
                mask_G = type7[:, i] == 1
                out_i = []

                if mask_A.any():
                    pred_A = self.heads_A_7[i](feat[mask_A])
                    if a_idx < len(self.Y_real_max_7):
                        t_min = self.Y_real_min_7[a_idx]
                        t_max = self.Y_real_max_7[a_idx]
                    else:
                        t_min = self.Y_real_min_7[-1]
                        t_max = self.Y_real_max_7[-1]

                    t_range7 = t_max - t_min
                    if air_base_ct is not None:
                        base_t = air_base_ct[idx7, i, 1:2][mask_A]
                        delta_mm = air_delta_scale_mm * torch.tanh(pred_A / air_delta_scale_mm)
                        pred_A = torch.clamp(base_t + delta_mm, min=t_min, max=t_max)
                    else:
                        pred_A = _convert2real_t(pred_A, t_min, t_range7)

                    out_i.append(pred_A)
                    a_idx += 1

                if mask_G.any():
                    feat_G = feat[mask_G]
                    ri_G = surf7[:, i, :][mask_G]
                    group_idx = self.get_group_idx(ri_G)
                    curv_sel, thick_sel = self.class_heads_G_7[i](
                        feat_G, group_idx,
                        self.group_mask, self.group_c, self.group_t, tau, hard
                    )
                    pred_G = torch.cat([curv_sel[:, 0:1], thick_sel, curv_sel[:, 1:2]], dim=1)
                    out_i.append(pred_G)

                outs7.append(out_i)
            outs7 = [t for sub in outs7 for t in sub]
            outs7 = torch.cat(outs7, dim=1)
            curv_zero = torch.zeros_like(outs7[:, 0:1])
            outs7 = torch.cat((curv_zero, outs7), dim=1)
            outs7 = outs7.view(outs7.shape[0], 7, self.output_size - 1)
            outputs[idx7, :7, :] = outs7

        idx13 = (seq_length == 13)
        if idx13.any():
            x13 = x[idx13, :13, :]
            type13 = type_seq[idx13, :13]
            surf13 = bgr_seq[idx13, :13]
            outs13 = []
            a_idx = 0
            for i in range(13):
                feat = x13[:, i, :]
                mask_A = type13[:, i] == 0
                mask_G = type13[:, i] == 1
                out_i = []

                if mask_A.any():
                    pred_A = self.heads_A_13[i](feat[mask_A])
                    if a_idx < len(self.Y_real_max_13):
                        t_min = self.Y_real_min_13[a_idx]
                        t_max = self.Y_real_max_13[a_idx]
                    else:
                        t_min = self.Y_real_min_13[-1]
                        t_max = self.Y_real_max_13[-1]

                    t_range13 = t_max - t_min
                    if air_base_ct is not None:
                        base_t = air_base_ct[idx13, i, 1:2][mask_A]
                        delta_mm = air_delta_scale_mm * torch.tanh(pred_A / air_delta_scale_mm)
                        pred_A = torch.clamp(base_t + delta_mm, min=t_min, max=t_max)
                    else:
                        pred_A = _convert2real_t(pred_A, t_min, t_range13)
                    out_i.append(pred_A)
                    a_idx += 1

                if mask_G.any():
                    feat_G = feat[mask_G]
                    ri_G = surf13[:, i, :][mask_G]
                    group_idx = self.get_group_idx(ri_G)
                    curv_sel, thick_sel = self.class_heads_G_13[i](
                        feat_G, group_idx,
                        self.group_mask, self.group_c, self.group_t, tau, hard
                    )
                    pred_G = torch.cat([curv_sel[:, 0:1], thick_sel, curv_sel[:, 1:2]], dim=1)

                    out_i.append(pred_G)
                outs13.append(out_i)
            outs13 = [t for sub in outs13 for t in sub]
            outs13 = torch.cat(outs13, dim=1)
            curv_zero = torch.zeros_like(outs13[:, 0:1])
            outs13 = torch.cat((curv_zero, outs13), dim=1)
            outs13 = outs13.view(outs13.shape[0], 13, self.output_size - 1)
            outputs[idx13, :13, :] = outs13

        return outputs, padding_mask

if __name__ == "__main__":
    # 构造混合 batch（2个系统：一个9面，一个11面）
    B = 2
    max_seq_len = 13
    seq_lengths = torch.tensor([7, 13]).cuda()

    sys_params = torch.randn(B, 3).cuda()
    surf_seq = torch.randn(B, max_seq_len, 3).cuda()
    type_seq = torch.randint(0, 2, (B, max_seq_len)).cuda()

    opt = argparse.Namespace(sys_dim=3, nWL=3, input_size=128, hidden_size=256, max_seq_length=13, output_size=4,num_heads=8,num_layers=6, seq1=7, seq2=13)

    group_mask, group_c, group_t, uniq_keys = utils.get_OTS_CT()
    model = LensTransformer(opt, group_mask, group_c, group_t, uniq_keys).cuda()
    out, mask = model(sys_params, surf_seq, type_seq, seq_lengths)
    print(out.shape)  # [2, 13, 3]


