'''
Description: 从 Train_model.py里分出来的一个新的尝试。原本是回归，现在变成分类！
Debug Date : 2025-11-25, SU, 采用此模型训练。记录在log里。
Status: ok to run and train!

运行方法:

    python Train_Model.py
    # 同时关闭RMS和EFL筛选
    python Train_Model.py --disable_rms_filter --disable_efl_filter

TODO：distortion不ok！原因在于EFL的计算，EFL=F# x EPD. F# 没有控制，导致比如，GT是11.5，
TODO: 然后网络预测的CT值计算实际是15，导致EFL偏差很大。但是没有在计算dist的时候考虑。
TODO: 接下类需要，增加一个优化层，把空气间隔做成自由优化。优化好的T作为GT值，再对原来的网络做一轮监督训练！
'''


import os
import utils
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import pandas as pd
from dataset_norm import load_train_data,load_validate_data,convert2real_dataSys,convert2real_dataBGR
from USL_Loss import USL_Loss
from data_process.load_dataset import TrainDataset, ValDataset
import math
from plot_result_summary import analyze_metrics_pass, plot_scatter_plots

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _rms_filter_tag(opt) -> str:
    """
    根据 RMS 筛选开关生成输出文件后缀，避免不同模式互相覆盖。
    """
    enabled = getattr(opt, "enable_rms_filter", True)
    return "rmsfilter_on" if enabled else "rmsfilter_off"

def _with_tag(filename: str, tag: str) -> str:
    """
    将 tag 插入到文件名（扩展名前）。
    e.g. log_loss.csv + tag -> log_loss_rmsfilter_on.csv
    """
    base, ext = os.path.splitext(filename)
    return f"{base}_{tag}{ext}"


def configure_stage1_loss(opt):
    opt.usl_loss_variant = "stage1_geometric_v1"
    return opt


LOSS_CSV_HEADER = (
    "epoch,train_filtered_loss,val_full_loss,val_filtered_loss,"
    "train_filtered_spot_loss,val_full_spot_loss,val_filtered_spot_loss,"
    "val_pass_rate,val_kept,val_total\n"
)


def _ensure_csv_header(path: str, header: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header)
        return

    with open(path, "r+", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line == header:
            return
        rest = f.read()
        f.seek(0)
        f.write(header)
        f.write(first_line)
        f.write(rest)
        f.truncate()


def load_initial_model_if_needed(opt, model):
    """
    可选加载一个已有 checkpoint 作为初始权重，然后继续按当前训练脚本训练。

    注意：这里不是严格 resume optimizer/scheduler 状态，只加载网络参数；
    后续 epoch、学习率调度和筛选阈值仍按本次训练从头计算。
    """
    load_name = getattr(opt, "load_name", "")
    if not load_name:
        return model

    if not os.path.exists(load_name):
        raise FileNotFoundError(f"指定的初始模型不存在: {load_name}")

    checkpoint = torch.load(load_name, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]

    # 兼容 DataParallel 保存出的 module.* key。
    if isinstance(checkpoint, dict):
        checkpoint = {
            key.replace("module.", "", 1) if key.startswith("module.") else key: value
            for key, value in checkpoint.items()
        }

    missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    print(f"[初始化模型] 已加载初始权重: {load_name}")
    if missing_keys:
        print(f"[初始化模型] missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"[初始化模型] unexpected keys: {unexpected_keys}")

    return model

# seed = 1079
# torch.manual_seed(seed)  # 固定随机种子（CPU）
# torch.cuda.manual_seed(seed)  # 为当前GPU设置
# torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
# np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
# torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
# torch.backends.cudnn.deterministic = True  # 固定网络结构


#Adjust learning rate
# def adjust_learning_rate(opt, optimizer, epoch):
#     lr = opt.lr * (opt.lr_decrease_factor ** (epoch // opt.lr_decrease_epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#     return lr

def lr_schedule_cosine_floor(epoch, max_epoch, base_lr=4e-4, min_lr=1e-6, warmup_lr=4e-6, warmup=0.05):
    """
    epoch: 当前迭代数 (0 ~ max_epoch)
    max_epoch: 总迭代数
    base_lr: warmup 结束后的最大学习率
    min_lr: 最低学习率 (余弦退火的终点)
    warmup_lr: warmup 起始学习率
    warmup: warmup 占比 (0~1)
    """
    p = epoch / max_epoch
    if p < warmup:
        # 线性 warmup
        return warmup_lr + (base_lr - warmup_lr) * p / warmup
    else:
        # 余弦退火，最终到 min_lr
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * (p - warmup) / (1 - warmup)))

def save_model(opt, epoch, net):
    # Define the name of trained model
    tag = _rms_filter_tag(opt)
    model_name = f"SLT_{tag}_epoch{epoch}_bs{opt.batch_size}.pth"
    path = os.path.join(opt.save_path, opt.save_model_path)
    save_model_path = os.path.join(path, model_name)
    # Save mode
    if torch.cuda.device_count() > 1:
       if (epoch % opt.save_by_epoch == 0):
           torch.save(net.module.state_dict(), save_model_path)
           print('The trained model is successfully saved at epoch %d' % (epoch))
    else:
        if (epoch % opt.save_by_epoch == 0):
            torch.save(net.state_dict(), save_model_path)
            print('The trained model is successfully saved at epoch %d' % (epoch))

def train(opt,train_dataloader, model, USL_loss,epoch,optim):
    model.train()
    USL_loss.train()
    Total_loss = 0
    spot_total_loss = 0
    efl_total_loss = 0.0
    efl_count = 0
    count =0
    lens_batch = utils.LensBatch(None, None, None)  # 只创建一次

    for X_sys, X_bgr,X_type, X_seg_length in train_dataloader:
        X_sys = X_sys.to(device)
        X_bgr = X_bgr.to(device)
        X_seg_length = X_seg_length.to(device)
        X_type = X_type.to(device)

        B, n_fea = X_bgr.shape
        X_bgr = X_bgr.view(B, opt.max_seq_length, opt.nWL)
        CT_data, mask = model(X_sys, X_bgr,X_type,X_seg_length,epoch)

        X_sys_real = convert2real_dataSys(X_sys)
        X_bgr = X_bgr.view(B, n_fea)
        X_bgr_real = convert2real_dataBGR(X_bgr)
        X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)

        mask = ~mask
        # 赋值给 LensBatch
        lens_batch.X = X_sys_real
        lens_batch.N_bgr = X_bgr_real
        lens_batch.CT = CT_data

        loss, loss_spot, metrics = USL_loss(
            lens_batch, opt.max_seq_length, X_seg_length, mask, epoch,
            save=1,
            apply_hard_filter=True
        )
        efl_batch = metrics["loss_EFL"].detach()
        if efl_batch.numel() > 0:
            efl_total_loss += float(efl_batch.sum().item())
            efl_count += int(efl_batch.numel())

        # 每 10 个 epoch 打印一次 Loss 构成，避免日志过密。
        if (epoch + 1) % 10 == 0 and count % 50 == 0:
            batch_efl = float(efl_batch.mean().item()) if efl_batch.numel() > 0 else float("nan")
            print(f"  Batch {count} Breakdown: Total={loss.item():.4f}, Spot={loss_spot.item():.4f}, EFL={batch_efl:.4f}, Penalty={(loss - loss_spot).item():.4f}")

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optim.step()

        count +=1

        Total_loss = Total_loss + loss.item()
        spot_total_loss = spot_total_loss + loss_spot.item()
    avg_efl_loss = efl_total_loss / efl_count if efl_count > 0 else float("nan")
    opt.last_train_efl_loss = avg_efl_loss
    return Total_loss / count, spot_total_loss / count


def validate(opt,val_dataloader,model,USL_loss,epoch):
    model.eval()
    USL_loss.eval()
    Total_loss = 0
    spot_total_loss = 0
    efl_total_loss = 0.0
    efl_count = 0
    count =0
    lens_batch = utils.LensBatch(None, None, None)  # 只创建一次

    with torch.no_grad():
         for X_sys, X_bgr,X_type,X_seg_length in val_dataloader:
             X_sys = X_sys.to(device)
             X_bgr = X_bgr.to(device)
             X_seg_length = X_seg_length.to(device)
             X_type = X_type.to(device)

             B, n_fea = X_bgr.shape
             X_bgr = X_bgr.view(B, opt.max_seq_length, opt.nWL)

             CT_data, mask = model(X_sys, X_bgr, X_type,X_seg_length,epoch,hard=True)

             X_sys_real = convert2real_dataSys(X_sys)
             X_bgr = X_bgr.view(B, n_fea)
             X_bgr_real = convert2real_dataBGR(X_bgr)
             X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)

             mask = ~mask
             # 赋值给 LensBatch
             lens_batch.X = X_sys_real
             lens_batch.N_bgr = X_bgr_real
             lens_batch.CT = CT_data
             loss, loss_spot, metrics = USL_loss(
                 lens_batch, opt.max_seq_length, X_seg_length, mask, epoch,
                 save=1,
                 apply_hard_filter=False
             )
             efl_batch = metrics["loss_EFL"].detach()
             if efl_batch.numel() > 0:
                 efl_total_loss += float(efl_batch.sum().item())
                 efl_count += int(efl_batch.numel())
             count += 1

             Total_loss = Total_loss + loss.item()
             spot_total_loss = spot_total_loss + loss_spot.item()

    avg_efl_loss = efl_total_loss / efl_count if efl_count > 0 else float("nan")
    opt.last_val_full_efl_loss = avg_efl_loss
    return Total_loss / count, spot_total_loss / count


def validate_filtered(opt, val_dataloader, model, USL_loss, epoch):
    model.eval()
    USL_loss.eval()
    composite_chunks = []
    rms_chunks = []
    efl_chunks = []
    kept_count = 0
    total_count = 0
    lens_batch = utils.LensBatch(None, None, None)
    loss_variant = getattr(opt, "usl_loss_variant", "stage1_geometric_v1")

    with torch.no_grad():
        for X_sys, X_bgr, X_type, X_seg_length in val_dataloader:
            X_sys = X_sys.to(device)
            X_bgr = X_bgr.to(device)
            X_seg_length = X_seg_length.to(device)
            X_type = X_type.to(device)

            B, n_fea = X_bgr.shape
            total_count += B
            X_bgr = X_bgr.view(B, opt.max_seq_length, opt.nWL)

            CT_data, mask = model(X_sys, X_bgr, X_type, X_seg_length, epoch, hard=True)

            X_sys_real = convert2real_dataSys(X_sys)
            X_bgr = X_bgr.view(B, n_fea)
            X_bgr_real = convert2real_dataBGR(X_bgr)
            X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)

            mask = ~mask
            lens_batch.X = X_sys_real
            lens_batch.N_bgr = X_bgr_real
            lens_batch.CT = CT_data
            _, _, metrics = USL_loss(
                lens_batch, opt.max_seq_length, X_seg_length, mask, epoch,
                save=1,
                apply_hard_filter=True
            )

            batch_kept = int(metrics["X"].shape[0])
            kept_count += batch_kept
            if batch_kept > 0:
                composite_chunks.append(metrics["composite"].detach())
                rms_chunks.append(metrics["rms"].detach())
                efl_chunks.append(metrics["loss_EFL"].detach())

    pass_rate = kept_count / total_count if total_count > 0 else 0.0
    if kept_count == 0:
        opt.last_val_filtered_efl_loss = float("nan")
        return 0.0, 0.0, pass_rate, kept_count, total_count

    composite_all = torch.cat(composite_chunks)
    rms_all = torch.cat(rms_chunks)
    efl_all = torch.cat(efl_chunks)
    eps = torch.finfo(composite_all.dtype).eps
    if loss_variant == "stage1_geometric_v1":
        filtered_loss = torch.exp(torch.mean(torch.log(composite_all + eps)))
    else:
        filtered_loss = torch.mean(composite_all)
    filtered_spot_loss = torch.mean(rms_all)
    filtered_efl_loss = torch.mean(efl_all)
    opt.last_val_filtered_efl_loss = filtered_efl_loss.item()

    return (
        filtered_loss.item(),
        filtered_spot_loss.item(),
        pass_rate,
        kept_count,
        total_count,
    )


def save_train_set_filtered_data(opt, train_dataloader, model, USL_loss, epoch):
    """
    保存最后一个epoch筛选后的训练集数据（格式与Test_Model.py一致）
    
    Args:
        opt: 配置参数
        train_dataloader: 训练数据加载器
        model: 训练好的模型
        USL_loss: USL损失函数
        epoch: 当前epoch（应该是最后一个epoch）
    """
    model.eval()
    USL_loss.eval()
    
    # 收集容器
    X_sys_params = []
    X_bgr_list = []
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
    
    lens_batch = utils.LensBatch(None, None, None)
    
    with torch.no_grad():
        for X_sys, X_bgr, X_type, X_seg_length in train_dataloader:
            X_sys = X_sys.to(device)
            X_bgr = X_bgr.to(device)
            X_seg_length = X_seg_length.to(device)
            X_type = X_type.to(device)
            
            B, n_fea = X_bgr.shape
            X_bgr_seq = X_bgr.view(B, opt.max_seq_length, opt.nWL)
            
            # 模型预测
            CT_data, mask_net = model(X_sys, X_bgr_seq, X_type, X_seg_length, epoch, hard=True)
            
            # 反归一化
            X_sys_real = convert2real_dataSys(X_sys)
            X_bgr_flat = X_bgr_seq.view(B, n_fea)
            X_bgr_real = convert2real_dataBGR(X_bgr_flat)
            X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)
            
            mask = ~mask_net
            
            # 赋值给 LensBatch
            lens_batch.X = X_sys_real
            lens_batch.N_bgr = X_bgr_real
            lens_batch.CT = CT_data
            
            # 使用USL_Loss计算指标（训练集导出仍保留硬筛选）
            loss, loss_spot, metrics = USL_loss(
                lens_batch,
                opt.max_seq_length,
                X_seg_length,
                mask,
                epoch,
                save=1,
                apply_hard_filter=True,
            )
            
            # 提取筛选后的数据（metrics中已经是筛选后的）
            X_datasel = metrics["X"]        # (nKeep, 2)
            N_datasel = metrics["N"]        # (nKeep, max_seq, nWL)
            CT_datasel = metrics["CT"]      # (nKeep, max_seq, output_size-1)
            
            # 展平
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
    
    # 汇总所有batch
    if len(X_sys_params) == 0:
        print("[警告] 没有收集到任何训练集数据（可能全部被筛选掉了）")
        return
    
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
    
    # 保存CSV（格式与Test_Model.py一致）
    x_data = torch.cat(
        (X_sys_params, X_bgr_seq, X_ct, loss_all, spot_all, dist_all, tele_all, ovlp_all, rays_all, efl_est_all, efl_ideal_all),
        dim=1,
    )
    x1_data = torch.cat(
        (X_sys_params, X_bgr_seq, X_ct, loss_all, spot_all, loss_dist_all, loss_tele_all, ovlp_all, rays_all),
        dim=1,
    )
    
    train_output = pd.DataFrame(x_data.cpu().numpy())
    train_output1 = pd.DataFrame(x1_data.cpu().numpy())
    
    tag = _rms_filter_tag(opt)
    save_name = os.path.join(opt.save_path, _with_tag("train_output_metrics_pred.csv", tag))
    save_name1 = os.path.join(opt.save_path, _with_tag("train_output_loss_pred.csv", tag))
    
    train_output.to_csv(save_name, header=None, index=False, encoding="utf-8")
    train_output1.to_csv(save_name1, header=None, index=False, encoding="utf-8")
    
    print(f"\n[保存训练集数据] Epoch {epoch+1} 筛选后的训练集数据已保存:")
    print(f"  - {save_name} ({len(train_output)} 行)")
    print(f"  - {save_name1} ({len(train_output1)} 行)")


def save_val_set_filtered_data(opt, val_dataloader, model, USL_loss, epoch):
    """
    保存最后一个epoch筛选后的验证集数据（格式与Test_Model.py一致）
    - 用于后续“验证集空气层优化 -> 筛选 -> 作为第二次监督训练的验证监督信号”
    """
    model.eval()
    USL_loss.eval()

    # 收集容器
    X_sys_params = []
    X_bgr_list = []
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

    lens_batch = utils.LensBatch(None, None, None)

    with torch.no_grad():
        for X_sys, X_bgr, X_type, X_seg_length in val_dataloader:
            X_sys = X_sys.to(device)
            X_bgr = X_bgr.to(device)
            X_seg_length = X_seg_length.to(device)
            X_type = X_type.to(device)

            B, n_fea = X_bgr.shape
            X_bgr_seq = X_bgr.view(B, opt.max_seq_length, opt.nWL)

            # 模型预测
            CT_data, mask_net = model(X_sys, X_bgr_seq, X_type, X_seg_length, epoch, hard=True)

            # 反归一化
            X_sys_real = convert2real_dataSys(X_sys)
            X_bgr_flat = X_bgr_seq.view(B, n_fea)
            X_bgr_real = convert2real_dataBGR(X_bgr_flat)
            X_bgr_real = X_bgr_real.view(B, opt.max_seq_length, opt.nWL)

            mask = ~mask_net

            lens_batch.X = X_sys_real
            lens_batch.N_bgr = X_bgr_real
            lens_batch.CT = CT_data

            # 使用USL_Loss计算指标（验证集导出不做筛选，保留全量数据）
            loss, loss_spot, metrics = USL_loss(
                lens_batch,
                opt.max_seq_length,
                X_seg_length,
                mask,
                epoch,
                save=1,
                apply_hard_filter=True,
            )

            X_datasel = metrics["X"]        # (nKeep, 2)
            N_datasel = metrics["N"]        # (nKeep, max_seq, nWL)
            CT_datasel = metrics["CT"]      # (nKeep, max_seq, output_size-1)

            N_datasel = N_datasel.view(N_datasel.shape[0], opt.max_seq_length * opt.nWL)
            CT_datasel = CT_datasel.view(CT_datasel.shape[0], opt.max_seq_length * (opt.output_size - 1))

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

    if len(X_sys_params) == 0:
        print("[警告] 没有收集到任何验证集数据（可能全部被筛选掉了）")
        return

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

    x_data = torch.cat(
        (X_sys_params, X_bgr_seq, X_ct, loss_all, spot_all, dist_all, tele_all, ovlp_all, rays_all, efl_est_all, efl_ideal_all),
        dim=1,
    )
    x1_data = torch.cat(
        (X_sys_params, X_bgr_seq, X_ct, loss_all, spot_all, loss_dist_all, loss_tele_all, ovlp_all, rays_all),
        dim=1,
    )

    val_output = pd.DataFrame(x_data.cpu().numpy())
    val_output1 = pd.DataFrame(x1_data.cpu().numpy())

    tag = _rms_filter_tag(opt)
    save_name = os.path.join(opt.save_path, _with_tag("val_output_metrics_pred.csv", tag))
    save_name1 = os.path.join(opt.save_path, _with_tag("val_output_loss_pred.csv", tag))

    val_output.to_csv(save_name, header=None, index=False, encoding="utf-8")
    val_output1.to_csv(save_name1, header=None, index=False, encoding="utf-8")

    total_val_rows = len(getattr(val_dataloader, "dataset", []))
    print(f"\n[保存验证集数据] Epoch {epoch+1} 筛选后的验证集数据已保存:")
    print(f"  - {save_name} ({len(val_output)}/{total_val_rows} rows kept)")
    print(f"  - {save_name1} ({len(val_output1)}/{total_val_rows} rows kept)")


def main(opt):
    # 准备数据
    X_train_sys, X_train_bgr,X_train_type = load_train_data()
    X_val_sys, X_val_bgr,X_val_type = load_validate_data()
    seq_lengths_train = (X_train_sys[:, 2] ).astype(int)
    seq_lengths_val = (X_val_sys[:, 2]).astype(int)
    X_train_sys = X_train_sys[:, :2]
    X_val_sys = X_val_sys[:, :2]

    # print(torch.where(Y_train.isnan()))

    train_loss_list = []
    epoch_list = []
    val_loss_list = []
    val_filtered_loss_list = []
    val_pass_rate_list = []
    train_spot_loss_list = []
    val_spot_loss_list = []
    val_filtered_spot_loss_list = []

    train_dataset = TrainDataset(X_train_sys, X_train_bgr,X_train_type,seq_lengths_train)
    seed = int(getattr(opt, "seed", utils.DEFAULT_SEED))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        worker_init_fn=utils.seed_worker,
        generator=utils.make_torch_generator(seed, offset=0),
    )
    val_dataset = ValDataset(X_val_sys, X_val_bgr,X_val_type,seq_lengths_val)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=opt.num_workers,
        worker_init_fn=utils.seed_worker,
        generator=utils.make_torch_generator(seed, offset=1),
    )

    # SL_train_dataset = TensorDataset(X_SL_train, Y_SL_train)
    # SL_train_dataloader = DataLoader(USL_train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=0,drop_last=True)

    # cudnn.benchmark = True
    # Create model
    model = utils.create_transformer_model(opt)

    # === 🚀 训练阶段开关（兼容不同模型版本）===
    # 旧版本模型可能有 set_train_stage / Base+Delta 结构；
    # 你现在“还原后”的 TransformerClass_Model 通常没有该接口，因此这里做兼容：
    # - 有 set_train_stage：沿用旧逻辑
    # - 没有：跳过（默认训练全部参数），避免 AttributeError
    if hasattr(model, "set_train_stage"):
        # stage=1:   只练 Base，冻结 Delta
        # stage=1.5: Base + Delta 一起练
        # stage=2:   固定 Base，只练 Delta（监督微调用）
        model.set_train_stage(opt.train_stage)
        print(f"[Stage] model.set_train_stage({opt.train_stage})")
    else:
        print(f"[Stage] 当前模型不支持 set_train_stage，已跳过（opt.train_stage={opt.train_stage} 仅用于日志/目录分流）")

    model = load_initial_model_if_needed(opt, model)

    if torch.cuda.device_count() > 1:  # multi GPU
        model = nn.DataParallel(model)
        model = model.to(device)
    else:
        model = model.to(device)

    for param_tensor in model.state_dict():  # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
        print(param_tensor, '\t',model.state_dict()[param_tensor].size())

    # Optimizer and Loss function
    # === 🚀 只训练 requires_grad=True 的参数，提升稳定性 ===
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.Adam(trainable_params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
    # 仅第一次训练切回旧版乘法复合损失；优化/测试等其它脚本不设置该开关，保持现状。
    configure_stage1_loss(opt)
    # 导出阶段是否筛选，由 USL_Loss 内部 save=1 逻辑单独决定。
    USL_loss = USL_Loss(opt).to(device)  # define USL loss
    USL_loss.attach_loss_metadata()
    USL_loss.print_loss_metadata()
    utils.record_parameters(opt)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=opt.epochs, eta_min=1e-6)

    start_train_time = time.time()
    # torch.autograd.set_detect_anomaly(True) #检测是否反向传播

    tag = _rms_filter_tag(opt)
    loss_csv_path = os.path.join(opt.save_path, _with_tag('log_loss.csv', tag))
    _ensure_csv_header(loss_csv_path, LOSS_CSV_HEADER)
    loss_csv = open(loss_csv_path, 'a+', encoding="utf-8")   # define save file
    time_csv = open(os.path.join(opt.save_path, _with_tag('log_time.csv', tag)), 'a+')   # define save file

    # start epoch
    for epoch in range(opt.epochs):
        start_epoch_time = time.time()
        train_loss,train_spot_loss = train(opt,train_dataloader, model, USL_loss,epoch,optim) # 自定义训练函数，更新网络，返回loss train_spot_loss
        val_loss,val_spot_loss= validate(opt,val_dataloader,model,USL_loss,epoch)  # 自定义验证函数，不更新网络，返回loss val_spot_loss

        val_filtered_loss, val_filtered_spot_loss, val_pass_rate, val_kept, val_total = validate_filtered(
            opt, val_dataloader, model, USL_loss, epoch
        )

        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time
        if (epoch + 1) % 10 == 0:
            print(
                "Epoch [%02d/%02d], Time:%.9f, Train Loss: %.9f, Val Full Loss: %.9f, Val Filtered Loss: %.9f, Val Pass Rate: %.4f (%d/%d), Train Spot Loss: %.9f, Val Full Spot Loss: %.9f, Val Filtered Spot Loss: %.9f"
                % (
                    epoch + 1,
                    opt.epochs,
                    epoch_time,
                    train_loss,
                    val_loss,
                    val_filtered_loss,
                    val_pass_rate,
                    val_kept,
                    val_total,
                    train_spot_loss,
                    val_spot_loss,
                    val_filtered_spot_loss,
                ))
            print(
                "  EFL Loss: Train=%.9f, Val Full=%.9f, Val Filtered=%.9f"
                % (
                    getattr(opt, "last_train_efl_loss", float("nan")),
                    getattr(opt, "last_val_full_efl_loss", float("nan")),
                    getattr(opt, "last_val_filtered_efl_loss", float("nan")),
                )
            )
        save_model(opt, (epoch + 1), model)
        scheduler.step()
        # lr = adjust_learning_rate(opt, optim, (epoch + 1)) # 更新学习率
        # lr = lr_schedule_cosine_floor(epoch, opt.epochs)
        #
        # # 更新 optimizer 中的学习率
        # for param_group in optim.param_groups:
        #     param_group["lr"] = lr

        # record loss，存储
        if (epoch + 1) % 10 == 0:
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            val_filtered_loss_list.append(val_filtered_loss)
            val_pass_rate_list.append(val_pass_rate)
            epoch_list.append(epoch + 1)
            train_spot_loss_list.append(train_spot_loss)
            val_spot_loss_list.append(val_spot_loss)
            val_filtered_spot_loss_list.append(val_filtered_spot_loss)
            utils.record_loss(
                loss_csv,
                epoch,
                train_loss,
                val_loss,
                train_spot_loss,
                val_spot_loss,
                val_filtered_loss=val_filtered_loss,
                val_filtered_spot_loss=val_filtered_spot_loss,
                val_pass_rate=val_pass_rate,
                val_kept=val_kept,
                val_total=val_total,
            )
        
        # 在最后一个epoch保存筛选后的训练集数据
        if (epoch + 1) == opt.epochs:
            print(f"\n[保存训练集数据] 开始保存最后一个epoch ({epoch+1}) 筛选后的训练集数据...")
            save_train_set_filtered_data(opt, train_dataloader, model, USL_loss, epoch)
            print(f"\n[保存验证集数据] 开始保存最后一个epoch ({epoch+1}) 筛选后的验证集数据...")
            save_val_set_filtered_data(opt, val_dataloader, model, USL_loss, epoch)

    end_train_time = time.time()
    train_time = (end_train_time - start_train_time)/3600
    utils.record_time(time_csv, train_time)
    print(" Train_Time:%.9f" % train_time)
    return (
        epoch_list,
        train_loss_list,
        val_loss_list,
        val_filtered_loss_list,
        val_pass_rate_list,
        train_spot_loss_list,
        val_spot_loss_list,
        val_filtered_spot_loss_list,
    )


if __name__ == "__main__":
    # define hyper-parameters
    opt = utils.set_parser()
    utils.set_random_seed(opt.seed)
    configure_stage1_loss(opt)

    # 关闭筛选时：训练结果写到单独子目录，避免覆盖开启筛选的结果
    if not getattr(opt, "enable_rms_filter", True):
        if os.path.basename(opt.save_path) != "rmsfilter_off":
            opt.save_path = os.path.join(opt.save_path, "rmsfilter_off")

    # === 根据训练阶段自动分流文件夹 ===
    stage_dir = f"stage_{opt.train_stage}"
    opt.save_path = os.path.join(opt.save_path, stage_dir)

    # Create folders
    save_folder = opt.save_path
    save_model_folder = os.path.join(opt.save_path, opt.save_model_path)
    utils.check_path(opt.save_path)
    utils.check_path(save_model_folder)
    # loss_csv = utils.save_loss_path(opt)
    # time_csv = utils.save_time_path(opt)

    # 显示筛选配置
    print("\n" + "="*60)
    print(f"训练配置 - [Stage {opt.train_stage}]")
    print("="*60)
    rms_status = "开启" if getattr(opt, "enable_rms_filter", True) else "关闭"
    efl_status = "开启" if getattr(opt, "enable_efl_filter", True) else "关闭"
    print(f"  当前阶段: Stage {opt.train_stage}")
    print(f"  RMS 筛选: {rms_status}")
    print(f"  EFL 筛选: {efl_status}")
    print(f"  input_size (d_model): {opt.input_size}")
    print(f"  hidden_size (ffn): {opt.hidden_size}")
    print(f"  num_heads: {opt.num_heads}")
    if int(opt.input_size) % int(opt.num_heads) == 0:
        print(f"  head_dim: {int(opt.input_size) // int(opt.num_heads)}")
    else:
        print(f"  head_dim: 非整数 ({opt.input_size}/{opt.num_heads})")
    efl_fo_status = "开启" if getattr(opt, "enable_efl_first_order_control", True) else "关闭"
    print(f"  一阶EFL控制(ABCD): {efl_fo_status}")
    print(f"  一阶EFL权重: {float(getattr(opt, 'efl_first_order_weight', 0.5))}")
    print(f"  一阶EFL容差: {float(getattr(opt, 'efl_first_order_tolerance', 0.1))}")
    print(f"  保存路径: {opt.save_path}")
    print("="*60 + "\n")

    (
        epochs,
        train_loss,
        val_loss,
        val_filtered_loss,
        val_pass_rate,
        train_spot,
        val_spot,
        val_filtered_spot,
    ) = main(opt) #,train_spot, val_spot

    train, = plt.plot(epochs, train_loss, color='red', linewidth=1)
    val_filtered, = plt.plot(epochs, val_filtered_loss, color='blue', linewidth=1)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend(
        [train, val_filtered],
        [
            'train_filtered_loss',
            'val_filtered_loss',
        ],
        loc='upper right',
    )
    tag = _rms_filter_tag(opt)
    plt.savefig(os.path.join(opt.save_path, _with_tag('SLT_USL_loss.png', tag)))
    plt.close()

    train_metrics_csv = os.path.join(opt.save_path, _with_tag("train_output_metrics_pred.csv", tag))
    val_metrics_csv = os.path.join(opt.save_path, _with_tag("val_output_metrics_pred.csv", tag))

    if os.path.exists(train_metrics_csv):
        train_out_dir = os.path.join(opt.save_path, "train_analysis")
        plot_scatter_plots(train_metrics_csv, train_out_dir)
        analyze_metrics_pass(train_metrics_csv, train_out_dir)
    if os.path.exists(val_metrics_csv):
        val_out_dir = os.path.join(opt.save_path, "val_analysis")
        plot_scatter_plots(val_metrics_csv, val_out_dir)
        analyze_metrics_pass(val_metrics_csv, val_out_dir)

## Demo: 调用 draw_loss函数来美化 loss plot.
