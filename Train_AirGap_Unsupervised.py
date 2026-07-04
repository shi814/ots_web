#!/usr/bin/env python3
"""Unsupervised air-gap-only fine tuning.

This script starts from a stage-1 checkpoint and stage-1 prediction CSVs.
The first-stage lens prescription is treated as fixed: materials, glass
curvatures, and glass thicknesses are copied from the CSV. During training
only the air-gap regression heads are updated, and only air-layer thicknesses
are replaced before differentiable ray tracing.
"""

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import utils
from USL_Loss import USL_Loss
from data_process.add_type import generate_type_seq_from_padded
from dataset_norm import (
    convert2real_dataBGR,
    convert2real_dataSys,
    normalize_dataBGR,
    normalize_dataSys,
)
from plot_result_summary import plot_result_summary

try:
    from exports.scanlens_export_from_csv import _infer_pred_nsurf_and_offset
except Exception:
    def _infer_pred_nsurf_and_offset(vals: Sequence[float], tol: float = 1e-10) -> Tuple[int, int]:
        """
        Infer 9-surface vs 11-surface prediction CSV rows.

        Mixed prediction CSVs use a max-11 layout. For 9-surface rows, the last
        two N_bgr surfaces and CT pairs are zero padded, and CT starts after a
        6-column offset. Returns (n_surf, offset_ct).
        """
        max_n = 11
        y0 = 2
        y1 = y0 + 3 * max_n
        ct0 = y1
        ct1 = ct0 + 2 * max_n

        if len(vals) < ct1:
            if len(vals) >= 2 + 5 * 9 + 6:
                return 9, 6
            if len(vals) >= 2 + 5 * 9:
                return 9, 0
            return 11, 0

        y_pad = vals[y0 + 3 * 9: y1]
        ct_pad = vals[ct0 + 2 * 9: ct1]

        def _all_near_zero(seq: Sequence[float]) -> bool:
            return all(abs(float(x)) <= tol for x in seq)

        is_9 = _all_near_zero(y_pad) and _all_near_zero(ct_pad)
        return (9, 6) if is_9 else (11, 0)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STAGE1_ROOT = PROJECT_ROOT / "log/260521_1013/stage_1"
DEFAULT_STAGE1_CKPT = (
    DEFAULT_STAGE1_ROOT / "checkpoints/SLT_rmsfilter_on_epoch5000_bs512.pth"
)
DEFAULT_SAVE_ROOT = PROJECT_ROOT / "log/260521_1013"
STAGE2_DIR = Path("stage_2") / "airgap_unsupervised"


@dataclass
class BatchStats:
    loss_sum: float = 0.0
    rms_sum: float = 0.0
    dist_sum: float = 0.0
    efl_sum: float = 0.0
    tele_sum: float = 0.0
    count: int = 0

    def update(self, loss, metrics):
        n = int(metrics["rms"].numel())
        if n <= 0:
            return
        self.loss_sum += float(loss.detach().cpu()) * n
        self.rms_sum += float(metrics["rms"].detach().mean().cpu()) * n
        self.dist_sum += float(metrics["dist"].detach().mean().cpu()) * n
        self.efl_sum += float(metrics["loss_EFL"].detach().mean().cpu()) * n
        self.tele_sum += float(metrics["tele"].detach().mean().cpu()) * n
        self.count += n

    def as_dict(self):
        denom = max(self.count, 1)
        return {
            "loss": self.loss_sum / denom,
            "rms_mm": self.rms_sum / denom,
            "distortion": self.dist_sum / denom,
            "efl_error": self.efl_sum / denom,
            "tele_deg": self.tele_sum / denom,
            "count": self.count,
        }


class Stage1PredictionDataset(Dataset):
    """Loads stage-1 prediction CSV rows and keeps their CT as fixed geometry."""

    def __init__(self, csv_path, opt, max_rows=0):
        self.csv_path = str(csv_path)
        self.opt = opt
        self.max_seq_length = int(opt.max_seq_length)
        self.n_wl = int(opt.nWL)

        data = np.loadtxt(self.csv_path, delimiter=",", dtype=float)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if max_rows and max_rows > 0:
            data = data[: int(max_rows)]
        self.raw = data

        self.x_sys = []
        self.x_bgr = []
        self.x_type = []
        self.x_ct_fixed = []
        self.seq_lengths = []
        self.row_indices = []

        for row_idx, row in enumerate(data):
            n_surf, offset_ct = _infer_pred_nsurf_and_offset(row)
            if n_surf not in (9, 11):
                continue

            x_real = row[:2].reshape(1, 2)
            y_real = row[2 : 2 + n_surf * 3].reshape(1, n_surf, 3)
            ct_start = 2 + n_surf * 3 + offset_ct
            ct_end = 2 + n_surf * 5 + offset_ct
            ct_real = row[ct_start:ct_end].reshape(1, n_surf, 2)

            x_norm = normalize_dataSys(x_real)
            y_flat = y_real.reshape(1, -1)
            if n_surf == 9:
                y_padded = np.pad(
                    y_flat, ((0, 0), (0, (self.max_seq_length - 9) * self.n_wl)),
                    mode="constant",
                    constant_values=0,
                )
                y_norm_flat = normalize_dataBGR(y_padded)[:, : 9 * self.n_wl]
            else:
                y_norm_flat = normalize_dataBGR(y_flat)
            y_norm = y_norm_flat.reshape(1, n_surf, self.n_wl)

            y_for_type = y_norm.reshape(-1)
            if y_for_type.size < self.max_seq_length * self.n_wl:
                y_for_type = np.pad(
                    y_for_type,
                    (0, self.max_seq_length * self.n_wl - y_for_type.size),
                    mode="constant",
                    constant_values=0,
                )
            type_seq = generate_type_seq_from_padded(
                y_for_type,
                seq_len=n_surf,
                nWL=self.n_wl,
                nSurf=self.max_seq_length,
                use_pattern=True,
            )

            y_pad = np.zeros((self.max_seq_length, self.n_wl), dtype=np.float32)
            y_pad[:n_surf, :] = y_norm[0].astype(np.float32)
            ct_pad = np.zeros((self.max_seq_length, 2), dtype=np.float32)
            ct_pad[:n_surf, :] = ct_real[0].astype(np.float32)

            self.x_sys.append(x_norm[0].astype(np.float32))
            self.x_bgr.append(y_pad.reshape(-1))
            self.x_type.append(np.asarray(type_seq, dtype=np.int64))
            self.x_ct_fixed.append(ct_pad.reshape(-1))
            self.seq_lengths.append(n_surf)
            self.row_indices.append(row_idx)

        if not self.x_sys:
            raise ValueError(f"No valid 9/11-surface rows found in {self.csv_path}")

    def __len__(self):
        return len(self.x_sys)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x_sys[idx], dtype=torch.float32),
            torch.tensor(self.x_bgr[idx], dtype=torch.float32),
            torch.tensor(self.x_type[idx], dtype=torch.long),
            torch.tensor(self.seq_lengths[idx], dtype=torch.long),
            torch.tensor(self.x_ct_fixed[idx], dtype=torch.float32),
            torch.tensor(self.row_indices[idx], dtype=torch.long),
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unsupervised fine tuning of air-gap regression heads."
    )
    parser.add_argument(
        "--train_csv_path",
        default=str(DEFAULT_STAGE1_ROOT / "train_output_metrics_pred_rmsfilter_on.csv"),
    )
    parser.add_argument(
        "--val_csv_path",
        default=str(DEFAULT_STAGE1_ROOT / "val_output_metrics_pred_rmsfilter_on.csv"),
    )
    parser.add_argument("--load_name", default=str(DEFAULT_STAGE1_CKPT))
    parser.add_argument(
        "--save_path",
        default=str(DEFAULT_SAVE_ROOT),
    )
    parser.add_argument("--save_model_path", default="checkpoints")

    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--save_by_epoch", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=utils.DEFAULT_SEED, help="random seed")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional smoke-test row limit.")

    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--output_size", type=int, default=3)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--sys_dim", type=int, default=2)
    parser.add_argument("--nWL", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=11)
    parser.add_argument("--seq1", type=int, default=9)
    parser.add_argument("--seq2", type=int, default=11)

    parser.add_argument("--nRayDensity", type=int, default=11)
    parser.add_argument("--nField", type=int, default=3)
    parser.add_argument("--EPD", type=float, default=4.0)
    parser.add_argument("--max_rmsSpotR", type=float, default=0.04)
    parser.add_argument("--efl_loss_tolerance", type=float, default=0.1)
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
    parser.add_argument("--efl_first_order_weight", type=float, default=0.5)
    parser.add_argument("--efl_first_order_tolerance", type=float, default=0.1)
    parser.add_argument(
        "--efl_filter_max_loss",
        type=float,
        default=0.1,
        help="Stage-2 hard filter: only samples with loss_EFL < threshold are kept.",
    )
    rms_filter_group = parser.add_mutually_exclusive_group()
    rms_filter_group.add_argument(
        "--enable_rms_filter",
        dest="enable_rms_filter",
        action="store_true",
        help="Enable RMS hard filtering in USL_Loss (default: enabled).",
    )
    rms_filter_group.add_argument(
        "--disable_rms_filter",
        dest="enable_rms_filter",
        action="store_false",
        help="Disable RMS hard filtering in USL_Loss.",
    )
    parser.set_defaults(enable_rms_filter=True)

    efl_filter_group = parser.add_mutually_exclusive_group()
    efl_filter_group.add_argument(
        "--enable_efl_filter",
        dest="enable_efl_filter",
        action="store_true",
        help="Enable EFL hard filtering in USL_Loss (default: enabled).",
    )
    efl_filter_group.add_argument(
        "--disable_efl_filter",
        dest="enable_efl_filter",
        action="store_false",
        help="Disable EFL hard filtering in USL_Loss.",
    )
    parser.set_defaults(enable_efl_filter=True)
    parser.add_argument("--clip_grad", type=float, default=5.0)

    parser.add_argument("--air_delta_scale_mm", type=float, default=60.0)
    parser.add_argument(
        "--zero_init_air_delta",
        action="store_true",
        default=True,
        help="Zero heads_A so training starts from the fixed stage-1 air gaps.",
    )
    parser.add_argument(
        "--no_zero_init_air_delta",
        dest="zero_init_air_delta",
        action="store_false",
    )
    return parser.parse_args()


def zero_init_air_heads(model):
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for group_name in ("heads_A_9", "heads_A_11"):
        group = getattr(raw_model, group_name, None)
        if group is None:
            continue
        for head in group:
            for module in head.modules():
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        torch.nn.init.zeros_(module.bias)


def freeze_except_air_heads(model):
    trainable = []
    frozen = []
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    for name, param in raw_model.named_parameters():
        if "heads_A_9" in name or "heads_A_11" in name:
            param.requires_grad = True
            trainable.append(name)
        else:
            param.requires_grad = False
            frozen.append(name)
    print("[Freeze] fixed backbone + classification/glass heads")
    print(f"  trainable tensors: {len(trainable)}")
    print(f"  frozen tensors: {len(frozen)}")
    return trainable, frozen


def build_air_only_ct(ct_fixed, ct_pred, type_seq, seq_length, max_seq_length):
    """Copy fixed CT and replace only valid air-layer thicknesses."""
    length_mask = torch.arange(max_seq_length, device=ct_fixed.device)[None, :] < seq_length[:, None]
    air_mask = (type_seq == 0) & length_mask
    ct_out = ct_fixed.clone()
    ct_out[:, :, 1] = torch.where(air_mask, ct_pred[:, :, 1], ct_fixed[:, :, 1])
    return ct_out


def forward_air_only(opt, model, x_sys, x_bgr_flat, x_type, seq_length, ct_fixed):
    bsz = x_bgr_flat.shape[0]
    x_bgr_seq = x_bgr_flat.view(bsz, opt.max_seq_length, opt.nWL)
    ct_fixed_seq = ct_fixed.view(bsz, opt.max_seq_length, 2)
    ct_pred, padding_mask = model(
        x_sys,
        x_bgr_seq,
        x_type,
        seq_length,
        opt.epochs,
        hard=True,
        air_base_ct=ct_fixed_seq,
        air_delta_scale_mm=float(opt.air_delta_scale_mm),
    )
    ct_air_only = build_air_only_ct(
        ct_fixed_seq, ct_pred, x_type, seq_length, opt.max_seq_length
    )
    valid_surface_mask = ~padding_mask
    return ct_air_only, valid_surface_mask


def make_lens_batch(opt, x_sys, x_bgr_flat, ct_air_only):
    bsz = x_bgr_flat.shape[0]
    x_real = convert2real_dataSys(x_sys).to(dtype=torch.float64)
    bgr_real = convert2real_dataBGR(x_bgr_flat).view(
        bsz, opt.max_seq_length, opt.nWL
    ).to(dtype=torch.float64)
    ct_real = ct_air_only.to(dtype=torch.float64)
    return utils.LensBatch(x_real, bgr_real, ct_real)


def configure_loss_metadata(opt):
    opt.usl_loss_variant = "stage1_geometric_v1"
    opt.loss_EFL_formula = (
        "loss_EFL_trace=max(abs(EFL_ideal-EFL_est)/EFL_ideal-efl_loss_tolerance,0); "
        "loss_EFL_control=max(abs(EFL_est-EFL_first_order)/EFL_first_order-efl_first_order_tolerance,0); "
        "loss_EFL=loss_EFL_trace+efl_first_order_weight*loss_EFL_control"
    )
    opt.metric_source = (
        "USL_Loss multiplicative metrics with apply_hard_filter=True, "
        f"efl_filter_max_loss={float(getattr(opt, 'efl_filter_max_loss', 0.1))}"
    )
    opt.distortion_mode = getattr(opt, "distortion_mode", "zemax_ftan")
    opt.distortion_ref_angle_deg = getattr(opt, "distortion_ref_angle_deg", 0.01)
    return opt


def run_epoch(opt, loader, model, usl_loss, optimizer=None, epoch=0):
    is_train = optimizer is not None
    model.eval()
    stats = BatchStats()

    for batch in loader:
        x_sys, x_bgr, x_type, seq_length, ct_fixed, _row_idx = batch
        x_sys = x_sys.to(DEVICE)
        x_bgr = x_bgr.to(DEVICE)
        x_type = x_type.to(DEVICE)
        seq_length = seq_length.to(DEVICE)
        ct_fixed = ct_fixed.to(DEVICE)

        grad_ctx = torch.enable_grad() if is_train else torch.no_grad()
        with grad_ctx:
            ct_air_only, valid_mask = forward_air_only(
                opt, model, x_sys, x_bgr, x_type, seq_length, ct_fixed
            )
            lens_batch = make_lens_batch(opt, x_sys, x_bgr, ct_air_only)
            loss, _spot_unused, metrics = usl_loss(
                lens_batch,
                opt.max_seq_length,
                seq_length,
                valid_mask,
                epoch,
                save=1,
                apply_hard_filter=True,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    float(opt.clip_grad),
                )
                optimizer.step()

        stats.update(loss, metrics)

    return stats.as_dict()


def collect_metrics_csv(opt, loader, model, usl_loss, output_csv, output_loss_csv=None):
    rows_metrics = []
    rows_loss = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_sys, x_bgr, x_type, seq_length, ct_fixed, _row_idx = batch
            x_sys = x_sys.to(DEVICE)
            x_bgr = x_bgr.to(DEVICE)
            x_type = x_type.to(DEVICE)
            seq_length = seq_length.to(DEVICE)
            ct_fixed = ct_fixed.to(DEVICE)

            ct_air_only, valid_mask = forward_air_only(
                opt, model, x_sys, x_bgr, x_type, seq_length, ct_fixed
            )
            lens_batch = make_lens_batch(opt, x_sys, x_bgr, ct_air_only)
            _loss_unused, _spot_unused, metrics = usl_loss(
                lens_batch,
                opt.max_seq_length,
                seq_length,
                valid_mask,
                opt.epochs,
                save=1,
                apply_hard_filter=True,
            )

            x_out = metrics["X"].detach().cpu()
            n_out = metrics["N"].detach().cpu().view(metrics["N"].shape[0], -1)
            ct_out = metrics["CT"].detach().cpu().view(metrics["CT"].shape[0], -1)
            metric_tail = torch.cat(
                [
                    metrics["composite"].detach().cpu().view(-1, 1),
                    metrics["rms"].detach().cpu().view(-1, 1),
                    metrics["dist"].detach().cpu().view(-1, 1),
                    metrics["tele"].detach().cpu().view(-1, 1),
                    metrics["loss_ovlp"].detach().cpu().view(-1, 1),
                    metrics["loss_ray"].detach().cpu().view(-1, 1),
                    metrics["EFL_est"].detach().cpu().view(-1, 1),
                    metrics["EFL_ideal"].detach().cpu().view(-1, 1),
                ],
                dim=1,
            )
            loss_tail = torch.cat(
                [
                    metrics["composite"].detach().cpu().view(-1, 1),
                    metrics["rms"].detach().cpu().view(-1, 1),
                    metrics["loss_dist"].detach().cpu().view(-1, 1),
                    metrics["loss_tele"].detach().cpu().view(-1, 1),
                    metrics["loss_ovlp"].detach().cpu().view(-1, 1),
                    metrics["loss_ray"].detach().cpu().view(-1, 1),
                    metrics["loss_EFL"].detach().cpu().view(-1, 1),
                ],
                dim=1,
            )
            rows_metrics.append(torch.cat([x_out, n_out, ct_out, metric_tail], dim=1).numpy())
            rows_loss.append(torch.cat([x_out, n_out, ct_out, loss_tail], dim=1).numpy())

    out = np.concatenate(rows_metrics, axis=0)
    pd.DataFrame(out).to_csv(output_csv, header=None, index=False, encoding="utf-8")
    print(f"[Export] saved metrics CSV: {output_csv} ({out.shape[0]} rows)")

    if output_loss_csv:
        out_loss = np.concatenate(rows_loss, axis=0)
        pd.DataFrame(out_loss).to_csv(output_loss_csv, header=None, index=False, encoding="utf-8")
        print(f"[Export] saved loss CSV: {output_loss_csv} ({out_loss.shape[0]} rows)")


def generate_result_analysis(metrics_csv, output_dir, loss_csv=None):
    try:
        plot_result_summary(
            loss_csv_path=loss_csv,
            metrics_csv_path=metrics_csv,
            output_dir=output_dir,
        )
    except Exception as e:
        print(f"[Warning] result analysis failed for {metrics_csv}: {e}")


def save_checkpoint(opt, model, epoch, final=False):
    ckpt_dir = Path(opt.save_path) / opt.save_model_path
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (not final) and int(opt.save_by_epoch) > 0 and epoch % int(opt.save_by_epoch) != 0:
        return
    name = "AirGapUnsupervised_final.pth" if final else f"AirGapUnsupervised_epoch{epoch}.pth"
    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(raw_model.state_dict(), ckpt_dir / name)
    print(f"[Checkpoint] saved {ckpt_dir / name}")


def write_parameters(opt, trainable, frozen):
    for key, value in utils.infer_loss_metadata(opt).items():
        setattr(opt, key, value)
    out = Path(opt.save_path) / "parameters_airgap_unsupervised.txt"
    with open(out, "w", encoding="utf-8") as f:
        for key, value in sorted(vars(opt).items()):
            f.write(f"{key}: {value}\n")
        f.write("\n[Trainable]\n")
        for name in trainable:
            f.write(f"{name}\n")
        f.write("\n[Frozen]\n")
        for name in frozen:
            f.write(f"{name}\n")
    print(f"[Config] saved {out}")


def main():
    opt = parse_args()
    utils.set_random_seed(opt.seed)
    configure_loss_metadata(opt)
    save_root = Path(opt.save_path)
    if save_root.name != "airgap_unsupervised":
        opt.save_path = str(save_root / STAGE2_DIR)
    Path(opt.save_path).mkdir(parents=True, exist_ok=True)
    Path(opt.save_path, opt.save_model_path).mkdir(parents=True, exist_ok=True)

    print("[AirGapUnsupervised]")
    print(f"  train_csv: {opt.train_csv_path}")
    print(f"  val_csv:   {opt.val_csv_path}")
    print(f"  load:      {opt.load_name}")
    print(f"  save:      {opt.save_path}")
    print(f"  device:    {DEVICE}")

    train_ds = Stage1PredictionDataset(opt.train_csv_path, opt, max_rows=opt.max_rows)
    val_ds = Stage1PredictionDataset(opt.val_csv_path, opt, max_rows=opt.max_rows)
    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        worker_init_fn=utils.seed_worker,
        generator=utils.make_torch_generator(opt.seed, offset=30),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        worker_init_fn=utils.seed_worker,
        generator=utils.make_torch_generator(opt.seed, offset=31),
        drop_last=False,
    )

    model = utils.create_transformer_model(opt).to(DEVICE)
    if opt.zero_init_air_delta:
        zero_init_air_heads(model)
        print("[Init] zero-initialized heads_A_*; initial air gaps equal fixed stage-1 CSV.")
    trainable, frozen = freeze_except_air_heads(model)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(opt.lr),
        weight_decay=float(opt.weight_decay),
    )
    usl_loss = USL_Loss(opt).to(DEVICE)
    usl_loss.attach_loss_metadata()
    usl_loss.print_loss_metadata()
    write_parameters(opt, trainable, frozen)
    usl_loss.eval()

    log_path = Path(opt.save_path) / "log_loss_airgap_unsupervised.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            "epoch,train_loss,val_loss,train_rms_mm,val_rms_mm,"
            "train_dist,val_dist,train_efl_error,val_efl_error,train_tele_deg,val_tele_deg\n"
        )

    for epoch in range(1, int(opt.epochs) + 1):
        t0 = time.time()
        train_stats = run_epoch(opt, train_loader, model, usl_loss, optimizer, epoch=epoch)
        val_stats = run_epoch(opt, val_loader, model, usl_loss, optimizer=None, epoch=epoch)
        dt = time.time() - t0
        should_log = (epoch % 10 == 0) or (epoch == int(opt.epochs))
        if should_log:
            print(
                f"Epoch [{epoch:04d}/{opt.epochs}] {dt:.1f}s "
                f"train_loss={train_stats['loss']:.6g} val_loss={val_stats['loss']:.6g} "
                f"train_rms={train_stats['rms_mm']*1000:.3f}um val_rms={val_stats['rms_mm']*1000:.3f}um "
                f"val_efl={val_stats['efl_error']:.4f}"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{epoch},{train_stats['loss']},{val_stats['loss']},"
                    f"{train_stats['rms_mm']},{val_stats['rms_mm']},"
                    f"{train_stats['distortion']},{val_stats['distortion']},"
                    f"{train_stats['efl_error']},{val_stats['efl_error']},"
                    f"{train_stats['tele_deg']},{val_stats['tele_deg']}\n"
                )
        save_checkpoint(opt, model, epoch)

    save_checkpoint(opt, model, int(opt.epochs), final=True)
    train_metrics_csv = str(Path(opt.save_path) / "train_output_metrics_pred_airgap_unsupervised.csv")
    train_loss_csv = str(Path(opt.save_path) / "train_output_loss_pred_airgap_unsupervised.csv")
    val_metrics_csv = str(Path(opt.save_path) / "val_output_metrics_pred_airgap_unsupervised.csv")
    val_loss_csv = str(Path(opt.save_path) / "val_output_loss_pred_airgap_unsupervised.csv")

    collect_metrics_csv(
        opt,
        train_loader,
        model,
        usl_loss,
        train_metrics_csv,
        train_loss_csv,
    )
    collect_metrics_csv(
        opt,
        val_loader,
        model,
        usl_loss,
        val_metrics_csv,
        val_loss_csv,
    )
    generate_result_analysis(
        train_metrics_csv,
        str(Path(opt.save_path) / "train_analysis"),
        loss_csv=train_loss_csv,
    )
    generate_result_analysis(
        val_metrics_csv,
        str(Path(opt.save_path) / "val_analysis"),
        loss_csv=val_loss_csv,
    )


if __name__ == "__main__":
    main()
