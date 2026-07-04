# ScanLens

ScanLens 是一个基于 PyTorch 的扫描镜头（Scan Lens）无监督建模项目，当前主流程为：

1. `Stage 1`：训练主 Transformer，直接预测系统结构参数并用光学指标做无监督优化。  
2. `Stage 2`：从 Stage 1 结果出发，仅微调空气层（air-gap）相关预测头。  
3. `Test`：测试模型并自动执行结果汇总、GT 匹配、玻璃匹配、ZMX/JSON 导出。

---

## 1) 环境与运行

建议使用 `scanlens` 环境：

```powershell
conda activate scanlens
```

如果 Windows 下遇到编码问题，可直接调用环境内 Python：

```powershell
D:\projects\Miniconda3\envs\scanlens\python.exe <script>.py
```

---

## 2) 数据配置

默认数据路径定义在 `dataset_norm.py`，支持通过环境变量覆盖：

```text
SCANLENS_ORIGIN_CSV   # 归一化参考全集
SCANLENS_TRAIN_CSV    # 训练集
SCANLENS_VAL_CSV      # 验证集
SCANLENS_TEST_CSV     # 测试集
```

当前默认文件（未设置环境变量时）：

```text
data/surf10_12_ul_1104.csv
data/scan_lens_train_ul_20260512.csv
data/scan_lens_val_ul_20260512.csv
data/scan_lens_test_ul_20260512.csv
```

---

## 3) Stage 1 训练（主模型）

脚本：`Train_Model.py`

基础训练：

```powershell
python Train_Model.py --epochs 5000 --save_by_epoch 1000
```

快速 smoke test：

```powershell
python Train_Model.py --epochs 1 --save_by_epoch 1
```

关键行为（按当前代码）：

- 默认开启 RMS/EFL 硬筛选（可用 `--disable_rms_filter --disable_efl_filter` 关闭）。  
- 结果会自动按训练阶段分流到 `stage_<train_stage>` 子目录（默认 `stage_1`）。  
- 最后一个 epoch 会导出筛选后的 `train/val` 预测 CSV，并生成 `train_analysis` / `val_analysis`。

常见输出（以筛选开启为例）：

```text
log/.../stage_1/
  checkpoints/SLT_rmsfilter_on_epoch*.pth
  log_loss_rmsfilter_on.csv
  log_time_rmsfilter_on.csv
  train_output_metrics_pred_rmsfilter_on.csv
  train_output_loss_pred_rmsfilter_on.csv
  val_output_metrics_pred_rmsfilter_on.csv
  val_output_loss_pred_rmsfilter_on.csv
  train_analysis/
  val_analysis/
```

---

## 4) Stage 2 无监督空气层微调

脚本：`Train_AirGap_Unsupervised.py`

用途（按当前实现）：

- 从 Stage 1 checkpoint + Stage 1 预测 CSV 出发。  
- 冻结 backbone 与非空气层头，只训练 `heads_A_9 / heads_A_11`。  
- 使用固定的 Stage 1 几何作为基底，仅替换有效空气层厚度参与追迹与损失。

运行示例：

```powershell
python Train_AirGap_Unsupervised.py --epochs 5000 --save_by_epoch 1000
```

快速 smoke test：

```powershell
python Train_AirGap_Unsupervised.py --epochs 1 --save_by_epoch 1 --max_rows 64 --save_path "log/airgap_smoke"
```

常见输出：

```text
log/.../stage_2/airgap_unsupervised/
  checkpoints/AirGapUnsupervised_epoch*.pth
  checkpoints/AirGapUnsupervised_final.pth
  log_loss_airgap_unsupervised.csv
  parameters_airgap_unsupervised.txt
  train_output_metrics_pred_airgap_unsupervised.csv
  train_output_loss_pred_airgap_unsupervised.csv
  val_output_metrics_pred_airgap_unsupervised.csv
  val_output_loss_pred_airgap_unsupervised.csv
  train_analysis/
  val_analysis/
```

---

## 5) 测试与后处理

脚本：`Test_Model.py`

基础用法：

```powershell
python Test_Model.py --load_name "path/to/checkpoint.pth" --save_path "log/test_run"
```

### 5.1 两种测试路径（自动判断）

- **Stage 1 checkpoint**：直接测试。  
- **Stage 2 AirGapUnsupervised checkpoint**：自动启用“空气层残差测试模式”，先加载 Stage 1 基底，再叠加空气层 delta。

### 5.2 默认后处理链路

测试流程默认会执行以下步骤：

1. 保存测试 CSV（metrics/loss）。  
2. 按 `--test_efl_error_threshold`（默认 `0.1`）过滤结果。  
3. 生成 `test_result_summary_*` 图表与报告。  
4. 生成 GT 匹配报告（Markdown + selected rows CSV）。  
5. 执行玻璃匹配。  
6. 按 GT 报告选中的行导出预测与 GT 的 ZMX/JSON。

若只想做轻量测试，可设置：

```powershell
set SCANLENS_TEST_LIGHTWEIGHT=1
python Test_Model.py --load_name "..."
```

这会跳过结果汇总、GT 匹配、玻璃匹配和导出步骤。

### 5.3 常用参数

```powershell
# 调整 EFL 过滤阈值
python Test_Model.py --test_efl_error_threshold 0.05

# 关闭 ZMX/JSON 导出
python Test_Model.py --no_export_zmx_json

# 手动限制导出数量（仅在批量导出时生效）
python Test_Model.py --export_max_rows 20

# 指定导出单行（0-based）
python Test_Model.py --export_row 19
```

---

## 6) 关键脚本

```text
Train_Model.py
  Stage 1 无监督训练主流程。

Train_AirGap_Unsupervised.py
  Stage 2 空气层专用无监督微调。

Test_Model.py
  测试 + EFL筛选 + 汇总 + GT匹配 + 玻璃匹配 + ZMX/JSON导出。

USL_Loss.py
  光线追迹指标、损失和硬筛选逻辑。

dataset_norm.py
  数据加载、归一化和反归一化；支持环境变量切换数据源。

plot_result_summary.py
  结果统计图与阈值报告生成。

glass_matching/
  玻璃名匹配工具。

exports/
  JSON/ZMX 导出工具。
```

---

## 7) 当前目录约定（建议）

- 训练/测试输出统一放在 `log/<run_id>/...`。  
- Stage 1 推荐保持在 `log/<run_id>/stage_1`。  
- Stage 2 推荐保持在 `log/<run_id>/stage_2/airgap_unsupervised`。  
- 导出结果默认跟随测试输出目录，便于追溯同一次实验。

---

## 8) 备注

- `models/BP_Model.py`、`models/Stacking_BP_Model.py`、`models/Transformer_Model.py` 为历史实现，当前主流程使用 `models/TransformerClass_Model.py`。  
- `lens_visualization/` 为可选工具，不是训练/测试必需依赖。  
