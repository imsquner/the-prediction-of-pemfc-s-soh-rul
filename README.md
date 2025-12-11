# PEMFC-Integrated GRU Prognostics Toolkit

一个用于质子交换膜燃料电池（PEMFC）堆电压预测、SOH/RUL 估计与未来无工况滚动外推的综合工具集。项目包含从数据处理、模型训练评估、跨数据集推理到可视化与报告生成的完整流水线，支持在未知未来工况条件下为工程决策提供电压趋势与寿命区间参考。

## 主要特性
- 端到端流程：原始数据清洗、特征选取、标准化、序列生成、GRU 训练与评估、可视化与报告。
- RUL/SOH 计算：基于电压序列与固定阈值的寿命外推，支持真实与预测对比及误差指标（MAE、MAPE、PHM）。
- FC2 无工况未来预测：尾段周期复用 + 微负漂移 + 窄幅 clamp + 连续性校正，自回归滚动，输出完整时间轴与未来段预测。
- 残差线性校正：推理端快速抑制跨数据集的整体偏差（slope/intercept）。
- 可视化与报告：训练曲线、全段电压对比、SOH/RUL 曲线，自动生成 JSON/文本报告与图像。

## 项目结构
仅列核心文件/目录，完整内容以仓库为准：
- `train.py`：主入口，训练、评估、滚动预测、可视化与报告生成。
- `model.py`、`data_processing.py`、`data_processors.py`：模型与数据处理组件。
- `PEMFC_Integrated_Tool.py`：集成工具（打包入口与 GUI 辅助）。
- `gui/`：界面相关组件（`pages.py`、`plot_worker.py`、`ui_components.py`）。
- `processed_results/`：处理后数据（FC1/FC2 的 npz/csv 等）。
- `train_results_paper/`：实验输出（configs/csv/images/logs/models/tables）。
- `catboost_results/`：特征重要性输出（可选）。
- `requirements.txt`：Python 依赖。

## 环境与依赖
建议使用 Python 3.10+ 与虚拟环境（Windows PowerShell 示例）：
```powershell
# 创建并启用虚拟环境（如未创建）
python -m venv .venv
& .\.venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

## 快速上手
两种常用运行模式：
1) 训练 + 评估（固定 45 轮，保存并自动加载最佳权重）
```powershell
D:/pythonfile/123/.venv/Scripts/python.exe d:/pythonfile/123/train.py
```
2) 仅评估（跳过训练，直接加载最佳模型）
```powershell
$env:EVAL_ONLY="1"; D:/pythonfile/123/.venv/Scripts/python.exe d:/pythonfile/123/train.py
```

运行结束后，完整输出位于：`train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/`。

## 数据与配置
- 默认训练数据：`processed_results/FC1/FC1_processed_*.npz`
- 跨数据集推理：`processed_results/FC2/FC2_processed_*.npz`
- 关键配置在 `train.py` 的 `ModelConfig(...)` 中，包括：
	- `sequence_length`、`selected_features`、`train/val/test` 比例
	- `gru_hidden_size/num_layers/dropout`、`learning_rate`、`epochs`（已设为 45）
	- `fixed_dt`（时间步长）、`forecast_horizon/max_steps`（滚动外推范围）
	- `soh_threshold`（失效阈值）、`rul_prediction_scale`（RUL 计算前预测缩放）

## 训练与评估输出
- 训练日志与模型：`train_results_paper/.../logs/`、`models/best_model.pth`
- 评估与预测：`csv_files/`（`predictions.csv`、`fc2_full_with_future.csv`、`future_rollout.csv`）
- 可视化：`images/`（`training_history.png`、`voltage_prediction.png`、`soh_rul_curve.png`、`fc2_full_prediction.png`）
- 报告：`configs/final_report.json`、`configs/final_report.txt`

## RUL/SOH 说明
- FC1：基于 FC1 测试集真实/预测电压序列计算 SOH 与 RUL，对比误差（示例：MAE≈52h、MAPE≈7.9%）。
- FC2：无未来工况时，默认假设“与尾段周期相似但缓慢走弱”，生成未来电压并可据此外推 RUL（可提供乐观/中性/保守多情景）。

## FC2 未来段生成策略（无工况假设）
- 周期窗口：复用 FC2 测试尾段的一个代表性周期（默认约 `3*seq_len`）。
- 微负漂移：每完成一个周期叠加线性微降（默认 `drift_per_cycle=0.015`）。
- 窄幅 clamp：对非时间特征按 P5–P95 并加极小余量做限幅，抑制漂移跑偏（时间特征不参与 clamp）。
- 连续性校正：将未来段首点与测试末点的“原始预测”对齐，消除跳变。
- 残差线性校正：对未来段应用 fc2 的偏差校正（slope/intercept），降低整体正偏差。

## 常见问题（FAQ）
- 训练何时停止？若不启用早停，本项目固定训练到 45 轮，并加载验证集最优权重（当前常见最佳在 ~epoch 40）。
- 未来段下行过缓怎么办？调高 `drift_per_cycle`（如 0.015→0.018）、收紧 clamp（如 P4–P94 或降低 `clamp_margin`），或在连续性校正后对未来段整体加一个小负偏移。
- FC2 的 RUL 是否可靠？在无工况前提下属于情景外推，建议输出区间（乐观/中性/保守）供工程取值。

## 快速复现实验
1) 准备 `processed_results/FC1` 与 `processed_results/FC2` 数据（npz/csv）。
2) 安装依赖并运行训练/评估命令。
3) 查看 `train_results_paper/.../images` 与 `configs` 下的图表与报告。
4) 如需更快衰减或更保守 RUL，微调 `drift_per_cycle`、`cycle_window_len` 与 clamp 设置，重新运行。

## 许可证与引用
本仓库未声明许可证。若用于学术/工程，请在成果中注明“PEMFC-Integrated GRU Prognostics Toolkit”。

---
如需生成 FC2 多情景（乐观/中性/保守）未来与 RUL 区间，请提出需求，我可补充实现并更新报告输出。
