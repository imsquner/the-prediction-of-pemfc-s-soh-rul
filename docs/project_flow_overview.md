# PEMFC 寿命预测软件全流程说明

> 本文档总结当前项目的端到端流程、关键脚本、前端交互以及模型与数据处理细节。无需修改现有代码即可阅读使用。

## 1. 总体目标
- 基于 GRU 的 PEMFC 寿命预测：用 FC1 的处理数据训练，在 FC1 测试集验证精度，并将模型用于 FC2 后段电压预测与 RUL 估计（阈值 96% V_initial）。
- 前端一键化：CatBoost 特征重要性分析 → 原始数据可视化 → 数据处理滤波 → GRU 训练与预测对比，全程日志与进度条提示。

## 2. 关键目录与产物
- 原始数据：`data/FC1_Ageing_part*.csv`，`datatest/FC2_Ageing_part*.csv`
- 原始合并可视化输出：`visualization/truedata/FC{1,2}_merged_YYYYMMDD_HHMMSS.csv`
- 处理结果：`processed_results/FC{1,2}/FC{1,2}_processed_YYYYMMDD_HHMMSS.(csv|npz)`
- 特征重要性：`catboost_results/feature_importance_results_*.csv` + 对应图表/报告
- 训练输出（GRU）：`train_results_paper/<experiment>/` 下的 `models/`、`csv_files/predictions.csv`（含 FC1/FC2 dataset 列）、`images/`、`tables/`
- 日志：`train_results_paper/<experiment>/logs/`（训练）、前端 MonitorPanel 实时日志

## 3. 前端交互（`PEMFC_Integrated_Tool.py` + `gui/`）
- 导航栏三页：
  1) 特征重要性页（`FeatureImportancePage`）
     - 按钮：运行 CatBoost 分析（调用 `pemfc_catboost_analysis.py`），生成 CSV+图表；生成重要性图表（使用最新 CSV）。
     - 可配置显示 Top-N，默认 5，日志提示中文特征名。
  2) 原始数据处理页（`DataProcessingPage`）
     - 合并并可视化原始数据：合并 FC1/FC2 源 CSV → `visualization/truedata`，绘制电压+关键信号（自动选列/归一化，时间轴对齐）。
     - 运行 `data_processing.py`：生成处理后 CSV/NPZ（FC1/FC2）。
     - 生成滤波对比图：原始 vs 处理后电压叠加。
     - 数据集切换 FC1/FC2，按钮状态依赖文件检测。
  3) 训练与预测页（`LifePredictionPage`）
     - 运行训练：调用 `train.py`，进度条+日志实时显示。
     - 绘制预测对比：读取最新 `predictions.csv`（含 dataset 列），可选 FC1/FC2，展示真实 vs 预测，自动用时间轴/样本序号，显示 MAE/RMSE。

## 4. 核心脚本概览
- `pemfc_catboost_analysis.py`
  - 读取/映射原始 CSV，线性插值补缺，MinMax 归一化。
  - CatBoostRegressor 训练，输出特征重要性（CSV+图表+报告）。
  - 图表：Times New Roman，PNG+PDF，默认 300 dpi，按重要性降序。

- `data_processing.py`
  - 读取源 CSV（FC1/FC2），多阶段滤波：中值滤波 → 小波去噪（db8，可自适应层数）→ Savitzky-Golay 平滑；电压列特殊处理（滚动均值、范围校验、尖峰处理）。
  - 生成处理后 CSV + NPZ，文件名含时间戳；保存列统计与图表。

- `train.py`
  - 配置集中（`ModelConfig`）：数据路径、特征列表（默认 CatBoost Top5：air_outlet_flow, hydrogen_inlet_temp, current, coolant_flow, current_density）、序列长度、GRU/训练超参、SOH/RUL 阈值设定。
  - 数据管线：加载最新处理后 NPZ → 选定特征 + 目标 `stack_voltage` → 小波去噪 → 时间序列划分（7/1.5/1.5，按时间，不 shuffle）→ 标准化（特征）。
  - 模型：GRU（可双向），全连接收尾，MSE 损失，AdamW，ReduceLROnPlateau，梯度裁剪，早停。
  - 评估与输出：MAE/RMSE/MAPE/R²；保存预测 `predictions.csv`；图表（训练曲线、电压预测、SOH/RUL）。
  - FC2 推理：训练后自动载入 `processed_results/FC2`，复用 FC1 标准化器与特征，输出 FC2 预测与指标到同一 `predictions.csv`（含 dataset 标记）与 `metrics_overall.csv`。
  - RUL 计算：SOH=V/V_initial，阈值 0.96；V_initial 来自前 100 点（SG/小波，可变窗口，CV<1% 判稳）；线性插值跨阈值求 RUL，提供 MAE_RUL/MAPE_RUL/PHM_Score。
  - 日志：完整训练生命周期到文件+控制台。

- `test_npz.py` / `original_csv.py`
  - NPZ/CSV 质检与可视化工具：自动找最新文件，输出关键特征时序、热图、质量报告。

## 5. 指标函数与表格（已在 `train.py` 中实现）
- `calculate_metrics(y_true, y_pred)`：支持 numpy/torch，返回 MAE/RMSE/MAPE/R²。
- `generate_metrics_table(metrics_records)`：输出 markdown 表格，可多实验条件汇总。
- 训练中记录 train/val/test 指标，另存 `metrics_overall.csv`（含 FC1/FC2）。

## 6. 小波去噪要点（WTD）
- 小波基：sym8；6 层分解；仅细节系数软阈值，近似系数保留；阈值可取排序后 90 分位。
- 评估：SNR、RMSE、平滑度（差分方差）；对比去噪前后电压曲线。
- 实现：`WaveletDenoiser`（train.py）和 `data_processing.py` 的多阶段滤波。

## 7. 可视化要点
- 训练/预测图（train.py → images/）：真实（蓝）、训练段预测（绿）、测试段预测（红），训练/预测分割线，EOL 阈值线，RUL 标注；支持置信区间（mean±1.96*std，占位逻辑）。
- SOH 子图：SOH 随时间，阈值 0.96，真实/预测 RUL 垂直线与误差标注。
- 前端预测对比：`plot_worker.py::plot_prediction_vs_true` 按 dataset 过滤，优先用时间轴。

## 8. 前端期望与当前状态对应
- 重要性分析：已可一键运行/绘图，读取最新 `catboost_results/*.csv`。
- 原始数据展示：合并源 CSV → `visualization/truedata`，电压+关键信号对比已支持（时间轴对齐、自动选列）；需在 UI 中确保“1/2 视图切换”与 FC1/FC2 切换同时可用（现有按钮：合并并可视化原始数据）。
- 数据处理：一键运行 `data_processing.py`，生成最新处理文件，并可做原始 vs 滤波对比（已有按钮）。
- 训练预测：一键运行 `train.py`，显示日志/进度；按钮绘制 FC1/FC2 预测对比（读取合并的 predictions.csv）。

## 9. 健壮性与测试
- 异常捕获：数据缺失、列缺失、文件未找到、RUL 计算不可用均有日志与提示。
- 单元测试占位：可针对去噪函数、指标函数、数据加载与列检测编写；当前未附具体测试文件，可按模块补充。

## 10. 新手参数修改提示（简版）
- 过拟合：增大 dropout(0.4-0.5)，减小 gru_hidden_size，增加 weight_decay。
- 欠拟合：减小 dropout，增大 gru_hidden_size/层数，或稍升 learning_rate。
- 不稳定：降低 learning_rate，减小 batch_size，收紧 grad_clip。
- 速度/内存：调小 sequence_length、gru_hidden_size 或增大 batch_size（视显存）。

## 11. 未来可扩展点
- UI：原始数据两种视图（汇总/分列）快捷切换按钮；FC2 时间轴适配提示。
- 置信区间：如模型输出均值+std，可在前端叠加阴影区。
- 指标表：在前端渲染 `metrics_overall.csv`，支持多实验条件展示。
- 单元测试：为去噪、指标、数据加载、前端文件检测增加测试用例。

---
如需运行：按前端按钮顺序“CatBoost 分析 → 原始数据可视化 → 数据处理 → 训练与预测”，日志与图表会在对应目录自动生成。