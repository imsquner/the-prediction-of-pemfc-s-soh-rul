# PEMFC Integrated GRU Toolkit

用于 PEMFC 电堆电压预测与 SOH/RUL 评估的工程化项目。当前仓库已做精简，保留了可直接运行的最小必要数据、模型和结果目录。

## 一分钟上手

```powershell
# 1) 安装依赖
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2) 仅评估（推荐首次验证）
$env:EVAL_ONLY="1"; & .\.venv\Scripts\python.exe .\train.py

# 3) 完整训练+评估
Remove-Item Env:EVAL_ONLY -ErrorAction SilentlyContinue; & .\.venv\Scripts\python.exe .\train.py
```

默认输出目录：`train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/`

## 当前工程结构（精简后）

- `train.py`：主入口，负责训练/评估/可视化/报告导出。
- `model.py`、`data_processing.py`、`data_processors.py`：模型与数据处理。
- `processed_results/FC1`、`processed_results/FC2`：仅保留最新一组处理数据（npz+csv）。
- `train_results_paper/gru_pemfc_paper_experiment_fixed_r2_rul/models/best_model.pth`：当前保留的最新模型。
- `catboost_results/`：仅保留最新一次特征重要性分析结果。
- `docs/`：环境文档与工程阅读文档。

## 模型与数据保留策略

- 模型：保留最新 `best_model.pth`，其余历史模型已清理。
- 处理数据：每个数据集仅保留最新版本（FC1 与 FC2 各 1 组 npz+csv）。
- 日志与中间产物：构建缓存、临时目录、历史草稿日志已清理。

## 可读性与命名约定

- 代码文件统一使用 `snake_case.py`（保留已有入口名以避免外部脚本失效）。
- 结果文件建议使用：`{artifact}_{YYYYMMDD_HHMMSS}.{ext}`。
- 新文档统一放在 `docs/`，避免散落在根目录。

## 文档索引

- 环境与运行说明：`docs/environment_setup.md`
- 工程阅读路线：`docs/project_reading_guide.md`

## License

本项目采用 MIT License，见 `LICENSE`。
