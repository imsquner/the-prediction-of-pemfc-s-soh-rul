<div align="center">
  
  <h1>🔋 PEMFC Integrated GRU Prognostics Toolkit</h1>
  
  <p>
    <strong>一个用于质子交换膜燃料电池（PEMFC）堆电压预测、SOH/RUL 估计与未来无工况滚动外推的综合工具集。</strong>
  </p>

  <p>
    <a href="https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/imsquner/the-prediction-of-pemfc-s-soh-rul?style=flat-square&color=yellow" /></a>
    <a href="https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul/network/members"><img alt="Forks" src="https://img.shields.io/github/forks/imsquner/the-prediction-of-pemfc-s-soh-rul?style=flat-square&color=orange" /></a>
    <a href="https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul/issues"><img alt="Issues" src="https://img.shields.io/github/issues/imsquner/the-prediction-of-pemfc-s-soh-rul?style=flat-square&color=red" /></a>
    <a href="https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/imsquner/the-prediction-of-pemfc-s-soh-rul?style=flat-square&color=blue" /></a>
    <a href="https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul"><img alt="Python" src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square" /></a>
  </p>

  <img src="https://images.unsplash.com/photo-1620916566398-39f1143ab7be?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80" alt="Banner Image" width="100%" />
  <br/>
  <em>*基于深度学习的时间序列预测模型，提供从数据提取、特征分析到寿命预测的端到端解决方案*</em>
</div>

## 📌 项目背景 (Background)

质子交换膜燃料电池（PEMFC）在长期运行中，其电堆电压会随着材料老化而逐渐下降。准确预测 PEMFC 的 **SOH（健康状态，State of Health）** 和 **RUL（剩余使用寿命，Remaining Useful Life）** 对于制定预测性维护策略、降低系统运行成本具有重大意义。

本项目旨在提供一套**工程化、自动化、可复用**的深度学习基准测试平台。它结合了传统特征工程与先进的深度时序模型（主要基于 GRU 门控循环单元），能够在缺失未来具体工况的情况下，通过滚动外推（Rolling Extrapolation）完成对燃料电池性能的长期预测。

## 🌟 核心特性 (Key Features)

- 🚀 **端到端自动化流水线**：涵盖从原始 CSV 数据清洗、异常剔除、特征规约到模型组装、自动训练、评估测试和长效预测的全生命周期。
- 🧠 **混合驱动的分析架构**：
  - 引进 `CatBoost` 算法针对多维工况（如电流、温度、压力等）特征进行重要性排序（Feature Importance），提供可解释的特征贡献度证据。
  - 使用优化的 `GRU` 时序架构提取时间依赖特征，极大地缓解了长序列预测中的梯度消失问题。
- 📉 **工业级指标输出与可视化**：自带成熟度极高的评价体系（包括 `MSE`, `RMSE`, `MAE`, `R2` 等），并自动绘制包含地面真值（Ground Truth）与预测值（Predictions）比对的高清折线图，生成研究级报告。
- 🛠 **工程轻量化与高拓展**：仓库极简设计，去除了多余的冗余数据卷与草稿代码，提供一键评估与训练接口，方便二次开发对接其他网络架构（如 LSTM, Transformer 等）。

## 🛠️ 安装与配置 (Installation)

本项目强烈建议在 Python 3.8+ 虚拟环境下运行。通过以下步骤即可快速配置开发环境。

### 1. 克隆代码仓库
```powershell
git clone https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul.git
cd the-prediction-of-pemfc-s-soh-rul
```

### 2. 构建虚拟环境并激活
```powershell
python -m venv .venv
# Windows PowerShell 下激活
& .\.venv\Scripts\Activate.ps1
# Mac/Linux 环境使用: source .venv/bin/activate
```

### 3. 安装核心依赖包
```powershell
pip install -r requirements.txt
```
> **提示**：核心依赖包含 `torch`, `pandas`, `numpy`, `scikit-learn`, `catboost`, `matplotlib` 等库，环境详情亦可参考 [环境与运行说明](docs/environment_setup.md)。

## 🚀 快速使用 (Usage Guide)

本工具高度集成，您无需逐个调用脚本，直接通过主入口 `train.py` 即可完成从数据处理到图表生成的全流程：

### 方案 A：直接运行预测模型（仅评估验证）
若在 `configs/` 或 `models/` 下已有预训练好的检查点（`.pth` 模型），推荐首先通过此模式验证闭环：
```powershell
$env:EVAL_ONLY="1"
python train.py
```
> 系统将会直接加载最佳参数缓存文件与模型，完成对测试集的滚动预测，并在结果文件夹下生成图表与评估报表。

### 方案 B：重置并启动全量训练
当修改了架构、需引入新数据集或调节超参数时，执行常规的训练流程（清理环境变量标志以解锁训练模块）：
```powershell
Remove-Item Env:EVAL_ONLY -ErrorAction SilentlyContinue
python train.py
```
> 大致流程：**数据预处理 -> 构建张量缓存 -> 启动 GRU 多世代训练 (包含 Early Stopping) -> 最优模型落地 -> 测试集验证 / 滚动预测外推 -> 保存日志与图像**。

## 📁 工程目录全貌 (Project Hierarchy)

```text
📦 PEMFC-Integrated-Tool
├── 📂 catboost_results/     # CatBoost特征重要性分析产物与文本报告
├── 📂 data/                 # 放置原始电池老化数据集 (请确保文件存在)
├── 📂 datatest/             # 预留用于跨环境或异构电堆测试的小批数据
├── 📂 docs/                 # 项目文档库 📖 (开发手册、阅读指引)
├── 📂 processed_results/    # 经过清洗和平滑提取后暂存的 `.npz` 张量与 `.csv` 报表
├── 📂 train_results_paper/  # 🚀 统一输出中心 (模型权重、指标表格、验证图表全览)
│   └── 📂 gru_pemfc_paper_experiment_fixed_r2_rul/
│       ├── 📂 configs/      # 参数备份与最终报告 JSON
│       ├── 📂 models/       # 产出的 best_model.pth 存档
│       └── 📂 tables/       # 指标输出 (metrics_overall.csv 等)
├── 📜 train.py              # 🎉 主程序入口，流转调度的“引擎”
├── 📜 model.py              # 深度学习网络架构层 (核心包含 GRU 类与初始化工具)
├── 📜 data_processing.py    # 将 CSV 数据切片化映射至 PyTorch Dataset 的加载器实现
└── 📜 data_processors.py    # 针对燃料电池时间序列退化特征的核心过滤器、对齐器
```

## ⚙️ 模型与数据策略 (Strategies)

为避免大量废弃模型和缓存堆积，本项目引入了严格的文件生命周期管理：
- **最优模型留存 (Model Retention)**：无论跑多少个 Epoch，结果目录永远只更新、保留泛化最优的 `best_model.pth`。
- **流水线数据缓存 (Data Caching)**：为降低内存消耗与提升加载速度，大型 CSV 会在运行初被压缩处理至 `processed_results/`，以 `.npz` 供后续模块免计算极速读取。
- **历史清扫隔离**：通过模块化时间戳 `{artifact_name}_{YYYYMMDD_HHMMSS}` 避免日志覆写；同时系统会在重新训练前提供临时目录（tmp, cache）的主动告警及归档能力。

## 📖 深入阅读 (Documentation)

欲深入探究本项目实现细节的小伙伴，请按照以下配套指南进行进阶阅读：
- 🛠 [环境与运行详细说明 (Environment Setup)](docs/environment_setup.md)
- 🗺 [源码级架构阅读路线图 (Project Reading Guide)](docs/project_reading_guide.md)

## 🤝 参与贡献 (Contributing)

欢迎研究 PEMFC 数据预测领域或对序列时间预测感兴趣的朋友进行 Fork 与 Pull Requests。
如果你在使用中遇到了 Bug，或者有对于工况预测逻辑（如：添加变工况干扰下的小波去噪）等改良提议，请不要犹豫发一个 [Issue](https://github.com/imsquner/the-prediction-of-pemfc-s-soh-rul/issues)！

## 📄 开源许可证 (License)

本项目采用 [MIT License](LICENSE) 开源许可证。允许自由的商用、修改与分发，仅需保留原作者版权声明。
