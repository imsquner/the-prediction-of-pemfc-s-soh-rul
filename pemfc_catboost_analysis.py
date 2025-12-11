"""
基于CatBoost的PEMFC监测参数重要性分析 - 论文第三章实现
目标：评估燃料电池状态监测参数对性能退化的表征重要性
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import warnings
import textwrap
from datetime import datetime
import seaborn as sns

# 避免Windows终端GBK编码报错，强制UTF-8输出
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(__builtins__, "print"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")
    except Exception:
        pass

warnings.filterwarnings('ignore')

# ==================== 全局配置 ====================

# 设置字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# CSV文件列名映射（基于论文和原有代码）
CSV_COLUMN_MAPPING = {
    'Time (h)': 'time',
    'Utot (V)': 'stack_voltage',
    'J (A/cm2)': 'current_density',
    'I (A)': 'current',
    'TinH2 (°C)': 'hydrogen_inlet_temp',
    'ToutH2 (°C)': 'hydrogen_outlet_temp',
    'TinAIR (°C)': 'air_inlet_temp',
    'ToutAIR (°C)': 'air_outlet_temp',
    'TinWAT (°C)': 'coolant_inlet_temp',
    'ToutWAT (°C)': 'coolant_outlet_temp',
    'PinAIR (mbara)': 'air_inlet_pressure',
    'PoutAIR (mbara)': 'air_outlet_pressure',
    'PoutH2 (mbara)': 'hydrogen_outlet_pressure',
    'PinH2 (mbara)': 'hydrogen_inlet_pressure',
    'DinH2 (l/mn)': 'hydrogen_inlet_flow',
    'DoutH2 (l/mn)': 'hydrogen_outlet_flow',
    'DinAIR (l/mn)': 'air_inlet_flow',
    'DoutAIR (l/mn)': 'air_outlet_flow',
    'DWAT (l/mn)': 'coolant_flow',
    'HrAIRFC (%)': 'air_inlet_humidity'
}

# 需要排除的列（U1-U5电压列，因为它们是单个电池电压，总和就是堆电压）
EXCLUDED_COLUMNS = ['U1 (V)', 'U2 (V)', 'U3 (V)', 'U4 (V)', 'U5 (V)']

# 尝试的编码列表（解决编码问题）
ENCODINGS_TO_TRY = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1', 'cp1252', 'iso-8859-1']

# ==================== 1. 数据加载与预处理 ====================

def load_and_preprocess_data(data_dir="data", dataset_name="FC1"):
    """
    加载并预处理PEMFC老化实验数据
    论文数据源：IEEE PHM 2014数据挑战赛，FC1数据集
    """
    print("=" * 70)
    print("Data Loading and Preprocessing")
    print("=" * 70)

    # 确定文件列表
    if dataset_name == "FC1":
        file_list = [
            "FC1_Ageing_part1.csv",
            "FC1_Ageing_part2.csv",
            "FC1_Ageing_part3.csv"
        ]
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}, please enter 'FC1'")

    data_frames = []

    for file_name in file_list:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file does not exist: {file_path}")

        print(f"Loading: {file_name}")

        # 尝试多种编码读取CSV文件（解决编码问题）
        df = None
        for encoding in ENCODINGS_TO_TRY:
            try:
                print(f"  Trying encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, engine='python')
                print(f"  OK read with {encoding}")
                break
            except UnicodeDecodeError as e:
                print(f"  FAIL encoding {encoding}: {str(e)[:50]}...")
                continue
            except Exception as e:
                print(f"  FAIL encoding {encoding}, other error: {str(e)[:50]}...")
                continue

        if df is None:
            # 如果所有编码都失败，尝试不指定编码
            print("  WARN all encodings failed, try default encoding...")
            try:
                df = pd.read_csv(file_path, engine='python')
                print("  OK read with default encoding")
            except Exception as e:
                raise ValueError(f"Cannot read file {file_name}: {str(e)}")

        if df is None:
            raise ValueError(f"Cannot read file {file_name}, please check file format and encoding")

        # 去除列名两端的空格
        df.columns = df.columns.str.strip()

        # 应用列名映射
        rename_count = 0
        for old_name, new_name in CSV_COLUMN_MAPPING.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
                rename_count += 1

        if rename_count > 0:
            print(f"  Renamed {rename_count} columns")

        # 删除U1-U5电压列（避免信息泄露）
        columns_before = len(df.columns)
        for col in EXCLUDED_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
                print(f"  Removed column: {col} (individual cell voltage)")

        columns_after = len(df.columns)
        print(f"  Removed {columns_before - columns_after} individual cell voltage columns")

        data_frames.append(df)
        print(f"  Loading complete: {df.shape[0]} rows, {df.shape[1]} columns")

    # 合并所有数据
    merged_df = pd.concat(data_frames, ignore_index=True)

    # 按时间排序（时间序列数据）
    if 'time' in merged_df.columns:
        merged_df = merged_df.sort_values('time').reset_index(drop=True)

    print(f"\nData merging complete: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")

    # 检查目标列是否存在
    if 'stack_voltage' not in merged_df.columns:
        raise ValueError("Stack voltage column (stack_voltage) not found")

    return merged_df


def preprocess_data(df, target_col='stack_voltage'):
    """
    数据预处理（基于论文第三章描述）
    包括：缺失值处理、最大最小归一化、数据重构
    """
    print("\nData Preprocessing...")

    df_processed = df.copy()

    # 1. 缺失值处理（使用线性插值，论文未明确但合理）
    print("  1. Missing value processing...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df_processed[col].isna().sum() > 0:
            original_missing = df_processed[col].isna().sum()
            df_processed[col] = df_processed[col].interpolate(method='linear', limit_direction='both')
            df_processed[col] = df_processed[col].fillna(method='ffill').fillna(method='bfill')
            filled_count = original_missing - df_processed[col].isna().sum()
            if filled_count > 0:
                print(f"    Column '{col}': filled {filled_count} missing values")

    # 2. 最大最小归一化（论文公式3.1）
    print("  2. Min-Max normalization...")
    scaler = MinMaxScaler()

    # 选择所有数值型特征（除了时间和目标变量）
    feature_cols = [col for col in numeric_cols
                   if col not in ['time', target_col] and col in df_processed.columns]

    # 保存原始数据的min和max用于反归一化（如果需要）
    original_mins = df_processed[feature_cols].min()
    original_maxs = df_processed[feature_cols].max()

    # 应用归一化：y = (x - min(x)) / (max(x) - min(x))
    df_processed[feature_cols] = scaler.fit_transform(df_processed[feature_cols])

    print(f"  Normalized {len(feature_cols)} features")

    # 3. 数据重构（论文图3.1，将时序数据转换为适合模型的格式）
    print("  3. Data restructuring...")
    # 论文中的重构可能指将多时间步数据整合，这里我们简化处理
    # 使用每个时间点的所有特征作为输入，预测当前时间点的堆电压
    X = df_processed[feature_cols].values
    y = df_processed[target_col].values

    print(f"    Input feature dimension: {X.shape}")
    print(f"    Target variable dimension: {y.shape}")

    # 4. 数据集划分（论文未明确，使用时间序列划分）
    print("  4. Dataset splitting...")
    # 由于是时间序列数据，按时间顺序划分
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    print(f"    Training set: {len(X_train)} samples")
    print(f"    Validation set: {len(X_val)} samples")
    print(f"    Test set: {len(X_test)} samples")

    # 数据基本信息
    print("\nData Basic Information:")
    print(f"  Total samples: {n_samples}")
    print(f"  Number of features: {len(feature_cols)}")
    print(f"  Feature list: {feature_cols}")
    print(f"  Target variable statistics:")
    print(f"    Minimum: {y.min():.4f} V")
    print(f"    Maximum: {y.max():.4f} V")
    print(f"    Mean: {y.mean():.4f} V")
    print(f"    Standard deviation: {y.std():.4f} V")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'feature_names': feature_cols,
        'scaler': scaler,
        'original_mins': original_mins,
        'original_maxs': original_maxs,
        'df_processed': df_processed
    }


# ==================== 2. CatBoost特征重要性分析 ====================

def catboost_feature_importance_analysis(data_dict, target_col='stack_voltage'):
    """
    使用CatBoost评估监测参数重要性（基于论文第三章）
    论文图3.9显示了各参数的重要性排序
    """
    print("\n" + "=" * 70)
    print("CatBoost Feature Importance Analysis")
    print("=" * 70)

    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    feature_names = data_dict['feature_names']

    # CatBoost模型参数（基于论文意图，使用对称树结构）
    print("\nTraining CatBoost model...")
    model_params = {
        'iterations': 100,          # 迭代次数
        'learning_rate': 0.03,      # 学习率
        'depth': 6,                 # 树深度
        'loss_function': 'RMSE',    # 损失函数
        'verbose': False,           # 不显示训练过程
        'random_seed': 42,          # 随机种子
        'grow_policy': 'SymmetricTree',  # 对称树结构
        'task_type': 'CPU',         # 使用CPU
        'early_stopping_rounds': 10,  # 早停
    }

    model = CatBoostRegressor(**model_params)

    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )

    # 在验证集上评估模型
    y_val_pred = model.predict(X_val)

    # 计算评估指标
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    r2 = r2_score(y_val, y_val_pred)

    print("\nModel Evaluation Results (Validation Set):")
    print(f"  Mean Absolute Error (MAE): {mae:.4f} V")
    print(f"  Root Mean Square Error (RMSE): {rmse:.4f} V")
    print(f"  Coefficient of Determination (R2): {r2:.4f}")

    # 获取特征重要性
    print("\nCalculating feature importance...")
    importance_scores = model.get_feature_importance()

    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    })

    # 按重要性降序排序
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 计算重要性百分比
    total_importance = importance_df['importance'].sum()
    importance_df['importance_percent'] = (importance_df['importance'] / total_importance) * 100
    importance_df['importance_percent'] = importance_df['importance_percent'].round(2)

    # 计算累积重要性
    importance_df['cumulative_percent'] = importance_df['importance_percent'].cumsum()

    # 显示重要性排序
    print("\nFeature Importance Ranking:")
    print("-" * 80)
    print(f"{'Rank':<5} {'Feature':<25} {'Importance':<12} {'Percent(%)':<12} {'Cumulative(%)':<12}")
    print("-" * 80)

    for idx, (_, row) in enumerate(importance_df.iterrows(), start=1):
        print(f"{idx:<5} {row['feature']:<25} {row['importance']:<12.4f} "
              f"{row['importance_percent']:<12.2f} {row['cumulative_percent']:<12.2f}")

    return model, importance_df, {'MAE': mae, 'RMSE': rmse, 'R2': r2}


# ==================== 3. 可视化 ====================

def create_visualizations(importance_df, data_dict, save_dir='results'):
    """
    创建可视化图表（基于论文图3.9）
    所有图表文字使用英文
    """
    print("\n" + "=" * 70)
    print("Generating Visualization Charts")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # ========== Chart 1: Feature Importance Bar Chart (Paper Figure 3.9) ==========
    print("\nGenerating Chart 1: Feature Importance Bar Chart...")

    plt.figure(figsize=(14, 8))

    # 选择前15个最重要的特征进行显示
    plot_df = importance_df.head(15).copy()

    # 创建颜色映射：根据重要性值
    colors = cm.get_cmap('viridis')(np.linspace(0.3, 0.9, len(plot_df)))

    # 绘制水平柱状图（更易阅读）
    bars = plt.barh(range(len(plot_df)), plot_df['importance_percent'],
                   color=colors, edgecolor='black', linewidth=0.5)

    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, plot_df['importance_percent'])):
        plt.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}%', ha='left', va='center', fontsize=10)

    # 设置y轴标签（特征名称）
    plt.yticks(range(len(plot_df)), plot_df['feature'], fontsize=11)
    plt.xlabel('Feature Importance (%)', fontsize=14, fontweight='bold')
    plt.title('Importance of PEMFC Monitoring Parameters', fontsize=16, fontweight='bold', pad=20)

    # 添加网格线
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # 保存图表
    chart1_path = os.path.join(save_dir, f'feature_importance_{timestamp}.png')
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    print(f"  Chart 1 saved: {chart1_path}")
    plt.show()
    plt.close()

    # ========== Chart 2: Cumulative Importance Curve ==========
    import sys
 
    # 强制stdout/stderr使用utf-8，避免Windows控制台GBK编码报错
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("\nGenerating Chart 2: Cumulative Importance Curve...")

    plt.figure(figsize=(10, 6))

    # 绘制累积重要性曲线
    plt.plot(range(1, len(importance_df)+1), importance_df['cumulative_percent'],
             'b-', linewidth=2, marker='o', markersize=6)

    # 标记95%累积重要性的点
    cumulative_mask = importance_df['cumulative_percent'] >= 95
    idx_95 = int(cumulative_mask.idxmax()) if cumulative_mask.any() else len(importance_df) - 1
    idx_95 = max(0, min(idx_95, len(importance_df) - 1))
    if idx_95 < len(importance_df):  # 确保索引有效
        plt.axvline(x=idx_95 + 1, color='r', linestyle='--', alpha=0.7)
        plt.axhline(y=95, color='r', linestyle='--', alpha=0.7)
        plt.plot(idx_95 + 1, importance_df.loc[idx_95, 'cumulative_percent'], 'ro', markersize=10)

        # 添加文本标注
        plt.text(idx_95 + 1, 50, f'Top {idx_95+1} features\nCumulative importance: {importance_df.loc[idx_95, "cumulative_percent"]:.1f}%',
                 fontsize=10, ha='center', va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.xlabel('Number of Features', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Importance (%)', fontsize=12, fontweight='bold')
    plt.title('Cumulative Feature Importance Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    chart2_path = os.path.join(save_dir, f'cumulative_importance_{timestamp}.png')
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    print(f"  Chart 2 saved: {chart2_path}")
    plt.show()
    plt.close()

    # ========== Chart 3: Stack Voltage Time Series ==========
    print("\nGenerating Chart 3: Stack Voltage Time Series...")

    plt.figure(figsize=(14, 5))

    df_processed = data_dict['df_processed']

    # 绘制堆电压时间序列（前5000个点以清晰显示）
    plot_points = min(5000, len(df_processed))
    plt.plot(df_processed['time'].values[:plot_points],
             df_processed['stack_voltage'].values[:plot_points],
             'b-', alpha=0.7, linewidth=0.8)

    # 添加滚动平均值
    rolling_window = min(100, plot_points // 10)
    rolling_mean = df_processed['stack_voltage'].rolling(window=rolling_window, center=True).mean()
    plt.plot(df_processed['time'].values[:plot_points],
             rolling_mean.values[:plot_points],
             'r-', linewidth=1.5, label=f'{rolling_window}-point rolling average')

    plt.xlabel('Time (h)', fontsize=12, fontweight='bold')
    plt.ylabel('Stack Voltage (V)', fontsize=12, fontweight='bold')
    plt.title('PEMFC Stack Voltage Time Series (FC1 Dataset)', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存图表
    chart3_path = os.path.join(save_dir, f'voltage_timeseries_{timestamp}.png')
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    print(f"  Chart 3 saved: {chart3_path}")
    plt.show()
    plt.close()

    return [chart1_path, chart2_path, chart3_path]


# ==================== 4. 结果保存与报告生成 ====================

def save_results_and_report(importance_df, model_metrics, data_dict,
                          selected_features, chart_paths, save_dir='results'):
    """
    保存分析结果和生成报告（基于论文结论）
    报告使用英文
    """
    print("\n" + "=" * 70)
    print("Saving Analysis Results and Generating Report")
    print("=" * 70)

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. 保存特征重要性结果为CSV
    csv_path = os.path.join(save_dir, f'feature_importance_results_{timestamp}.csv')
    importance_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"  Feature importance results saved to: {csv_path}")

    # 2. 生成详细分析报告（英文）
    report_path = os.path.join(save_dir, f'analysis_report_{timestamp}.txt')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PEMFC Monitoring Parameter Importance Analysis Report (Based on Chapter 3 Method)\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: FC1 (IEEE PHM 2014 Data Challenge)\n")
        f.write(f"Analysis Method: CatBoost Feature Importance Analysis\n\n")

        # 数据基本信息
        f.write("1. Data Basic Information\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Samples: {len(data_dict['df_processed'])}\n")
        f.write(f"Number of Features: {len(data_dict['feature_names'])}\n")
        f.write(f"Training Set Samples: {len(data_dict['X_train'])}\n")
        f.write(f"Validation Set Samples: {len(data_dict['X_val'])}\n")
        f.write(f"Test Set Samples: {len(data_dict['X_test'])}\n\n")

        # 模型性能
        f.write("2. CatBoost Model Performance\n")
        f.write("-" * 40 + "\n")
        f.write(f"Mean Absolute Error (MAE): {model_metrics['MAE']:.4f} V\n")
        f.write(f"Root Mean Square Error (RMSE): {model_metrics['RMSE']:.4f} V\n")
        f.write(f"Coefficient of Determination (R2): {model_metrics['R2']:.4f}\n\n")

        # 特征重要性排名（前10）
        f.write("3. Feature Importance Ranking (Top 10)\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Rank':<5} {'Feature':<25} {'Importance(%)':<12} {'Cumulative(%)':<12}\n")
        f.write("-" * 60 + "\n")

        for i, row in importance_df.head(10).iterrows():
            f.write(f"{i+1:<5} {row['feature']:<25} {row['importance_percent']:<12.2f} "
                   f"{row['cumulative_percent']:<12.2f}\n")

        f.write("\n")

        # 关键发现与结论
        f.write("4. Key Findings and Conclusions\n")
        f.write("-" * 40 + "\n")

        # 重要性排名前三的参数
        top_3_features = importance_df.head(3)['feature'].tolist()
        top_3_percent = importance_df.head(3)['importance_percent'].tolist()

        f.write(f"(1) Top three most important monitoring parameters:\n")
        for i, (feat, perc) in enumerate(zip(top_3_features, top_3_percent), 1):
            f.write(f"    #{i}: {feat} ({perc:.1f}%)\n")

        f.write("\n")

        # 基于论文的结论（空气出口流量最重要）
        f.write("(2) Comparison with paper findings:\n")
        f.write("    Paper Figure 3.9 shows that air outlet flow (DoutAIR) is the most important parameter.\n")

        # 检查我们的结果
        if 'air_outlet_flow' in importance_df['feature'].values:
            air_outlet_flow_rank = importance_df[importance_df['feature'] == 'air_outlet_flow'].index[0] + 1
            air_outlet_flow_percent = importance_df[importance_df['feature'] == 'air_outlet_flow']['importance_percent'].values[0]
            f.write(f"    In our analysis, air outlet flow ranks #{air_outlet_flow_rank}, with importance of {air_outlet_flow_percent:.1f}%.\n")

            # 判断是否与论文一致
            if air_outlet_flow_rank == 1:
                f.write(f"    ✓ Consistent with paper conclusion: air outlet flow is the most important parameter.\n")
            elif air_outlet_flow_rank <= 3:
                f.write(f"    ✓ Partially consistent: air outlet flow is among the top 3 important parameters.\n")
            else:
                f.write(f"    ⚠ Not fully consistent: air outlet flow is not among the top 3 parameters.\n")
        else:
            f.write("    ⚠ Note: Air outlet flow parameter not found in our data.\n")

        f.write("\n")

        # 选择用于预测模型的关键参数（基于论文）
        f.write("(3) Recommended parameters for performance degradation prediction model:\n")
        f.write("    According to paper conclusions and degradation mechanism analysis, the following parameters are recommended:\n")

        # 论文中最终选择的参数
        paper_selected_params = [
            'time', 'stack_voltage', 'current', 'current_density',
            'hydrogen_inlet_temp', 'coolant_flow', 'air_outlet_flow'
        ]

        # 检查这些参数在我们的重要性排名中的位置
        available_params = []
        for param in paper_selected_params:
            if param in importance_df['feature'].values:
                rank = importance_df[importance_df['feature'] == param].index[0] + 1
                percent = importance_df[importance_df['feature'] == param]['importance_percent'].values[0]
                f.write(f"    - {param}: Rank #{rank}, importance {percent:.1f}%\n")
                available_params.append(param)
            elif param == 'time':
                f.write(f"    - {param}: (time variable, not included in feature importance analysis)\n")
                available_params.append(param)
            else:
                f.write(f"    - {param}: (parameter not found in data)\n")

        f.write(f"\n    Total {len(available_params)} parameters selected for subsequent prediction model.\n")

        f.write("\n(4) Recommendations and Next Steps:\n")
        f.write("    a. Build performance degradation prediction model using selected key parameters\n")
        f.write("    b. Consider interaction effects between parameters (automatically handled by CatBoost)\n")
        f.write("    c. Validate model generalization ability under different operating conditions\n")
        f.write("    d. Combine results with mechanistic models to explain degradation processes\n")
        f.write("    e. Consider removing U1-U5 voltage columns as they sum to stack voltage (already implemented)\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("Analysis Complete\n")
        f.write("=" * 80 + "\n")

    print(f"  Detailed analysis report saved to: {report_path}")

    # 3. 打印关键结论（英文）
    print("\nKey Conclusions:")
    print("-" * 40)

    if len(importance_df) > 0:
        top_feature = importance_df.iloc[0]['feature']
        top_percent = importance_df.iloc[0]['importance_percent']
        print(f"1. Most important monitoring parameter: {top_feature} ({top_percent:.1f}%)")

        # 检查前5个特征的累积重要性
        if len(importance_df) >= 5:
            top_5_cumulative = importance_df.head(5)['cumulative_percent'].iloc[-1]
            print(f"2. Cumulative importance of top 5 features: {top_5_cumulative:.1f}%")

    # 基于论文选择参数
    print(f"3. Recommended parameters for prediction model (based on paper):")
    paper_params_display = ['Time', 'Stack Voltage', 'Current', 'Current Density',
                           'Hydrogen Inlet Temperature', 'Coolant Flow', 'Air Outlet Flow']
    for i, param in enumerate(paper_params_display, 1):
        print(f"   {i}. {param}")

    print("\n4. Note: U1-U5 individual cell voltage columns have been excluded from analysis")
    print("   (their sum equals stack voltage, which would cause information leakage).")

    return csv_path, report_path


# ==================== 5. 主函数 ====================

def main():
    """
    主函数：执行完整的PEMFC监测参数重要性分析
    """
    print("=" * 80)
    print("PEMFC Monitoring Parameter Importance Analysis Based on CatBoost")
    print("Implementation of Chapter 3 Method")
    print("=" * 80)

    # 配置参数
    DATA_DIR = "data"
    DATASET_NAME = "FC1"
    RESULTS_DIR = "catboost_results"

    try:
        # 1. 数据加载与预处理
        print("\n[Stage 1] Data Loading and Preprocessing")
        print("-" * 40)

        df_raw = load_and_preprocess_data(DATA_DIR, DATASET_NAME)
        data_dict = preprocess_data(df_raw, 'stack_voltage')

        # 2. CatBoost特征重要性分析
        print("\n[Stage 2] CatBoost Feature Importance Analysis")
        print("-" * 40)

        model, importance_df, model_metrics = catboost_feature_importance_analysis(
            data_dict, 'stack_voltage'
        )

        # 3. 选择关键参数（基于累积重要性）
        print("\n[Stage 3] Key Parameter Selection")
        print("-" * 40)

        # 选择累积重要性达到90%的特征
        cumulative_threshold = 90.0
        selected_by_cumulative = importance_df[importance_df['cumulative_percent'] <= cumulative_threshold]
        selected_features = selected_by_cumulative['feature'].tolist()

        # 如果阈值内没有特征，至少选择最重要的一个
        if not selected_features and len(importance_df) > 0:
            selected_features = [importance_df.iloc[0]['feature']]

        print(f"Features selected based on cumulative importance > {cumulative_threshold}%: {len(selected_features)} features")
        if selected_features:
            print(f"Feature list: {selected_features}")

        # 4. 可视化
        print("\n[Stage 4] Visualization")
        print("-" * 40)

        chart_paths = create_visualizations(importance_df, data_dict, RESULTS_DIR)

        # 5. 结果保存与报告生成
        print("\n[Stage 5] Result Saving and Report Generation")
        print("-" * 40)

        # 基于论文选择参数（而不仅仅是累积重要性）
        paper_selected_features = [
            'time', 'stack_voltage', 'current', 'current_density',
            'hydrogen_inlet_temp', 'coolant_flow', 'air_outlet_flow'
        ]

        # 过滤出实际存在的特征
        actual_paper_features = [f for f in paper_selected_features
                                if f in data_dict['feature_names'] or f == 'time']

        csv_path, report_path = save_results_and_report(
            importance_df, model_metrics, data_dict,
            actual_paper_features, chart_paths, RESULTS_DIR
        )

        # 6. 完成总结
        print("\n" + "=" * 80)
        print("Analysis Complete!")
        print("=" * 80)

        print(f"\nGenerated Files:")
        print(f"  1. Feature importance results (CSV): {csv_path}")
        print(f"  2. Detailed analysis report (TXT): {report_path}")

        for i, path in enumerate(chart_paths, 3):
            print(f"  {i}. Visualization chart (PNG): {path}")

        print("\nPaper Key Findings Verification:")
        print("-" * 40)
        print("Paper Figure 3.9 shows air outlet flow as the most important parameter.")
        print("In our analysis:")

        if 'air_outlet_flow' in importance_df['feature'].values:
            rank = importance_df[importance_df['feature'] == 'air_outlet_flow'].index[0] + 1
            percent = importance_df[importance_df['feature'] == 'air_outlet_flow']['importance_percent'].values[0]
            print(f"  Air outlet flow rank: #{rank} ({percent:.1f}%)")
            if rank == 1:
                print(f"  ✓ Consistent with paper conclusion")
            elif rank <= 3:
                print(f"  ✓ Partially consistent (top 3)")
            else:
                print(f"  ⚠ Not fully consistent")
        else:
            print(f"  ⚠ Air outlet flow parameter not included in data")

        print("\nNote: U1-U5 individual cell voltage columns have been excluded from analysis")
        print("      (their sum equals stack voltage, which would cause information leakage).")

        print("\nNext Steps Recommendations:")
        print("  1. Build performance degradation prediction model using selected key parameters")
        print("  2. Validate model performance on different datasets")
        print("  3. Combine with mechanistic analysis to explain physical reasons behind parameter importance")
        print("  4. Consider time series characteristics in model design")

    except Exception as e:
        print(f"\nError occurred during analysis: {e}")
        import traceback
        traceback.print_exc()


# ==================== 执行程序 ====================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PEMFC Monitoring Parameter Importance Analysis Based on CatBoost - Starting Execution")
    print("=" * 80 + "\n")

    # 执行主分析
    main()