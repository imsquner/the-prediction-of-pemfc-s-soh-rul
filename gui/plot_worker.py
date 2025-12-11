# 子进程绘图脚本改为直接返回Figure对象，用于Qt界面嵌入
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 滤波库
import matplotlib
import sys
from typing import List, Tuple, Optional
# 设置Matplotlib后端为QtAgg（用于嵌入Qt界面）
matplotlib.use('QtAgg')

TIME_ALIASES = ['time', 'timestamp', 'timeh', 'time(h)', '采样时间', '时间', 'hour', 'hrs']
VOLTAGE_ALIASES = ['voltage', 'volt', 'utot', 'utotv', 'stackvoltage', 'cellvoltage', '电压']


def _normalize_col(col: str) -> str:
    return ''.join(ch for ch in col.lower() if ch.isalnum())


def _select_column(df: pd.DataFrame, preferred: Optional[str], aliases: List[str]) -> Optional[str]:
    if preferred and preferred in df.columns:
        return preferred
    normalized_aliases = {_normalize_col(a) for a in aliases}
    for col in df.columns:
        norm = _normalize_col(col)
        if norm in normalized_aliases:
            return col
    for col in df.columns:
        norm = _normalize_col(col)
        if any(alias in norm for alias in normalized_aliases):
            return col
    return None

# 特征中英文映射字典（根据用户需求配置）
FEATURE_CN_MAP = {
    'time': '时间',
    'stack_voltage': '堆栈电压',
    'current_density': '电流密度',
    'current': '电流',
    'hydrogen_inlet_temp': '氢气入口温度',
    'hydrogen_outlet_temp': '氢气出口温度',
    'air_inlet_temp': '空气入口温度',
    'air_outlet_temp': '空气出口温度',
    'coolant_inlet_temp': '冷却水入口温度',
    'coolant_outlet_temp': '冷却水出口温度',
    'air_inlet_pressure': '空气入口压力',
    'air_outlet_pressure': '空气出口压力',
    'hydrogen_outlet_pressure': '氢气出口压力',
    'hydrogen_inlet_pressure': '氢气入口压力',
    'hydrogen_inlet_flow': '氢气入口流量',
    'hydrogen_outlet_flow': '氢气出口流量',
    'air_inlet_flow': '空气入口流量',
    'air_outlet_flow': '空气出口流量',
    'coolant_flow': '冷却水流量',
    'air_inlet_humidity': '空气入口湿度'
}


def plot_feature_importance(csv_path, top_n=5):
    """
    绘制特征重要性柱状图（直接返回Figure对象）
    :param csv_path: 特征重要性CSV路径
    :param top_n: 显示前N个特征
    :return: (fig, feature_cn_list) 图对象+中文特征名列表
    """
    try:
        # 1. 读取并验证CSV数据
        print(f"[特征重要性-绘图] 开始读取CSV文件：{csv_path}，显示前{top_n}个特征")
        df = pd.read_csv(csv_path)
        required_cols = ["feature", "importance", "importance_percent"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CSV缺少必要列：{col}（需包含{required_cols}）")

        # 2. 按重要性降序排序，取前top_n个
        # 去重并按重要性排序，避免重复条目
        df = df.drop_duplicates(subset=["feature"], keep="first")
        df_sorted = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
        feature_en_list = df_sorted['feature'].to_numpy().tolist()
        # 转换为中文特征名并确保都是字符串（避免 None）
        feature_cn_list = [str(FEATURE_CN_MAP.get(f, f)) if f is not None else "" for f in feature_en_list]
        print(f"[特征重要性-绘图] 筛选前{top_n}个特征（中文）：{feature_cn_list}")

        # 3. 配置中文字体（修复字体找不到问题）
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]  # 避免缺失字体告警
        plt.rcParams["axes.unicode_minus"] = False

        # 4. 绘制柱状图
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100, facecolor='white')
        bars = ax.bar(
            feature_cn_list,  # 显示中文特征名
            df_sorted["importance"],
            color="#165DFF",
            alpha=0.8,
            edgecolor="#1D2129",
            linewidth=1
        )

        # 5. 添加数值标签（显示百分比）
        for i, bar in enumerate(bars):
            height = bar.get_height()
            percent = df_sorted["importance_percent"].iloc[i]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + height * 0.01,
                f"{percent:.1f}%",
                ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color="#1D2129"
            )

        # 6. 图表样式优化（解决模糊问题）
        ax.set_title(f"PEMFC 监测参数重要性排序（前{top_n}）",
                     fontsize=16, color="#165DFF", fontweight="bold", pad=20)
        ax.set_xlabel("监测参数（特征）", fontsize=12, color="#1D2129")
        ax.set_ylabel("重要性分数", fontsize=12, color="#1D2129")
        ax.set_ylim(0, np.max(df_sorted["importance"]) * 1.1)
        # 特征名旋转45度，确保长名称可读
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.grid(True, axis='y', alpha=0.3, linestyle='--', color="#E5E6EB")
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#E5E6EB")
        ax.spines['bottom'].set_color("#E5E6EB")

        plt.tight_layout()
        return fig, feature_cn_list  # 返回图对象和中文特征列表

    except Exception as e:
        print(f"[特征重要性-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def plot_voltage_filter(raw_csv_path, filtered_csv_path, window_size=10, threshold=5.0,
                       time_col: Optional[str] = None, voltage_col: Optional[str] = None):
    """
    绘制电压滤波对比图（原始数据 vs 滤波后数据）
    :param raw_csv_path: 原始CSV路径（固定）
    :param filtered_csv_path: 滤波后CSV路径（最新）
    :param window_size: 移动平均窗口大小
    :param threshold: 尖峰阈值（百分比）
    :return: fig 图对象
    """
    try:
        print(f"[电压滤波-绘图] 开始处理：原始文件={raw_csv_path}，滤波文件={filtered_csv_path}")
        print(f"[电压滤波-绘图] 滤波参数：窗口大小={window_size}，阈值={threshold}%")

        # 1. 读取原始数据和滤波后数据
        df_raw = pd.read_csv(raw_csv_path, encoding_errors="ignore")
        df_filtered = pd.read_csv(filtered_csv_path, encoding_errors="ignore")
        print(f"[电压滤波-绘图] 原始数据行数：{len(df_raw)}，滤波后数据行数：{len(df_filtered)}")

        # 2. 自动/手动识别时间列和电压列（允许两文件列名不同，如 Time (h) vs time）
        raw_time = _select_column(df_raw, time_col, TIME_ALIASES)
        filt_time = _select_column(df_filtered, time_col, TIME_ALIASES)
        raw_voltage = _select_column(df_raw, voltage_col, VOLTAGE_ALIASES)
        filt_voltage = _select_column(df_filtered, voltage_col, VOLTAGE_ALIASES)

        if raw_time is None or filt_time is None or raw_voltage is None or filt_voltage is None:
            raise ValueError(
                "无法识别时间列或电压列，请在界面手动选择或确保列名包含 time/Time (h)/Utot (V)/voltage；"
                f"原始列: {list(df_raw.columns)}；滤波列: {list(df_filtered.columns)}"
            )
        print(f"[电压滤波-绘图] 识别列：原始时间={raw_time}，滤波时间={filt_time}，原始电压={raw_voltage}，滤波电压={filt_voltage}")

        # 3. 数据清洗（对齐时间轴）
        df_raw = df_raw[[raw_time, raw_voltage]].dropna().rename(columns={raw_time: 'time', raw_voltage: 'voltage'})
        df_filtered = df_filtered[[filt_time, filt_voltage]].dropna().rename(columns={filt_time: 'time', filt_voltage: 'voltage'})
        # 按时间列对齐数据
        df_merged = pd.merge(df_raw, df_filtered, on='time', suffixes=('_raw', '_filtered'))
        if len(df_merged) == 0:
            raise ValueError("原始数据和滤波后数据无匹配的时间点")
        print(f"[电压滤波-绘图] 对齐后有效数据点：{len(df_merged)}")

        # 4. 提取数据
        time_data = df_merged['time'].to_numpy()
        raw_voltage = df_merged['voltage_raw'].to_numpy()
        filtered_voltage = df_merged['voltage_filtered'].to_numpy()

        # 5. 计算差异百分比（用于下方子图）
        pct_diff = np.abs((raw_voltage - filtered_voltage) / raw_voltage * 100)
        spike_count = np.sum(pct_diff > threshold)
        print(f"[电压滤波-绘图] 检测到{spike_count}个尖峰（阈值>{threshold}%）")

        # 6. 配置中文字体（修复字体问题）
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        # 7. 绘制对比图（双子图）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), dpi=100, facecolor='white')

        # 子图1：原始电压 vs 滤波后电压
        ax1.plot(time_data, raw_voltage, label='原始电压', color='#165DFF', alpha=0.7, linewidth=1.5)
        ax1.plot(time_data, filtered_voltage, label='滤波后电压', color='#F53F3F', alpha=0.8, linewidth=2)
        # 标记尖峰点
        spike_indices = pct_diff > threshold
        if spike_count > 0:
            ax1.scatter(time_data[spike_indices], raw_voltage[spike_indices],
                        color='#FFD700', s=50, zorder=5, label=f'尖峰点（{spike_count}个）')
        ax1.set_title(f'电压滤波对比图（窗口大小：{window_size}，阈值：{threshold}%）',
                      fontsize=14, fontweight='bold', color='#1D2129')
        ax1.set_xlabel('时间', fontsize=12, color='#1D2129')
        ax1.set_ylabel('电压 (V)', fontsize=12, color='#1D2129')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--', color='#E5E6EB')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#E5E6EB')

        # 子图2：差异百分比
        ax2.plot(time_data, pct_diff, color='#00B42A', alpha=0.8, linewidth=1.5)
        ax2.axhline(y=threshold, color='#F53F3F', linestyle='--', alpha=0.7,
                    label=f'阈值线（{threshold}%）')
        ax2.fill_between(time_data, threshold, pct_diff,
                         where=(pct_diff > threshold), color='#FFE6E6', alpha=0.5)
        ax2.set_title('电压差异百分比（原始 vs 滤波后）', fontsize=14, fontweight='bold', color='#1D2129')
        ax2.set_xlabel('时间', fontsize=12, color='#1D2129')
        ax2.set_ylabel('差异百分比 (%)', fontsize=12, color='#1D2129')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--', color='#E5E6EB')
        for spine in ax2.spines.values():
            spine.set_edgecolor('#E5E6EB')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"[电压滤波-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def plot_soh_prediction(csv_path, prediction_duration=100):
    """
    绘制SOH寿命预测曲线（返回Figure对象）
    :param csv_path: SOH数据CSV路径
    :param prediction_duration: 预测时长
    :return: fig 图对象
    """
    try:
        # 配置中文字体
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        # 读取SOH数据
        df = pd.read_csv(csv_path, encoding_errors="ignore")
        print(f"[SOH预测-绘图] 读取数据行数：{len(df)}")

        # 识别时间列和SOH列
        time_col = None
        soh_col = None
        for col in df.columns:
            if 'time' in col.lower() or 'timestamp' in col.lower():
                time_col = col
            elif 'soh' in col.lower() or '健康度' in col.lower():
                soh_col = col
        if time_col is None or soh_col is None:
            raise ValueError("无法识别时间列或SOH列")
        print(f"[SOH预测-绘图] 识别列：时间列={time_col}，SOH列={soh_col}")

        # 数据清洗
        df = df[[time_col, soh_col]].dropna()
        time_data = df[time_col].to_numpy()
        soh_data = df[soh_col].to_numpy()

        # 模拟预测（实际项目需替换为真实模型）
        future_time = np.linspace(time_data[-1], time_data[-1] + prediction_duration, 50)
        # 假设SOH线性衰减
        soh_pred = soh_data[-1] * (1 - np.linspace(0, 0.3, 50))  # 30%衰减

        # 绘制图表
        fig, ax = plt.subplots(figsize=(12, 6), dpi=100, facecolor='white')
        ax.plot(time_data, soh_data, label='历史SOH数据', color='#165DFF', linewidth=2, alpha=0.8)
        ax.plot(future_time, soh_pred, label=f'预测SOH（{prediction_duration}小时）',
                color='#F53F3F', linewidth=2, linestyle='--')
        ax.set_title('PEMFC SOH衰减预测曲线', fontsize=16, color='#1D2129', fontweight='bold')
        ax.set_xlabel('时间（小时）', fontsize=12, color='#1D2129')
        ax.set_ylabel('健康状态（SOH）', fontsize=12, color='#1D2129')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--', color='#E5E6EB')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#E5E6EB')
        ax.spines['bottom'].set_color('#E5E6EB')

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"[SOH预测-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def _detect_time_and_voltage(df: pd.DataFrame, time_col: Optional[str] = None,
                             voltage_col: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """在原始或处理后数据中自动/手动识别时间列和电压列"""
    detected_time = _select_column(df, time_col, TIME_ALIASES)
    detected_voltage = _select_column(df, voltage_col, VOLTAGE_ALIASES)

    return detected_time, detected_voltage


def _select_numeric_signals(df: pd.DataFrame, exclude: List[str], max_signals: int = 5) -> List[str]:
    """选择用于展示的数值信号列，排除时间/电压等列"""
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    if not numeric_cols:
        return []
    # 依据方差排序，选取波动较大的信号便于展示
    variances = []
    for col in numeric_cols:
        try:
            var_val = float(np.nan_to_num(np.var(df[col].to_numpy(dtype=float))))
        except Exception:
            var_val = 0.0
        variances.append((col, var_val))
    variances = sorted(variances, key=lambda x: x[1], reverse=True)
    return [c for c, _ in variances[:max_signals]]


def _estimate_rul_from_series(x_axis: np.ndarray, series: np.ndarray,
                              threshold_ratio: float = 0.96) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """基于阈值比例估计EOL时间与RUL（简单线性插值）。

    返回: (阈值, EOL时间, RUL=EOL-当前时间)。若无法计算，则返回(None, None, None)。
    """
    try:
        if len(x_axis) == 0 or len(series) == 0:
            return None, None, None
        x_axis = np.asarray(x_axis, dtype=float)
        series = np.asarray(series, dtype=float)
        if not np.all(np.isfinite(x_axis)) or not np.all(np.isfinite(series)):
            return None, None, None

        # 按x排序确保插值正确
        order = np.argsort(x_axis)
        x_sorted = x_axis[order]
        y_sorted = series[order]

        threshold = y_sorted[0] * threshold_ratio
        below = np.where(y_sorted <= threshold)[0]
        if len(below) == 0:
            return threshold, None, None
        idx = below[0]
        if idx == 0:
            eol_time = float(x_sorted[0])
        else:
            x1, x2 = x_sorted[idx - 1], x_sorted[idx]
            y1, y2 = y_sorted[idx - 1], y_sorted[idx]
            if y2 == y1:
                eol_time = float(x2)
            else:
                # 线性插值求过阈值时刻
                eol_time = float(x1 + (threshold - y1) * (x2 - x1) / (y2 - y1))

        current_time = float(x_sorted[-1])
        rul = max(eol_time - current_time, 0.0)
        return float(threshold), float(eol_time), float(rul)
    except Exception:
        return None, None, None


def plot_raw_views(csv_path: str, dataset_label: str = "FC1", max_signals: int = 5,
                   time_col: Optional[str] = None, voltage_col: Optional[str] = None,
                   selected_signals: Optional[List[str]] = None):
    """绘制原始数据视图：仅保留电压与关键信号对比视图，支持无额外信号时仅绘制电压。"""
    try:
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        df = pd.read_csv(csv_path, encoding_errors="ignore")
        if df.empty:
            raise ValueError("原始数据为空，无法绘图")

        time_col, voltage_col = _detect_time_and_voltage(df, time_col=time_col, voltage_col=voltage_col)
        if time_col is None or voltage_col is None:
            raise ValueError(
                "无法识别时间列或电压列，请检查原始数据；"
                f"可用列: {list(df.columns)}"
            )

        base_cols = [time_col, voltage_col]
        available_signals = _select_numeric_signals(df, exclude=base_cols, max_signals=max_signals)
        signal_cols = available_signals
        if selected_signals:
            signal_cols = [c for c in selected_signals if c in available_signals]
            if not signal_cols:
                signal_cols = available_signals

        # 清洗与排序
        plot_df = df[base_cols + signal_cols].dropna()
        plot_df = plot_df.sort_values(by=time_col)
        time_data = plot_df[time_col].to_numpy()
        voltage = plot_df[voltage_col].to_numpy()

        # 归一化信号便于同图对比
        norm_signals = {}
        for col in signal_cols:
            series = plot_df[col].to_numpy()
            min_v = np.nanmin(series)
            max_v = np.nanmax(series)
            norm_signals[col] = (series - min_v) / (max_v - min_v + 1e-9)

        # 映射中文名称
        def cn_label(col: str) -> str:
            return str(FEATURE_CN_MAP.get(col, col))

        # 如果没有可视化信号，仅绘制电压单图
        if not signal_cols:
            fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
            ax.plot(time_data, voltage, label="堆栈电压", color="#165DFF", linewidth=2.4, alpha=0.95)
            ax.set_title(f"{dataset_label} 电压随时间", fontsize=14, fontweight="bold", color="#1D2129")
            ax.set_xlabel("时间", fontsize=12, color="#1D2129")
            ax.set_ylabel("电压 (V)", fontsize=12, color="#1D2129")
            ax.grid(True, alpha=0.3, linestyle='--', color="#E5E6EB")
            ax.legend(loc='best')
            plt.tight_layout()
            return fig

        fig, ax = plt.subplots(figsize=(14, 6), dpi=110)

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(10)]
        twin = ax.twinx()
        markers = ['o', 's', '^', 'D', 'v', '<', '>']
        for idx, col in enumerate(signal_cols):
            marker = markers[idx % len(markers)]
            twin.plot(time_data, norm_signals[col], marker=marker, markersize=3,
                      linewidth=1.0, alpha=0.8, label=f"{cn_label(col)} (归一化)",
                      color=colors[idx % len(colors)], zorder=1)

        ax.plot(time_data, voltage, label="堆栈电压", color="#165DFF", linewidth=2.8,
                alpha=0.95, zorder=3)

        # 标记电压异常点（简单z-score>3）
        if len(voltage) > 0:
            v_mean = np.nanmean(voltage)
            v_std = np.nanstd(voltage) + 1e-9
            z = np.abs((voltage - v_mean) / v_std)
            spike_idx = z > 3
            if spike_idx.any():
                ax.scatter(time_data[spike_idx], voltage[spike_idx], color="#F53F3F",
                           s=28, marker='x', label="电压异常点", zorder=4)

        ax.set_title(f"{dataset_label} 电压与关键信号（归一化信号右轴）", fontsize=14,
                     fontweight="bold", color="#1D2129")
        ax.set_xlabel("时间", fontsize=12, color="#1D2129")
        ax.set_ylabel("电压 (V)", fontsize=12, color="#1D2129")
        twin.set_ylabel("归一化信号", fontsize=12, color="#1D2129")
        ax.grid(True, alpha=0.3, linestyle='--', color="#E5E6EB")

        handles_v, labels_v = ax.get_legend_handles_labels()
        handles_s, labels_s = twin.get_legend_handles_labels()
        ax.legend(handles_v + handles_s, labels_v + labels_s, loc='best')

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[原始数据-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def plot_prediction_vs_true(pred_csv_path: str, max_points: int = 1500, dataset_label: str = "FC1"):
    """预测值对比真值，可选截断长度（用于FC2较短时间轴）"""
    try:
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        df = pd.read_csv(pred_csv_path, encoding_errors="ignore")
        if df.empty:
            raise ValueError("预测结果文件为空，无法绘图")

        # 如存在 dataset 列，先按标签过滤
        if "dataset" in df.columns:
            df = df[df["dataset"].str.upper() == dataset_label.upper()]
            if df.empty:
                raise ValueError(f"预测结果中未找到数据集 {dataset_label} 的记录")

        # 按时间排序（若存在时间列）
        if "time" in df.columns:
            df = df.sort_values(by="time")

        required = ["target", "prediction"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"预测结果缺少必要列：{col}；现有列: {list(df.columns)}")

        if len(df) > max_points:
            # 等距抽样，覆盖全时间轴（含未来）
            indices = np.linspace(0, len(df) - 1, max_points).astype(int)
            plot_df = df.iloc[indices].copy()
        else:
            plot_df = df.copy()
        # 使用时间列优先，其次使用样本序号
        if "time" in plot_df.columns:
            x_axis = plot_df["time"].to_numpy()
            x_label = "时间 (h)"
        else:
            x_axis = np.arange(len(plot_df))
            x_label = "样本序号"

        fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
        target_series = plot_df["target"].to_numpy()
        pred_series = plot_df["prediction"].to_numpy()

        # 标记未来滚动预测（split包含future 或 target缺失）
        if "split" in plot_df.columns:
            future_mask = plot_df["split"].str.contains("future", case=False, na=False).to_numpy()
        else:
            future_mask = np.isnan(target_series)
        known_mask = ~future_mask

        # 已知区间
        ax.plot(x_axis[known_mask], target_series[known_mask], label="真实值", color="#165DFF", linewidth=1.8)
        ax.plot(x_axis[known_mask], pred_series[known_mask], label="预测值(已知)", color="#F53F3F", linewidth=1.8, linestyle='--')

        # 未来滚动预测区间
        if future_mask.any():
            ax.plot(x_axis[future_mask], pred_series[future_mask], label="滚动预测(未来)",
                    color="#00B42A", linestyle=':', linewidth=1.6, alpha=0.9)

        # 置信区间可选（仅对已知区间）
        if "ci_lower" in plot_df.columns and "ci_upper" in plot_df.columns:
            ax.fill_between(x_axis[known_mask], plot_df.loc[known_mask, "ci_lower"], plot_df.loc[known_mask, "ci_upper"],
                            color="#F53F3F", alpha=0.15, label="95% 置信区间")

        # 计算指标（仅已知区间）
        valid_mask = known_mask & ~np.isnan(target_series) & ~np.isnan(pred_series)
        if valid_mask.any():
            mae = float(np.mean(np.abs(target_series[valid_mask] - pred_series[valid_mask])))
            rmse = float(np.sqrt(np.mean((target_series[valid_mask] - pred_series[valid_mask]) ** 2)))
            metrics_txt = f"MAE={mae:.4f}, RMSE={rmse:.4f}"
        else:
            metrics_txt = "MAE=NA, RMSE=NA"

        # 估算RUL并标注（基于0.96阈值线性插值）
        rul_info_lines = []
        true_thr, true_eol, true_rul = _estimate_rul_from_series(x_axis[~np.isnan(target_series)],
                                    target_series[~np.isnan(target_series)])
        pred_thr, pred_eol, pred_rul = _estimate_rul_from_series(x_axis[~np.isnan(pred_series)],
                                    pred_series[~np.isnan(pred_series)])

        if true_eol is not None:
            ax.axvline(true_eol, color="#165DFF", linestyle=":", alpha=0.7, linewidth=1.3,
                       label=f"真实EOL≈{true_eol:.1f}h")
            rul_info_lines.append(f"真实RUL≈{true_rul:.1f}h @EOL≈{true_eol:.1f}h")
        if pred_eol is not None:
            ax.axvline(pred_eol, color="#F53F3F", linestyle=":", alpha=0.7, linewidth=1.3,
                       label=f"预测EOL≈{pred_eol:.1f}h")
            rul_info_lines.append(f"预测RUL≈{pred_rul:.1f}h @EOL≈{pred_eol:.1f}h")

        # 标题附加RUL文本
        rul_txt = " | ".join(rul_info_lines) if rul_info_lines else "RUL不可用"
        ax.set_title(f"{dataset_label} 预测 vs 真实 ({max_points} 点) | {metrics_txt} | {rul_txt}",
                     fontsize=14, fontweight="bold", color="#1D2129")
        ax.set_xlabel(x_label, fontsize=12, color="#1D2129")
        ax.set_ylabel("电压 / 目标值", fontsize=12, color="#1D2129")
        ax.grid(True, alpha=0.3, linestyle='--', color="#E5E6EB")
        ax.legend(loc='best')

        # 拉长时间轴到1500h以便观察滚动RUL
        if np.issubdtype(x_axis.dtype, np.number):
            x_min = float(np.nanmin(x_axis)) if len(x_axis) else 0.0
            x_max = float(np.nanmax(x_axis)) if len(x_axis) else 0.0
            ax.set_xlim(left=x_min, right=max(1500.0, x_max * 1.02))

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[预测对比-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def plot_metrics_table(metrics_csv_path: str):
    """绘制评估指标表格（单独窗口显示）。"""
    try:
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        df = pd.read_csv(metrics_csv_path, encoding_errors="ignore")
        if df.empty:
            raise ValueError("指标文件为空，无法绘制")

        # 只保留常见指标列
        keep_cols = [c for c in ["dataset", "MAE", "RMSE", "MAPE", "R2"] if c in df.columns]
        df = df[keep_cols]

        fig, ax = plt.subplots(figsize=(6, 2 + 0.35 * len(df)), dpi=130)
        ax.axis('off')

        table = ax.table(cellText=df.values.tolist(),
                 colLabels=df.columns.tolist(),
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.3)

        ax.set_title('评估指标表格', fontsize=14, fontweight='bold', color='#1D2129', pad=12)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[指标表-绘图错误] {str(e)}", file=sys.stderr)
        raise e


def plot_voltage_overlay(raw_csv_path: str, filtered_csv_path: str,
                         dataset_label: str = "FC1",
                         time_col: Optional[str] = None,
                         voltage_col: Optional[str] = None):
    """单图叠加对比原始与滤波后电压，原始蓝色、滤波红色。"""
    try:
        plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False

        df_raw = pd.read_csv(raw_csv_path, encoding_errors="ignore")
        df_filtered = pd.read_csv(filtered_csv_path, encoding_errors="ignore")
        if df_raw.empty or df_filtered.empty:
            raise ValueError("原始或滤波后数据为空")

        raw_time = _select_column(df_raw, time_col, TIME_ALIASES)
        filt_time = _select_column(df_filtered, time_col, TIME_ALIASES)
        raw_voltage = _select_column(df_raw, voltage_col, VOLTAGE_ALIASES)
        filt_voltage = _select_column(df_filtered, voltage_col, VOLTAGE_ALIASES)
        if raw_time is None or filt_time is None or raw_voltage is None or filt_voltage is None:
            raise ValueError("无法识别时间或电压列，请检查列名")

        df_raw = df_raw[[raw_time, raw_voltage]].dropna().rename(columns={raw_time: 'time', raw_voltage: 'voltage_raw'})
        df_filtered = df_filtered[[filt_time, filt_voltage]].dropna().rename(columns={filt_time: 'time', filt_voltage: 'voltage_filt'})

        df_merged = pd.merge(df_raw, df_filtered, on='time', how='inner')
        if df_merged.empty:
            raise ValueError("原始与滤波数据时间轴无交集，无法对齐")

        df_merged = df_merged.sort_values(by='time')
        time_data = df_merged['time'].to_numpy()
        v_raw = df_merged['voltage_raw'].to_numpy()
        v_filt = df_merged['voltage_filt'].to_numpy()

        fig, ax = plt.subplots(figsize=(14, 6), dpi=110, facecolor='white')
        ax.plot(time_data, v_raw, label='原始电压 (蓝)', color='#165DFF', linewidth=2.0, alpha=0.85)
        ax.plot(time_data, v_filt, label='滤波后电压 (红)', color='#F53F3F', linewidth=2.0, alpha=0.9)

        ax.set_title(f"{dataset_label} 原始 vs 滤波后电压", fontsize=14, fontweight='bold', color='#1D2129')
        ax.set_xlabel('时间', fontsize=12, color='#1D2129')
        ax.set_ylabel('电压 (V)', fontsize=12, color='#1D2129')
        ax.grid(True, alpha=0.3, linestyle='--', color='#E5E6EB')
        ax.legend(loc='best')

        for spine in ax.spines.values():
            spine.set_color('#E5E6EB')

        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"[原始vs滤波-绘图错误] {str(e)}", file=sys.stderr)
        raise e