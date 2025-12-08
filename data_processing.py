"""质子交换膜燃料电池(PEMFC)数据集预处理 - 增强全列处理版本
目标：对PEMFC原始时序数据集进行标准化处理，增强对所有数值列的滤波处理能力
功能：数据加载→基础预处理→自适应参数计算→全列多阶段滤波→电压列特殊处理（含滚动窗口平均）→格式转换
优化：增强全列处理覆盖，自适应参数优化，多阶段协同尖峰处理，电压列滚动窗口平滑
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # 小波变换库
import warnings
from datetime import datetime
from scipy.signal import savgol_filter, medfilt
from scipy import stats

# 设置中文字体和英文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['SimHei', 'Times New Roman']

warnings.filterwarnings('ignore')

# ==================== 信噪比计算函数 ====================
def signaltonoise(signal, axis=0, ddof=0):
    """计算信号的信噪比(SNR)"""
    signal = np.asanyarray(signal)
    mean = signal.mean(axis)
    std = signal.std(axis=axis, ddof=ddof)
    return 20 * np.log10(np.where(std == 0, 0, mean / std))

# ==================== 增强版配置参数 ====================
PROCESSING_CONFIG = {
    # 核心参数
    'base_wavelet': 'db8',  # 基础小波基函数
    'base_decomposition_level': 4,  # 基础分解层数

    # 自适应参数范围
    'median_window_range': [3, 7],  # 中值滤波窗口范围(奇数)
    'wavelet_level_range': [3, 5],  # 小波分解层数范围
    'savgol_window_range': [7, 21],  # Savitzky-Golay窗口范围

    # 自适应参数计算配置
    'volatility_thresholds': {
        'low': 0.001,    # 低波动阈值
        'medium': 0.01,  # 中等波动阈值
        'high': 0.05     # 高波动阈值
    },
    'data_length_thresholds': {
        'short': 1000,   # 短数据阈值
        'medium': 5000,  # 中等数据阈值
        'long': 10000    # 长数据阈值
    },

    # 关键列特殊处理
    'key_columns': ['stack_voltage', 'current_density', 'current'],
    'special_treatment_columns': {
        'stack_voltage': {
            'voltage_normal_range': [3.0, 3.5],
            'spike_threshold': 0.003,
            'end_points_to_check': 20,
            'rolling_window_size': 500  # 新增：滚动窗口平均大小
        },
        'temperature_columns': ['hydrogen_inlet_temp', 'hydrogen_outlet_temp',
                               'air_inlet_temp', 'air_outlet_temp',
                               'coolant_inlet_temp', 'coolant_outlet_temp'],
        'pressure_columns': ['air_inlet_pressure', 'air_outlet_pressure',
                            'hydrogen_outlet_pressure', 'hydrogen_inlet_pressure'],
        'flow_columns': ['hydrogen_inlet_flow', 'hydrogen_outlet_flow',
                        'air_inlet_flow', 'air_outlet_flow', 'coolant_flow']
    },

    # 多阶段滤波配置
    'multistage_filtering': {
        'pre_scan_window': 5,      # 极端值预扫描窗口
        'median_filter_first': True,  # 先应用中值滤波
        'wavelet_denoise_second': True,  # 然后小波去噪
        'smooth_final': True,      # 最后平滑
        'adaptive_threshold': 0.8  # 自适应阈值系数
    },

    # 输出配置
    'save_individual_charts': True,
    'save_column_statistics': True  # 新增：保存各列统计信息
}

# ==================== 自适应参数计算函数 ====================
def calculate_column_statistics(data_column):
    """
    计算数据列的统计特征，用于自适应参数选择

    参数:
    - data_column: 数据列（一维数组）

    返回:
    - stats_dict: 统计特征字典
    """
    stats_dict = {}

    # 基本统计量
    stats_dict['length'] = len(data_column)
    stats_dict['mean'] = np.mean(data_column)
    stats_dict['std'] = np.std(data_column)
    stats_dict['min'] = np.min(data_column)
    stats_dict['max'] = np.max(data_column)

    # 波动系数（标准差/均值）
    if stats_dict['mean'] != 0:
        stats_dict['coefficient_of_variation'] = stats_dict['std'] / abs(stats_dict['mean'])
    else:
        stats_dict['coefficient_of_variation'] = 0

    # 差分统计（评估噪声水平）
    data_array = np.asarray(data_column, dtype=np.float64)
    diffs = np.diff(data_array)
    stats_dict['diff_mean'] = float(np.mean(np.abs(diffs)))
    stats_dict['diff_std'] = float(np.std(diffs))

    # 尖峰检测（忽略NaN并转为浮点）
    z_scores = np.abs(stats.zscore(data_array, nan_policy='omit'))  # type: ignore[arg-type]
    z_scores = np.nan_to_num(z_scores, nan=0.0)
    stats_dict['spike_count'] = int(np.sum(z_scores > 3))  # 超过3倍标准差的点

    # 数据平稳性评估（基于滑动窗口）
    window_size = min(100, len(data_column) // 10)
    if window_size > 1:
        means = np.array([np.mean(data_array[i:i+window_size])
                 for i in range(0, len(data_array)-window_size, window_size)])
        stats_dict['window_std'] = float(np.std(means))
    else:
        stats_dict['window_std'] = 0

    return stats_dict

def determine_filter_parameters(column_name, column_data, column_stats):
    """
    根据列特征确定滤波参数

    参数:
    - column_name: 列名
    - column_data: 列数据
    - column_stats: 列统计特征

    返回:
    - filter_params: 滤波参数字典
    """
    filter_params = {}

    # 1. 中值滤波窗口大小
    volatility = column_stats['coefficient_of_variation']
    length = column_stats['length']

    if volatility < PROCESSING_CONFIG['volatility_thresholds']['low']:
        # 低波动数据，使用小窗口
        median_window = PROCESSING_CONFIG['median_window_range'][0]
    elif volatility < PROCESSING_CONFIG['volatility_thresholds']['medium']:
        # 中等波动数据，使用中等窗口
        median_window = np.mean(PROCESSING_CONFIG['median_window_range']).astype(int)
        # 确保是奇数
        median_window = median_window if median_window % 2 == 1 else median_window + 1
    else:
        # 高波动数据，使用大窗口
        median_window = PROCESSING_CONFIG['median_window_range'][1]

    filter_params['median_window'] = median_window

    # 2. 小波分解层数
    if length < PROCESSING_CONFIG['data_length_thresholds']['short']:
        wavelet_level = PROCESSING_CONFIG['wavelet_level_range'][0]
    elif length < PROCESSING_CONFIG['data_length_thresholds']['medium']:
        wavelet_level = np.mean(PROCESSING_CONFIG['wavelet_level_range']).astype(int)
    else:
        wavelet_level = PROCESSING_CONFIG['wavelet_level_range'][1]
    # 限制最大分解层数（根据数据长度）
    max_possible_level = int(np.log2(len(column_data)))
    filter_params['wavelet_level'] = min(wavelet_level, max_possible_level, 5)
    # 关键调整：电压列强制小波层数为4（原默认5层，减少过度分解）
    if 'voltage' in column_name.lower():
        filter_params['wavelet_level'] = 4

    # 3. 平滑窗口大小
    if column_stats['spike_count'] / length > 0.1:
        # 尖峰较多，使用较大窗口
        savgol_window = PROCESSING_CONFIG['savgol_window_range'][1]
    else:
        # 尖峰较少，使用中等窗口
        savgol_window = np.mean(PROCESSING_CONFIG['savgol_window_range']).astype(int)

    # 确保窗口是奇数且不超过数据长度
    savgol_window = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
    savgol_window = min(savgol_window, len(column_data) - 1 if len(column_data) % 2 == 0 else len(column_data))
    filter_params['savgol_window'] = max(3, savgol_window)  # 最小为3

    # 4. 小波基函数选择
    if 'voltage' in column_name.lower():
        filter_params['wavelet_type'] = 'db8'  # 电压数据使用db8
    elif 'temp' in column_name.lower():
        filter_params['wavelet_type'] = 'sym8'  # 温度数据使用sym8
    elif 'pressure' in column_name.lower():
        filter_params['wavelet_type'] = 'haar'  # 压力数据使用haar
    else:
        filter_params['wavelet_type'] = PROCESSING_CONFIG['base_wavelet']

    # 5. 尖峰检测阈值
    if column_stats['diff_std'] > 0:
        # 基于差分标准差设置阈值
        filter_params['spike_threshold'] = column_stats['diff_std'] * 2
    else:
        filter_params['spike_threshold'] = 0.01

    return filter_params

# ==================== 多阶段滤波函数 ====================
def extreme_value_pre_scan(data, window_size=5):
    """
    极端值预扫描：检测并初步处理极端异常值

    参数:
    - data: 一维数据序列
    - window_size: 扫描窗口大小

    返回:
    - processed_data: 处理后的数据
    - scan_stats: 扫描统计信息
    """
    scan_stats = {
        'extreme_values_detected': 0,
        'extreme_values_corrected': 0
    }

    if len(data) < window_size * 2:
        return data.copy(), scan_stats

    processed_data = data.copy()
    n = len(data)

    for i in range(n):
        # 动态窗口
        start_idx = max(0, i - window_size)
        end_idx = min(n, i + window_size + 1)
        window = data[start_idx:end_idx]

        # 排除当前点计算窗口统计
        window_without_current = np.concatenate([window[:i-start_idx], window[i-start_idx+1:]])

        if len(window_without_current) > 0:
            window_mean = np.mean(window_without_current)
            window_std = np.std(window_without_current)

            # 检测极端值（5倍标准差）
            if window_std > 0 and abs(data[i] - window_mean) > 5 * window_std:
                scan_stats['extreme_values_detected'] += 1
                # 用窗口均值替换极端值
                processed_data[i] = window_mean
                scan_stats['extreme_values_corrected'] += 1

    return processed_data, scan_stats

def adaptive_wavelet_denoise(data, wavelet_type='db8', level=4, threshold_factor=0.8):
    """
    自适应小波去噪：根据数据特征调整阈值

    参数:
    - data: 一维数据序列
    - wavelet_type: 小波基函数
    - level: 分解层数
    - threshold_factor: 阈值系数（0-1之间）

    返回:
    - denoised_data: 去噪后的数据
    """
    data_array = np.array(data, dtype=np.float64)
    n = len(data_array)

    if n < 2**(level + 1):
        return data_array

    try:
        # 小波分解
        coeffs = pywt.wavedec(data_array, wavelet_type, level=level, mode='per')

        # 自适应阈值计算
        all_coeffs = np.concatenate(coeffs[1:])
        if len(all_coeffs) > 0:
            # 使用稳健的噪声估计
            sigma = np.median(np.abs(all_coeffs)) / 0.6745

            # 自适应阈值：基于数据长度和阈值系数
            adaptive_threshold = sigma * np.sqrt(2 * np.log(n)) * threshold_factor

            # 软阈值处理
            coeffs_th = list(coeffs)
            for i in range(1, len(coeffs_th)):
                coeffs_th[i] = pywt.threshold(coeffs_th[i], adaptive_threshold, mode='soft')

            # 信号重构
            denoised_data = pywt.waverec(coeffs_th, wavelet_type, mode='per')

            # 长度匹配
            if len(denoised_data) > n:
                denoised_data = denoised_data[:n]
            elif len(denoised_data) < n:
                denoised_data = np.pad(denoised_data, (0, n - len(denoised_data)), 'edge')
        else:
            denoised_data = data_array

        return denoised_data
    except Exception as e:
        print(f"    小波去噪错误: {e}，返回原始数据")
        return data_array

def adaptive_savgol_filter(data, window_length, polyorder=2):
    """
    自适应Savitzky-Golay滤波：处理边界情况

    参数:
    - data: 一维数据序列
    - window_length: 窗口长度
    - polyorder: 多项式阶数

    返回:
    - smoothed_data: 平滑后的数据
    """
    data_array = np.array(data, dtype=np.float64)
    n = len(data_array)

    if n < window_length:
        # 数据长度不足，调整窗口
        window_length = n if n % 2 == 1 else n - 1
        window_length = max(window_length, 3)
        polyorder = min(polyorder, window_length - 1)

    try:
        smoothed_data = savgol_filter(data_array, window_length, polyorder, mode='nearest')
        return smoothed_data
    except Exception as e:
        print(f"    Savitzky-Golay滤波错误: {e}，返回原始数据")
        return data_array

def multistage_column_filtering(column_data, column_name, filter_params):
    """
    多阶段列滤波：极端值预扫描 → 中值滤波 → 小波去噪 → 平滑

    参数:
    - column_data: 列数据
    - column_name: 列名
    - filter_params: 滤波参数

    返回:
    - filtered_data: 滤波后的数据
    - filtering_stats: 滤波统计信息
    """
    filtering_stats = {
        'original_std': np.std(column_data),
        'stages_applied': [],
        'spikes_removed': 0
    }

    current_data = column_data.copy()

    # 阶段1：极端值预扫描
    if PROCESSING_CONFIG['multistage_filtering']['pre_scan_window'] > 0:
        pre_scan_window = PROCESSING_CONFIG['multistage_filtering']['pre_scan_window']
        current_data, scan_stats = extreme_value_pre_scan(current_data, pre_scan_window)
        filtering_stats['extreme_values_detected'] = scan_stats['extreme_values_detected']
        filtering_stats['extreme_values_corrected'] = scan_stats['extreme_values_corrected']
        filtering_stats['stages_applied'].append('extreme_value_pre_scan')

    # 阶段2：中值滤波（去孤立尖峰）
    if PROCESSING_CONFIG['multistage_filtering']['median_filter_first']:
        median_window = filter_params['median_window']
        # 确保窗口为奇数且不超过数据长度
        if median_window % 2 == 1 and median_window <= len(current_data):
            try:
                median_filtered = medfilt(current_data, kernel_size=median_window)
                # 计算中值滤波去除的尖峰数
                diff_before = np.abs(np.diff(current_data))
                diff_after = np.abs(np.diff(median_filtered))
                spikes_removed = np.sum(diff_before > filter_params['spike_threshold']) - np.sum(diff_after > filter_params['spike_threshold'])
                filtering_stats['spikes_removed'] += max(0, spikes_removed)

                current_data = median_filtered
                filtering_stats['stages_applied'].append('median_filter')
            except Exception as e:
                print(f"    列 '{column_name}' 中值滤波失败: {e}")

    # 阶段3：小波去噪（除高频噪声）
    if PROCESSING_CONFIG['multistage_filtering']['wavelet_denoise_second']:
        wavelet_type = filter_params['wavelet_type']
        wavelet_level = filter_params['wavelet_level']
        threshold_factor = PROCESSING_CONFIG['multistage_filtering']['adaptive_threshold']

        try:
            wavelet_denoised = adaptive_wavelet_denoise(
                current_data,
                wavelet_type,
                wavelet_level,
                threshold_factor
            )
            current_data = wavelet_denoised
            filtering_stats['stages_applied'].append('wavelet_denoise')
        except Exception as e:
            print(f"    列 '{column_name}' 小波去噪失败: {e}")

    # 阶段4：最终平滑
    if PROCESSING_CONFIG['multistage_filtering']['smooth_final']:
        savgol_window = filter_params['savgol_window']

        try:
            smoothed = adaptive_savgol_filter(current_data, savgol_window, polyorder=2)
            current_data = smoothed
            filtering_stats['stages_applied'].append('savgol_smooth')
        except Exception as e:
            print(f"    列 '{column_name}' Savitzky-Golay平滑失败: {e}")

    # 统计滤波效果
    filtering_stats['final_std'] = np.std(current_data)
    filtering_stats['noise_reduction'] = filtering_stats['original_std'] - filtering_stats['final_std']

    return current_data, filtering_stats

# ==================== 新增：滚动窗口平均函数 ====================
def rolling_window_average(voltage_data, window_size=500):
    """
    对电压列应用滚动窗口平均处理，降低数据波动幅度，提升平滑度

    参数:
    - voltage_data: 电压列数据（一维数组）
    - window_size: 滚动窗口大小，默认为500

    返回:
    - smoothed_data: 平滑后的电压数据
    - stats: 处理统计信息
    """
    print(f"    应用滚动窗口平均处理，窗口大小={window_size}")

    # 转换为numpy数组
    data_array = np.array(voltage_data, dtype=np.float64)
    n = len(data_array)

    # 处理边界情况：数据长度小于窗口大小
    if n < window_size:
        print(f"      警告：数据长度({n})小于窗口大小({window_size})，使用全窗口平均")
        window_size = max(1, n // 2)

    # 计算滚动窗口平均
    # 使用pandas的rolling函数实现，边缘值使用扩展窗口
    voltage_series = pd.Series(data_array)
    rolling_avg = voltage_series.rolling(window=window_size, center=True, min_periods=1).mean()

    # 将NaN值填充为原始值（边缘情况）
    smoothed_data = np.asarray(rolling_avg.values, dtype=np.float64)

    # 统计信息
    original_std = np.std(data_array)
    smoothed_std = np.std(smoothed_data)
    noise_reduction = original_std - smoothed_std

    stats = {
        'window_size': window_size,
        'original_std': original_std,
        'smoothed_std': smoothed_std,
        'noise_reduction': noise_reduction,
        'smoothing_ratio': smoothed_std / original_std if original_std > 0 else 0
    }

    print(f"      效果: 原始标准差={original_std:.4f}, 平滑后标准差={smoothed_std:.4f}, "
          f"噪声减少={noise_reduction:.4f}, 平滑比={stats['smoothing_ratio']:.2f}")

    return smoothed_data, stats

# ==================== 保留的核心功能（修改版） ====================
def trim_ends_voltage(df, voltage_col='stack_voltage', normal_range=[3.0, 3.5], end_points=20):
    """处理数据开头和结尾的异常电压值（保持不变）"""
    print("\n步骤2.4: 处理首尾异常电压数据...")

    if voltage_col not in df.columns:
        print(f"  警告: 数据中不包含电压列 '{voltage_col}'，跳过异常处理")
        return df, {'total_replaced': 0, 'start_replaced': 0, 'end_replaced': 0}

    df_processed = df.copy()
    voltage_data = df_processed[voltage_col].values
    n = len(voltage_data)

    if end_points is None or end_points <= 0:
        end_points = 20
    if normal_range is None or len(normal_range) != 2:
        normal_range = [3.0, 3.5]

    min_voltage, max_voltage = normal_range[0], normal_range[1]
    print(f"  正常电压范围: [{min_voltage:.2f}, {max_voltage:.2f}]V")
    print(f"  检查首尾各 {end_points} 个数据点...")

    stats = {
        'total_replaced': 0,
        'start_replaced': 0,
        'end_replaced': 0
    }

    abnormal_indices = []
    for i in range(n):
        current_voltage = voltage_data[i]
        if current_voltage < min_voltage or current_voltage > max_voltage:
            abnormal_indices.append(i)

    print(f"  发现 {len(abnormal_indices)} 个超出范围的异常点")

    # 处理开头部分
    start_checked = min(end_points, n)
    start_abnormal_count = 0

    for i in range(start_checked):
        current_voltage = voltage_data[i]

        if current_voltage < min_voltage or current_voltage > max_voltage:
            print(f"    开头第{i}个点: 异常值 {current_voltage:.3f}V (超出正常范围)")

            replacement_found = False
            for j in range(i + 1, n):
                if min_voltage <= voltage_data[j] <= max_voltage:
                    df_processed.loc[i, voltage_col] = voltage_data[j]
                    stats['start_replaced'] += 1
                    stats['total_replaced'] += 1
                    start_abnormal_count += 1
                    replacement_found = True
                    print(f"      替换为第{j}个点的 {voltage_data[j]:.3f}V")
                    break

            if not replacement_found:
                for j in range(i - 1, -1, -1):
                    if min_voltage <= voltage_data[j] <= max_voltage:
                        df_processed.loc[i, voltage_col] = voltage_data[j]
                        stats['start_replaced'] += 1
                        stats['total_replaced'] += 1
                        start_abnormal_count += 1
                        replacement_found = True
                        print(f"      向前查找替换为第{j}个点的 {voltage_data[j]:.3f}V")
                        break

            if not replacement_found:
                default_value = (min_voltage + max_voltage) / 2
                df_processed.loc[i, voltage_col] = default_value
                stats['start_replaced'] += 1
                stats['total_replaced'] += 1
                start_abnormal_count += 1
                print(f"      替换为默认值 {default_value:.3f}V")

    # 处理结尾部分
    end_checked = min(end_points, n)
    end_abnormal_count = 0

    for i in range(n - 1, n - 1 - end_checked, -1):
        if i < 0:
            break

        current_voltage = voltage_data[i]

        if current_voltage < min_voltage or current_voltage > max_voltage:
            print(f"    结尾第{i}个点: 异常值 {current_voltage:.3f}V (超出正常范围)")

            replacement_found = False
            for j in range(i - 1, -1, -1):
                if min_voltage <= voltage_data[j] <= max_voltage:
                    df_processed.loc[i, voltage_col] = voltage_data[j]
                    stats['end_replaced'] += 1
                    stats['total_replaced'] += 1
                    end_abnormal_count += 1
                    replacement_found = True
                    print(f"      替换为第{j}个点的 {voltage_data[j]:.3f}V")
                    break

            if not replacement_found:
                for j in range(i + 1, n):
                    if min_voltage <= voltage_data[j] <= max_voltage:
                        df_processed.loc[i, voltage_col] = voltage_data[j]
                        stats['end_replaced'] += 1
                        stats['total_replaced'] += 1
                        end_abnormal_count += 1
                        replacement_found = True
                        print(f"      向后查找替换为第{j}个点的 {voltage_data[j]:.3f}V")
                        break

            if not replacement_found:
                default_value = (min_voltage + max_voltage) / 2
                df_processed.loc[i, voltage_col] = default_value
                stats['end_replaced'] += 1
                stats['total_replaced'] += 1
                end_abnormal_count += 1
                print(f"      替换为默认值 {default_value:.3f}V")

    print(f"  首尾异常处理完成: 共替换 {stats['total_replaced']} 个异常值")
    print(f"    开头部分: 检查了{start_checked}个点，替换了{start_abnormal_count}个异常值")
    print(f"    结尾部分: 检查了{end_checked}个点，替换了{end_abnormal_count}个异常值")

    processed_min = df_processed[voltage_col].min()
    processed_max = df_processed[voltage_col].max()
    print(f"  处理后电压范围: [{processed_min:.3f}, {processed_max:.3f}]V")


    return df_processed, stats

def check_and_fix_voltage_range(df, voltage_col='stack_voltage'):
    """检查并修复电压范围（保持不变）"""
    print("  校验并修复电压数据范围...")

    if voltage_col not in df.columns:
        print(f"  警告: 电压列 '{voltage_col}' 不存在")
        return df, {'low_fixed': 0, 'high_fixed': 0}

    normal_range = PROCESSING_CONFIG['special_treatment_columns']['stack_voltage']['voltage_normal_range']
    min_voltage, max_voltage = normal_range[0], normal_range[1]

    original_data = df[voltage_col].values.copy()
    original_min = np.min(original_data)
    original_max = np.max(original_data)

    print(f"    原始电压范围: [{original_min:.3f}, {original_max:.3f}]V")
    print(f"    正常电压范围: [{min_voltage:.2f}, {max_voltage:.2f}]V")

    mask_low = original_data < min_voltage
    mask_high = original_data > max_voltage

    low_count = mask_low.sum()
    high_count = mask_high.sum()

    stats = {
        'low_fixed': low_count,
        'high_fixed': high_count,
        'total_fixed': low_count + high_count
    }

    if low_count > 0 or high_count > 0:
        print(f"    发现异常电压值: {low_count}个低于{min_voltage}V, {high_count}个高于{max_voltage}V")

        normal_mask = (original_data >= min_voltage) & (original_data <= max_voltage)
        if normal_mask.sum() > 0:
            normal_median = np.median(original_data[normal_mask])
        else:
            normal_median = (min_voltage + max_voltage) / 2

        fixed_data = original_data.copy()

        if low_count > 0:
            fixed_data[mask_low] = np.minimum(normal_median, original_data[mask_low] + 0.1)

        if high_count > 0:
            fixed_data[mask_high] = np.maximum(normal_median, original_data[mask_high] - 0.1)

        df[voltage_col] = fixed_data

        fixed_min = df[voltage_col].min()
        fixed_max = df[voltage_col].max()
        print(f"    修复后电压范围: [{fixed_min:.3f}, {fixed_max:.3f}]V")
    else:
        print(f"    电压范围正常，无需修复")

    return df, stats

# ==================== 数据加载与合并（保持不变） ====================
def load_and_merge_data(data_path, dataset_name="FC1"):
    """加载并合并指定数据集的3个CSV文件（保持不变）"""
    print(f"\n步骤1: 加载并合并 {dataset_name} 数据文件...")

    if dataset_name == "FC1":
        file_list = [
            "FC1_Ageing_part1.csv",
            "FC1_Ageing_part2.csv",
            "FC1_Ageing_part3.csv"
        ]
    elif dataset_name == "FC2":
        file_list = [
            "FC2_Ageing_part1.csv",
            "FC2_Ageing_part2.csv",
            "FC2_Ageing_part3.csv"
        ]
    else:
        raise ValueError(f"不支持的dataset_name: {dataset_name}，请输入 'FC1' 或 'FC2'")

    data_frames = []

    for file_name in file_list:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            print(f"  正在加载: {file_name}")

            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except:
                try:
                    df = pd.read_csv(file_path, encoding='gbk')
                except:
                    try:
                        df = pd.read_csv(file_path, encoding='ISO-8859-1')
                    except Exception as e:
                        print(f"  警告: 无法读取文件 {file_name}，编码错误: {e}")
                        continue

            df.columns = df.columns.str.strip()

            columns_before = df.columns.tolist()
            df = df.dropna(axis=1, how='all')
            columns_after = df.columns.tolist()

            if len(columns_before) != len(columns_after):
                removed_cols = set(columns_before) - set(columns_after)
                print(f"    已删除全缺失值列: {list(removed_cols)}")

            column_mapping = {
                'Time (h)': 'time',
                'Utot (V)': 'stack_voltage',
                'J (A/cm²)': 'current_density',
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

            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df.rename(columns={old_name: new_name}, inplace=True)

            voltage_cols = [col for col in df.columns if 'U' in col and col.endswith('(V)')]
            if voltage_cols:
                df = df.drop(columns=voltage_cols)
                print(f"    已排除电压列: {voltage_cols}")

            data_frames.append(df)
            print(f"    加载完成: {df.shape[0]}行, {df.shape[1]}列")
        else:
            print(f"  警告: 文件 {file_name} 不存在，跳过")

    if not data_frames:
        raise FileNotFoundError(f"在文件夹 '{data_path}' 中未找到 {dataset_name} 的数据文件")

    merged_df = pd.concat(data_frames, ignore_index=True)

    if 'time' in merged_df.columns:
        merged_df = merged_df.sort_values('time').reset_index(drop=True)

    print(f"数据合并完成: {len(merged_df)} 行, {len(merged_df.columns)} 列")
    print(f"列名: {list(merged_df.columns)}")

    return merged_df

# ==================== 基础预处理（保持不变） ====================
def enhanced_missing_value_fill(df, col):
    """增强的缺失值填充方法（保持不变）"""
    original_missing = df[col].isna().sum()

    df[col] = df[col].interpolate(method='linear', limit_direction='both')

    remaining_missing = df[col].isna().sum()

    if remaining_missing > 0:
        df[col] = df[col].fillna(method='ffill')
        remaining_missing = df[col].isna().sum()

    if remaining_missing > 0:
        df[col] = df[col].fillna(method='bfill')
        remaining_missing = df[col].isna().sum()

    if remaining_missing > 0 and not df[col].isnull().all():
        df[col] = df[col].fillna(df[col].mean())
        remaining_missing = df[col].isna().sum()

    if remaining_missing > 0:
        df[col] = df[col].fillna(0)

    filled_count = original_missing - df[col].isna().sum()

    return df[col], filled_count

def detect_local_outliers(data, window_size=10, threshold_std=1.0, spike_threshold=0.01):
    """增强版局部异常检测（保持不变）"""
    data_processed = data.copy()
    n = len(data)

    for i in range(n):
        left_win_size = min(window_size, i)
        right_win_size = min(window_size, n - 1 - i)

        if left_win_size == 0 and right_win_size == 0:
            continue

        left_window = data[i - left_win_size:i] if left_win_size > 0 else []
        right_window = data[i + 1:i + right_win_size + 1] if right_win_size > 0 else []
        window = np.concatenate([left_window, right_window])

        if len(window) > 0:
            window_mean = np.mean(window)
            window_std = np.std(window)

            is_spike = (abs(data[i] - window_mean) > threshold_std * window_std
                        and abs(data[i] - window_mean) > spike_threshold)

            if is_spike:
                data_processed[i] = window_mean

    return data_processed

def preprocess_basic(df):
    """基础数据预处理（保持不变）"""
    print("\n步骤2: 基础数据预处理 (增强缺失值填充 + IQR原则 + 局部异常检测)...")

    df_clean = df.copy()

    if 'stack_voltage' not in df_clean.columns:
        raise ValueError("未找到堆电压(stack_voltage)列")

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if df_clean[col].isna().sum() > 0:
            original_missing = df_clean[col].isna().sum()

            df_clean[col], filled_count = enhanced_missing_value_fill(df_clean, col)

            remaining_missing = df_clean[col].isna().sum()

            if remaining_missing > 0:
                print(f"  警告: 列 '{col}' 仍有 {remaining_missing} 个缺失值无法填充")
            else:
                print(f"  列 '{col}': 成功填充 {filled_count} 个缺失值")

    exclude_cols = ['time']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    outliers_count = 0
    for col in feature_cols:
        if col in df_clean.columns:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outlier_mask = (df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)
            outliers_count += outlier_mask.sum()

            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound

            if outlier_mask.sum() > 0:
                print(
                    f"  列 '{col}': 处理 {outlier_mask.sum()} 个异常值 (IQR范围: [{lower_bound:.4f}, {upper_bound:.4f}])")

    print(f"  共处理 {outliers_count} 个异常值")

    print("\n步骤2.3: 局部异常检测（滑动窗口方法）...")

    if 'stack_voltage' in df_clean.columns:
        print("  检测stack_voltage局部异常（如900h附近的突变）...")

        stack_voltage_original = df_clean['stack_voltage'].values.copy()
        df_clean['stack_voltage'] = detect_local_outliers(
            df_clean['stack_voltage'].values,
            window_size=10,
            threshold_std=1.0,
            spike_threshold=0.008  # 调整为0.008V，更早识别小尖峰
        )

        local_outliers = np.sum(stack_voltage_original != df_clean['stack_voltage'].values)
        print(f"  处理 {local_outliers} 个局部异常点（如900h附近的突变）")

    return df_clean

# ==================== 全列增强滤波处理 ====================
def process_all_columns_with_adaptive_filtering(df):
    """
    对所有数值列进行自适应多阶段滤波处理

    参数:
    - df: 预处理后的DataFrame

    返回:
    - df_filtered: 滤波后的DataFrame
    - column_stats: 各列处理统计信息
    """
    print("\n步骤3: 全列自适应多阶段滤波处理...")

    df_filtered = df.copy()
    column_stats = {}

    # 识别所有数值列（除时间列外）
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
    if 'time' in numeric_cols:
        numeric_cols.remove('time')

    print(f"  将对 {len(numeric_cols)} 个数值列进行自适应滤波处理:")
    print(f"  列列表: {numeric_cols}")

    for i, col in enumerate(numeric_cols):
        print(f"\n  [{i+1}/{len(numeric_cols)}] 处理列: '{col}'")

        try:
            # 1. 计算列统计特征
            column_data = df_filtered[col].values
            col_stats = calculate_column_statistics(column_data)

            # 2. 确定自适应滤波参数
            filter_params = determine_filter_parameters(col, column_data, col_stats)

            # 3. 特殊处理配置（针对特定类型的列）
            if col in PROCESSING_CONFIG['key_columns']:
                print(f"    关键列，应用增强处理")
            elif col in PROCESSING_CONFIG['special_treatment_columns'].get('temperature_columns', []):
                print(f"    温度列，应用温度数据处理策略")
            elif col in PROCESSING_CONFIG['special_treatment_columns'].get('pressure_columns', []):
                print(f"    压力列，应用压力数据处理策略")
            elif col in PROCESSING_CONFIG['special_treatment_columns'].get('flow_columns', []):
                print(f"    流量列，应用流量数据处理策略")

            # 4. 多阶段滤波
            print(f"    滤波参数: 中值窗口={filter_params['median_window']}, "
                  f"小波层数={filter_params['wavelet_level']}, "
                  f"平滑窗口={filter_params['savgol_window']}")

            filtered_data, filtering_stats = multistage_column_filtering(
                column_data,
                col,
                filter_params
            )

            # 5. 对电压列应用额外的滚动窗口平均
            if col == 'stack_voltage':
                window_size = PROCESSING_CONFIG['special_treatment_columns']['stack_voltage']['rolling_window_size']
                filtered_data, rolling_stats = rolling_window_average(filtered_data, window_size)
                filtering_stats['rolling_stats'] = rolling_stats

            # 6. 更新数据和统计信息
            df_filtered[col] = filtered_data
            column_stats[col] = {
                'statistics': col_stats,
                'filter_params': filter_params,
                'filtering_stats': filtering_stats
            }

            # 7. 输出滤波效果
            print(f"    滤波完成: 原始标准差={filtering_stats['original_std']:.6f}, "
                  f"处理后标准差={filtering_stats['final_std']:.6f}, "
                  f"噪声减少={filtering_stats['noise_reduction']:.6f}")
            print(f"    应用的滤波阶段: {filtering_stats['stages_applied']}")

        except Exception as e:
            print(f"    处理列 '{col}' 时出错: {e}，使用原始数据")
            column_stats[col] = {
                'error': str(e),
                'status': 'processing_failed'
            }

    return df_filtered, column_stats

# ==================== 主处理函数 ====================
def process_pemfc_data(data_path, output_path, dataset_name="FC1", save_plots=True):
    """
    质子交换膜燃料电池数据集完整处理流程

    参数:
    - data_path: 原始数据文件夹路径
    - output_path: 处理结果保存路径
    - dataset_name: 数据集名称，'FC1' 或 'FC2'
    - save_plots: 是否保存可视化图表

    返回:
    - processed_data: 处理后的DataFrame
    - processing_report: 处理报告字典
    """
    print(f"====== 开始PEMFC数据集预处理 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ======")

    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)

    try:
        # 步骤1: 加载并合并数据
        merged_df = load_and_merge_data(data_path, dataset_name)

        # 步骤2: 基础预处理
        preprocessed_df = preprocess_basic(merged_df)

        # 步骤2.4: 处理首尾异常电压
        voltage_col = 'stack_voltage'
        if voltage_col in preprocessed_df.columns:
            voltage_config = PROCESSING_CONFIG['special_treatment_columns'][voltage_col]
            preprocessed_df, trim_stats = trim_ends_voltage(
                preprocessed_df,
                voltage_col=voltage_col,
                normal_range=voltage_config['voltage_normal_range'],
                end_points=voltage_config['end_points_to_check']
            )

            # 检查并修复电压范围
            preprocessed_df, voltage_fix_stats = check_and_fix_voltage_range(
                preprocessed_df,
                voltage_col=voltage_col
            )
        else:
            trim_stats = None
            voltage_fix_stats = None

        # 步骤3: 全列自适应滤波处理
        filtered_df, column_stats = process_all_columns_with_adaptive_filtering(preprocessed_df)

        # 步骤4: 保存处理结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        npz_filename = f"{dataset_name}_processed_{timestamp}.npz"
        npz_path = os.path.join(output_path, npz_filename)

        # 保存为NPZ文件
        data_dict = {col: filtered_df[col].values for col in filtered_df.columns}
        np.savez(npz_path, **data_dict)
        print(f"\n处理结果已保存为NPZ文件: {npz_path}")

        # 新增：保存为CSV文件（与NPZ同路径）
        csv_filename = f"{dataset_name}_processed_{timestamp}.csv"
        csv_path = os.path.join(output_path, csv_filename)
        filtered_df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"处理结果已保存为CSV文件: {csv_path}")

        # 生成处理报告
        processing_report = {
            'dataset_name': dataset_name,
            'processing_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_shape': merged_df.shape,
            'processed_shape': filtered_df.shape,
            'columns_processed': list(column_stats.keys()),
            'trim_stats': trim_stats,
            'voltage_fix_stats': voltage_fix_stats,
            'column_stats': column_stats,
            'output_files': {
                'npz_path': npz_path,
                'csv_path': csv_path
            }
        }

        print(f"\n====== 预处理完成 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ======")
        return filtered_df, processing_report

    except Exception as e:
        print(f"\n预处理过程出错: {e}")
        raise

# 示例调用（如果直接运行该脚本）
if __name__ == "__main__":
    # 示例路径，实际使用时请修改为你的数据路径和输出路径
    DATA_PATH = "./data"    # 原始数据所在文件夹
    OUTPUT_PATH = "processed_results/FC1"  # 处理结果保存文件夹

    try:
        # 处理FC1数据集
        processed_df, report = process_pemfc_data(
            data_path=DATA_PATH,
            output_path=OUTPUT_PATH,
            dataset_name="FC1",
            save_plots=True
        )
        print(f"FC1数据集处理完成，共 {processed_df.shape[0]} 行, {processed_df.shape[1]} 列")

        # 如需处理FC2数据集，取消下面的注释
        processed_df_fc2, report_fc2 = process_pemfc_data(
            data_path="./datatest",
            output_path="processed_results/FC2",
            dataset_name="FC2",
            save_plots=True
        )
        print(f"FC2数据集处理完成，共 {processed_df_fc2.shape[0]} 行, {processed_df_fc2.shape[1]} 列")

    except Exception as e:
        print(f"处理失败: {e}")