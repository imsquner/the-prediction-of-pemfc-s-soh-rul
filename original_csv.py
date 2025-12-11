# csv_data_visualizer_fixed_v2.py
"""
CSV原始数据可视化检测工具 - 修复版V2
修复单个文件时的子图绘制问题
"""

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
import chardet
from pathlib import Path
from matplotlib import cm

warnings.filterwarnings('ignore')

# 改进字体设置，确保中英文正常显示
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 列名映射关系（原始CSV列名 -> 统一列名）
COLUMN_MAPPING = {
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


def detect_encoding(file_path):
    """检测文件编码"""
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # 读取前10000字节进行检测
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        confidence = result['confidence']
        print(f"    检测到编码: {encoding} (置信度: {confidence:.2f})")
        return encoding
    except Exception as e:
        print(f"    编码检测失败: {e}")
        return None


def read_csv_with_encoding(file_path: str, encoding: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """使用指定编码或自动检测编码读取CSV文件"""
    encodings_to_try = []

    if encoding:
        encodings_to_try.append(encoding)

    # 尝试常见编码
    encodings_to_try.extend([
        'latin1',  # 最可能解决您的问题
        'ISO-8859-1',
        'cp1252',
        'Windows-1252',
        'utf-8-sig',
        'utf-8'
    ])

    for enc in encodings_to_try:
        try:
            print(f"    尝试编码: {enc}")
            df = pd.read_csv(file_path, encoding=enc, engine='python')
            print(f"    使用编码 {enc} 成功读取文件")
            return df, enc
        except Exception as e:
            print(f"    编码 {enc} 失败: {str(e)[:100]}")
            continue

    # 如果所有编码都失败，尝试使用错误处理
    try:
        print(f"    尝试使用错误处理读取")
        df = pd.read_csv(str(file_path), encoding='utf-8', errors='ignore', engine='python')  # type: ignore[arg-type]
        return df, 'utf-8-with-errors-ignored'
    except Exception as e:
        print(f"    使用错误处理也失败: {e}")

    return None, None


class CSVDataVisualizer:
    def __init__(self, csv_folder_path, dataset_name):
        """
        初始化CSV数据可视化器

        Parameters:
        -----------
        csv_folder_path : str
            CSV文件夹路径（包含多个CSV文件）
        dataset_name : str
            数据集名称
        """
        self.csv_folder_path = csv_folder_path
        self.dataset_name = dataset_name
        self.df: Optional[pd.DataFrame] = None
        self.columns: Optional[List[str]] = None
        self.csv_files: List[str] = []
        self.merged_data: Optional[pd.DataFrame] = None

    def _ensure_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("Data not loaded. Run load_and_merge_csv_files() first.")
        return self.df

    def load_and_merge_csv_files(self):
        """加载并合并文件夹中的所有CSV文件"""
        print(f"\n{'=' * 60}")
        print(f"开始检测 {self.dataset_name} CSV 原始数据")
        print(f"文件夹路径: {self.csv_folder_path}")
        print(f"{'=' * 60}")

        try:
            # 获取所有CSV文件
            csv_files = []
            for file in Path(self.csv_folder_path).glob("*.csv"):
                csv_files.append(str(file))

            if not csv_files:
                print(f"❌ 未找到CSV文件: {self.csv_folder_path}")
                return False

            print(f"找到 {len(csv_files)} 个CSV文件: {[Path(f).name for f in csv_files]}")

            # 按文件名排序以确保正确的顺序
            csv_files.sort()
            self.csv_files = csv_files

            all_data_frames = []
            successful_files = 0

            for i, csv_file in enumerate(csv_files, 1):
                print(f"\n  加载文件 {i}/{len(csv_files)}: {Path(csv_file).name}")

                try:
                    # 检测文件编码
                    detected_encoding = detect_encoding(csv_file)

                    # 读取CSV文件
                    df_part, used_encoding = read_csv_with_encoding(csv_file, detected_encoding)

                    if df_part is None:
                        print(f"    ❌ 无法读取文件: {csv_file}")
                        continue

                    print(f"    原始形状: {df_part.shape}")
                    print(f"    原始列数: {len(df_part.columns)}")
                    print(f"    使用编码: {used_encoding}")

                    # 显示原始列名
                    if successful_files == 0:
                        print(f"    原始列名: {list(df_part.columns)}")

                    # 重命名列
                    df_part_renamed = df_part.copy()

                    # 应用列名映射
                    rename_dict = {}
                    for old_col in df_part.columns:
                        if old_col in COLUMN_MAPPING:
                            rename_dict[old_col] = COLUMN_MAPPING[old_col]
                        elif old_col.strip() in COLUMN_MAPPING:  # 处理可能的空格
                            rename_dict[old_col] = COLUMN_MAPPING[old_col.strip()]
                        else:
                            # 对于U1-U5电压列，我们可能需要特殊处理
                            if 'U' in old_col and 'V' in old_col:
                                # 尝试提取电池编号
                                if 'U1' in old_col:
                                    rename_dict[old_col] = 'cell_1_voltage'
                                elif 'U2' in old_col:
                                    rename_dict[old_col] = 'cell_2_voltage'
                                elif 'U3' in old_col:
                                    rename_dict[old_col] = 'cell_3_voltage'
                                elif 'U4' in old_col:
                                    rename_dict[old_col] = 'cell_4_voltage'
                                elif 'U5' in old_col:
                                    rename_dict[old_col] = 'cell_5_voltage'
                                else:
                                    rename_dict[old_col] = old_col.replace(' ', '_').replace('(', '').replace(')',
                                                                                                              '').replace(
                                        '°', '').replace('²', '2').replace('/', '_')
                            else:
                                # 保持原列名，但去除特殊字符
                                rename_dict[old_col] = old_col.replace(' ', '_').replace('(', '').replace(')',
                                                                                                          '').replace(
                                    '°', '').replace('²', '2').replace('/', '_')

                    df_part_renamed = df_part_renamed.rename(columns=rename_dict)

                    # 检查并处理可能的重复列名
                    if df_part_renamed.columns.duplicated().any():
                        print(f"    ⚠ 发现重复列名: {df_part_renamed.columns[df_part_renamed.columns.duplicated()]}")
                        # 为重复列名添加后缀
                        df_part_renamed = df_part_renamed.loc[:, ~df_part_renamed.columns.duplicated()]

                    # 排除单个电池电压列（U1-U5），只保留总电压
                    cell_voltage_cols = [col for col in df_part_renamed.columns if 'cell_' in col and 'voltage' in col]
                    if cell_voltage_cols:
                        print(f"    已排除单个电池电压列: {cell_voltage_cols}")
                        df_part_renamed = df_part_renamed.drop(columns=cell_voltage_cols, errors='ignore')

                    # 添加文件来源标记
                    df_part_renamed['file_source'] = Path(csv_file).stem

                    all_data_frames.append(df_part_renamed)
                    successful_files += 1

                except Exception as e:
                    print(f"    ❌ 加载文件失败 {csv_file}: {e}")

            if not all_data_frames:
                print("❌ 所有CSV文件加载失败")
                return False

            # 合并所有数据
            print(f"\n  成功加载 {successful_files}/{len(csv_files)} 个文件")
            self.df = pd.concat(all_data_frames, ignore_index=True, sort=False)

            print(f"\n步骤1: 数据合并完成")
            print(f"  总样本数量: {len(self.df):,}")
            print(f"  特征数量: {len(self.df.columns)}")
            print(f"  最终列名: {list(self.df.columns)}")

            # 检查是否有时间列，并确保它是数值类型
            if 'time' in self.df.columns:
                self.df['time'] = pd.to_numeric(self.df['time'], errors='coerce')
                print(f"  时间范围: [{self.df['time'].min():.2f}, {self.df['time'].max():.2f}] 小时")

            # 显示数据基本信息
            print(f"\n  数据基本信息:")
            print(f"    数据类型:\n{self.df.dtypes.value_counts()}")

            # 显示前几行数据
            print(f"\n  数据前5行:")
            print(self.df.head())

            return True

        except Exception as e:
            print(f"❌ 加载CSV文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def basic_statistics(self):
        """计算并显示基本统计信息"""
        print(f"\n步骤2: 数据基本统计信息")

        if self.df is None or len(self.df) == 0:
            print("  数据为空，跳过统计")
            return

        # 选择数值列进行分析
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if 'time' in numeric_columns and len(numeric_columns) > 1:
            # 排除时间列，专注于其他特征
            numeric_columns.remove('time')

        print(f"  分析的数值特征数量: {len(numeric_columns)}")

        if len(numeric_columns) == 0:
            print("  未找到数值列")
            return

        # 重点关注电压相关列
        voltage_columns = [col for col in self.df.columns if 'voltage' in col.lower()]

        if not voltage_columns:
            print("  未找到电压列")
        else:
            for col in voltage_columns[:5]:  # 显示前5个电压相关列
                if col in self.df.columns:
                    col_data = self.df[col].dropna()
                    if len(col_data) > 0:
                        print(f"\n  🔋 {col}:")
                        print(f"      范围: [{col_data.min():.4f}, {col_data.max():.4f}]")
                        print(f"      均值: {col_data.mean():.4f} ± {col_data.std():.4f}")
                        missing_count = self.df[col].isnull().sum()
                        missing_pct = missing_count / len(self.df) * 100
                        print(f"      缺失值: {missing_count} ({missing_pct:.2f}%)")

                        # 检查电压是否在合理范围内
                        if 'stack_voltage' in col or 'utot' in col.lower():
                            voltage_range_ok = 3.0 <= col_data.min() <= 3.5 and 3.0 <= col_data.max() <= 3.5
                            voltage_status = "✓ 正常" if voltage_range_ok else "⚠ 异常"
                            print(f"      电压范围状态: {voltage_status}")
                    else:
                        print(f"\n  🔋 {col}: 无有效数据")

        # 显示其他关键特征
        key_features = ['current_density', 'current', 'hydrogen_inlet_temp', 'air_inlet_temp', 'coolant_inlet_temp']
        print(f"\n  其他关键特征统计:")
        for feature in key_features:
            if feature in self.df.columns:
                col_data = self.df[feature].dropna()
                if len(col_data) > 0:
                    print(f"\n  📊 {feature}:")
                    print(f"      范围: [{col_data.min():.4f}, {col_data.max():.4f}]")
                    print(f"      均值: {col_data.mean():.4f} ± {col_data.std():.4f}")
                else:
                    print(f"\n  📊 {feature}: 无有效数据")

    def check_data_consistency(self):
        """检查数据一致性（跨文件）"""
        print(f"\n步骤3: 跨文件数据一致性检查")

        if self.df is None or len(self.df) == 0:
            print("  数据为空，跳过一致性检查")
            return

        if 'file_source' not in self.df.columns:
            print("  无文件来源信息，跳过一致性检查")
            return

        # 按文件检查基本统计信息
        file_sources = self.df['file_source'].unique()
        print(f"  发现 {len(file_sources)} 个数据文件")

        for source in file_sources:
            source_data = self.df[self.df['file_source'] == source]
            print(f"\n  文件: {source}")
            print(f"    样本数: {len(source_data):,}")

            if 'stack_voltage' in source_data.columns:
                voltage = source_data['stack_voltage'].dropna()
                if len(voltage) > 0:
                    print(f"    电压范围: [{voltage.min():.3f}, {voltage.max():.3f}] V")
                    print(f"    电压均值: {voltage.mean():.3f} ± {voltage.std():.3f} V")
                else:
                    print(f"    电压: 无有效数据")

            if 'time' in source_data.columns:
                time_range = source_data['time'].dropna()
                if len(time_range) > 0:
                    print(f"    时间范围: [{time_range.min():.2f}, {time_range.max():.2f}] 小时")
                else:
                    print(f"    时间: 无有效数据")

        # 检查时间连续性
        if 'time' in self.df.columns:
            print(f"\n  ⏱️ 时间连续性检查:")
            self.df = self.df.sort_values('time')
            time_data = self.df['time'].dropna().values

            if len(time_data) > 1:
                time_diff = np.diff(np.asarray(time_data, dtype=float))

                if len(time_diff) > 0:
                    positive_diffs = time_diff[time_diff > 0]
                    if len(positive_diffs) > 0:
                        avg_time_step = np.mean(positive_diffs)
                        max_gap = np.max(time_diff) if len(time_diff) > 0 else 0
                        print(f"    平均时间步长: {avg_time_step:.4f} 小时")
                        print(f"    最大时间间隔: {max_gap:.4f} 小时")

                    # 检查是否有时间倒流
                    negative_steps = np.sum(time_diff < 0)
                    if negative_steps > 0:
                        print(f"    ⚠ 发现 {negative_steps} 个时间倒流点")
            else:
                print("    时间数据不足，无法进行连续性检查")

    def visualize_raw_data(self, save_dir="csv_raw_visualization"):
        """可视化原始CSV数据 - 专门针对原始数据特性"""
        print(f"\n步骤4: 原始数据可视化")

        if self.df is None or len(self.df) == 0:
            print("  数据为空，跳过可视化")
            return

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 1. 电压时间序列图（所有文件合并显示）
        fig1, axes1 = plt.subplots(2, 2, figsize=(16, 10))
        axes1 = axes1.flatten()

        # 检查是否有时间列
        has_time = 'time' in self.df.columns

        # 子图1: 总电压时间序列
        if 'stack_voltage' in self.df.columns:
            ax = axes1[0]

            voltage_data = self.df['stack_voltage'].dropna()
            if len(voltage_data) > 0:
                if has_time:
                    # 按文件用不同颜色显示
                    if 'file_source' in self.df.columns:
                        file_sources = self.df['file_source'].unique()
                        colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(file_sources)))

                        for i, source in enumerate(file_sources):
                            source_data = self.df[self.df['file_source'] == source]
                            time_data = source_data['time'].dropna()
                            voltage_data_source = source_data['stack_voltage'].dropna()

                            if len(time_data) > 0 and len(voltage_data_source) > 0:
                                # 确保数据长度一致
                                min_len = min(len(time_data), len(voltage_data_source))
                                ax.plot(time_data.values[:min_len], voltage_data_source.values[:min_len],
                                        linewidth=0.8, alpha=0.7, color=colors[i], label=source)

                        if len(file_sources) > 1:
                            ax.legend(fontsize=8, loc='upper right')
                    else:
                        time_data = self.df['time'].dropna()
                        min_len = min(len(time_data), len(voltage_data))
                        ax.plot(time_data.values[:min_len], voltage_data.values[:min_len],
                                linewidth=0.8, alpha=0.7, color='blue')
                    ax.set_xlabel('Time (hours)', fontsize=10)
                else:
                    ax.plot(voltage_data.values, linewidth=0.8, alpha=0.7, color='blue')
                    ax.set_xlabel('Sample Index', fontsize=10)

                ax.set_ylabel('Stack Voltage (V)', fontsize=10)
                ax.set_title(f'{self.dataset_name} - Raw Stack Voltage', fontsize=12, pad=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)

                # 添加统计信息
                stats_text = f"Mean: {voltage_data.mean():.3f}V\nStd: {voltage_data.std():.3f}V\nRange: [{voltage_data.min():.3f}, {voltage_data.max():.3f}]V"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No voltage data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{self.dataset_name} - Raw Stack Voltage', fontsize=12, pad=10)

        # 子图2: 电流密度时间序列
        if 'current_density' in self.df.columns:
            ax = axes1[1]

            current_density_data = self.df['current_density'].dropna()
            if len(current_density_data) > 0:
                if has_time:
                    time_data = self.df['time'].dropna()
                    min_len = min(len(time_data), len(current_density_data))
                    ax.plot(time_data.values[:min_len], current_density_data.values[:min_len],
                            linewidth=0.8, alpha=0.7, color='green')
                    ax.set_xlabel('Time (hours)', fontsize=10)
                else:
                    ax.plot(current_density_data.values, linewidth=0.8, alpha=0.7, color='green')
                    ax.set_xlabel('Sample Index', fontsize=10)

                ax.set_ylabel('Current Density (A/cm2)', fontsize=10)
                ax.set_title(f'{self.dataset_name} - Raw Current Density', fontsize=12, pad=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=9)

                stats_text = f"Mean: {current_density_data.mean():.4f}\nStd: {current_density_data.std():.4f}"
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        fontsize=8, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            else:
                ax.text(0.5, 0.5, 'No current density data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{self.dataset_name} - Raw Current Density', fontsize=12, pad=10)

        # 子图3: 温度对比（氢入口和空气入口）
        temp_cols = []
        for col in ['hydrogen_inlet_temp', 'air_inlet_temp', 'coolant_inlet_temp']:
            if col in self.df.columns and len(self.df[col].dropna()) > 0:
                temp_cols.append(col)

        if temp_cols and len(temp_cols) >= 2:
            ax = axes1[2]

            colors = ['red', 'orange', 'purple']
            for i, col in enumerate(temp_cols[:2]):  # 只显示前2个温度
                temp_data = self.df[col].dropna()
                if len(temp_data) > 0:
                    if has_time:
                        time_data = self.df['time'].dropna()
                        min_len = min(len(time_data), len(temp_data))
                        ax.plot(time_data.values[:min_len], temp_data.values[:min_len],
                                linewidth=0.8, alpha=0.7, color=colors[i],
                                label=col.replace('_', ' ').title())
                    else:
                        ax.plot(temp_data.values, linewidth=0.8, alpha=0.7,
                                color=colors[i], label=col.replace('_', ' ').title())

            if has_time:
                ax.set_xlabel('Time (hours)', fontsize=10)
            else:
                ax.set_xlabel('Sample Index', fontsize=10)

            ax.set_ylabel('Temperature (°C)', fontsize=10)
            ax.set_title(f'{self.dataset_name} - Raw Inlet Temperatures', fontsize=12, pad=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)
        else:
            ax = axes1[2]
            ax.text(0.5, 0.5, 'No temperature data', ha='center', va='center', fontsize=12)
            ax.set_title(f'{self.dataset_name} - Raw Inlet Temperatures', fontsize=12, pad=10)

        # 子图4: 按文件来源显示电压分布
        if 'file_source' in self.df.columns and 'stack_voltage' in self.df.columns:
            ax = axes1[3]

            file_sources = self.df['file_source'].unique()
            voltage_by_file = []
            labels = []

            for source in file_sources:
                source_voltage = self.df[self.df['file_source'] == source]['stack_voltage'].dropna()
                if len(source_voltage) > 0:
                    voltage_by_file.append(source_voltage.values)
                    labels.append(source)

            if voltage_by_file:
                # 创建箱线图
                bp = ax.boxplot(voltage_by_file, labels=labels, patch_artist=True)

                # 设置颜色
                colors = cm.get_cmap('Set3')(np.linspace(0, 1, len(voltage_by_file)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                ax.set_ylabel('Stack Voltage (V)', fontsize=10)
                ax.set_title(f'{self.dataset_name} - Voltage Distribution by File', fontsize=12, pad=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.tick_params(labelsize=9, rotation=45)
            else:
                ax.text(0.5, 0.5, 'No voltage data by file', ha='center', va='center', fontsize=12)
                ax.set_title(f'{self.dataset_name} - Voltage Distribution by File', fontsize=12, pad=10)

        plt.suptitle(f'{self.dataset_name} - Raw Data Overview', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.dataset_name}_raw_overview.png', dpi=150, bbox_inches='tight')
        print(f"  Saved raw overview plot: {save_dir}/{self.dataset_name}_raw_overview.png")
        plt.close(fig1)

        # 2. 电压详细分析图
        if 'stack_voltage' in self.df.columns:
            voltage_data = self.df['stack_voltage'].dropna()
            if len(voltage_data) > 0:
                fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

                # 左图: 电压直方图 + 核密度估计
                ax1.hist(voltage_data, bins=50, alpha=0.7, density=True,
                         color='blue', edgecolor='black')

                # 添加核密度估计
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(voltage_data)
                    x_range = np.linspace(voltage_data.min(), voltage_data.max(), 1000)
                    ax1.plot(x_range, kde(x_range), color='red', linewidth=2, label='KDE')
                except:
                    pass  # 如果scipy不可用，跳过KDE

                ax1.set_xlabel('Stack Voltage (V)', fontsize=12)
                ax1.set_ylabel('Density', fontsize=12)
                ax1.set_title('Voltage Distribution', fontsize=14, pad=15)
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # 添加统计信息
                stats_text = f"Mean: {voltage_data.mean():.4f} V\nStd: {voltage_data.std():.4f} V\nSkew: {voltage_data.skew():.4f}\nKurtosis: {voltage_data.kurtosis():.4f}"
                ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

                # 右图: 电压滚动统计
                if has_time and len(self.df) > 100:
                    # 按时间排序
                    df_sorted = self.df.sort_values('time') if has_time else self.df
                    df_sorted = df_sorted.dropna(subset=['stack_voltage', 'time'])

                    if len(df_sorted) > 10:
                        # 计算滚动统计
                        window_size = min(1000, len(df_sorted) // 100)
                        if window_size < 10:
                            window_size = 10

                        rolling_mean = df_sorted['stack_voltage'].rolling(window=window_size, min_periods=1,
                                                                          center=True).mean()
                        rolling_std = df_sorted['stack_voltage'].rolling(window=window_size, min_periods=1,
                                                                         center=True).std()

                        # 主Y轴：电压
                        ax2.plot(df_sorted['time'], df_sorted['stack_voltage'], linewidth=0.5, alpha=0.3,
                                 color='gray', label='Raw Voltage')
                        ax2.plot(df_sorted['time'], rolling_mean, linewidth=1.5, color='blue',
                                 label=f'Rolling Mean (window={window_size})')
                        ax2.set_xlabel('Time (hours)', fontsize=12)

                        ax2.set_ylabel('Voltage (V)', fontsize=12, color='blue')
                        ax2.tick_params(axis='y', labelcolor='blue')

                        # 次Y轴：滚动标准差
                        ax2_twin = ax2.twinx()
                        ax2_twin.plot(df_sorted['time'], rolling_std, linewidth=1.0, color='red', alpha=0.7,
                                      label='Rolling Std')

                        ax2_twin.set_ylabel('Standard Deviation', fontsize=12, color='red')
                        ax2_twin.tick_params(axis='y', labelcolor='red')

                        # 合并图例
                        lines1, labels1 = ax2.get_legend_handles_labels()
                        lines2, labels2 = ax2_twin.get_legend_handles_labels()
                        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

                        ax2.set_title('Voltage with Rolling Statistics', fontsize=14, pad=15)
                        ax2.grid(True, alpha=0.3)
                    else:
                        ax2.text(0.5, 0.5, 'Insufficient data for rolling statistics',
                                 ha='center', va='center', fontsize=12)
                        ax2.set_title('Voltage with Rolling Statistics', fontsize=14, pad=15)
                else:
                    ax2.text(0.5, 0.5, 'No time data or insufficient data',
                             ha='center', va='center', fontsize=12)
                    ax2.set_title('Voltage with Rolling Statistics', fontsize=14, pad=15)

                plt.suptitle(f'{self.dataset_name} - Voltage Detailed Analysis', fontsize=16, y=1.02)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{self.dataset_name}_voltage_analysis.png', dpi=150, bbox_inches='tight')
                print(f"  Saved voltage analysis plot: {save_dir}/{self.dataset_name}_voltage_analysis.png")
                plt.close(fig2)

        # 3. 按文件显示电压时间序列（分图） - 修复版本
        if 'file_source' in self.df.columns and 'stack_voltage' in self.df.columns and has_time:
            file_sources = self.df['file_source'].unique()
            n_files = len(file_sources)

            if n_files > 0:
                # 根据文件数量确定子图布局
                n_cols = min(2, n_files)
                n_rows = (n_files + n_cols - 1) // n_cols

                # 创建子图
                if n_rows == 1 and n_cols == 1:
                    # 只有一个子图的情况
                    fig3, ax = plt.subplots(figsize=(8, 6))
                    axes_list = [ax]  # 将单个axes对象放入列表
                else:
                    # 多个子图的情况
                    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                    # 将axes展平为一维列表
                    if n_rows == 1 or n_cols == 1:
                        axes_list = axes.flatten()
                    else:
                        axes_list = axes.ravel()

                for idx, source in enumerate(file_sources):
                    if idx < len(axes_list):
                        ax = axes_list[idx]

                        source_data = self.df[self.df['file_source'] == source]
                        time_data = source_data['time'].dropna()
                        voltage_data = source_data['stack_voltage'].dropna()

                        if len(time_data) > 0 and len(voltage_data) > 0:
                            min_len = min(len(time_data), len(voltage_data))
                            ax.plot(time_data.values[:min_len], voltage_data.values[:min_len],
                                    linewidth=0.8, alpha=0.7, color='blue')

                            ax.set_xlabel('Time (hours)', fontsize=9)
                            ax.set_ylabel('Voltage (V)', fontsize=9)
                            ax.set_title(f'File: {source}', fontsize=11, pad=10)
                            ax.grid(True, alpha=0.3)

                            # 添加文件统计信息
                            file_stats = f"Min: {voltage_data.min():.3f}V\nMax: {voltage_data.max():.3f}V\nMean: {voltage_data.mean():.3f}V"
                            ax.text(0.02, 0.98, file_stats, transform=ax.transAxes,
                                    fontsize=8, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
                        else:
                            ax.text(0.5, 0.5, 'No voltage data', ha='center', va='center', fontsize=10)
                            ax.set_title(f'File: {source}', fontsize=11, pad=10)

                # 隐藏多余的子图
                for idx in range(len(file_sources), len(axes_list)):
                    axes_list[idx].set_visible(False)

                plt.suptitle(f'{self.dataset_name} - Voltage by File Source', fontsize=16, y=1.02)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{self.dataset_name}_voltage_by_file.png', dpi=150, bbox_inches='tight')
                print(f"  Saved voltage by file plot: {save_dir}/{self.dataset_name}_voltage_by_file.png")
                plt.close(fig3)

        print(f"  All raw data plots saved to: {save_dir}/")

    def check_data_quality(self):
        """检查原始数据质量"""
        print(f"\n步骤5: 原始数据质量评估")

        if self.df is None or len(self.df) == 0:
            print("  数据为空，无法进行质量评估")
            return {'dataset': self.dataset_name, 'quality_score': 0}

        quality_report = {
            'dataset': self.dataset_name,
            'samples': len(self.df),
            'features': len(self.df.columns),
            'missing_values': 0,
            'duplicates': 0,
            'quality_score': 100
        }

        # 检查缺失值
        missing_values = self.df.isnull().sum().sum()
        missing_pct = missing_values / (len(self.df) * len(self.df.columns)) * 100
        quality_report['missing_values'] = missing_values
        quality_report['missing_pct'] = missing_pct

        print(f"  Missing values check:")
        print(f"    Total missing values: {missing_values}")
        print(f"    Missing percentage: {missing_pct:.4f}%")

        if missing_pct == 0:
            print(f"    ✓ No missing values")
        elif missing_pct < 0.1:
            print(f"    ⚠ Very few missing values ({missing_pct:.4f}%)")
            quality_report['quality_score'] -= 5
        elif missing_pct < 1:
            print(f"    ⚠ Few missing values ({missing_pct:.4f}%)")
            quality_report['quality_score'] -= 10
        else:
            print(f"    ⚠ Significant missing values ({missing_pct:.4f}%)")
            quality_report['quality_score'] -= 20

        # 检查重复值
        duplicates = self.df.duplicated().sum()
        duplicate_pct = duplicates / len(self.df) * 100
        quality_report['duplicates'] = duplicates
        quality_report['duplicate_pct'] = duplicate_pct

        print(f"  Duplicates check:")
        print(f"    Duplicate rows: {duplicates}")
        print(f"    Duplicate percentage: {duplicate_pct:.4f}%")

        if duplicates == 0:
            print(f"    ✓ No duplicate rows")
        elif duplicate_pct < 0.01:
            print(f"    ⚠ Very few duplicate rows ({duplicate_pct:.4f}%)")
            quality_report['quality_score'] -= 2
        else:
            print(f"    ⚠ Duplicate rows exist ({duplicate_pct:.4f}%)")
            quality_report['quality_score'] -= 5

        # 检查电压数据质量
        if 'stack_voltage' in self.df.columns:
            voltage_data = self.df['stack_voltage'].dropna()
            if len(voltage_data) > 0:
                voltage_min = voltage_data.min()
                voltage_max = voltage_data.max()

                print(f"  Voltage quality check:")
                print(f"    Min voltage: {voltage_min:.3f} V")
                print(f"    Max voltage: {voltage_max:.3f} V")
                print(f"    Voltage range: {voltage_max - voltage_min:.3f} V")

                # 检查电压范围是否合理
                if 3.0 <= voltage_min <= 3.5 and 3.0 <= voltage_max <= 3.5:
                    print(f"    ✓ Voltage range normal (3.0-3.5V)")
                elif 2.5 <= voltage_min <= 4.0 and 2.5 <= voltage_max <= 4.0:
                    print(
                        f"    ⚠ Voltage range somewhat abnormal but acceptable: {voltage_min:.3f}-{voltage_max:.3f} V")
                    quality_report['quality_score'] -= 10
                else:
                    print(f"    ⚠ Voltage range abnormal: {voltage_min:.3f}-{voltage_max:.3f} V")
                    quality_report['quality_score'] -= 20

                # 检查电压突变
                if len(voltage_data) > 10:
                    voltage_diff = np.abs(np.diff(np.asarray(voltage_data.values, dtype=float)))
                    large_jumps = np.sum(voltage_diff > 0.1)  # 大于0.1V的突变
                    jump_pct = large_jumps / len(voltage_diff) * 100

                    if large_jumps > 0:
                        print(f"    ⚠ Found {large_jumps} large voltage jumps (>0.1V) ({jump_pct:.2f}%)")
                        quality_report['quality_score'] -= 5
            else:
                print(f"  Voltage quality check: No voltage data available")
                quality_report['quality_score'] -= 30

        # 检查时间序列连续性
        if 'time' in self.df.columns:
            time_data = self.df['time'].dropna()
            if len(time_data) > 1:
                time_diff = np.diff(np.asarray(time_data.values, dtype=float))
                negative_steps = np.sum(time_diff < 0)

                if negative_steps > 0:
                    print(f"  Time continuity check:")
                    print(f"    ⚠ Found {negative_steps} negative time steps (time going backwards)")
                    quality_report['quality_score'] -= 10

        # 最终质量评分
        quality_report['quality_score'] = max(0, min(100, quality_report['quality_score']))

        print(f"\n  📊 Raw Data Quality Score: {quality_report['quality_score']}/100")

        if quality_report['quality_score'] >= 90:
            print(f"  ✅ Excellent raw data quality")
        elif quality_report['quality_score'] >= 75:
            print(f"  ⚠ Good raw data quality, minor issues")
        elif quality_report['quality_score'] >= 60:
            print(f"  ⚠ Average raw data quality, check recommended")
        else:
            print(f"  ❌ Poor raw data quality, preprocessing needed")

        return quality_report

    def save_summary_report(self, quality_report, save_dir="csv_raw_visualization"):
        """保存原始数据检测报告"""
        os.makedirs(save_dir, exist_ok=True)

        report_path = f"{save_dir}/{self.dataset_name}_raw_quality_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"CSV Raw Data Quality Report - {self.dataset_name}\n")
            f.write(f"Report Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write("1. Basic Information\n")
            f.write(f"    Dataset: {self.dataset_name}\n")
            f.write(f"    Folder Path: {self.csv_folder_path}\n")
            f.write(f"    CSV Files: {[Path(f).name for f in self.csv_files]}\n")
            f.write(f"    Total Samples: {quality_report['samples']:,}\n")
            f.write(f"    Features: {quality_report['features']}\n")

            if self.df is not None and len(self.df.columns) > 0:
                f.write(f"    Feature List: {list(self.df.columns)}\n\n")
            else:
                f.write(f"    Feature List: No data\n\n")

            f.write("2. Data Quality Checks\n")
            f.write(
                f"    Missing Values: {quality_report['missing_values']} ({quality_report.get('missing_pct', 0):.4f}%)\n")
            f.write(
                f"    Duplicate Rows: {quality_report['duplicates']} ({quality_report.get('duplicate_pct', 0):.4f}%)\n\n")

            if self.df is not None and 'stack_voltage' in self.df.columns:
                voltage_data = self.df['stack_voltage'].dropna()
                if len(voltage_data) > 0:
                    f.write(f"    Voltage Statistics:\n")
                    f.write(f"        Min: {voltage_data.min():.3f} V\n")
                    f.write(f"        Max: {voltage_data.max():.3f} V\n")
                    f.write(f"        Mean: {voltage_data.mean():.3f} V\n")
                    f.write(f"        Std: {voltage_data.std():.3f} V\n")
                    f.write(f"        Range: {voltage_data.max() - voltage_data.min():.3f} V\n\n")

            f.write("3. Quality Score\n")
            f.write(f"    Overall Quality Score: {quality_report['quality_score']}/100\n\n")

            f.write("4. Recommendations\n")
            if quality_report['quality_score'] >= 90:
                f.write("    Excellent raw data quality, ready for preprocessing.\n")
            elif quality_report['quality_score'] >= 75:
                f.write("    Good raw data quality, suitable for preprocessing with minor adjustments.\n")
            elif quality_report['quality_score'] >= 60:
                f.write("    Average raw data quality, requires careful preprocessing and validation.\n")
            else:
                f.write("    Poor raw data quality, significant preprocessing and cleaning needed.\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Raw Data Analysis Complete\n")
            f.write(f"{'=' * 60}\n")

        print(f"  Saved raw quality report: {report_path}")

    def run_full_analysis(self, save_dir="csv_raw_visualization"):
        """运行完整的原始数据分析流程"""
        print(f"\n{'=' * 60}")
        print(f"Starting Full Raw Data Analysis: {self.dataset_name}")
        print(f"{'=' * 60}")

        # 1. 加载并合并CSV数据
        if not self.load_and_merge_csv_files():
            print("❌ CSV data loading failed, terminating analysis")
            return None

        # 2. 基本统计
        self.basic_statistics()

        # 3. 数据一致性检查
        self.check_data_consistency()

        # 4. 数据可视化
        self.visualize_raw_data(save_dir)

        # 5. 数据质量检查
        quality_report = self.check_data_quality()

        # 6. 保存报告
        self.save_summary_report(quality_report, save_dir)

        print(f"\n{'=' * 60}")
        print(f"{self.dataset_name} CSV raw data analysis completed!")
        print(f"{'=' * 60}")

        return quality_report


def main():
    """主函数：检测所有原始CSV数据文件夹"""
    # 根据您的日志，CSV文件路径
    csv_folders = [
        {
            'path': 'data',  # FC1原始数据文件夹
            'name': 'FC1_Raw'
        },
        {
            'path': 'datatest',  # FC2原始数据文件夹
            'name': 'FC2_Raw'
        }
    ]

    all_reports = []

    for folder_info in csv_folders:
        folder_path = folder_info['path']
        dataset_name = folder_info['name']

        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"❌ Folder not found: {folder_path}")
            continue

        # 创建可视化器并运行分析
        print(f"\n{'#' * 60}")
        print(f"Analyzing {dataset_name} from {folder_path}")
        print(f"{'#' * 60}")

        visualizer = CSVDataVisualizer(folder_path, dataset_name)
        report = visualizer.run_full_analysis()

        if report:
            all_reports.append(report)

    # 生成综合报告
    if all_reports:
        print(f"\n{'=' * 60}")
        print("Comprehensive Raw Data Analysis Report")
        print(f"{'=' * 60}")

        for report in all_reports:
            print(f"\n{report['dataset']}:")
            print(f"  Samples: {report['samples']:,}")
            print(f"  Features: {report['features']}")
            print(f"  Missing Values: {report['missing_values']}")
            print(f"  Duplicate Rows: {report['duplicates']}")
            print(f"  Quality Score: {report['quality_score']}/100")

            # 添加评级
            if report['quality_score'] >= 90:
                rating = "Excellent"
            elif report['quality_score'] >= 75:
                rating = "Good"
            elif report['quality_score'] >= 60:
                rating = "Average"
            else:
                rating = "Poor"

            print(f"  Rating: {rating}")

        print(f"\n{'=' * 60}")
        print("All CSV raw data analysis completed!")
        print(f"Visualization results saved to: csv_raw_visualization/")
        print(f"{'=' * 60}")
    else:
        print(f"\n{'=' * 60}")
        print("No valid reports generated")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()