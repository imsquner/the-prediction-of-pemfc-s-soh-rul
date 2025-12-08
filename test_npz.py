# npz_data_visualizer_fixed.py
"""
NPZ数据可视化检测工具 - 修复版
修复字体显示问题，优化电压图布局
自动使用最新的NPZ文件
"""

from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas.api.types import is_timedelta64_dtype
import os
import glob
import warnings

warnings.filterwarnings('ignore')

# 改进字体设置，确保中文正常显示
try:
    # 尝试使用更常见的中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果上述字体不存在，使用更通用的设置
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

# 设置图表样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def find_latest_npz_files(base_dir="processed_results"):
    """
    查找每个FC目录下最新的NPZ文件

    Parameters:
    -----------
    base_dir : str
        基础目录路径

    Returns:
    --------
    list of dict
        每个元素包含 'path' 和 'name' 键值对
    """
    npz_files = []

    # 查找所有FC目录
    fc_dirs = glob.glob(os.path.join(base_dir, "FC*"))

    for fc_dir in fc_dirs:
        if os.path.isdir(fc_dir):
            # 获取目录名作为数据集名称
            dataset_name = os.path.basename(fc_dir)

            # 查找该目录下所有的NPZ文件
            fc_npz_files = glob.glob(os.path.join(fc_dir, "*.npz"))

            if fc_npz_files:
                # 按文件名中的时间戳排序，选择最新的
                # 文件名格式：FC1_processed_20251206_215017.npz
                # 时间戳格式：YYYYMMDD_HHMMSS
                def extract_timestamp(filepath):
                    filename = os.path.basename(filepath)
                    # 从文件名中提取时间戳部分
                    # 假设文件名格式为：{name}_processed_{timestamp}.npz
                    if '_processed_' in filename:
                        timestamp_part = filename.split('_processed_')[-1].replace('.npz', '')
                        return timestamp_part
                    return "00000000_000000"  # 默认值

                # 按时间戳降序排列（最新的在前）
                fc_npz_files.sort(key=extract_timestamp, reverse=True)

                # 使用最新的文件
                latest_npz = fc_npz_files[0]

                print(f"找到 {dataset_name} 的最新NPZ文件:")
                print(f"  路径: {latest_npz}")
                print(f"  时间戳: {extract_timestamp(latest_npz)}")

                npz_files.append({
                    'path': latest_npz,
                    'name': dataset_name
                })
            else:
                print(f"警告: {fc_dir} 目录下未找到NPZ文件")

    return npz_files


class NPZDataVisualizerFixed:
    def __init__(self, npz_path, dataset_name):
        """
        初始化NPZ数据可视化器

        Parameters:
        -----------
        npz_path : str
            NPZ文件路径
        dataset_name : str
            数据集名称
        """
        self.npz_path = npz_path
        self.dataset_name = dataset_name
        self.data: Optional[np.ndarray] = None
        self.columns: Optional[List[str]] = None
        self.df: Optional[pd.DataFrame] = None

    def _ensure_df(self) -> pd.DataFrame:
        """Return a loaded DataFrame or raise for clearer diagnostics."""
        if self.df is None:
            raise ValueError("DataFrame is not loaded. Call load_npz_data() first.")
        return self.df

    def _ensure_columns(self) -> List[str]:
        """Return loaded column names or raise for clearer diagnostics."""
        if self.columns is None:
            raise ValueError("Columns are not available. Call load_npz_data() first.")
        return self.columns

    def _get_time_values(self, df: pd.DataFrame) -> pd.Series:
        """Convert 'time' column to numeric seconds/hours safely."""
        time_series = df['time']
        if is_timedelta64_dtype(time_series):
            numeric_time = time_series.dt.total_seconds()  # type: ignore[attr-defined]
        else:
            numeric_time = pd.to_numeric(time_series, errors='coerce')
        numeric_time = numeric_time.fillna(0)
        return numeric_time / 3600.0 if numeric_time.max() > 1000 else numeric_time

    def load_npz_data(self):
        """加载NPZ文件数据"""
        print(f"\n{'=' * 60}")
        print(f"开始检测 {self.dataset_name} NPZ 文件")
        print(f"文件路径: {self.npz_path}")
        print(f"{'=' * 60}")

        try:
            # 加载NPZ文件
            loaded_data = np.load(self.npz_path, allow_pickle=True)

            # 获取所有可用的键
            available_keys = list(loaded_data.keys())
            print(f"NPZ文件中包含的键: {available_keys}")

            # 检查是否有数据
            if not available_keys:
                print("❌ NPZ文件为空，没有可用的数据键")
                return False

            # 检查是否是特征分离的格式（每个特征是一个单独的一维数组）
            # 假设所有数组长度相同
            first_key = available_keys[0]
            first_array = loaded_data[first_key]

            if len(first_array.shape) == 1:
                # 一维数组，可能是特征分离格式
                print("检测到特征分离格式（每个特征是一个一维数组）")

                # 收集所有特征数组
                feature_dict = {}
                sample_count = None

                for key in available_keys:
                    array = loaded_data[key]
                    if len(array.shape) == 1:
                        if sample_count is None:
                            sample_count = len(array)
                        elif len(array) != sample_count:
                            print(f"⚠ 警告: 特征 '{key}' 的长度不一致: {len(array)} vs {sample_count}")
                            # 截断或填充到相同长度
                            if len(array) > sample_count:
                                array = array[:sample_count]
                            else:
                                array = np.pad(array, (0, sample_count - len(array)), 'constant')

                        feature_dict[key] = array

                # 创建DataFrame
                self.df = pd.DataFrame(feature_dict)
                self.columns = list(feature_dict.keys())

                # 将DataFrame转换为numpy数组用于后续处理
                self.data = self.df.values

            else:
                # 尝试传统的二维数组格式
                if 'full_data' in loaded_data:
                    self.data = loaded_data['full_data']
                else:
                    # 尝试查找其他可能的数据键
                    data_keys = [k for k in available_keys if 'data' in k.lower()]
                    if data_keys:
                        self.data = loaded_data[data_keys[0]]
                    else:
                        # 使用第一个键
                        self.data = loaded_data[first_key]

                # 获取列名
                if 'columns' in loaded_data:
                    self.columns = [str(item) for item in loaded_data['columns']]
                elif 'column_names' in loaded_data:
                    self.columns = [str(item) for item in loaded_data['column_names']]
                else:
                    # 根据数据形状创建默认列名
                    assert self.data is not None
                    data_shape = self.data.shape
                    if len(data_shape) == 1:
                        self.columns = ['value']
                    else:
                        self.columns = [f'feature_{i}' for i in range(data_shape[1])]

                # 创建DataFrame
                self.df = pd.DataFrame(self.data, columns=self.columns)

            print(f"\n步骤1: 数据基本信息")
            assert self.data is not None
            data_shape = self.data.shape
            print(f"  数据形状: {data_shape}")
            print(f"  样本数量: {data_shape[0]:,}")

            if len(data_shape) >= 2:
                print(f"  特征数量: {data_shape[1]}")
            else:
                print(f"  特征数量: 1")

            print(f"  特征名称: {list(self.columns)}")
            print(f"  DataFrame形状: {self.df.shape}")

            return True

        except Exception as e:
            print(f"❌ 加载NPZ文件失败: {e}")
            import traceback
            traceback.print_exc()
            return False

    def basic_statistics(self):
        """计算并显示基本统计信息"""
        print(f"\n步骤2: 数据基本统计信息")

        if self.df is None:
            print("❌ Data not loaded. Run load_npz_data() first.")
            return

        df = self.df

        # 选择数值列进行分析
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if 'time' in numeric_columns and len(numeric_columns) > 1:
            # 排除时间列，专注于其他特征
            numeric_columns.remove('time')

        print(f"  分析的数值特征数量: {len(numeric_columns)}")

        # 创建统计摘要
        stats_summary = df[numeric_columns].describe().T
        stats_summary['missing'] = df[numeric_columns].isnull().sum()
        stats_summary['missing_pct'] = (stats_summary['missing'] / len(df) * 100).round(2)
        stats_summary['skewness'] = df[numeric_columns].skew()
        stats_summary['kurtosis'] = df[numeric_columns].kurtosis()

        print(f"\n  各特征统计摘要:")
        for col in numeric_columns[:5]:  # 只显示前5个特征
            col_data = df[col]
            print(f"    {col}:")
            print(f"      范围: [{col_data.min():.4f}, {col_data.max():.4f}]")
            print(f"      均值: {col_data.mean():.4f} ± {col_data.std():.4f}")
            print(f"      缺失值: {stats_summary.loc[col, 'missing']} ({stats_summary.loc[col, 'missing_pct']}%)")

        # 电压特定检查
        if 'stack_voltage' in df.columns:
            voltage_data = df['stack_voltage']
            print(f"\n  🔋 电压专项检查:")
            print(f"      电压范围: [{voltage_data.min():.3f}, {voltage_data.max():.3f}] V")
            print(f"      电压均值: {voltage_data.mean():.3f} ± {voltage_data.std():.3f} V")

            # 检查电压是否在合理范围内
            voltage_range_ok = 3.0 <= voltage_data.min() <= 3.5 and 3.0 <= voltage_data.max() <= 3.5
            voltage_status = "✓ 正常" if voltage_range_ok else "⚠ 异常"
            print(f"      电压范围状态: {voltage_status}")

    def detect_outliers(self):
        """检测异常值"""
        print(f"\n步骤3: 异常值检测")

        if self.df is None:
            print("❌ Data not loaded. Run load_npz_data() first.")
            return {}

        df = self.df
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'time' in numeric_columns:
            numeric_columns.remove('time')

        outlier_summary = {}

        for col in numeric_columns[:10]:  # 只检查前10个特征
            col_data = df[col].dropna()

            if len(col_data) < 10:
                continue

            # 使用IQR方法检测异常值
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            outlier_pct = len(outliers) / len(col_data) * 100

            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': outlier_pct,
                'bounds': (lower_bound, upper_bound)
            }

            if len(outliers) > 0:
                print(f"    {col}: {len(outliers)} 个异常值 ({outlier_pct:.2f}%)")

        return outlier_summary

    def visualize_data(self, save_dir="npz_visualization_fixed"):
        """可视化数据 - 修复字体显示，优化电压图"""
        print(f"\n步骤4: 数据可视化")

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 1. 时间序列图 - 关键特征
        fig1, axes1 = plt.subplots(4, 2, figsize=(16, 12))
        axes1 = axes1.flatten()

        # 选择关键特征进行可视化
        key_features = [
            'stack_voltage', 'current_density', 'current',
            'hydrogen_inlet_temp', 'hydrogen_outlet_temp',
            'air_inlet_temp', 'air_outlet_temp',
            'coolant_inlet_temp'
        ]

        # 如果有时间列，使用时间作为x轴
        if self.df is None:
            print("❌ Data not loaded. Run load_npz_data() first.")
            return

        df = self.df

        has_time = 'time' in df.columns

        for i, feature in enumerate(key_features):
            if i >= len(axes1) or feature not in df.columns:
                continue

            ax = axes1[i]
            if has_time:
                time_hours = self._get_time_values(df)
                ax.plot(time_hours, df[feature], linewidth=0.5, alpha=0.7)
                ax.set_xlabel('Time (hours)', fontsize=10)
            else:
                ax.plot(df[feature], linewidth=0.5, alpha=0.7)
                ax.set_xlabel('Sample Index', fontsize=10)

            # 使用英文标签避免字体问题
            if feature == 'stack_voltage':
                ax.set_ylabel('Voltage (V)', fontsize=10)
            elif feature == 'current_density':
                ax.set_ylabel('Current Density', fontsize=10)
            elif feature == 'current':
                ax.set_ylabel('Current', fontsize=10)
            elif 'temp' in feature:
                ax.set_ylabel('Temperature', fontsize=10)
            else:
                ax.set_ylabel(feature, fontsize=10)

            # 使用英文标题
            ax.set_title(f'Time Series - {feature}', fontsize=12, pad=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=9)

            # 添加统计信息（使用英文）
            stats_text = f"Mean: {df[feature].mean():.3f}\nStd: {df[feature].std():.3f}"
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 隐藏多余的子图
        for i in range(len(key_features), len(axes1)):
            axes1[i].set_visible(False)

        plt.suptitle(f'{self.dataset_name} - Key Features Time Series', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/{self.dataset_name}_time_series.png', dpi=150, bbox_inches='tight')
        print(f"  Saved time series plot: {save_dir}/{self.dataset_name}_time_series.png")
        plt.close(fig1)

        # 2. 电压分析图 - 只显示左上和右下两个子图
        if 'stack_voltage' in df.columns:
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # 左上图：电压时间序列
            if has_time:
                time_hours = self._get_time_values(df)
                ax1.plot(time_hours, df['stack_voltage'], linewidth=0.8, alpha=0.8, color='red')
                ax1.set_xlabel('Time (hours)', fontsize=12)
            else:
                ax1.plot(df['stack_voltage'], linewidth=0.8, alpha=0.8, color='red')
                ax1.set_xlabel('Sample Index', fontsize=12)

            ax1.set_ylabel('Voltage (V)', fontsize=12)
            ax1.set_title('Voltage Time Series', fontsize=14, pad=15)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=10)

            # 添加电压范围标注（使用英文）
            voltage_min = df['stack_voltage'].min()
            voltage_max = df['stack_voltage'].max()
            voltage_mean = df['stack_voltage'].mean()
            voltage_std = df['stack_voltage'].std()

            stats_text = f"Min: {voltage_min:.3f} V\nMax: {voltage_max:.3f} V\nMean: {voltage_mean:.3f} V\nStd: {voltage_std:.3f} V"
            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

            # 右下图：电压滚动均值
            # 根据数据长度调整滚动窗口
            window_size = min(500, len(self.df) // 100)  # 窗口大小不超过数据长度的1%
            if window_size < 10:
                window_size = 10

            rolling_mean = df['stack_voltage'].rolling(window=window_size, min_periods=1, center=True).mean()

            if has_time:
                time_hours = self._get_time_values(df)
                ax2.plot(time_hours, df['stack_voltage'], linewidth=0.5, alpha=0.4, color='gray',
                         label='Processed Data')
                ax2.plot(time_hours, rolling_mean, linewidth=1.5, color='darkgreen',
                         label=f'Rolling Mean (window={window_size})')
                ax2.set_xlabel('Time (hours)', fontsize=12)
            else:
                ax2.plot(df['stack_voltage'], linewidth=0.5, alpha=0.4, color='gray', label='Original')
                ax2.plot(rolling_mean, linewidth=1.5, color='darkgreen', label=f'Rolling Mean (window={window_size})')
                ax2.set_xlabel('Sample Index', fontsize=12)

            ax2.set_ylabel('Voltage (V)', fontsize=12)
            ax2.set_title('Voltage with Rolling Mean', fontsize=14, pad=15)
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=10)

            # 添加滚动均值信息
            rolling_stats = f"Rolling Window: {window_size}\nSmoothness Improved"
            ax2.text(0.02, 0.98, rolling_stats, transform=ax2.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

            plt.suptitle(f'{self.dataset_name} - Voltage Analysis', fontsize=16, y=1.02)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/{self.dataset_name}_voltage_analysis.png', dpi=150, bbox_inches='tight')
            print(f"  Saved voltage analysis plot: {save_dir}/{self.dataset_name}_voltage_analysis.png")
            plt.close(fig2)

        # 3. 特征相关性热图
        if len(df.columns) > 2:
            # 选择数值列进行相关性分析
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 15:  # 限制特征数量
                # 选择最重要的特征
                important_features = ['stack_voltage', 'current', 'current_density'] + \
                                     [col for col in numeric_cols if 'temp' in col][:5] + \
                                     [col for col in numeric_cols if 'pressure' in col][:3] + \
                                     [col for col in numeric_cols if 'flow' in col][:3]
                important_features = [f for f in important_features if f in numeric_cols]
                correlation_cols = list(set(important_features))[:15]
            else:
                correlation_cols = numeric_cols

            if len(correlation_cols) > 2:
                fig3, ax3 = plt.subplots(figsize=(12, 10))
                corr_matrix = df[correlation_cols].corr()

                # 创建热图
                im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)

                # 添加颜色条
                cbar = fig3.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
                cbar.set_label('Correlation Coefficient', rotation=270, labelpad=15, fontsize=12)

                # 设置刻度
                ax3.set_xticks(np.arange(len(correlation_cols)))
                ax3.set_yticks(np.arange(len(correlation_cols)))

                # 缩短过长的列名
                short_labels = []
                for col in correlation_cols:
                    if len(col) > 15:
                        # 对长列名进行缩写
                        if 'hydrogen' in col:
                            short_labels.append(col.replace('hydrogen', 'H2'))
                        elif 'coolant' in col:
                            short_labels.append(col.replace('coolant', 'Cool'))
                        elif 'pressure' in col:
                            short_labels.append(col.replace('pressure', 'Press'))
                        elif 'temperature' in col:
                            short_labels.append(col.replace('temperature', 'Temp'))
                        elif 'humidity' in col:
                            short_labels.append(col.replace('humidity', 'Hum'))
                        else:
                            short_labels.append(col[:12] + '...')
                    else:
                        short_labels.append(col)

                ax3.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=9)
                ax3.set_yticklabels(short_labels, fontsize=9)

                # 添加相关系数文本
                for i in range(len(correlation_cols)):
                    for j in range(len(correlation_cols)):
                        corr_value = float(pd.to_numeric(corr_matrix.iloc[i, j], errors='coerce'))  # type: ignore[arg-type]
                        # 只显示绝对值较大的相关系数
                        if abs(corr_value) > 0.3:
                            text = ax3.text(j, i, f'{corr_value:.2f}',
                                            ha="center", va="center", color="black", fontsize=8)
                        elif abs(corr_value) > 0.7:
                            # 强相关用白色显示
                            text = ax3.text(j, i, f'{corr_value:.2f}',
                                            ha="center", va="center", color="white", fontsize=8, fontweight='bold')

                ax3.set_title(f'{self.dataset_name} - Feature Correlation Heatmap', fontsize=14, pad=20)
                plt.tight_layout()
                plt.savefig(f'{save_dir}/{self.dataset_name}_correlation_heatmap.png', dpi=150, bbox_inches='tight')
                print(f"  Saved correlation heatmap: {save_dir}/{self.dataset_name}_correlation_heatmap.png")
                plt.close(fig3)

        print(f"  All plots saved to: {save_dir}/")

    def check_data_quality(self):
        """检查数据质量"""
        print(f"\n步骤5: 数据质量评估")

        try:
            df = self._ensure_df()
            columns = self._ensure_columns()
        except ValueError as e:
            print(f"❌ {e}")
            return {}

        quality_report = {
            'dataset': self.dataset_name,
            'samples': len(df),
            'features': len(columns),
            'missing_values': 0,
            'duplicates': 0,
            'quality_score': 100  # 初始质量分数
        }

        # 检查缺失值
        missing_values = df.isnull().sum().sum()
        missing_pct = missing_values / (len(df) * len(columns)) * 100
        quality_report['missing_values'] = missing_values
        quality_report['missing_pct'] = missing_pct

        print(f"  Missing values check:")
        print(f"    Total missing values: {missing_values}")
        print(f"    Missing percentage: {missing_pct:.4f}%")

        if missing_pct == 0:
            print(f"    ✓ No missing values")
        elif missing_pct < 1:
            print(f"    ⚠ Few missing values ({missing_pct:.4f}%)")
            quality_report['quality_score'] -= 5
        else:
            print(f"    ⚠ Missing values exist ({missing_pct:.4f}%)")
            quality_report['quality_score'] -= 10

        # 检查重复值
        duplicates = df.duplicated().sum()
        duplicate_pct = duplicates / len(df) * 100
        quality_report['duplicates'] = duplicates
        quality_report['duplicate_pct'] = duplicate_pct

        print(f"  Duplicates check:")
        print(f"    Duplicate rows: {duplicates}")
        print(f"    Duplicate percentage: {duplicate_pct:.4f}%")

        if duplicates == 0:
            print(f"    ✓ No duplicate rows")
        elif duplicate_pct < 0.1:
            print(f"    ⚠ Very few duplicate rows ({duplicate_pct:.4f}%)")
            quality_report['quality_score'] -= 2
        else:
            print(f"    ⚠ Duplicate rows exist ({duplicate_pct:.4f}%)")
            quality_report['quality_score'] -= 5

        # 检查数据范围（针对电压）
        if 'stack_voltage' in df.columns:
            voltage_min = df['stack_voltage'].min()
            voltage_max = df['stack_voltage'].max()

            print(f"  Voltage range check:")
            print(f"    Min voltage: {voltage_min:.3f} V")
            print(f"    Max voltage: {voltage_max:.3f} V")

            # 燃料电池电压通常应在合理范围内
            if 3.0 <= voltage_min <= 3.5 and 3.0 <= voltage_max <= 3.5:
                print(f"    ✓ Voltage range normal (3.0-3.5V)")
            else:
                print(f"    ⚠ Voltage range abnormal: {voltage_min:.3f}-{voltage_max:.3f} V")
                quality_report['quality_score'] -= 15

        # 检查数据一致性
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            std_devs = df[numeric_cols].std()
            high_variance_cols = std_devs[std_devs == 0].index.tolist()

            if high_variance_cols:
                print(f"  Zero-variance features check:")
                print(f"    Found {len(high_variance_cols)} zero-variance features: {high_variance_cols[:5]}")
                quality_report['quality_score'] -= len(high_variance_cols) * 2

        # 最终质量评分
        quality_report['quality_score'] = max(0, min(100, quality_report['quality_score']))

        print(f"\n  📊 Data Quality Score: {quality_report['quality_score']}/100")

        if quality_report['quality_score'] >= 90:
            print(f"  ✅ Excellent data quality")
        elif quality_report['quality_score'] >= 75:
            print(f"  ⚠ Good data quality, but can be improved")
        elif quality_report['quality_score'] >= 60:
            print(f"  ⚠ Average data quality, check recommended")
        else:
            print(f"  ❌ Poor data quality, reprocessing needed")

        return quality_report

    def save_summary_report(self, quality_report, save_dir="npz_visualization_fixed"):
        """保存检测报告"""
        os.makedirs(save_dir, exist_ok=True)

        try:
            df = self._ensure_df()
            columns = self._ensure_columns()
        except ValueError as e:
            print(f"❌ {e}")
            return

        report_path = f"{save_dir}/{self.dataset_name}_quality_report.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"NPZ Data Quality Report - {self.dataset_name}\n")
            f.write(f"Report Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'=' * 60}\n\n")

            f.write("1. Basic Information\n")
            f.write(f"    Dataset: {self.dataset_name}\n")
            f.write(f"    File Path: {self.npz_path}\n")
            f.write(f"    Samples: {quality_report['samples']:,}\n")
            f.write(f"    Features: {quality_report['features']}\n")
            f.write(f"    Feature List: {list(columns)}\n\n")

            f.write("2. Data Quality Checks\n")
            f.write(
                f"    Missing Values: {quality_report['missing_values']} ({quality_report.get('missing_pct', 0):.4f}%)\n")
            f.write(
                f"    Duplicate Rows: {quality_report['duplicates']} ({quality_report.get('duplicate_pct', 0):.4f}%)\n\n")

            if 'stack_voltage' in df.columns:
                voltage_min = df['stack_voltage'].min()
                voltage_max = df['stack_voltage'].max()
                f.write(f"    Voltage Range: {voltage_min:.3f} - {voltage_max:.3f} V\n")

            f.write(f"3. Quality Score\n")
            f.write(f"    Overall Quality Score: {quality_report['quality_score']}/100\n\n")

            f.write("4. Recommendations\n")
            if quality_report['quality_score'] >= 90:
                f.write("    Excellent data quality, ready for analysis and modeling.\n")
            elif quality_report['quality_score'] >= 75:
                f.write("    Good data quality, suitable for analysis and modeling.\n")
            elif quality_report['quality_score'] >= 60:
                f.write("    Average data quality, further outlier checking recommended.\n")
            else:
                f.write("    Poor data quality, data reprocessing recommended.\n")

            f.write(f"\n{'=' * 60}\n")
            f.write("Analysis Complete\n")
            f.write(f"{'=' * 60}\n")

        print(f"  Saved quality report: {report_path}")

    def run_full_analysis(self, save_dir="npz_visualization_fixed"):
        """运行完整的分析流程"""
        print(f"\n{'=' * 60}")
        print(f"Starting Full Analysis: {self.dataset_name}")
        print(f"{'=' * 60}")

        # 1. 加载数据
        if not self.load_npz_data():
            print("❌ Data loading failed, terminating analysis")
            return None

        # 2. 基本统计
        self.basic_statistics()

        # 3. 异常值检测
        self.detect_outliers()

        # 4. 数据可视化
        self.visualize_data(save_dir)

        # 5. 数据质量检查
        quality_report = self.check_data_quality()

        # 6. 保存报告
        self.save_summary_report(quality_report, save_dir)

        print(f"\n{'=' * 60}")
        print(f"{self.dataset_name} NPZ file analysis completed!")
        print(f"{'=' * 60}")

        return quality_report


def main():
    """主函数：自动检测最新的NPZ文件并进行分析"""
    print(f"\n{'=' * 60}")
    print("自动查找最新的NPZ文件")
    print(f"{'=' * 60}")

    # 自动查找最新的NPZ文件
    npz_files = find_latest_npz_files("processed_results")

    if not npz_files:
        print("❌ 未找到任何NPZ文件，请检查 processed_results 目录")
        return

    print(f"\n找到 {len(npz_files)} 个最新的NPZ文件:")
    for file_info in npz_files:
        print(f"  - {file_info['name']}: {file_info['path']}")

    all_reports = []

    for file_info in npz_files:
        npz_path = file_info['path']
        dataset_name = file_info['name']

        # 创建可视化器并运行分析
        visualizer = NPZDataVisualizerFixed(npz_path, dataset_name)
        report = visualizer.run_full_analysis()

        if report:
            all_reports.append(report)

    # 生成综合报告
    if all_reports:
        print(f"\n{'=' * 60}")
        print("综合报告总结")
        print(f"{'=' * 60}")

        for report in all_reports:
            print(f"\n{report['dataset']}:")
            print(f"  样本数: {report['samples']:,}")
            print(f"  特征数: {report['features']}")
            print(f"  缺失值: {report['missing_values']}")
            print(f"  重复行: {report['duplicates']}")
            print(f"  质量分数: {report['quality_score']}/100")

            # 添加评级
            if report['quality_score'] >= 90:
                rating = "优秀"
            elif report['quality_score'] >= 75:
                rating = "良好"
            elif report['quality_score'] >= 60:
                rating = "一般"
            else:
                rating = "差"

            print(f"  评级: {rating}")

        print(f"\n{'=' * 60}")
        print("所有NPZ文件分析完成!")
        print(f"可视化结果已保存至: npz_visualization_fixed/")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()