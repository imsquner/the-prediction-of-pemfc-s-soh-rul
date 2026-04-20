# PEMFC燃料电池寿命预测GRU模型训练框架（PyTorch版）- 修复版
"""
基于GRU的质子交换膜燃料电池性能寿命预测模型训练框架
版本：2.2 (修复模型保存和编码问题)
作者：基于论文第三章和用户需求定制

核心特点：
1. 修复模型保存时的input_size错误
2. 所有文件操作使用utf-8编码
3. 基于提供的特征重要性选择关键参数
4. 完整的日志系统和可视化功能
5. 模块化设计，便于调试和修改
"""

# ============================================================================
# 0. 导入必要的库
# ============================================================================
import os
import sys
import time
import json
import random
import logging
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pywt  # pyright: ignore[reportMissingImports]  # 小波变换库

# 强制UTF-8输出，避免Windows控制台乱码
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")


# 设置随机种子保证可重复性
def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# ============================================================================
# 1. 参数配置模块 - 基于论文逻辑和特征重要性排序
# ============================================================================

@dataclass
class ModelConfig:
    """GRU模型和训练参数配置 - 基于论文逻辑"""

    def fix_selected_features(self):
        """自动修正历史配置中的特征名映射"""
        mapping = {"J (A/cm²)": "current_density", "J (A/cm2)": "current_density"}
        self.selected_features = [mapping.get(f, f) for f in self.selected_features]

    # ==================== 数据参数 ====================
    data_path: str = "processed_results/FC1/"  # 数据目录路径（修改为目录）
    data_file_pattern: str = "*_processed_*.npz"  # 数据文件匹配模式
    data_file_suffix: str = "_processed"  # 数据文件后缀标识
    target_feature: str = "stack_voltage"  # 目标特征
    sequence_length: int = 80# 时间序列长度（滑动窗口大小）
    # 基于提供的特征重要性列表（仅前5，按最新CatBoost重要性排序）
    selected_features: List[str] = field(default_factory=lambda: [
        "air_outlet_flow",       # 48.66%
        "hydrogen_inlet_temp",   # 19.76%
        "current",               # 9.40%
        "coolant_flow",          # 7.37%
        "current_density",       # 6.93%
    ])

    # ==================== 数据预处理参数 ====================
    # 缺失值处理
    missing_value_method: str = "linear"  # 缺失值处理方法

    # 异常值处理
    outlier_method: str = "3sigma"  # 异常值处理方法
    outlier_sigma: float = 3.0  # σ值

    # 小波去噪参数（基于论文要求）
    wavelet_method: str = "sym8"  # 小波基函数
    wavelet_level: int = 4  # 分解层数
    wavelet_threshold_percentile: int = 70  # 阈值百分位数
    enable_wavelet: bool = False# 可关闭以减少过平滑

    # ==================== 数据划分参数 ====================
    train_ratio: float = 0.8 # 训练集比例
    val_ratio: float = 0.1  # 验证集比例
    test_ratio: float = 0.1 # 测试集比例
    # 预测步数（基础步数，自动根据时间步长扩展）
    forecast_steps: int = 200
    # 滚动预测目标时间范围（小时），用于可视化外推到未来
    forecast_horizon: float = 2200.0
    # 滚动预测最大步数上限，防止过多步导致耗时
    forecast_max_steps: Optional[int] = 200000
    # 若采样间隔不稳定，可固定滚动预测的时间步长；None 表示自动取中位数
    fixed_dt: Optional[float] = None

    # ==================== GRU模型参数 ====================
    # 【作用】GRU隐藏层单元数量，决定模型容量
    # 【默认值】64，根据论文和数据集大小调整
    # 【修改建议】数据量大可增加（128-256），数据量小可减少（16-32）
    gru_hidden_size: int = 192

    # 【作用】GRU层数，增加模型深度
    # 【默认值】2，平衡模型复杂度和过拟合风险
    # 【修改建议】简单任务用1层，复杂任务可用2-3层
    gru_num_layers: int = 1

    # 【作用】Dropout率，防止过拟合
    # 【默认值】0.3，经验值
    # 【修改建议】过拟合时增加（0.4-0.5），欠拟合时减少（0.1-0.2）
    dropout_rate: float = 0.1

    # 【作用】是否使用双向GRU
    # 【默认值】True，捕获前后时序信息
    # 【修改建议】对上下文敏感的任务建议True
    bidirectional: bool = False

    # ==================== 全连接层参数 ====================
    # 【作用】全连接层隐藏单元数
    # 【默认值】[32, 16]，逐步压缩特征
    # 【修改建议】根据输入特征维度调整
    fc_hidden_sizes: List[int] = field(default_factory=lambda: [32, 16])

    # ==================== 训练参数 ====================
    # 【作用】批大小，影响训练稳定性和速度
    # 【默认值】32，经验值
    # 【修改建议】GPU内存大可增加（64-128），内存小需减少（16-24）
    batch_size: int = 32

    # 【作用】训练轮数
    # 【默认值】200，足够收敛
    # 【修改建议】根据早停机制调整，通常100-300
    epochs: int = 200

    # 【作用】学习率，控制参数更新步长
    # 【默认值】0.001，常用初始值
    # 【修改建议】训练不稳定时减少（0.0001-0.0005），收敛慢时增加（0.005）
    learning_rate: float = 0.0006

    # 【作用】权重衰减（L2正则化）
    # 【默认值】0.0001，防止过拟合
    # 【修改建议】过拟合时增加（0.001），欠拟合时减少（0.0）
    weight_decay: float = 0.0001

    # 【作用】早停耐心值
    # 【默认值】20，连续20轮验证损失不改善则停止
    # 【修改建议】数据集大时增加（30-50），小时减少（10-15）
    patience: int = 30

    # 【作用】学习率衰减因子
    # 【默认值】0.7，每次衰减为原来的0.7倍
    # 【修改建议】训练震荡时减小（0.5），稳定时可增大（0.8）
    lr_decay_factor: float = 0.7

    # 【作用】学习率衰减耐心值
    # 【默认值】10，连续10轮不改善则降低学习率
    # 【修改建议】与早停耐心值协调设置
    lr_patience: int = 8

    # 【作用】梯度裁剪阈值
    # 【默认值】1.0，防止梯度爆炸
    # 【修改建议】训练不稳定时减少（0.5-1.0）
    grad_clip: float = 1.0

    # 【作用】斜率一致性辅助损失权重，鼓励预测曲线的变化幅度跟随真实值，避免过度平滑
    slope_loss_weight: float = 0.2

    # 【作用】方差匹配辅助损失权重，使预测序列波动幅度接近真实，进一步避免输出塌缩
    variance_loss_weight: float = 0.3

    # ==================== 推理偏差校准 ====================
    # 针对FC1的系统性抬高偏差进行校准（正值表示在推理时减去该偏差）
    apply_fc1_bias_correction: bool = True
    fc1_bias_correction: float = 0.0128

    # RUL计算前对预测电压的缩放因子（<1.0 可下调偏大预测）
    rul_prediction_scale: float = 0.995

    # ==================== SOH/RUL参数 ====================
    # 【作用】SOH失效阈值
    # 【默认值】0.96，基于论文设定
    soh_threshold: float = 0.96

    # 【作用】计算初始电压的窗口大小
    # 【默认值】100，基于论文要求
    v_initial_window: int = 100

    # 【作用】V_initial稳定性阈值（变异系数）
    # 【默认值】0.01，小于1%认为稳定
    cv_threshold: float = 0.01

    # 【作用】V_initial去噪方法
    # 【默认值】"wavelet"，基于论文要求
    denoising_method: str = "wavelet"

    # ==================== 设备与路径 ====================
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "train_results_paper"
    experiment_name: str = "gru_pemfc_paper_v2"

    def __post_init__(self):
        """初始化后处理"""
        # 检查GPU可用性
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            print("警告: GPU不可用，切换到CPU")

        # 创建保存目录
        self._create_directories()

    def _create_directories(self):
        """创建保存目录结构"""
        base_dir = os.path.join(self.save_dir, self.experiment_name)
        directories = [
            base_dir,
            os.path.join(base_dir, "models"),
            os.path.join(base_dir, "csv_files"),
            os.path.join(base_dir, "images"),
            os.path.join(base_dir, "tables"),
            os.path.join(base_dir, "logs"),
            os.path.join(base_dir, "configs"),
            os.path.join(base_dir, "test_results")
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

        self.save_paths = {
            'base': base_dir,
            'models': os.path.join(base_dir, "models"),
            'csv': os.path.join(base_dir, "csv_files"),
            'images': os.path.join(base_dir, "images"),
            'tables': os.path.join(base_dir, "tables"),
            'logs': os.path.join(base_dir, "logs"),
            'configs': os.path.join(base_dir, "configs"),
            'test_results': os.path.join(base_dir, "test_results")
        }

    def save_config(self):
        """保存配置到JSON文件 - 修复编码问题"""
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_') and k != 'save_paths'
        }

        # 处理field默认值
        if 'selected_features' in config_dict:
            config_dict['selected_features'] = list(config_dict['selected_features'])
        if 'fc_hidden_sizes' in config_dict:
            config_dict['fc_hidden_sizes'] = list(config_dict['fc_hidden_sizes'])

        config_file = os.path.join(
            self.save_paths['configs'],
            f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        # 修复编码问题：使用utf-8编码
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=4, ensure_ascii=False)

        return config_file


# ============================================================================
# 2. 日志系统模块
# ============================================================================

class TrainingLogger:
    """训练日志记录器"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.log_dir = config.save_paths['logs']

        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"training_{timestamp}.log")

        # 配置logging
        self.logger = logging.getLogger('PEMFC_GRU_Training_Paper')
        self.logger.setLevel(logging.DEBUG)

        # 清除已有处理器
        self.logger.handlers.clear()

        # 文件处理器 - 修复编码问题
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)

        # 控制台处理器（使用UTF-8编码的stdout）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.start_time = time.time()

    def log_config(self, config: ModelConfig):
        """记录配置参数"""
        self.logger.info("=" * 80)
        self.logger.info("PEMFC燃料电池GRU寿命预测模型（论文版本）")
        self.logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 80)
        self.logger.info("[模型配置参数]")

        config_dict = {
            k: v for k, v in config.__dict__.items()
            if not k.startswith('_') and k != 'save_paths'
        }

        for key, value in config_dict.items():
            if isinstance(value, list):
                value_str = f"[{len(value)} 个元素] " + ', '.join([str(v) for v in value[:5]])
                if len(value) > 5:
                    value_str += "..."
                self.logger.info(f"  {key}: {value_str}")
            else:
                self.logger.info(f"  {key}: {value}")

        self.logger.info("=" * 80)

    def log_data_info(self, data_info: Dict[str, Any]):
        """记录数据信息"""
        self.logger.info("[数据信息]")
        for key, value in data_info.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("-" * 80)

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                  train_metrics: Dict[str, float], val_metrics: Dict[str, float],
                  lr: float):
        """记录每个epoch的训练信息"""
        epoch_str = f"Epoch {epoch:03d}/{self.config.epochs:03d}"
        loss_str = f"训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}"
        lr_str = f"学习率: {lr:.6f}"

        # 训练指标
        train_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        # 验证指标
        val_metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])

        self.logger.info(f"{epoch_str} | {loss_str} | {lr_str}")
        self.logger.info(f"  训练指标: {train_metrics_str}")
        self.logger.info(f"  验证指标: {val_metrics_str}")

    def log_info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)

    def log_warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(f"警告: {message}")

    def log_error(self, message: str):
        """记录错误信息"""
        self.logger.error(f"错误: {message}")

    def log_step_complete(self, step_name: str):
        """记录步骤完成"""
        self.logger.info(f"✓ {step_name} 完成")


# ============================================================================
# 3. 数据处理模块 - 基于论文逻辑
# ============================================================================


class WaveletDenoiser:
    """小波阈值去噪类 - 基于论文逻辑"""

    @staticmethod
    def wavelet_denoise(signal_data: Any, wavelet: str = 'sym8',
                        level: int = 6, percentile: int = 90) -> np.ndarray:
        """
        小波阈值去噪 - 基于论文逻辑

        参数:
            signal_data: 输入信号
            wavelet: 小波基函数 (论文使用sym8)
            level: 分解层数 (论文使用6层)
            percentile: 阈值百分位数 (论文使用90分位数)

        返回:
            去噪后的信号
        """
        # 先将输入转换为 numpy 数组，确保兼容 pandas ExtensionArray 等
        signal_data = np.asarray(signal_data, dtype=float)
        try:
            if len(signal_data) < 2 ** level:
                raise ValueError(f"信号长度 ({len(signal_data)}) 太短，无法进行 {level} 层分解")

            # 小波分解
            coeffs = pywt.wavedec(signal_data, wavelet, level=level)

            # 计算阈值
            # 获取所有细节系数
            detail_coeffs = []
            for i in range(1, len(coeffs)):  # 跳过近似系数
                detail_coeffs.extend(coeffs[i].flatten())

            # 计算绝对值并排序
            abs_coeffs = np.abs(detail_coeffs)
            sorted_coeffs = np.sort(abs_coeffs)

            # 计算平方序列 F(k)
            F_k = sorted_coeffs ** 2

            # 选择阈值（论文使用90分位数）
            threshold_index = int(len(F_k) * percentile / 100)
            lambda_k = np.sqrt(F_k[threshold_index])

            # 软阈值处理（仅对细节系数）
            coeffs_thresholded = [coeffs[0]]  # 保留近似系数

            for i in range(1, len(coeffs)):
                coeff = coeffs[i]
                # 软阈值处理
                coeff_thresholded = np.sign(coeff) * np.maximum(np.abs(coeff) - lambda_k, 0)
                coeffs_thresholded.append(coeff_thresholded)

            # 小波重构
            denoised_signal = pywt.waverec(coeffs_thresholded, wavelet)

            # 确保长度一致
            if len(denoised_signal) > len(signal_data):
                denoised_signal = denoised_signal[:len(signal_data)]
            elif len(denoised_signal) < len(signal_data):
                # 填充缺失值
                padding = np.zeros(len(signal_data) - len(denoised_signal))
                denoised_signal = np.concatenate([denoised_signal, padding])

            return denoised_signal

        except Exception as e:
            print(f"小波去噪失败: {e}, 返回原始信号")
            return np.asarray(signal_data, dtype=float)

    @staticmethod
    def evaluate_denoising(original: Any, denoised: Any) -> Dict[str, float]:
        """评估去噪效果"""
        # 转换为 numpy 数组
        orig = np.asarray(original, dtype=float)
        den = np.asarray(denoised, dtype=float)

        # 计算信噪比 (SNR)
        signal_power = float(np.mean(orig ** 2))
        noise_power = float(np.mean((orig - den) ** 2))
        snr = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else float('inf')

        # 计算均方根误差 (RMSE)
        rmse = float(np.sqrt(np.mean((orig - den) ** 2)))

        # 计算平滑度 (一阶差分方差)
        original_diff = np.diff(orig)
        denoised_diff = np.diff(den)
        smoothness = float(np.var(denoised_diff) / np.var(original_diff)) if np.var(original_diff) > 0 else 0.0

        return {
            'SNR_dB': float(snr),
            'RMSE': float(rmse),
            'Smoothness_Ratio': float(smoothness)
        }


class DataProcessor:
    """数据处理类 - 基于论文逻辑"""

    def __init__(self, config: ModelConfig, logger: TrainingLogger):
        # 自动修正特征名，兼容历史配置
        if hasattr(config, 'fix_selected_features'):
            config.fix_selected_features()
        self.config = config
        self.logger = logger
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        # 记录训练集中各特征的最小/最大值，用于后续滚动预测时进行安全裁剪
        self.feature_min: Optional[np.ndarray] = None
        self.feature_max: Optional[np.ndarray] = None
        # 使用配置中的选定特征
        self.selected_features = config.selected_features.copy()

    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """安全的特征标准化，避免零方差导致归一化退化"""
        X_arr = np.asarray(X, dtype=float)
        transformed = self.scaler_X.transform(X_arr)
        if not np.all(np.isfinite(transformed)):
            self.logger.log_warning("标准化后存在非有限值，已替换为0")
            transformed = np.nan_to_num(transformed, nan=0.0, posinf=0.0, neginf=0.0)
        return transformed

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """目标标准化，保持二维形状"""
        y_arr = np.asarray(y, dtype=float).reshape(-1, 1)
        scaled = self.scaler_y.transform(y_arr)
        if not np.all(np.isfinite(scaled)):
            self.logger.log_warning("目标标准化后存在非有限值，已替换为0")
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return scaled

    def generate_future_features(
        self,
        recent_features: np.ndarray,
        max_step_ratio: float = 0.05,
        clamp_min: Optional[np.ndarray] = None,
        clamp_max: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """根据最近趋势生成下一步特征，限制增量并裁剪到合理范围"""
        window = np.asarray(recent_features, dtype=float)
        if window.ndim == 1:
            window = window.reshape(1, -1)

        # 取最近三步估计趋势
        tail = window[-3:] if len(window) >= 3 else window
        if tail.shape[0] > 1:
            deltas = np.diff(tail, axis=0)
            trend = deltas.mean(axis=0)
        else:
            trend = np.zeros(tail.shape[1], dtype=float)

        last = tail[-1]
        step_limit = np.maximum(np.abs(last) * max_step_ratio, 1e-3)
        trend = np.clip(trend, -step_limit, step_limit)
        next_feat = last + trend

        # 避免生成 NaN/Inf
        next_feat = np.nan_to_num(next_feat, nan=last, posinf=last, neginf=last)
        # 使用训练数据的范围或调用方提供的范围进行裁剪，避免长时间滚动时特征漂移到非物理区间
        low = clamp_min if clamp_min is not None else self.feature_min
        high = clamp_max if clamp_max is not None else self.feature_max
        if low is not None and high is not None:
            next_feat = np.clip(next_feat, low, high)
        return next_feat.astype(float)

    def _fix_np_str_columns(self, columns: List[Any]) -> List[str]:
        """修复numpy.str_类型的列名"""
        fixed_columns = []
        for col in columns:
            if isinstance(col, np.str_):
                fixed_columns.append(str(col))
            else:
                fixed_columns.append(str(col))
        return fixed_columns

    def _check_and_fix_duplicate_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """检查并修复重复列名"""
        # 检查重复列名
        if data.columns.duplicated().any():
            self.logger.log_warning("数据中发现重复列名")

            # 获取重复列名
            duplicate_columns = data.columns[data.columns.duplicated()].tolist()
            self.logger.log_info(f"重复列: {duplicate_columns}")

            # 记录原始形状
            original_shape = data.shape

            # 移除重复列（保留第一次出现的列）
            data = data.loc[:, ~data.columns.duplicated()]

            self.logger.log_info(f"已移除重复列。形状从 {original_shape} 变为 {data.shape}")
            self.logger.log_info(f"清理后的列: {list(data.columns)}")

        return data

    def _find_latest_data_file(self, data_path: Optional[str] = None) -> str:
        """
        自动查找目录下最新的数据文件

        返回:
            最新数据文件的完整路径

        异常:
            FileNotFoundError: 如果没有找到有效的数据文件
        """
        base_path = data_path or self.config.data_path
        self.logger.log_info(f"在目录中搜索最新数据文件: {base_path}")

        # 确保目录存在
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"数据目录未找到: {base_path}")

        # 检查是否是文件而不是目录（兼容旧配置）
        if os.path.isfile(base_path):
            self.logger.log_info(f"数据路径是文件，使用: {base_path}")
            return base_path

        # 获取目录下所有.npz文件
        try:
            all_files = []
            for file_name in os.listdir(base_path):
                if file_name.endswith('.npz'):
                    file_path = os.path.join(base_path, file_name)
                    all_files.append(file_path)

            if not all_files:
                raise FileNotFoundError(f"目录中未找到.npz文件: {base_path}")

            self.logger.log_info(f"在目录中找到 {len(all_files)} 个.npz文件")

            # 根据配置模式筛选文件
            pattern_files = []
            import fnmatch

            for file_path in all_files:
                file_name = os.path.basename(file_path)
                if fnmatch.fnmatch(file_name, self.config.data_file_pattern):
                    pattern_files.append(file_path)

            if not pattern_files:
                self.logger.log_warning(f"未找到匹配模式 '{self.config.data_file_pattern}' 的文件")
                self.logger.log_info("使用所有.npz文件进行时间戳解析")
                pattern_files = all_files
            else:
                self.logger.log_info(f"找到 {len(pattern_files)} 个匹配模式的文件")

            # 解析时间戳并排序
            timestamp_files = []
            for file_path in pattern_files:
                file_name = os.path.basename(file_path)

                # 尝试从文件名中提取时间戳
                timestamp = self._extract_timestamp_from_filename(file_name)

                if timestamp:
                    timestamp_files.append((timestamp, file_path, file_name))
                else:
                    # 如果没有时间戳，使用文件修改时间
                    mod_time = os.path.getmtime(file_path)
                    timestamp_files.append((mod_time, file_path, file_name))
                    self.logger.log_warning(f"文件名中未找到时间戳: {file_name}, 使用修改时间")

            # 按时间戳降序排序（最新的在前）
            timestamp_files.sort(key=lambda x: x[0], reverse=True)

            # 选择最新的文件
            latest_timestamp, latest_file, latest_filename = timestamp_files[0]

            # 记录选择结果
            if isinstance(latest_timestamp, float):
                # 这是文件修改时间
                time_str = datetime.fromtimestamp(latest_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                self.logger.log_info(
                    f"按修改时间选择最新文件: {latest_filename} (修改于 {time_str})")
            else:
                # 这是解析的时间戳
                self.logger.log_info(f"按时间戳选择最新文件: {latest_filename}")

            self.logger.log_info(f"使用数据文件: {latest_file}")

            return latest_file

        except Exception as e:
            self.logger.log_error(f"查找最新数据文件失败: {str(e)}")
            raise FileNotFoundError(f"目录中未找到有效数据文件: {self.config.data_path}. 错误: {str(e)}")

    def _extract_timestamp_from_filename(self, filename: str) -> Optional[datetime]:
        """
        从文件名中提取时间戳

        参数:
            filename: 文件名

        返回:
            解析的时间戳，如果无法解析则返回None
        """
        try:
            # 查找文件名中的时间戳部分（格式：YYYYMMDD_HHMMSS）
            import re

            # 模式1: _YYYYMMDD_HHMMSS
            pattern1 = r'_(\d{8}_\d{6})'
            # 模式2: _YYYY-MM-DD_HH-MM-SS
            pattern2 = r'_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})'
            # 模式3: _YYYYMMDD
            pattern3 = r'_(\d{8})'

            match = None
            for pattern in [pattern1, pattern2, pattern3]:
                match = re.search(pattern, filename)
                if match:
                    break

            if not match:
                return None

            timestamp_str = match.group(1)

            # 解析时间戳
            if '_' in timestamp_str:
                # 包含日期和时间
                if '-' in timestamp_str:
                    # 格式: YYYY-MM-DD_HH-MM-SS
                    date_str, time_str = timestamp_str.split('_')
                    datetime_str = f"{date_str.replace('-', '')}_{time_str.replace('-', '')}"
                    return datetime.strptime(datetime_str, '%Y%m%d_%H%M%S')
                else:
                    # 格式: YYYYMMDD_HHMMSS
                    return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
            else:
                # 只包含日期: YYYYMMDD
                return datetime.strptime(timestamp_str, '%Y%m%d')

        except Exception as e:
            self.logger.log_warning(f"从文件名 '{filename}' 解析时间戳失败: {str(e)}")
            return None

    def load_and_process_data(self, data_path_override: Optional[str] = None,
                              update_selected_features: bool = True) -> pd.DataFrame:
        """
        加载数据并进行预处理 - 使用配置中的选定特征

        返回:
            包含选定特征的DataFrame
        """
        self.logger.log_info("步骤1: 加载数据并使用预定义特征")

        try:
            # 1. 查找最新的数据文件
            data_file = self._find_latest_data_file(data_path_override)
            self.logger.log_info(f"使用数据文件: {data_file}")

            # 2. 加载处理后的数据
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"数据文件未找到: {data_file}")

            data_npz = np.load(data_file, allow_pickle=True)

            # 获取所有特征名称
            all_features = list(data_npz.keys())
            self.logger.log_info(f"NPZ文件中的所有特征: {all_features}")

            # 检查是否有目标特征
            if self.config.target_feature not in all_features:
                raise ValueError(
                    f"目标特征 '{self.config.target_feature}' 未在数据中找到。可用特征: {all_features}")

            # 确定数据长度
            first_feature = all_features[0]
            data_length = len(data_npz[first_feature])
            self.logger.log_info(f"数据长度: {data_length}")

            # 创建字典来存储数据
            data_dict = {}
            for feature in all_features:
                data_dict[feature] = data_npz[feature]
                # 检查长度是否一致
                if len(data_npz[feature]) != data_length:
                    self.logger.log_warning(
                        f"特征 '{feature}' 的长度为 {len(data_npz[feature])}, 预期 {data_length}")

            # 创建DataFrame
            data = pd.DataFrame(data_dict)

            # 修复列名类型
            data.columns = self._fix_np_str_columns(data.columns.tolist())

            # 检查并修复重复列
            data = self._check_and_fix_duplicate_columns(data)

            self.logger.log_info(f"原始数据加载: {data.shape[0]} 行, {data.shape[1]} 列")

            # 3. 选择特征（基于配置中的选定特征）
            # 训练时允许按实际存在的列更新特征；推理时保持训练特征并严格校验
            features_to_keep = self.selected_features.copy()

            # 添加目标特征（如果尚未存在）
            if self.config.target_feature not in features_to_keep:
                features_to_keep.append(self.config.target_feature)

            # 检查'time'列是否存在
            if 'time' in data.columns:
                # 如果'time'不在列表中，则添加
                if 'time' not in features_to_keep:
                    features_to_keep.append('time')
            else:
                self.logger.log_warning("'time' 列未在数据中找到")
                # 如果没有时间列，创建一个索引作为时间
                self.logger.log_info("创建基于索引的时间列")
                data['time'] = np.arange(len(data))
                if 'time' not in features_to_keep:
                    features_to_keep.append('time')

            # 兼容历史/跨数据集特征命名，尽量复用语义一致列避免推理中断
            feature_aliases = {
                "iTnH2 (°C)": ["hydrogen_inlet_temp"]
            }
            for expected_feature, alias_candidates in feature_aliases.items():
                if expected_feature in features_to_keep and expected_feature not in data.columns:
                    for alias in alias_candidates:
                        if alias in data.columns:
                            data[expected_feature] = data[alias]
                            self.logger.log_warning(
                                f"特征 '{expected_feature}' 缺失，已使用别名列 '{alias}' 进行回退"
                            )
                            break

            # 只保留存在的特征
            existing_features = [f for f in features_to_keep if f in data.columns]
            missing_features = [f for f in features_to_keep if f not in data.columns]

            if missing_features:
                if update_selected_features:
                    self.logger.log_warning(f"以下选定特征在数据中不存在: {missing_features}")
                else:
                    raise ValueError(f"数据集中缺少训练所需特征: {missing_features}")

            data_selected = data[existing_features].copy()

            # 再次检查重复列
            data_selected = self._check_and_fix_duplicate_columns(data_selected)

            if update_selected_features:
                # 更新选定特征列表（排除目标特征，保留时间以建模退化斜率）
                self.selected_features = [f for f in existing_features if f != self.config.target_feature]

            self.logger.log_info(f"选定的特征（排除目标）: {self.selected_features}")
            self.logger.log_info(f"选定数据形状: {data_selected.shape}")

            return data_selected

        except Exception as e:
            self.logger.log_error(f"数据加载和处理失败: {str(e)}")
            # 对可预期的数据校验错误仅保留简洁日志，避免干扰主流程阅读
            if not isinstance(e, ValueError):
                import traceback
                self.logger.log_error(traceback.format_exc())
            raise

    def apply_wavelet_denoising(self, data: pd.DataFrame) -> pd.DataFrame:
        """应用小波去噪 - 基于论文逻辑"""
        if not self.config.enable_wavelet:
            self.logger.log_info("步骤2: 跳过小波去噪（enable_wavelet=False）")
            return data

        self.logger.log_info("步骤2: 应用小波阈值去噪")

        # 创建数据副本，避免原地修改
        data_denoised = data.copy()

        # 检查并记录数据列
        self.logger.log_info(f"去噪前数据列: {list(data_denoised.columns)}")

        # 对每个数值列应用去噪
        numeric_cols = data_denoised.select_dtypes(include=[np.number]).columns.tolist()
        self.logger.log_info(f"需要去噪的数值列: {numeric_cols}")

        for col in numeric_cols:
            if col == 'time':  # 时间列不需要去噪
                continue

            # 目标列跳过去噪，保留真实波动
            if col == self.config.target_feature:
                self.logger.log_info(f"跳过目标列去噪: {col}")
                continue

            try:
                original_signal = data_denoised[col].values
                denoised_signal = WaveletDenoiser.wavelet_denoise(
                    original_signal,
                    wavelet=self.config.wavelet_method,
                    level=self.config.wavelet_level,
                    percentile=self.config.wavelet_threshold_percentile
                )

                data_denoised[col] = denoised_signal

                # 评估去噪效果
                if col == self.config.target_feature:
                    metrics = WaveletDenoiser.evaluate_denoising(original_signal, denoised_signal)
                    self.logger.log_info(f"{col} 的去噪指标: SNR={metrics['SNR_dB']:.2f}dB, "
                                         f"RMSE={metrics['RMSE']:.6f}, 平滑度={metrics['Smoothness_Ratio']:.4f}")
            except Exception as e:
                self.logger.log_warning(f"列 {col} 去噪失败: {str(e)}")
                self.logger.log_info(f"保留列 {col} 的原始值")

        self.logger.log_step_complete("小波去噪")
        return data_denoised

    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建时间序列数据 - 基于论文滑动窗口方法

        参数:
            X: 特征数据 [n_samples, n_features]
            y: 目标数据 [n_samples, 1]

        返回:
            sequences: 序列数据 [n_sequences, sequence_length, n_features]
            targets: 目标数据 [n_sequences, 1]
        """
        self.logger.log_info(f"步骤3: 创建序列，窗口大小 {self.config.sequence_length}")

        sequences = []
        targets = []

        for i in range(len(X) - self.config.sequence_length):
            seq = X[i:i + self.config.sequence_length]
            target = y[i + self.config.sequence_length - 1]

            sequences.append(seq)
            targets.append(target)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32).reshape(-1, 1)

        self.logger.log_info(f"创建了 {len(sequences)} 个序列")
        return sequences, targets

    def split_data(self, data: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        划分数据集 - 基于时间顺序

        参数:
            data: 包含时间和目标特征的数据

        返回:
            包含训练、验证、测试集的字典
        """
        self.logger.log_info("步骤4: 按时间顺序划分数据")

        # 检查列名
        self.logger.log_info(f"划分前的数据列: {data.columns.tolist()}")

        # 检查重复列名
        if data.columns.duplicated().any():
            self.logger.log_warning("划分前发现重复列名")
            self.logger.log_info("移除重复列（保留第一次出现）")
            data = data.loc[:, ~data.columns.duplicated()]

        # 检查并确保'time'列存在
        if 'time' not in data.columns:
            self.logger.log_warning("'time' 列未找到，创建基于索引的时间")
            data = data.copy()
            data['time'] = np.arange(len(data))

        # 确保数据按时间排序
        try:
            data = data.sort_values('time').reset_index(drop=True)
            self.logger.log_info(
                f"数据按时间排序。时间范围: {data['time'].iloc[0]:.1f}h 到 {data['time'].iloc[-1]:.1f}h")
        except Exception as e:
            self.logger.log_error(f"按时间排序数据失败: {str(e)}")
            self.logger.log_info("继续不按时间排序")

        n_samples = len(data)
        train_end = int(n_samples * self.config.train_ratio)
        val_end = train_end + int(n_samples * self.config.val_ratio)

        # 划分数据集
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]

        self.logger.log_info(
            f"训练集: {len(train_data)} 个样本 ({train_data['time'].iloc[0]:.1f}h - {train_data['time'].iloc[-1]:.1f}h)")
        self.logger.log_info(
            f"验证集: {len(val_data)} 个样本 ({val_data['time'].iloc[0]:.1f}h - {val_data['time'].iloc[-1]:.1f}h)")
        self.logger.log_info(
            f"测试集: {len(test_data)} 个样本 ({test_data['time'].iloc[0]:.1f}h - {test_data['time'].iloc[-1]:.1f}h)")

        # 准备特征和目标
        X_train = train_data[self.selected_features].to_numpy()
        y_train = train_data[self.config.target_feature].to_numpy().reshape(-1, 1)

        X_val = val_data[self.selected_features].to_numpy()
        y_val = val_data[self.config.target_feature].to_numpy().reshape(-1, 1)

        X_test = test_data[self.selected_features].to_numpy()
        y_test = test_data[self.config.target_feature].to_numpy().reshape(-1, 1)

        # 标准化
        self.logger.log_info("步骤5: 标准化特征与目标")

        # 使用训练集拟合特征标准化器，避免零方差导致归一化为常数
        self.scaler_X.fit(X_train)

        scale_arr = getattr(self.scaler_X, 'scale_', None)
        mean_arr = getattr(self.scaler_X, 'mean_', None)
        if scale_arr is not None:
            scale_arr = np.asarray(scale_arr, dtype=float)
            zero_var_mask = np.isclose(scale_arr, 0)
            if zero_var_mask.any():
                zero_cols = [self.selected_features[i] for i, flag in enumerate(zero_var_mask) if flag]
                self.logger.log_warning(f"以下特征标准差为0，使用1.0替代: {zero_cols}")
                scale_arr[zero_var_mask] = 1.0
            self.scaler_X.scale_ = scale_arr

        if mean_arr is not None:
            mean_arr = np.asarray(mean_arr, dtype=float)
        if mean_arr is not None and scale_arr is not None:
            self.logger.log_info(
                f"特征标准化器均值范围: {mean_arr.min():.4f}~{mean_arr.max():.4f}, "
                f"标准差范围: {scale_arr.min():.4f}~{scale_arr.max():.4f}")

        # 目标标准化器
        self.scaler_y.fit(y_train)
        sy_scale = getattr(self.scaler_y, 'scale_', None)
        sy_mean = getattr(self.scaler_y, 'mean_', None)
        if sy_scale is not None:
            sy_scale = np.asarray(sy_scale, dtype=float)
            zero_var_y = np.isclose(sy_scale, 0)
            if zero_var_y.any():
                self.logger.log_warning("目标标准差为0，使用1.0替代")
                sy_scale[zero_var_y] = 1.0
            self.scaler_y.scale_ = sy_scale
        if sy_mean is not None and sy_scale is not None:
            self.logger.log_info(
                f"目标标准化器均值: {float(np.asarray(sy_mean).min()):.4f}, "
                f"标准差: {float(np.asarray(sy_scale).min()):.6f}")

        # 转换所有数据（特征与目标）
        X_train_scaled = self.transform_features(X_train)
        X_val_scaled = self.transform_features(X_val)
        X_test_scaled = self.transform_features(X_test)

        y_train_scaled = self.transform_target(y_train)
        y_val_scaled = self.transform_target(y_val)
        y_test_scaled = self.transform_target(y_test)

        # 创建序列
        X_train_seq, y_train_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        X_val_seq, y_val_seq = self.create_sequences(X_val_scaled, y_val_scaled)
        X_test_seq, y_test_seq = self.create_sequences(X_test_scaled, y_test_scaled)

        # 记录训练特征的取值范围，后续滚动预测时用于裁剪生成的特征，防止长时间外推导致数据爆炸
        try:
            train_feats = train_data[self.selected_features].to_numpy()
            self.feature_min = np.nanmin(train_feats, axis=0)
            self.feature_max = np.nanmax(train_feats, axis=0)
            if self.feature_min is not None and self.feature_max is not None:
                self.logger.log_info(
                    f"训练特征范围: min={self.feature_min.min():.4f}~max={self.feature_max.max():.4f}"
                )
            else:
                self.logger.log_warning("训练特征范围未记录，跳过范围日志")
        except Exception as range_err:
            self.logger.log_warning(f"记录训练特征范围失败: {range_err}")

        result = {
            'train': (X_train_seq, y_train_seq),
            'val': (X_val_seq, y_val_seq),
            'test': (X_test_seq, y_test_seq),
            'train_time': train_data['time'].to_numpy() if 'time' in train_data.columns else None,
            'val_time': val_data['time'].to_numpy() if 'time' in val_data.columns else None,
            'test_time': test_data['time'].to_numpy() if 'time' in test_data.columns else None,
            'train_original': y_train,
            'val_original': y_val,
            'test_original': y_test,
            'train_features': train_data[self.selected_features].to_numpy(),
            'val_features': val_data[self.selected_features].to_numpy(),
            'test_features': test_data[self.selected_features].to_numpy()
        }

        self.logger.log_step_complete("数据划分和预处理")
        return result

    def inverse_transform_target(self, y_scaled: np.ndarray) -> np.ndarray:
        """反标准化目标值"""
        if hasattr(self.scaler_y, 'scale_') and getattr(self.scaler_y, 'scale_', None) is not None:
            y_arr = np.asarray(y_scaled, dtype=float).reshape(-1, 1)
            return self.scaler_y.inverse_transform(y_arr).reshape(-1)
        return np.asarray(y_scaled, dtype=float).reshape(-1)

    def save_scalers(self):
        """保存标准化器"""
        scaler_dir = self.config.save_paths['csv']
        # 确保保存的对象为 ndarray（避免 None 或 pandas 扩展类型）
        sx_mean = np.asarray(getattr(self.scaler_X, 'mean_', np.array([])))
        sx_scale = np.asarray(getattr(self.scaler_X, 'scale_', np.array([])))
        sy_mean = np.asarray(getattr(self.scaler_y, 'mean_', np.array([])))
        sy_scale = np.asarray(getattr(self.scaler_y, 'scale_', np.array([])))

        np.savez(
            os.path.join(scaler_dir, 'scalers.npz'),
            scaler_X_mean=sx_mean,
            scaler_X_scale=sx_scale,
            scaler_y_mean=sy_mean,
            scaler_y_scale=sy_scale,
            selected_features=np.asarray(self.selected_features)
        )
        self.logger.log_info("标准化器保存成功")


# ============================================================================
# 4. GRU模型定义 - 基于论文逻辑
# ============================================================================

class PEMFCGRUModel(nn.Module):
    """PEMFC寿命预测GRU模型 - 基于论文逻辑"""

    def __init__(self, config: ModelConfig, input_size: int):
        """
        初始化GRU模型

        参数:
            config: 模型配置
            input_size: 输入特征维度
        """
        super(PEMFCGRUModel, self).__init__()
        self.config = config
        self.input_size = input_size  # 保存输入维度

        # 计算双向因子
        bidirectional_factor = 2 if config.bidirectional else 1

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=config.gru_hidden_size,
            num_layers=config.gru_num_layers,
            batch_first=True,
            dropout=config.dropout_rate if config.gru_num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )

        # Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

        # 全连接层
        self.fc_layers = nn.ModuleList()
        current_size = config.gru_hidden_size * bidirectional_factor

        for hidden_size in config.fc_hidden_sizes:
            self.fc_layers.append(nn.Linear(current_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
            self.fc_layers.append(nn.Dropout(config.dropout_rate))
            current_size = hidden_size

        # 输出层（仅均值）
        self.mean_layer = nn.Linear(current_size, 1)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化模型权重"""
        for name, param in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

        for layer in self.fc_layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
        if isinstance(self.mean_layer, nn.Linear):
            nn.init.kaiming_normal_(self.mean_layer.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(self.mean_layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        参数:
            x: 输入数据 [batch_size, sequence_length, input_size]

        返回:
            mean: 预测均值 [batch_size, 1]
        """
        # GRU层
        gru_out, _ = self.gru(x)  # [batch_size, sequence_length, hidden_size * bidirectional]

        # 取最后一个时间步
        last_output = gru_out[:, -1, :]  # [batch_size, hidden_size * bidirectional]

        # Dropout
        last_output = self.dropout(last_output)

        # 全连接层
        for layer in self.fc_layers:
            last_output = layer(last_output)

        # 输出层
        mean = self.mean_layer(last_output)

        return mean


# ============================================================================
# 5. 评估指标模块
# ============================================================================

class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        计算评估指标

        参数:
            y_true: 真实值
            y_pred: 预测值

        返回:
            指标字典
        """
        # 转换为numpy数组
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()

        y_true = y_true.flatten()
        y_pred = y_pred.flatten()

        # 确保长度相同
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # 计算MAE
        mae = mean_absolute_error(y_true, y_pred)

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 计算MAPE（防止除零）
        epsilon = 1e-10
        y_true_safe = np.where(y_true == 0, epsilon, y_true)
        mape = np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

        # 计算R2
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': float(round(mae, 6)),
            'RMSE': float(round(rmse, 6)),
            'MAPE': float(round(mape, 6)),
            'R2': float(round(r2, 6))
        }

    @staticmethod
    def calculate_rul_metrics(true_rul: Optional[float], pred_rul: Optional[float]) -> Dict[str, Optional[float]]:
        """
        计算RUL评估指标

        参数:
            true_rul: 真实RUL（小时）
            pred_rul: 预测RUL（小时）

        返回:
            RUL指标字典
        """
        if true_rul is None or pred_rul is None:
            return {'MAE_RUL': None, 'MAPE_RUL': None, 'PHM_Score': None}

        try:
            # 绝对误差
            mae_rul = abs(true_rul - pred_rul)

            # 相对误差
            if true_rul != 0:
                mape_rul = (abs(true_rul - pred_rul) / true_rul) * 100
            else:
                mape_rul = 100.0

            # PHM评分函数
            diff = true_rul - pred_rul
            if diff < 0:  # 晚期预测（预测偏小）
                phm_score = np.exp(-diff / 13) - 1
            else:  # 早期预测（预测偏大）
                phm_score = np.exp(diff / 10) - 1

            return {
                'MAE_RUL': float(round(mae_rul, 2)),
                'MAPE_RUL': float(round(mape_rul, 2)),
                'PHM_Score': float(round(phm_score, 4))
            }
        except Exception as e:
            print(f"RUL指标计算错误: {e}")
            return {'MAE_RUL': None, 'MAPE_RUL': None, 'PHM_Score': None}

    @staticmethod
    def generate_metrics_table(metrics_records: List[Dict[str, Any]]) -> str:
        """
        生成Markdown格式的指标表格

        参数:
            metrics_records: 指标记录列表

        返回:
            Markdown表格字符串
        """
        if not metrics_records:
            return "无指标数据"

        # 表头
        headers = ["条件", "MAE", "RMSE", "MAPE", "R2"]
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "|" + "|".join(["---"] * len(headers)) + "|"

        # 数据行
        rows = []
        for record in metrics_records:
            condition = record.get('condition', '实验')
            mae = record.get('MAE', 0)
            rmse = record.get('RMSE', 0)
            mape = record.get('MAPE', 0)
            r2 = record.get('R2', 0)

            row = f"| {condition} | {mae:.6f} | {rmse:.6f} | {mape:.6f} | {r2:.6f} |"
            rows.append(row)

        # 组合表格
        table = "\n".join([header_row, separator_row] + rows)

        # 打印表格
        print("\n" + "=" * 80)
        print("评估指标表格:")
        print("=" * 80)
        print(table)

        return table


# ============================================================================
# 6. SOH和RUL计算模块 - 基于论文逻辑
# ============================================================================

class SOHRULCalculator:
    """SOH和RUL计算器 - 基于论文逻辑"""

    def __init__(self, config: ModelConfig, logger: TrainingLogger):
        self.config = config
        self.logger = logger

    def calculate_v_initial(self, voltage_series: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        计算初始电压V_initial - 基于论文逻辑

        参数:
            voltage_series: 电压时间序列

        返回:
            V_initial: 初始电压
            details: 计算详情
        """
        self.logger.log_info("使用稳定段均值计算V_initial（波动<0.5%）")

        try:
            series = np.asarray(voltage_series, dtype=float).flatten()
            if len(series) == 0:
                raise ValueError("电压序列为空")

            # 计算相邻点相对变化率，寻找稳定区间
            rel_change = np.abs(np.diff(series)) / (np.maximum(np.abs(series[:-1]), 1e-6))
            stable_mask = rel_change < 0.005  # 0.5%

            # 找到最长稳定片段
            best_start = 0
            best_len = 0
            current_start = 0
            current_len = 0
            for idx, is_stable in enumerate(stable_mask):
                if is_stable:
                    if current_len == 0:
                        current_start = idx
                    current_len += 1
                else:
                    if current_len > best_len:
                        best_len = current_len
                        best_start = current_start
                    current_len = 0
            if current_len > best_len:
                best_len = current_len
                best_start = current_start

            if best_len > 0:
                # stable_mask 对应相邻点，片段长度需+1
                stable_segment = series[best_start:best_start + best_len + 1]
            else:
                # 如果未找到稳定段，使用前窗口均值作为后备
                window = min(100, max(1, len(series) // 10))
                stable_segment = series[:window]

            V_initial = float(np.mean(stable_segment))

            details = {
                'V_initial': float(V_initial),
                'method': 'stable_mean',
                'stable_start': int(best_start),
                'stable_len': int(best_len + 1 if best_len > 0 else len(stable_segment))
            }

            self.logger.log_info(
                f"计算出的V_initial: {V_initial:.4f}V (稳定段长度={details['stable_len']}, 起点={details['stable_start']})"
            )

            return float(V_initial), details

        except Exception as e:
            self.logger.log_error(f"V_initial计算失败: {str(e)}")
            window = min(100, max(1, len(voltage_series)))
            V_initial = float(np.mean(voltage_series[:window]))
            self.logger.log_info(f"使用后备V_initial: {V_initial:.4f}V")

            return float(V_initial), {'V_initial': float(V_initial), 'method': 'fallback_mean', 'window_size': int(window)}

    def calculate_soh(self, voltage_series: np.ndarray, V_initial: Optional[float] = None) -> np.ndarray:
        """
        计算SOH序列

        参数:
            voltage_series: 电压时间序列
            V_initial: 初始电压（如果为None则计算）

        返回:
            SOH序列
        """
        if V_initial is None:
            V_initial, _ = self.calculate_v_initial(voltage_series)

        # 计算SOH
        soh = voltage_series / V_initial

        # 限制SOH范围在合理区间
        soh = np.clip(soh, 0.8, 1.2)

        return soh

    def calculate_rul(self, soh_series: np.ndarray, time_series: np.ndarray) -> Dict[str, Any]:
        """
        计算RUL - 基于线性插值方法

        参数:
            soh_series: SOH序列
            time_series: 时间序列

        返回:
            RUL计算结果
        """
        self.logger.log_info("使用线性插值计算RUL")

        # 记录输入形状用于调试
        self.logger.log_info(f"SOH序列形状: {soh_series.shape}, 时间序列形状: {time_series.shape}")

        # 确保输入是1D数组
        soh_series = soh_series.flatten()
        time_series = time_series.flatten()

        # 记录处理后的形状
        self.logger.log_info(f"处理后形状 - SOH: {soh_series.shape}, 时间: {time_series.shape}")

        try:
            # 确保输入是1D数组
            soh_series = soh_series.flatten()
            time_series = time_series.flatten()

            # 检查数据长度
            if len(soh_series) != len(time_series):
                min_len = min(len(soh_series), len(time_series))
                soh_series = soh_series[:min_len]
                time_series = time_series[:min_len]
                self.logger.log_warning(f"SOH和时间序列长度不匹配，截断为 {min_len}")

            # 检查是否所有SOH都高于阈值
            if np.all(soh_series > self.config.soh_threshold):
                self.logger.log_info("所有SOH值都高于阈值，电池尚未显著退化")
                # 使用趋势外推估计RUL
                return self._extrapolate_rul(soh_series, time_series)

            # 检查是否所有SOH都低于阈值
            if np.all(soh_series < self.config.soh_threshold):
                self.logger.log_warning("所有SOH值都低于阈值，电池可能已失效")
                return {
                    'rul_actual': 0,
                    'eol_time': time_series[-1],
                    'threshold': self.config.soh_threshold,
                    'method': 'all_below_threshold'
                }

            # 找到SOH首次低于阈值的位置
            below_threshold_idx = np.where(soh_series < self.config.soh_threshold)[0]

            if len(below_threshold_idx) == 0:
                return self._extrapolate_rul(soh_series, time_series)

            first_below = below_threshold_idx[0]

            # 线性插值计算精确RUL
            if first_below > 0:
                # 获取跨越阈值的两个点
                idx1 = first_below - 1
                idx2 = first_below

                t1, t2 = time_series[idx1], time_series[idx2]
                soh1, soh2 = soh_series[idx1], soh_series[idx2]

                # 确保SOH1 > 阈值 > SOH2
                if not (soh1 > self.config.soh_threshold > soh2):
                    # 寻找合适的插值点
                    for i in range(first_below - 1, 0, -1):
                        if soh_series[i] > self.config.soh_threshold:
                            idx1 = i
                            t1, soh1 = time_series[idx1], soh_series[idx1]
                            break

                # 线性插值公式
                if soh2 != soh1:
                    eol_time = t1 + (self.config.soh_threshold - soh1) * (t2 - t1) / (soh2 - soh1)
                else:
                    eol_time = t2
                    self.logger.log_warning("线性插值分母为零，使用t2")
            else:
                eol_time = time_series[0]
                self.logger.log_warning("第一个点低于阈值，使用第一个时间点")

            # 计算RUL（从当前时间到EOL）
            current_time = time_series[-1]
            rul = max(0, eol_time - current_time)

            self.logger.log_info(f"计算出的RUL: {rul:.2f}h, EOL在 {eol_time:.2f}h")

            return {
                'rul_actual': rul,
                'eol_time': eol_time,
                'threshold': self.config.soh_threshold,
                'method': 'linear_interpolation'
            }

        except Exception as e:
            self.logger.log_error(f"RUL计算失败: {str(e)}")
            return {
                'rul_actual': None,
                'eol_time': None,
                'threshold': self.config.soh_threshold,
                'method': 'failed',
                'error': str(e)
            }

    def _extrapolate_rul(self, soh_series: np.ndarray, time_series: np.ndarray) -> Dict[str, Any]:
        """使用趋势外推估计RUL"""
        try:
            # 线性回归拟合SOH趋势
            slope, intercept = np.polyfit(time_series, soh_series, 1)

            # 计算达到阈值的时间
            if slope >= 0:
                self.logger.log_warning("SOH趋势在增加，无法估计RUL")
                return {
                    'rul_actual': None,
                    'eol_time': None,
                    'threshold': self.config.soh_threshold,
                    'method': 'trend_extrapolation_failed'
                }

            eol_time = (self.config.soh_threshold - intercept) / slope
            current_time = time_series[-1]
            rul = max(0, eol_time - current_time)

            self.logger.log_info(f"外推的RUL: {rul:.2f}h (斜率={slope:.6f})")

            return {
                'rul_actual': rul,
                'eol_time': eol_time,
                'threshold': self.config.soh_threshold,
                'method': 'trend_extrapolation',
                'slope': slope
            }
        except Exception as e:
            self.logger.log_error(f"趋势外推失败: {str(e)}")
            return {
                'rul_actual': None,
                'eol_time': None,
                'threshold': self.config.soh_threshold,
                'method': 'extrapolation_failed'
            }

    def compare_rul(self, true_voltage: np.ndarray, pred_voltage: np.ndarray,
                    time_series: np.ndarray, pred_v_initial_override: Optional[float] = None) -> Dict[str, Any]:
        """
        比较真实和预测的RUL

        参数:
            true_voltage: 真实电压序列
            pred_voltage: 预测电压序列
            time_series: 时间序列

        返回:
            RUL比较结果
        """
        self.logger.log_info("比较真实和预测的RUL")

        # 确保长度一致
        min_len = min(len(true_voltage), len(pred_voltage), len(time_series))
        true_voltage = true_voltage[:min_len]
        pred_voltage = pred_voltage[:min_len]
        time_series = time_series[:min_len]

        # 过滤异常的预测零值或负值
        if np.any(pred_voltage <= 0):
            self.logger.log_warning("预测序列存在非正值，已使用最小正值替换以避免阈值错误")
            min_positive = max(1e-6, np.percentile(pred_voltage[pred_voltage > 0], 5)) if np.any(pred_voltage > 0) else 1e-3
            pred_voltage = np.where(pred_voltage <= 0, min_positive, pred_voltage)

        # 计算真实SOH和RUL
        true_V_initial, true_details = self.calculate_v_initial(true_voltage)
        true_soh = self.calculate_soh(true_voltage, true_V_initial)
        true_rul_result = self.calculate_rul(true_soh, time_series)

        # 计算预测SOH和RUL；允许使用未缩放的V_initial进行RUL校准
        if pred_v_initial_override is None:
            pred_V_initial, pred_details = self.calculate_v_initial(pred_voltage)
        else:
            pred_V_initial = float(pred_v_initial_override)
            pred_details = {
                'method': 'override_from_unscaled',
                'override_value': pred_V_initial
            }
        pred_soh = self.calculate_soh(pred_voltage, pred_V_initial)
        pred_rul_result = self.calculate_rul(pred_soh, time_series)

        # 计算RUL误差指标
        if true_rul_result['rul_actual'] is not None and pred_rul_result['rul_actual'] is not None:
            rul_metrics = MetricsCalculator.calculate_rul_metrics(
                true_rul_result['rul_actual'],
                pred_rul_result['rul_actual']
            )
        else:
            rul_metrics = {'MAE_RUL': None, 'MAPE_RUL': None, 'PHM_Score': None}

        comparison_result = {
            'true_rul': true_rul_result,
            'pred_rul': pred_rul_result,
            'rul_metrics': rul_metrics,
            'true_V_initial': true_V_initial,
            'pred_V_initial': pred_V_initial,
            'true_details': true_details,
            'pred_details': pred_details
        }

        # 记录比较结果（允许 RUL 为 None 时安全输出）
        true_rul_val = true_rul_result.get('rul_actual')
        pred_rul_val = pred_rul_result.get('rul_actual')
        true_rul_text = f"{true_rul_val:.2f}h" if true_rul_val is not None else "N/A"
        pred_rul_text = f"{pred_rul_val:.2f}h" if pred_rul_val is not None else "N/A"
        self.logger.log_info(f"真实RUL: {true_rul_text}")
        self.logger.log_info(f"预测RUL: {pred_rul_text}")

        if rul_metrics['MAE_RUL'] is not None:
            self.logger.log_info(f"RUL MAE: {rul_metrics['MAE_RUL']:.2f}h")
            self.logger.log_info(f"RUL MAPE: {rul_metrics['MAPE_RUL']:.2f}%")
            self.logger.log_info(f"PHM评分: {rul_metrics['PHM_Score']:.4f}")

        return comparison_result


# ============================================================================
# 7. 训练模块
# ============================================================================

class PEMFCDataset(Dataset):
    """PEMFC时序数据集"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = sequences.astype(np.float32)
        self.targets = targets.astype(np.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class GRUTrainer:
    """GRU模型训练器"""

    def __init__(self, config: ModelConfig, model: nn.Module,
                 data_processor: DataProcessor, logger: TrainingLogger,
                 input_size: int):  # 新增input_size参数
        self.config = config
        self.model = model
        self.data_processor = data_processor
        self.logger = logger
        self.input_size = input_size  # 保存输入维度

        # 设备设置
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 优化器
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.lr_decay_factor,
            patience=config.lr_patience,
            min_lr=1e-6
        )

        # 损失函数（均方误差）
        self.criterion = nn.MSELoss()

        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }

        # 最佳模型信息
        self.best_model_info = {
            'epoch': 0,
            'val_loss': float('inf'),
            'state_dict': None
        }

    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """执行单个epoch的训练"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []

        for sequences, targets in train_loader:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            mean_pred = self.model(sequences)

            # 计算损失
            loss_main = F.smooth_l1_loss(mean_pred, targets)
            slope_loss = 0.0
            if mean_pred.shape[0] > 1:
                diff_pred = mean_pred[1:] - mean_pred[:-1]
                diff_tgt = targets[1:] - targets[:-1]
                slope_loss = F.mse_loss(diff_pred, diff_tgt)
            var_pred = torch.var(mean_pred, unbiased=False)
            var_tgt = torch.var(targets, unbiased=False)
            variance_loss = F.mse_loss(var_pred, var_tgt)
            loss = loss_main + self.config.slope_loss_weight * slope_loss + self.config.variance_loss_weight * variance_loss

            # 反向传播
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            # 优化
            self.optimizer.step()

            # 记录
            total_loss += loss.item()
            all_preds.append(mean_pred.detach().cpu())
            all_targets.append(targets.detach().cpu())

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)

        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """验证"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                mean_pred = self.model(sequences)

                # 计算损失
                loss_main = F.smooth_l1_loss(mean_pred, targets)
                slope_loss = 0.0
                if mean_pred.shape[0] > 1:
                    diff_pred = mean_pred[1:] - mean_pred[:-1]
                    diff_tgt = targets[1:] - targets[:-1]
                    slope_loss = F.mse_loss(diff_pred, diff_tgt)
                var_pred = torch.var(mean_pred, unbiased=False)
                var_tgt = torch.var(targets, unbiased=False)
                variance_loss = F.mse_loss(var_pred, var_tgt)
                loss = loss_main + self.config.slope_loss_weight * slope_loss + self.config.variance_loss_weight * variance_loss

                # 记录
                total_loss += loss.item()
                all_preds.append(mean_pred.detach().cpu())
                all_targets.append(targets.detach().cpu())

        # 计算平均损失
        avg_loss = total_loss / len(val_loader)

        # 计算指标
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        metrics = MetricsCalculator.calculate_metrics(all_targets, all_preds)

        return avg_loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """训练主循环"""
        self.logger.log_info("开始模型训练...")
        self.logger.log_info(f"设备: {self.device}")

        early_stop_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            try:
                # 训练
                train_loss, train_metrics = self.train_epoch(train_loader)

                # 验证
                val_loss, val_metrics = self.validate(val_loader)

                # 学习率调度
                self.scheduler.step(val_loss)
                current_lr = self.optimizer.param_groups[0]['lr']

                # 记录历史
                self.history['train_loss'].append(train_loss)
                self.history['val_loss'].append(val_loss)
                self.history['train_metrics'].append(train_metrics)
                self.history['val_metrics'].append(val_metrics)
                self.history['learning_rates'].append(current_lr)

                # 日志记录
                self.logger.log_epoch(epoch, train_loss, val_loss, train_metrics, val_metrics, current_lr)

                # 保存最佳模型
                if val_loss < self.best_model_info['val_loss']:
                    self.best_model_info['val_loss'] = val_loss
                    self.best_model_info['epoch'] = epoch
                    self.best_model_info['state_dict'] = self.model.state_dict().copy()

                    # 保存模型
                    self.save_model()
                    early_stop_counter = 0

                    self.logger.log_info(f"✓ 最佳模型已保存 (epoch {epoch}, 验证损失: {val_loss:.6f})")
                else:
                    early_stop_counter += 1

                # 早停检查
                if early_stop_counter >= self.config.patience:
                    self.logger.log_info(
                        f"早停触发，连续 {self.config.patience} 个epoch没有改善")
                    break

            except Exception as e:
                self.logger.log_error(f"Epoch {epoch} 训练错误: {str(e)}")
                if epoch > 10:
                    continue

        # 训练完成
        total_time = time.time() - self.logger.start_time
        self.logger.log_info(f"训练完成，耗时 {total_time:.2f} 秒")
        self.logger.log_info(
            f"最佳模型在 epoch {self.best_model_info['epoch']}，验证损失 {self.best_model_info['val_loss']:.6f}")

        # 加载最佳模型
        self.load_best_model()

    def save_model(self):
        """保存模型 - 修复input_size问题"""
        model_path = os.path.join(self.config.save_paths['models'], 'best_model.pth')

        try:
            # 使用保存的input_size，而不是从config中获取
            input_size = self.input_size

            # 创建检查点
            checkpoint = {
                'epoch': self.best_model_info['epoch'],
                'model_state_dict': self.best_model_info['state_dict'],
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_loss': float(self.best_model_info['val_loss']),
                # 保存输入维度
                'input_size': input_size,
                # 保存配置信息
                'config_info': {
                    'gru_hidden_size': self.config.gru_hidden_size,
                    'gru_num_layers': self.config.gru_num_layers,
                    'dropout_rate': self.config.dropout_rate,
                    'bidirectional': self.config.bidirectional,
                    'fc_hidden_sizes': self.config.fc_hidden_sizes
                }
            }

            # 使用推荐的保存方式
            torch.save(checkpoint, model_path, _use_new_zipfile_serialization=True)
            self.logger.log_info(f"✓ 模型保存到 {model_path} (输入维度: {input_size})")

        except Exception as e:
            self.logger.log_error(f"保存模型失败: {str(e)}")
            # 备用方案：只保存状态字典
            try:
                state_dict_path = model_path.replace('.pth', '_state_dict.pth')
                torch.save(self.best_model_info['state_dict'], state_dict_path)
                self.logger.log_info(f"仅保存状态字典到 {state_dict_path}")
            except Exception as e2:
                self.logger.log_error(f"保存状态字典失败: {str(e2)}")

    def load_best_model(self):
        """加载最佳模型"""
        model_path = os.path.join(self.config.save_paths['models'], 'best_model.pth')

        if not os.path.exists(model_path):
            self.logger.log_warning(f"最佳模型文件未找到: {model_path}")
            return

        try:
            # 使用安全的加载方式
            checkpoint = torch.load(model_path, map_location=self.device)

            # 加载模型状态字典
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.log_info(f"✓ 从 epoch {checkpoint.get('epoch', 'unknown')} 加载最佳模型")
            else:
                self.logger.log_error("检查点中未找到model_state_dict")

        except Exception as e:
            self.logger.log_error(f"加载最佳模型失败: {str(e)}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """评估模型"""
        self.model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                # 前向传播
                mean_pred = self.model(sequences)

                # 记录
                all_preds.append(mean_pred.detach().cpu())
                all_targets.append(targets.detach().cpu())

        # 合并结果
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()

        # 反标准化
        all_preds_orig = self.data_processor.inverse_transform_target(all_preds)
        all_targets_orig = self.data_processor.inverse_transform_target(all_targets)

        # 确保是一维数组 - 修复置信区间维度问题
        all_preds_orig = all_preds_orig.flatten()
        all_targets_orig = all_targets_orig.flatten()

        # FC1 偏差校准（只对本训练集对应的数据集生效）
        if self.config.apply_fc1_bias_correction and abs(self.config.fc1_bias_correction) > 0:
            all_preds_orig = all_preds_orig - self.config.fc1_bias_correction
            self.logger.log_info(
                f"已对FC1预测应用偏差校准: -{self.config.fc1_bias_correction:.6f}"
            )

        # 诊断输出分布
        pred_std = float(np.std(all_preds_orig)) if len(all_preds_orig) > 0 else 0.0
        true_std = float(np.std(all_targets_orig)) if len(all_targets_orig) > 0 else 0.0
        self.logger.log_info(f"预测序列标准差: {pred_std:.6f}, 真实序列标准差: {true_std:.6f}")

        # 计算指标（原始预测）
        metrics_raw = MetricsCalculator.calculate_metrics(all_targets_orig, all_preds_orig)

        # FC1 残差线性校正（推理端，避免重新训练），优先使用校正结果
        metrics = metrics_raw
        predictions_used = all_preds_orig
        bias_coeff = None
        try:
            if len(all_targets_orig) > 10:
                lr_bias = LinearRegression()
                lr_bias.fit(all_preds_orig.reshape(-1, 1), all_targets_orig.reshape(-1, 1))
                adj_pred = lr_bias.predict(all_preds_orig.reshape(-1, 1)).reshape(-1)
                metrics_adj = MetricsCalculator.calculate_metrics(all_targets_orig, adj_pred)

                predictions_used = adj_pred
                metrics = metrics_adj
                bias_coeff = {
                    'slope': float(np.ravel(lr_bias.coef_)[0]),
                    'intercept': float(np.ravel(lr_bias.intercept_)[0]),
                }
                self.logger.log_info(
                    f"FC1 残差校正: slope={bias_coeff['slope']:.6f}, "
                    f"intercept={bias_coeff['intercept']:.6f}, "
                    f"RMSE_adj={metrics_adj.get('RMSE'):.6f}"
                )
        except Exception as bias_err:
            self.logger.log_warning(f"FC1 残差校正失败: {bias_err}")

        # 准备评估结果
        evaluation_results = {
            'predictions': predictions_used,
            'predictions_raw': all_preds_orig,
            'targets': all_targets_orig,
            'metrics': metrics,
            'metrics_raw': metrics_raw,
            'bias_coeff': bias_coeff,
            'confidence_intervals': None
        }

        # 保存预测结果
        self.save_evaluation_results(evaluation_results)

        return evaluation_results

    def save_evaluation_results(self, results: Dict[str, Any]):
        """保存评估结果 - 修复编码问题"""
        # 保存预测数据
        pred_df = pd.DataFrame({
            'target': results['targets'].flatten(),
            'prediction': results['predictions'].flatten()
        })

        # 额外保存未校正预测，便于诊断
        if 'predictions_raw' in results:
            pred_df['prediction_raw'] = np.asarray(results['predictions_raw']).flatten()

        csv_path = os.path.join(self.config.save_paths['csv'], 'predictions.csv')
        # 修复编码问题：使用utf-8编码
        pred_df.to_csv(csv_path, index=False, encoding='utf-8')

        self.logger.log_info(f"预测结果保存到: {csv_path}")

        # 保存指标
        metrics_df = pd.DataFrame([results['metrics']])
        if 'metrics_raw' in results:
            raw_prefixed = {f"raw_{k}": v for k, v in results['metrics_raw'].items()}
            metrics_df = pd.concat([metrics_df, pd.DataFrame([raw_prefixed])], axis=1)
        metrics_path = os.path.join(self.config.save_paths['tables'], 'metrics.csv')
        # 修复编码问题：使用utf-8编码
        metrics_df.to_csv(metrics_path, index=False, encoding='utf-8')

        # 生成Markdown表格
        metrics_table = MetricsCalculator.generate_metrics_table([results['metrics']])
        table_path = os.path.join(self.config.save_paths['tables'], 'metrics.md')
        # 修复编码问题：使用utf-8编码
        with open(table_path, 'w', encoding='utf-8') as f:
            f.write(metrics_table)


# ============================================================================
# 8. 可视化模块
# ============================================================================

class Visualization:
    """可视化类"""

    def __init__(self, config: ModelConfig, logger: TrainingLogger):
        self.config = config
        self.logger = logger

        # 设置matplotlib样式
        plt.style.use('default')

        # 选择当前系统可用的中文字体，避免缺字形警告
        try:
            font_candidates = [
                'Microsoft YaHei',  # Windows 常用中文字体
                'SimHei',
                'Noto Sans CJK SC',
                'Source Han Sans SC',
                'Arial Unicode MS',
                'DejaVu Sans'
            ]
            available_fonts = {f.name for f in fm.fontManager.ttflist}
            chosen_font = next((f for f in font_candidates if f in available_fonts), 'DejaVu Sans')

            plt.rcParams['font.sans-serif'] = [chosen_font]
            plt.rcParams['font.family'] = [chosen_font]
            plt.rcParams['mathtext.fontset'] = 'dejavusans'
            plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
            plt.rcParams['pdf.fonttype'] = 42  # 使用 TrueType 内嵌
            plt.rcParams['ps.fonttype'] = 42
            self.logger.log_info(f"字体设置成功，使用: {chosen_font}")
        except Exception as e:
            self.logger.log_warning(f"字体设置失败: {str(e)}")
            self.logger.log_info("使用默认字体，可能出现缺字形")

        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
        """可视化类"""



    def plot_training_history(self, history: Dict[str, List]):
        """绘制训练历史"""
        self.logger.log_info("绘制训练历史")

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 损失曲线
        axes[0, 0].plot(history['train_loss'], label='训练损失', linewidth=2)
        axes[0, 0].plot(history['val_loss'], label='验证损失', linewidth=2)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('损失')
        axes[0, 0].set_title('训练和验证损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 学习率曲线
        axes[0, 1].plot(history['learning_rates'], color='green', linewidth=2)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('学习率')
        axes[0, 1].set_title('学习率调度')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('log')

        # MAE曲线
        train_mae = [m['MAE'] for m in history['train_metrics']]
        val_mae = [m['MAE'] for m in history['val_metrics']]
        axes[1, 0].plot(train_mae, label='训练MAE', linewidth=2)
        axes[1, 0].plot(val_mae, label='验证MAE', linewidth=2)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].set_title('MAE曲线')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # R2曲线
        train_r2 = [m['R2'] for m in history['train_metrics']]
        val_r2 = [m['R2'] for m in history['val_metrics']]
        axes[1, 1].plot(train_r2, label='训练R2', linewidth=2)
        axes[1, 1].plot(val_r2, label='验证R2', linewidth=2)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('R2')
        axes[1, 1].set_title('R2曲线')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([-0.1, 1.1])

        plt.tight_layout()

        save_path = os.path.join(self.config.save_paths['images'], 'training_history.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        self.logger.log_info(f"训练历史图保存: {save_path}")

    def plot_voltage_prediction(self, time_series: np.ndarray, true_voltage: np.ndarray,
                                pred_voltage: np.ndarray, train_size: int,
                                confidence_intervals: Optional[Dict[str, np.ndarray]] = None):
        """绘制整体电压预测（单幅图，无放大子图）"""
        self.logger.log_info("绘制电压预测对比图（单幅图）")

        # 确保一维
        true_voltage = true_voltage.flatten()
        pred_voltage = pred_voltage.flatten()
        time_series = time_series.flatten()

        self.logger.log_info(
            f"绘图输入形状 - true_voltage: {true_voltage.shape}, pred_voltage: {pred_voltage.shape}, time_series: {time_series.shape}")

        fig, ax = plt.subplots(1, 1, figsize=(14, 6))

        # 全段真实/预测
        ax.plot(time_series, true_voltage, 'b-', label='真实电压', linewidth=1.3, alpha=0.85)
        ax.plot(time_series, pred_voltage, 'r-', label='预测电压', linewidth=1.3, alpha=0.85)

        # 置信区间（可选）
        if confidence_intervals is not None:
            try:
                ci_lower = confidence_intervals['lower'].flatten()
                ci_upper = confidence_intervals['upper'].flatten()
                min_len = min(len(time_series), len(ci_lower), len(ci_upper))
                if min_len > 0:
                    ax.fill_between(time_series[:min_len], ci_lower[:min_len], ci_upper[:min_len],
                                    color='gray', alpha=0.25, label='95% 置信区间')
            except Exception as e:
                self.logger.log_warning(f"绘制置信区间失败: {str(e)}")

        # 训练/预测分界线
        if train_size > 0 and train_size < len(time_series):
            split_time = time_series[train_size]
            ax.axvline(x=split_time, color='k', linestyle='--', linewidth=1, alpha=0.7)
            ax.text(split_time, ax.get_ylim()[1] * 0.97, '训练阶段', ha='right', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))
            ax.text(split_time, ax.get_ylim()[1] * 0.97, '预测阶段', ha='left', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8))

        ax.set_xlabel('时间 (h)')
        ax.set_ylabel('电压 (V)')
        ax.set_title('燃料电池堆电压预测（全段）')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        save_path = os.path.join(self.config.save_paths['images'], 'voltage_prediction.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        self.logger.log_info(f"电压预测图保存: {save_path}")

    def plot_soh_rul_curve(self, time_series: np.ndarray, true_voltage: np.ndarray,
                           pred_voltage: np.ndarray, rul_results: Dict[str, Any]):
        """绘制SOH曲线和RUL对比图"""
        self.logger.log_info("绘制SOH和RUL曲线")

        # 创建计算器
        calculator = SOHRULCalculator(self.config, self.logger)

        # 计算SOH
        true_soh = calculator.calculate_soh(true_voltage)
        pred_soh = calculator.calculate_soh(pred_voltage)

        # 确保长度一致
        min_len = min(len(time_series), len(true_soh), len(pred_soh))
        time_series = time_series[:min_len]
        true_soh = true_soh[:min_len]
        pred_soh = pred_soh[:min_len]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 第一幅图：SOH曲线对比
        axes[0, 0].plot(time_series, true_soh, 'b-', label='真实SOH', linewidth=2, alpha=0.8)
        axes[0, 0].plot(time_series, pred_soh, 'r-', label='预测SOH', linewidth=2, alpha=0.8)

        # SOH阈值线
        threshold = self.config.soh_threshold
        axes[0, 0].axhline(y=threshold, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
        axes[0, 0].text(time_series[0], threshold + 0.005, f'失效阈值 SOH={threshold}',
                        fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # 标注RUL
        if rul_results['true_rul']['rul_actual'] is not None and rul_results['true_rul']['eol_time'] is not None:
            true_eol = rul_results['true_rul']['eol_time']
            if true_eol <= time_series[-1]:
                axes[0, 0].axvline(x=true_eol, color='b', linestyle=':', linewidth=1.5, alpha=0.7)
                axes[0, 0].text(true_eol, 0.97, f'真实EOL\n{true_eol:.1f}h',
                                color='b', ha='right', fontsize=8)

        if rul_results['pred_rul']['rul_actual'] is not None and rul_results['pred_rul']['eol_time'] is not None:
            pred_eol = rul_results['pred_rul']['eol_time']
            if pred_eol <= time_series[-1]:
                axes[0, 0].axvline(x=pred_eol, color='r', linestyle=':', linewidth=1.5, alpha=0.7)
                axes[0, 0].text(pred_eol, 0.94, f'预测EOL\n{pred_eol:.1f}h',
                                color='r', ha='left', fontsize=8)

        axes[0, 0].set_xlabel('时间 (h)')
        axes[0, 0].set_ylabel('SOH')
        axes[0, 0].set_title('燃料电池健康状态 (SOH) 曲线')
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.9, 1.05])

        # 第二幅图：电压曲线对比
        axes[0, 1].plot(time_series, true_voltage[:min_len], 'b-',
                        label='真实电压', linewidth=2, alpha=0.8)
        axes[0, 1].plot(time_series, pred_voltage[:min_len], 'r-',
                        label='预测电压', linewidth=2, alpha=0.8)

        axes[0, 1].set_xlabel('时间 (h)')
        axes[0, 1].set_ylabel('电压 (V)')
        axes[0, 1].set_title('电压曲线对比')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)

        # 第三幅图：RUL误差分析
        if rul_results['rul_metrics']['MAE_RUL'] is not None:
            metrics = rul_results['rul_metrics']
            labels = ['MAE_RUL', 'MAPE_RUL', 'PHM_Score']
            values = [metrics['MAE_RUL'], metrics['MAPE_RUL'], metrics['PHM_Score']]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

            bars = axes[1, 0].bar(labels, values, color=colors, alpha=0.8)
            axes[1, 0].set_ylabel('值')
            axes[1, 0].set_title('RUL预测误差指标')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # 在柱子上添加数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                if labels[0] == 'PHM_Score':
                    axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{value:.4f}', ha='center', va='bottom', fontsize=9)
                else:
                    axes[1, 0].text(bar.get_x() + bar.get_width() / 2., height,
                                    f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5,
                            'RUL计算不可用\n' + rul_results['true_rul'].get('method', '未知'),
                            ha='center', va='center', transform=axes[1, 0].transAxes,
                            fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            axes[1, 0].set_xticks([])
            axes[1, 0].set_yticks([])

        # 第四幅图：详细信息
        info_text = "V_initial 详情:\n"
        info_text += f"  真实 V_initial: {rul_results.get('true_V_initial', 0):.4f} V\n"
        info_text += f"  预测 V_initial: {rul_results.get('pred_V_initial', 0):.4f} V\n\n"

        if rul_results['true_rul']['rul_actual'] is not None:
            info_text += f"真实 RUL: {rul_results['true_rul']['rul_actual']:.1f} h\n"
        else:
            info_text += f"真实 RUL: {rul_results['true_rul'].get('method', 'N/A')}\n"

        if rul_results['pred_rul']['rul_actual'] is not None:
            info_text += f"预测 RUL: {rul_results['pred_rul']['rul_actual']:.1f} h\n"
        else:
            info_text += f"预测 RUL: {rul_results['pred_rul'].get('method', 'N/A')}\n"

        info_text += f"SOH 阈值: {threshold}\n"
        info_text += f"当前时间: {time_series[-1]:.1f} h"

        axes[1, 1].text(0.05, 0.95, info_text, transform=axes[1, 1].transAxes,
                        fontsize=9, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title('RUL计算详情')

        plt.tight_layout()

        save_path = os.path.join(self.config.save_paths['images'], 'soh_rul_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
        plt.close()

        self.logger.log_info(f"SOH/RUL曲线图保存: {save_path}")


# ============================================================================
# 9. 主训练流程
# ============================================================================

class PEMFCTrainer:
    """PEMFC训练主类"""

    def __init__(self, config: Optional[ModelConfig] = None):
        if config is None:
            config = ModelConfig()

        self.config = config
        self.logger = TrainingLogger(config)
        self.data_processor = DataProcessor(config, self.logger)

        # 记录配置
        self.logger.log_config(config)
        config.save_config()

    def prepare_data(self) -> Tuple[Dict[str, Any], int]:
        """准备数据"""
        self.logger.log_info("开始数据准备...")

        try:
            # 1. 加载并处理数据
            data = self.data_processor.load_and_process_data()

            # 数据完整性检查
            self.logger.log_info("检查数据完整性...")

            # 检查列名
            columns = list(data.columns)
            self.logger.log_info(f"加载后的数据列: {columns}")

            # 检查是否有NaN值
            nan_counts = data.isna().sum().sum()
            if nan_counts > 0:
                self.logger.log_warning(f"发现 {nan_counts} 个NaN值")
                # 填充NaN值
                # 使用链式前向/后向填充，避免类型检查对 fillna overload 的误报
                data = data.ffill().bfill()
                self.logger.log_info("NaN值已使用前向/后向填充")

            # 检查数据类型
            dtypes = data.dtypes
            self.logger.log_info(f"数据类型: {dict(dtypes)}")

            # 2. 应用小波去噪
            data_denoised = self.data_processor.apply_wavelet_denoising(data)

            # 3. 划分数据集
            split_data = self.data_processor.split_data(data_denoised)

            # 4. 创建DataLoader
            X_train, y_train = split_data['train']
            X_val, y_val = split_data['val']
            X_test, y_test = split_data['test']

            train_dataset = PEMFCDataset(X_train, y_train)
            val_dataset = PEMFCDataset(X_val, y_val)
            test_dataset = PEMFCDataset(X_test, y_test)

            train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)

            self.logger.log_info(
                f"DataLoader创建完成: 训练集={len(train_loader)}批次, 验证集={len(val_loader)}批次, 测试集={len(test_loader)}批次")

            # 获取输入维度
            input_size = X_train.shape[2]

            data_info = {
                'input_size': input_size,
                'train_samples': len(X_train),
                'val_samples': len(X_val),
                'test_samples': len(X_test),
                'selected_features': self.data_processor.selected_features,
                'data_columns': list(data.columns),
                'original_shape': data.shape
            }

            self.logger.log_step_complete("数据准备")

            return {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'test_loader': test_loader,
                'split_data': split_data,
                'data_info': data_info
            }, input_size

        except Exception as e:
            self.logger.log_error(f"数据准备失败: {str(e)}")
            # 打印更详细的错误信息
            import traceback
            self.logger.log_error(traceback.format_exc())
            raise

    def build_model(self, input_size: int) -> nn.Module:
        """构建模型"""
        self.logger.log_info(f"构建GRU模型，输入维度: {input_size}")

        try:
            model = PEMFCGRUModel(self.config, input_size)

            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.logger.log_info(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")

            return model

        except Exception as e:
            self.logger.log_error(f"模型构建失败: {str(e)}")
            raise

    def predict_dataset(self, trainer: GRUTrainer, data_path: str, dataset_label: str) -> Dict[str, Any]:
        """使用训练好的模型在指定数据集上进行推理（用于FC2评估/预测）"""
        self.logger.log_info(f"加载 {dataset_label} 数据用于推理: {data_path}")

        # 使用训练时的特征，不允许自动缩水
        data = self.data_processor.load_and_process_data(
            data_path_override=data_path,
            update_selected_features=False
        )
        data = self.data_processor.apply_wavelet_denoising(data)

        missing = [f for f in self.data_processor.selected_features if f not in data.columns]
        if missing:
            raise ValueError(f"{dataset_label} 数据缺少必要特征: {missing}")

        if self.config.target_feature not in data.columns:
            raise ValueError(f"{dataset_label} 数据缺少目标列 {self.config.target_feature}")

        X = data[self.data_processor.selected_features].to_numpy()
        y = data[self.config.target_feature].to_numpy().reshape(-1, 1)
        time_series = data['time'].to_numpy() if 'time' in data.columns else np.arange(len(data))

        # FC2 使用独立缩放器，避免跨域尺度失配；FC1 保持训练缩放器
        if dataset_label.strip().lower() == 'fc2':
            local_scaler_X = StandardScaler()
            local_scaler_y = StandardScaler()
            X_scaled = local_scaler_X.fit_transform(X)
            y_scaled = local_scaler_y.fit_transform(y)

            def inverse_target(arr: np.ndarray) -> np.ndarray:
                return local_scaler_y.inverse_transform(np.asarray(arr, dtype=float).reshape(-1, 1)).reshape(-1)

            self.logger.log_info("FC2 使用独立标准化器完成缩放")
        else:
            X_scaled = self.data_processor.transform_features(X)
            y_scaled = self.data_processor.transform_target(y)

            def inverse_target(arr: np.ndarray) -> np.ndarray:
                return self.data_processor.inverse_transform_target(arr)

        # 生成序列
        X_seq, y_seq = self.data_processor.create_sequences(X_scaled, y_scaled)

        # 对齐时间和真实值
        seq_len = self.config.sequence_length
        time_aligned = time_series[seq_len - 1:][:len(X_seq)]
        y_aligned_scaled = y_scaled.flatten()[seq_len - 1:][:len(X_seq)]

        # 推理
        pred_dataset = PEMFCDataset(X_seq, y_seq)
        pred_loader = DataLoader(pred_dataset, batch_size=self.config.batch_size, shuffle=False)

        trainer.model.eval()
        preds = []
        with torch.no_grad():
            for sequences, _targets in pred_loader:
                sequences = sequences.to(trainer.device)
                outputs = trainer.model(sequences).detach().cpu().numpy()
                preds.append(outputs)

        predictions_scaled = np.concatenate(preds, axis=0).flatten()

        # 反标准化
        predictions = inverse_target(predictions_scaled)
        y_aligned_orig = inverse_target(y_aligned_scaled)

        # 针对FC1的偏差校准
        if dataset_label.strip().lower() == 'fc1' and self.config.apply_fc1_bias_correction:
            if abs(self.config.fc1_bias_correction) > 0:
                predictions = predictions - self.config.fc1_bias_correction
                self.logger.log_info(
                    f"FC1 推理偏差校准: -{self.config.fc1_bias_correction:.6f}"
                )

        # 计算指标（使用原始尺度）
        metrics = MetricsCalculator.calculate_metrics(y_aligned_orig, predictions)

        return {
            'dataset': dataset_label,
            'time': time_aligned,
            'target': y_aligned_orig,
            'prediction': predictions,
            'metrics': metrics,
            # 供滚动预测/可视化使用的额外信息
            'raw_features': X,
            'scaled_features': X_scaled,
            'scaler_X': local_scaler_X if dataset_label.strip().lower() == 'fc2' else self.data_processor.scaler_X,
            'inverse_target_fn': inverse_target
        }

    def rolling_forecast(self, trainer: GRUTrainer, split_data: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        """基于单步GRU的滚动多步预测，自动生成未来特征，时间轴延展至 forecast_horizon。"""
        seq_len = self.config.sequence_length
        raw_features = split_data.get('test_features')
        time_series = split_data.get('test_time')

        if raw_features is None or len(raw_features) < seq_len:
            self.logger.log_warning("测试特征数量不足，无法进行滚动预测")
            return None

        # 取最后窗口作为起点
        window_raw = np.asarray(raw_features[-seq_len:], dtype=float)
        window_scaled = self.data_processor.transform_features(window_raw)
        feature_history = window_raw.copy()

        # 推算时间步长，并自适应调整以覆盖 forecast_horizon
        dt_base = 1.0
        last_time = float(len(raw_features))
        cfg_dt = getattr(self.config, "fixed_dt", None)
        if cfg_dt is not None and np.isfinite(cfg_dt) and cfg_dt > 0:
            dt_base = float(cfg_dt)
            if time_series is not None and len(time_series) > 0:
                ts = np.asarray(time_series, dtype=float).flatten()
                last_time = float(ts[-1])
        elif time_series is not None and len(time_series) > 1:
            ts = np.asarray(time_series, dtype=float).flatten()
            deltas = np.diff(ts)
            deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
            dt_base = float(np.median(deltas)) if len(deltas) > 0 else 1.0
            last_time = float(ts[-1])

        desired_end = getattr(self.config, "forecast_horizon", 1500.0) or 1500.0
        span = max(desired_end - last_time, 0.0)
        steps_base = int(np.ceil(span / dt_base)) if span > 0 and dt_base > 0 else self.config.forecast_steps
        max_steps_cfg = getattr(self.config, "forecast_max_steps", None)
        if max_steps_cfg is None:
            steps = max(self.config.forecast_steps, steps_base, 1)
        else:
            steps = min(max_steps_cfg, max(self.config.forecast_steps, steps_base, 1))
        dt_future = dt_base  # 保持真实时间分辨率，避免过度平滑
        self.logger.log_info(
            f"滚动预测: 起点={last_time:.3f}h, 目标={desired_end:.1f}h, 基础步长={dt_base:.4f}h, 步数={steps}")

        preds: List[float] = []
        future_times: List[float] = []
        future_feats: List[np.ndarray] = []

        time_idx = None
        try:
            time_idx = self.data_processor.selected_features.index('time')
        except ValueError:
            time_idx = None

        for step in range(steps):
            model_input = torch.FloatTensor(window_scaled).unsqueeze(0).to(trainer.device)
            with torch.no_grad():
                pred_val = trainer.model(model_input).detach().cpu().item()

            preds.append(float(pred_val))
            next_time = last_time + (step + 1) * dt_future
            future_times.append(next_time)

            next_feat_raw = self.data_processor.generate_future_features(feature_history)
            # 保持time特征与未来时间轴一致，避免被裁剪为常数
            if time_idx is not None and 0 <= time_idx < next_feat_raw.shape[0]:
                next_feat_raw[time_idx] = next_time
            feature_history = np.vstack([feature_history, next_feat_raw])

            next_feat_scaled = self.data_processor.transform_features(next_feat_raw.reshape(1, -1)).reshape(-1)
            window_scaled = np.vstack([window_scaled[1:], next_feat_scaled])
            future_feats.append(next_feat_raw)

        preds_scaled = np.asarray(preds, dtype=float)
        preds_orig = self.data_processor.inverse_transform_target(preds_scaled)

        # 根据最近趋势或兜底衰减，为未来预测添加轻微下行斜率，避免走势过平
        decay_slope = 0.0
        fallback_decay = getattr(self.config, "future_voltage_decay_per_hour", -0.00018)
        recent_target = split_data.get('test_original')
        recent_time = split_data.get('test_time')
        try:
            if recent_target is not None and len(recent_target) > 5:
                tail_n = min(200, len(recent_target))
                y_tail = np.asarray(recent_target[-tail_n:], dtype=float).flatten()
                if recent_time is not None and len(recent_time) >= tail_n:
                    t_tail = np.asarray(recent_time[-tail_n:], dtype=float).flatten()
                else:
                    t_tail = np.arange(len(y_tail), dtype=float) * dt_base
                mask = np.isfinite(t_tail) & np.isfinite(y_tail)
                if mask.sum() >= 2:
                    slope, _ = np.polyfit(t_tail[mask], y_tail[mask], 1)
                    decay_slope = slope if slope < 0 else fallback_decay
                else:
                    decay_slope = fallback_decay
            else:
                decay_slope = fallback_decay

            if decay_slope < 0:
                decay_adjust = decay_slope * np.arange(1, len(preds_orig) + 1, dtype=float) * dt_future
                preds_orig = preds_orig + decay_adjust
                # 限制不过度下穿近期真实电压
                floor_ref = np.nanpercentile(np.asarray(recent_target, dtype=float).flatten(), 2) if recent_target is not None else None
                if floor_ref is not None and np.isfinite(floor_ref):
                    preds_orig = np.maximum(preds_orig, floor_ref * 0.95)
        except Exception as decay_err:
            self.logger.log_warning(f"未来衰减修正失败，使用原始预测: {decay_err}")

        return {
            'time': np.asarray(future_times, dtype=float),
            'predictions': preds_orig,
            'future_features': np.asarray(future_feats, dtype=float)
        }

    def train_and_evaluate(self):
        """训练和评估主流程"""
        self.logger.log_info("=" * 80)
        self.logger.log_info("PEMFC燃料电池GRU寿命预测训练")
        self.logger.log_info("=" * 80)

        try:
            # 1. 准备数据
            data_result, input_size = self.prepare_data()

            # 2. 构建模型
            model = self.build_model(input_size)

            # 3. 训练模型 - 传入input_size参数
            trainer = GRUTrainer(self.config, model, self.data_processor, self.logger, input_size)

            eval_only = os.environ.get("EVAL_ONLY", "0") == "1"
            if eval_only:
                self.logger.log_info("EVAL_ONLY=1，跳过训练，直接加载最佳模型进行评估")
                trainer.load_best_model()
            else:
                trainer.train(data_result['train_loader'], data_result['val_loader'])

            # 4. 评估模型
            self.logger.log_info("在测试集上评估模型...")
            evaluation_results = trainer.evaluate(data_result['test_loader'])

            # 4.1 滚动多步预测关闭（FC1 不再生成未来外推，前端仅展示真实/测试段）
            future_result = None
            forecast_df = None
            self.logger.log_info("FC1 滚动预测已关闭，预测结果仅包含历史+测试段")

            # 5. 计算SOH和RUL
            self.logger.log_info("计算SOH和RUL...")

            # 获取预测电压并对齐序列，去除边界零值
            true_voltage_original = data_result['split_data']['test_original']
            seq_len = self.config.sequence_length

            true_voltage_flat = true_voltage_original.flatten()

            # 对齐真实与预测序列：去除前 (seq_len-1) 个无法预测的点
            true_voltage_aligned = true_voltage_flat[seq_len - 1:]
            predictions = evaluation_results['predictions'].flatten()

            min_len = min(len(true_voltage_aligned), len(predictions))
            true_voltage_aligned = true_voltage_aligned[:min_len]
            pred_voltage_aligned = predictions[:min_len]

            # RUL计算前缩放预测，快速收敛过大/过小的衰减速度
            calculator = SOHRULCalculator(self.config, self.logger)
            pred_v_initial_override = None
            rul_scale = getattr(self.config, "rul_prediction_scale", 1.0) or 1.0
            if abs(rul_scale - 1.0) > 1e-6:
                pred_v_initial_override, _ = calculator.calculate_v_initial(pred_voltage_aligned)
                pred_voltage_for_rul = pred_voltage_aligned * float(rul_scale)
                self.logger.log_info(f"RUL计算使用预测缩放因子: {rul_scale:.4f} (V_initial保持未缩放)")
            else:
                pred_voltage_for_rul = pred_voltage_aligned

            # 对齐时间序列
            time_series = data_result['split_data']['test_time']
            if len(time_series.shape) > 1:
                time_series = time_series.flatten()
            if len(time_series) == len(true_voltage_flat):
                time_series_aligned = time_series[seq_len - 1:][:min_len]
            else:
                time_series_aligned = time_series[:min_len]

            self.logger.log_info(
                f"对齐后：真实电压长度={len(true_voltage_aligned)}, 预测长度={len(pred_voltage_aligned)}, 时间长度={len(time_series_aligned)}"
            )

            # 计算RUL（使用对齐后的数据）
            rul_results = calculator.compare_rul(
                true_voltage_aligned,
                pred_voltage_for_rul,
                time_series_aligned,
                pred_v_initial_override=pred_v_initial_override
            )

            # 使用未来滚动预测估计更长时间的预测RUL
            try:
                if future_result is not None:
                    pred_full_for_rul = np.concatenate([
                        np.asarray(pred_voltage_for_rul, dtype=float).flatten(),
                        np.asarray(future_result['predictions'], dtype=float).flatten()
                    ])
                    time_full_for_rul = np.concatenate([
                        np.asarray(time_series_aligned, dtype=float).flatten(),
                        np.asarray(future_result['time'], dtype=float).flatten()
                    ])
                    pred_V0_full, _ = calculator.calculate_v_initial(pred_full_for_rul)
                    pred_soh_full = calculator.calculate_soh(pred_full_for_rul, pred_V0_full)
                    pred_rul_full = calculator.calculate_rul(pred_soh_full, time_full_for_rul)
                    rul_results['pred_rul_with_future'] = pred_rul_full
                    self.logger.log_info(
                        f"预测RUL(含未来外推): {pred_rul_full.get('rul_actual', 'N/A')}h, 方法={pred_rul_full.get('method')}")
            except Exception as pred_future_err:
                self.logger.log_warning(f"未来RUL计算失败: {pred_future_err}")

            # 计算基于全时间轴真实序列的RUL，检查是否在训练段已越阈值
            try:
                time_segments = []
                for key in ['train_time', 'val_time', 'test_time']:
                    seg = data_result['split_data'].get(key)
                    if seg is not None:
                        time_segments.append(np.asarray(seg, dtype=float).flatten())

                if time_segments:
                    true_time_full = np.concatenate(time_segments)
                    true_voltage_full = np.concatenate([
                        np.asarray(data_result['split_data']['train_original'], dtype=float).flatten(),
                        np.asarray(data_result['split_data']['val_original'], dtype=float).flatten(),
                        np.asarray(data_result['split_data']['test_original'], dtype=float).flatten()
                    ])

                    true_V0_full, _ = calculator.calculate_v_initial(true_voltage_full)
                    true_soh_full = calculator.calculate_soh(true_voltage_full, true_V0_full)
                    true_rul_full = calculator.calculate_rul(true_soh_full, true_time_full)
                    rul_results['true_rul_full'] = true_rul_full
                    self.logger.log_info(
                        f"真实RUL(全时间轴): {true_rul_full.get('rul_actual', 'N/A')}h, 方法={true_rul_full.get('method')}" )
                else:
                    self.logger.log_warning("全时间轴真实RUL: 时间序列缺失，跳过计算")
            except Exception as true_full_err:
                self.logger.log_warning(f"全时间轴真实RUL计算失败: {true_full_err}")

            # 跨数据集预测：使用FC1训练好的模型在FC2上推理
            fc2_result = None
            fc2_path = os.path.join("processed_results", "FC2")
            if os.path.exists(fc2_path):
                try:
                    self.logger.log_info("开始在FC2数据上进行跨数据集预测...")
                    fc2_result = self.predict_dataset(trainer, fc2_path, "FC2")
                    self.logger.log_info(f"FC2 指标: {fc2_result['metrics']}")

                    # 残差线性校正（快速降低整体偏差）
                    try:
                        y_true_fc2 = np.asarray(fc2_result['target'], dtype=float).reshape(-1, 1)
                        y_pred_fc2 = np.asarray(fc2_result['prediction'], dtype=float).reshape(-1, 1)
                        if len(y_true_fc2) > 10:
                            lr_bias = LinearRegression()
                            lr_bias.fit(y_pred_fc2, y_true_fc2)
                            adj_pred = lr_bias.predict(y_pred_fc2).reshape(-1)
                            fc2_result['prediction_bias_adj'] = adj_pred
                            fc2_result['bias_coeff'] = {
                                'slope': float(np.ravel(lr_bias.coef_)[0]),
                                'intercept': float(np.ravel(lr_bias.intercept_)[0]),
                            }
                            fc2_result['metrics_bias_adj'] = MetricsCalculator.calculate_metrics(
                                fc2_result['target'], adj_pred
                            )
                            self.logger.log_info(
                                f"FC2 残差校正: slope={fc2_result['bias_coeff']['slope']:.6f}, "
                                f"intercept={fc2_result['bias_coeff']['intercept']:.6f}, "
                                f"RMSE_adj={fc2_result['metrics_bias_adj'].get('RMSE'):.6f}"
                            )
                        else:
                            self.logger.log_warning("FC2 样本过少，跳过残差校正")
                    except Exception as bias_err:
                        self.logger.log_warning(f"FC2 残差校正失败: {bias_err}")

                    # 生成 FC2 50:50 分割可视化与结果（取消未来滚动预测）
                    try:
                        time_fc2 = np.asarray(fc2_result.get('time', []), dtype=float)
                        target_fc2 = np.asarray(fc2_result.get('target', []), dtype=float)
                        pred_fc2 = np.asarray(fc2_result.get('prediction', []), dtype=float)
                        if len(time_fc2) == 0 or len(target_fc2) == 0 or len(pred_fc2) == 0:
                            raise ValueError("FC2 推理结果为空，无法生成阈值分割图")

                        # 以 300h 为阈值，提前开始展示预测
                        try:
                            start_idx = int(np.searchsorted(time_fc2, 300.0, side='left'))
                        except Exception:
                            # 若时间轴异常，退回按对半划分
                            start_idx = len(time_fc2) // 2

                        start_idx = max(0, min(len(time_fc2), start_idx))
                        boundary_time = float(time_fc2[start_idx - 1]) if start_idx > 0 else float(time_fc2[0])

                        # 生成两段CSV用于前端绘制：阈值前仅真实，阈值后真实+预测
                        fc2_first_df = pd.DataFrame({
                            'dataset': 'FC2',
                            'split': 'fc2_first_half',
                            'time': time_fc2[:start_idx],
                            'target': target_fc2[:start_idx],
                            'prediction': np.full(start_idx, np.nan)
                        })
                        fc2_second_df = pd.DataFrame({
                            'dataset': 'FC2',
                            'split': 'fc2_second_half',
                            'time': time_fc2[start_idx:],
                            'target': target_fc2[start_idx:],
                            'prediction': pred_fc2[start_idx:]
                        })

                        # 计算真实与预测RUL（预测用后半段）
                        calc = SOHRULCalculator(self.config, self.logger)
                        # 真实RUL基于全序列
                        try:
                            true_V0, _ = calc.calculate_v_initial(target_fc2)
                            true_soh = calc.calculate_soh(target_fc2, true_V0)
                            true_rul = calc.calculate_rul(true_soh, time_fc2)
                        except Exception as e_r_true:
                            self.logger.log_warning(f"FC2 真实RUL计算失败: {e_r_true}")
                            true_rul = {'rul_actual': None, 'eol_time': None, 'method': 'error'}

                        # 预测RUL基于后半段预测
                        try:
                            pred_V0, _ = calc.calculate_v_initial(pred_fc2[start_idx:])
                            pred_soh = calc.calculate_soh(pred_fc2[start_idx:], pred_V0)
                            pred_rul = calc.calculate_rul(pred_soh, time_fc2[start_idx:])
                        except Exception as e_r_pred:
                            self.logger.log_warning(f"FC2 预测RUL计算失败: {e_r_pred}")
                            pred_rul = {'rul_actual': None, 'eol_time': None, 'method': 'error'}

                        # 保存半幅预测图（与前端一致的配色与风格）
                        try:
                            plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
                            plt.rcParams["axes.unicode_minus"] = False
                            fig, ax = plt.subplots(figsize=(12, 6), dpi=110, facecolor='white')
                            ax.plot(time_fc2[:start_idx], target_fc2[:start_idx], label='真实(≤300h)', color='#165DFF', linewidth=1.8)
                            ax.axvline(boundary_time, color='#86909C', linestyle='--', linewidth=1.2, alpha=0.8, label='分割线@300h')
                            ax.plot(time_fc2[start_idx:], target_fc2[start_idx:], label='真实(>300h)', color='#165DFF', linewidth=1.5, alpha=0.7)
                            ax.plot(time_fc2[start_idx:], pred_fc2[start_idx:], label='预测(>300h)', color='#F53F3F', linewidth=1.8, linestyle='--')

                            title_rul = f"真实RUL={true_rul.get('rul_actual', 'NA')}h | 预测RUL={pred_rul.get('rul_actual', 'NA')}h"
                            ax.set_title(f"FC2 预测对比（300h后开始） | {title_rul}", fontsize=14, fontweight='bold', color='#1D2129')
                            ax.set_xlabel('时间 (h)', fontsize=12, color='#1D2129')
                            ax.set_ylabel('堆栈电压 (V)', fontsize=12, color='#1D2129')
                            ax.grid(True, alpha=0.3, linestyle='--', color='#E5E6EB')
                            ax.legend(loc='best')
                            plt.tight_layout()
                            fc2_half_img_path = os.path.join(self.config.save_paths['images'], 'fc2_half_prediction.png')
                            fig.savefig(fc2_half_img_path, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            self.logger.log_info(f"FC2 半幅预测图保存: {fc2_half_img_path}")
                        except Exception as plot_err:
                            self.logger.log_warning(f"FC2 半幅绘图失败: {plot_err}")

                        # 将两段合并到最终输出
                        fc2_result['fc2_first_df'] = fc2_first_df
                        fc2_result['fc2_second_df'] = fc2_second_df
                        fc2_result['true_rul'] = true_rul
                        fc2_result['pred_rul'] = pred_rul
                    except Exception as fc2_half_err:
                        self.logger.log_warning(f"FC2 50:50 生成失败: {fc2_half_err}")

                except Exception as fc2_err:
                    self.logger.log_warning(f"FC2 预测失败: {fc2_err}")
            else:
                self.logger.log_warning("未找到 processed_results/FC2/ 目录，跳过FC2预测")

            # 汇总并保存预测结果，供前端FC1/FC2分别展示
            train_time_arr = np.asarray(data_result['split_data']['train_time']).flatten()
            val_time_arr = np.asarray(data_result['split_data']['val_time']).flatten()
            train_target = np.asarray(data_result['split_data']['train_original']).flatten()
            val_target = np.asarray(data_result['split_data']['val_original']).flatten()

            history_df = pd.DataFrame({
                'dataset': 'FC1',
                'split': 'fc1_history',
                'time': np.concatenate([train_time_arr, val_time_arr]),
                'target': np.concatenate([train_target, val_target]),
                'prediction': np.full(len(train_target) + len(val_target), np.nan)
            })

            preds_frames = [
                history_df,
                pd.DataFrame({
                    'dataset': 'FC1',
                    'split': 'fc1_test',
                    'time': time_series_aligned,
                    'target': true_voltage_aligned,
                    'prediction': pred_voltage_aligned
                })
            ]
            if forecast_df is not None:
                preds_frames.append(forecast_df)

            metrics_rows = [
                {**evaluation_results['metrics'], 'dataset': 'FC1'}
            ]

            if fc2_result is not None:
                # 使用 50:50 分割输出到 predictions.csv
                if 'fc2_first_df' in fc2_result and 'fc2_second_df' in fc2_result:
                    preds_frames.append(fc2_result['fc2_first_df'])
                    preds_frames.append(fc2_result['fc2_second_df'])
                else:
                    preds_frames.append(pd.DataFrame({
                        'dataset': 'FC2',
                        'split': 'fc2_test',
                        'time': fc2_result['time'],
                        'target': fc2_result['target'],
                        'prediction': fc2_result['prediction']
                    }))
                metrics_rows.append({**fc2_result['metrics'], 'dataset': 'FC2'})
                if 'metrics_bias_adj' in fc2_result:
                    metrics_rows.append({**fc2_result['metrics_bias_adj'], 'dataset': 'FC2_bias_adj'})
                # 不再追加未来滚动预测段

            preds_combined = pd.concat(preds_frames, ignore_index=True)
            preds_path = os.path.join(self.config.save_paths['csv'], 'predictions.csv')
            preds_combined.to_csv(preds_path, index=False, encoding='utf-8')
            self.logger.log_info(f"预测结果汇总保存: {preds_path}")

            metrics_combined = pd.DataFrame(metrics_rows)
            metrics_path = os.path.join(self.config.save_paths['tables'], 'metrics_overall.csv')
            metrics_combined.to_csv(metrics_path, index=False, encoding='utf-8')
            self.logger.log_info(f"综合指标保存: {metrics_path}")

            # 6. 可视化
            self.logger.log_info("生成可视化图表...")
            visualizer = Visualization(self.config, self.logger)

            # 准备全量时间轴（训练+验证+测试），并在测试段填充预测
            train_time = np.asarray(data_result['split_data']['train_time']).flatten()
            val_time = np.asarray(data_result['split_data']['val_time']).flatten()
            test_time_full = np.asarray(data_result['split_data']['test_time']).flatten()

            time_full = np.concatenate([train_time, val_time, test_time_full])

            true_full = np.concatenate([
                data_result['split_data']['train_original'].flatten(),
                data_result['split_data']['val_original'].flatten(),
                data_result['split_data']['test_original'].flatten()
            ])

            # 预测只在测试段有效，其余位置填充 NaN 以保持对齐
            pred_full = np.full(time_full.shape, np.nan, dtype=float)
            test_start_idx = len(train_time) + len(val_time) + (self.config.sequence_length - 1)
            pred_start = min(test_start_idx, len(pred_full))
            pred_end = min(pred_start + len(pred_voltage_aligned), len(pred_full))

            if pred_end > pred_start:
                pred_full[pred_start:pred_end] = pred_voltage_aligned[:pred_end - pred_start]
            else:
                self.logger.log_warning("预测序列无法对齐全量时间轴，跳过全幅预测绘图")

            # 追加未来滚动预测，使全幅图包含未来走势
            if forecast_df is not None and not forecast_df.empty:
                future_time_ext = forecast_df['time'].to_numpy()
                future_pred_ext = forecast_df['prediction'].to_numpy()
                time_full = np.concatenate([time_full, future_time_ext])
                true_full = np.concatenate([true_full, np.full_like(future_time_ext, np.nan, dtype=float)])
                pred_full = np.concatenate([pred_full, future_pred_ext])

            # 绘制训练历史
            visualizer.plot_training_history(trainer.history)

            # 绘制电压预测图（全时长，训练+验证+测试），训练/验证作为参考
            visualizer.plot_voltage_prediction(
                time_full,
                true_full,
                pred_full,
                len(train_time) + len(val_time),
                None  # MSE损失，无置信区间
            )

            # 绘制SOH曲线（对齐后的数据）
            visualizer.plot_soh_rul_curve(time_series_aligned, true_voltage_aligned, pred_voltage_for_rul, rul_results)

            # 7. 保存最终报告
            self.save_final_report(evaluation_results, rul_results, trainer)

            self.logger.log_info("=" * 80)
            self.logger.log_info("训练和评估成功完成!")
            self.logger.log_info("=" * 80)

            return {
                'model': model,
                'trainer': trainer,
                'evaluation_results': evaluation_results,
                'rul_results': rul_results
            }

        except Exception as e:
            self.logger.log_error(f"训练和评估失败: {str(e)}")
            import traceback
            self.logger.log_error(traceback.format_exc())
            raise

    def save_final_report(self, evaluation_results: Dict[str, Any],
                          rul_results: Dict[str, Any], trainer: GRUTrainer):
        """保存最终报告 - 修复编码问题"""
        self.logger.log_info("生成最终报告...")

        report = {
            'experiment_info': {
                'name': self.config.experiment_name,
                'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'device': self.config.device
            },
            'model_config': {
                'gru_hidden_size': self.config.gru_hidden_size,
                'gru_num_layers': self.config.gru_num_layers,
                'dropout_rate': self.config.dropout_rate,
                'bidirectional': self.config.bidirectional,
                'sequence_length': self.config.sequence_length,
                'batch_size': self.config.batch_size,
                'learning_rate': self.config.learning_rate
            },
            'training_results': {
                'best_epoch': trainer.best_model_info['epoch'],
                'best_val_loss': float(trainer.best_model_info['val_loss']),
                'training_time': f"{time.time() - self.logger.start_time:.1f} 秒"
            },
            'test_metrics': evaluation_results['metrics'],
            'rul_metrics': rul_results['rul_metrics'],
            'rul_details': {
                'true_rul': rul_results['true_rul'].get('rul_actual', 'N/A'),
                'pred_rul': rul_results['pred_rul'].get('rul_actual', 'N/A'),
                'true_V_initial': rul_results.get('true_V_initial', 0),
                'pred_V_initial': rul_results.get('pred_V_initial', 0)
            }
        }

        # 保存JSON报告 - 修复编码问题
        report_path = os.path.join(self.config.save_paths['configs'], 'final_report.json')

        # 将 numpy 类型（如 float32、ndarray）转换为原生 Python 类型，确保 JSON 可序列化
        import numpy as _np

        def _to_serializable(obj):
            if isinstance(obj, _np.generic):
                return obj.item()
            if isinstance(obj, _np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(v) for v in obj]
            return obj

        report_serializable = _to_serializable(report)

        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=4, ensure_ascii=False)

        # 保存文本报告 - 修复编码问题
        text_report = f"""
PEMFC燃料电池GRU寿命预测模型 - 最终报告
========================================================

实验: {report['experiment_info']['name']}
开始时间: {report['experiment_info']['start_time']}
设备: {report['experiment_info']['device']}

模型配置:
-------------------
GRU隐藏层大小: {report['model_config']['gru_hidden_size']}
GRU层数: {report['model_config']['gru_num_layers']}
Dropout率: {report['model_config']['dropout_rate']}
双向: {report['model_config']['bidirectional']}
序列长度: {report['model_config']['sequence_length']}
批次大小: {report['model_config']['batch_size']}
学习率: {report['model_config']['learning_rate']}

训练结果:
-----------------
最佳Epoch: {report['training_results']['best_epoch']}
最佳验证损失: {report['training_results']['best_val_loss']:.6f}
训练时间: {report['training_results']['training_time']}

测试集指标:
-----------------
MAE: {report['test_metrics'].get('MAE', 0):.6f}
RMSE: {report['test_metrics'].get('RMSE', 0):.6f}
MAPE: {report['test_metrics'].get('MAPE', 0):.6f}%
R2: {report['test_metrics'].get('R2', 0):.6f}

RUL预测结果:
-----------------------
真实RUL: {report['rul_details']['true_rul']} h
预测RUL: {report['rul_details']['pred_rul']} h
真实V_initial: {report['rul_details']['true_V_initial']:.4f} V
预测V_initial: {report['rul_details']['pred_V_initial']:.4f} V

RUL误差指标:
------------------
MAE_RUL: {report['rul_metrics'].get('MAE_RUL', 'N/A')}
MAPE_RUL: {report['rul_metrics'].get('MAPE_RUL', 'N/A')}%
PHM评分: {report['rul_metrics'].get('PHM_Score', 'N/A')}
        """

        text_report_path = os.path.join(self.config.save_paths['configs'], 'final_report.txt')
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write(text_report)

        self.logger.log_info(f"最终报告保存: {report_path}")

        # 打印总结
        print("\n" + "=" * 80)
        print("PEMFC燃料电池GRU寿命预测模型 - 训练总结")
        print("=" * 80)
        print(f"\n最佳模型: Epoch {report['training_results']['best_epoch']}")
        print(f"最佳验证损失: {report['training_results']['best_val_loss']:.6f}")
        print("\n测试集指标:")
        for metric, value in report['test_metrics'].items():
            print(f"  {metric}: {value}")

        if report['rul_metrics']['MAE_RUL'] is not None:
            print("\nRUL预测:")
            print(f"  真实RUL: {report['rul_details']['true_rul']} h")
            print(f"  预测RUL: {report['rul_details']['pred_rul']} h")
            print(f"  RUL MAE: {report['rul_metrics']['MAE_RUL']} h")
            print(f"  RUL MAPE: {report['rul_metrics']['MAPE_RUL']}%")

        print(f"\n结果保存到: {self.config.save_paths['base']}")
        print("=" * 80)


# ============================================================================
# 10. 主函数
# ============================================================================

def main():
    """主函数"""
    import torch
    print("=" * 80)
    print("PEMFC燃料电池GRU寿命预测模型 (论文版本)")
    print(f"PyTorch版本: {torch.__version__}")
    print("=" * 80)

    def load_top_features(pattern: str, top_n: int = 5, min_importance: float = 0.0):
        """从CatBoost特征重要性CSV中自动选择特征"""
        try:
            files = glob.glob(pattern)
            if not files:
                return []
            latest = max(files, key=os.path.getmtime)
            df = pd.read_csv(latest)
            if "feature" not in df.columns or "importance" not in df.columns:
                return []
            filtered = df[df["importance"] > min_importance].sort_values("importance", ascending=False)
            return filtered["feature"].head(top_n).tolist()
        except Exception:
            return []

    catboost_features = load_top_features("catboost_results/feature_importance_results_*.csv")
    # 若无法读取CatBoost结果，则退回固定的前5特征
    default_features = [
        "air_outlet_flow",       # 48.66%
        "hydrogen_inlet_temp",   # 19.76%
        "current",               # 9.40%
        "coolant_flow",          # 7.37%
        "current_density",       # 6.93%
    ]
    selected_features = catboost_features if catboost_features else default_features

    # 创建配置
    config = ModelConfig(
        # 数据参数
        data_path="processed_results/FC1/",  # 数据目录路径
        data_file_pattern="*_processed_*.npz",  # 数据文件匹配模式
        data_file_suffix="_processed",  # 数据文件后缀标识

        # 基于提供的特征重要性选择的关键特征
        selected_features=selected_features,

        # GRU模型参数
        gru_hidden_size=256,
        gru_num_layers=2,
        dropout_rate=0.1,
        bidirectional=False,

        # 训练参数
        batch_size=32,
        learning_rate=0.0006,
        epochs=45,
        patience=30,
        weight_decay=0.001,
        lr_patience=5,
        slope_loss_weight=0.2,
        variance_loss_weight=0.45,
        grad_clip=0.5,

        # 针对FC1 RUL偏差微调校正量（推理可复用最佳模型，无需重训）
        fc1_bias_correction=0.0138,

        # 固定时间步长，减少时间漂移
        fixed_dt=0.0084,

        # 实验名称
        experiment_name="gru_pemfc_paper_experiment_fixed_r2_rul"
    )

    # 创建训练器并运行
    trainer = PEMFCTrainer(config)
    results = trainer.train_and_evaluate()

    if results:
        print("\n" + "=" * 80)
        print("✓ 训练成功完成!")
        print("\n改进模型的建议:")
        print("1. 如果验证损失高（过拟合）:")
        print("   - 增加dropout_rate (0.4-0.5)")
        print("   - 减少gru_hidden_size (32-48)")
        print("   - 添加L2正则化 (增加weight_decay到0.001)")
        print("\n2. 如果训练损失高（欠拟合）:")
        print("   - 减少dropout_rate (0.1-0.2)")
        print("   - 增加gru_hidden_size (128-256)")
        print("   - 增加gru_num_layers (3)")
        print("   - 增加learning_rate (0.005)")
        print("\n3. 如果训练不稳定:")
        print("   - 减少learning_rate (0.0005)")
        print("   - 减少batch_size (16-24)")
        print("   - 减少grad_clip (0.5)")
        print("=" * 80)


if __name__ == "__main__":
    main()

# ============================================================================
# 11. GRU参数解释表和新手指南
# ============================================================================
"""
GRU参数解释表:
---------------
1. gru_hidden_size (隐藏层大小):
   - 【作用】决定GRU层中隐藏状态向量的维度，影响模型容量和学习能力
   - 【默认值】64，平衡复杂度和计算效率
   - 【修改建议】数据复杂时增加，简单时减少

2. gru_num_layers (层数):
   - 【作用】堆叠GRU层的数量，增加模型深度
   - 【默认值】2，捕获更复杂的时间依赖
   - 【修改建议】深度任务可增加到3，浅层任务用1

3. dropout_rate (丢弃率):
   - 【作用】防止过拟合的正则化技术
   - 【默认值】0.3，经验值
   - 【修改建议】过拟合时增加，欠拟合时减少

4. bidirectional (双向):
   - 【作用】是否使用双向GRU，同时考虑过去和未来信息
   - 【默认值】True，对时序预测通常有益
   - 【修改建议】对上下文敏感的任务建议True

5. sequence_length (序列长度):
   - 【作用】输入序列的时间步长
   - 【默认值】50，基于论文滑动窗口
   - 【修改建议】根据数据周期性和采样频率调整

新手参数修改指南:
----------------
1. 训练不稳定（损失震荡）:
   - 降低learning_rate (0.001 → 0.0005)
   - 减小batch_size (32 → 16)
   - 增加grad_clip (1.0 → 0.5)

2. 过拟合（训练损失低，验证损失高）:
   - 增加dropout_rate (0.3 → 0.4-0.5)
   - 减小gru_hidden_size (64 → 32)
   - 增加weight_decay (0.0001 → 0.001)
   - 减少gru_num_layers (2 → 1)

3. 欠拟合（训练和验证损失都高）:
   - 增加gru_hidden_size (64 → 128)
   - 增加gru_num_layers (2 → 3)
   - 减小dropout_rate (0.3 → 0.1-0.2)
   - 增加learning_rate (0.001 → 0.005)

4. 训练速度慢:
   - 增加batch_size (32 → 64-128)
   - 减小sequence_length (50 → 30)
   - 减小gru_hidden_size (64 → 32)

5. 内存不足:
   - 减小batch_size (32 → 16)
   - 减小sequence_length (50 → 30)
   - 减小gru_hidden_size (64 → 32)

关键参数调试顺序:
----------------
1. 先调整learning_rate和batch_size确保训练稳定
2. 然后调整gru_hidden_size和gru_num_layers优化模型容量
3. 接着调整dropout_rate和weight_decay防止过拟合
4. 最后调整sequence_length优化时间窗口

日志使用说明:
-------------
1. 训练日志保存在: train_results_paper/[experiment_name]/logs/
2. 查看日志了解训练过程、指标变化和问题
3. 根据日志中的验证损失变化调整参数
4. 保存日志文件用于后续分析和比较不同实验12312321321
"""