import pandas as pd
import numpy as np
import os


class PEMFCDataProcessor:
    """PEMFC数据处理工具（适配用户提供的CSV格式）"""

    @staticmethod
    def validate_csv(csv_path):
        """验证CSV文件是否包含必要列"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV文件不存在：{csv_path}")

        df = pd.read_csv(csv_path)
        required_cols = ["feature", "importance", "importance_percent", "cumulative_percent"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"CSV缺少必要列：{', '.join(missing_cols)}（需包含{required_cols}）")

        # 验证数据类型
        if not pd.api.types.is_numeric_dtype(df["importance"]):
            raise TypeError("'importance'列必须为数值类型")

        return df

    @staticmethod
    def get_top5_importance(csv_path):
        """读取CSV并获取前5个重要性最高的特征数据"""
        df = PEMFCDataProcessor.validate_csv(csv_path)
        # 按重要性降序排序，取前5
        df_sorted = df.sort_values("importance", ascending=False).head(5).reset_index(drop=True)

        # 转换为字典返回（便于传递）
        return {
            "csv_path": csv_path,
            "feature": df_sorted["feature"].tolist(),
            "importance": df_sorted["importance"].tolist(),
            "importance_percent": df_sorted["importance_percent"].tolist()
        }

    @staticmethod
    def parse_voltage_data(csv_path):
        """解析电压数据（用于原始数据处理页面）"""
        df = pd.read_csv(csv_path)
        # 自动识别时间列和电压列
        time_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["time", "t", "时间"])), None)
        voltage_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["voltage", "v", "电压"])),
                           None)

        if not time_col:
            raise ValueError("未找到时间列（列名需包含time/t/时间）")
        if not voltage_col:
            raise ValueError("未找到电压列（列名需包含voltage/v/电压）")

        # 清理数据（去除NaN）
        time_data = df[time_col].dropna().astype(float).tolist()
        voltage_data = df[voltage_col].dropna().astype(float).tolist()

        # 确保数据长度一致
        min_len = min(len(time_data), len(voltage_data))
        return {
            "time_col": time_col,
            "voltage_col": voltage_col,
            "time_data": time_data[:min_len],
            "voltage_data": voltage_data[:min_len]
        }

    @staticmethod
    def parse_soh_data(csv_path):
        """解析SOH数据（用于寿命预测页面）"""
        df = pd.read_csv(csv_path)
        time_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["time", "t", "时间"])), None)
        soh_col = next((col for col in df.columns if any(kw in col.lower() for kw in ["soh", "健康", "状态"])), None)

        if not time_col:
            raise ValueError("未找到时间列（列名需包含time/t/时间）")
        if not soh_col:
            raise ValueError("未找到SOH列（列名需包含soh/健康/状态）")

        time_data = df[time_col].dropna().astype(float).tolist()
        soh_data = df[soh_col].dropna().astype(float).tolist()
        min_len = min(len(time_data), len(soh_data))

        return {
            "time_col": time_col,
            "soh_col": soh_col,
            "time_data": time_data[:min_len],
            "soh_data": soh_data[:min_len]
        }