from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QGridLayout, QPushButton,
                             QLabel, QSpinBox, QMessageBox,
                             QComboBox, QDialog)
from PyQt6.QtCore import Qt, pyqtSlot, QProcess, QTimer
import sys
import os
import glob
import threading
import runpy
from datetime import datetime
from typing import List
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from .plot_worker import (
    plot_feature_importance,
    plot_raw_views,
    plot_prediction_vs_true,
    plot_voltage_overlay,
    plot_metrics_table,
    FEATURE_CN_MAP,
)
import pandas as pd
import shutil


# ================= 路径与运行环境配置 =================
# 在打包(onefile)后，PyInstaller会将资源解压到临时目录(sys._MEIPASS)，
# 这里统一计算基准目录，避免依赖当前工作目录导致找不到CSV或误触发自启动。
def _get_base_dir() -> str:
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    # 源码模式：pages.py 位于 gui 子目录，需回退一级到项目根目录
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = _get_base_dir()


def _abs_path(*parts: str) -> str:
    return os.path.abspath(os.path.join(BASE_DIR, *parts))


def _script_args(script_path: str):
    """为脚本生成子进程启动参数，冻结状态下用 -c 避免再次启动GUI。"""
    if getattr(sys, "frozen", False):
        return ["-c", f"import runpy; runpy.run_path(r'{script_path}', run_name='__main__')"]
    return [script_path]


def _run_script_in_thread(script_path: str, on_finish=None, on_error=None):
    """冻结状态下在后台线程直接 run_path，避免启动新的可执行文件。"""
    def _worker():
        try:
            runpy.run_path(script_path, run_name="__main__")
            if callable(on_finish):
                on_finish()
        except Exception as exc:  # noqa: BLE001 - 直接报告
            if callable(on_error):
                on_error(exc)
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t


# 原始/处理/可视化数据路径（改为绝对路径，兼容打包后）
RAW_DATA_DIR = _abs_path("data")
PROCESSED_DATA_DIR = _abs_path("processed_results")
SOH_DATA_DIR = _abs_path("soh_data")
RAW_VISUAL_DIR = _abs_path("visualization", "truedata")

CATBOOST_RESULT_PATTERN = _abs_path("catboost_results", "feature_importance_results_*.csv")
PREDICTION_PATTERNS = [
    _abs_path("train_results_paper", "**", "csv_files", "predictions.csv"),
    _abs_path("train_results", "**", "csv_files", "predictions.csv"),
]
METRICS_PATTERNS = [
    _abs_path("train_results_paper", "**", "tables", "metrics_overall.csv"),
    _abs_path("train_results", "**", "tables", "metrics_overall.csv"),
]

# 可清理的生成文件路径（不触碰原始数据和模型权重）
CLEAN_TARGETS = [
    (PROCESSED_DATA_DIR, ("*.csv", "*.npz")),
    (RAW_VISUAL_DIR, ("*.csv",)),
    (_abs_path("catboost_results"), ("*.csv",)),
    (_abs_path("train_results_paper"), ("**/*.csv", "**/*.npz", "**/*.json", "**/*.txt", "**/*.png")),
    (_abs_path("train_results"), ("**/*.csv", "**/*.npz", "**/*.json", "**/*.txt", "**/*.png")),
]
TRAIN_WAVELET = "sym8"
TRAIN_WINDOW = 50
DEFAULT_TOP_SIGNALS = [
    "hydrogen_inlet_temp",
    "air_outlet_flow",
    "current",
    "coolant_flow",
    "current_density",
]


def find_latest_file(patterns):
    """根据glob模式列表获取最新文件"""
    if isinstance(patterns, str):
        patterns = [patterns]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    if not files:
        return None
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files[0]


class FigureDialog(QDialog):
    """独立弹窗展示Matplotlib图，避免挤占主界面"""

    def __init__(self, fig, title="图表", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(1000, 720)
        layout = QVBoxLayout(self)
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)


def show_figure_in_dialog(owner, fig, title):
    dlg = FigureDialog(fig, title=title, parent=owner)
    if hasattr(owner, "_figure_dialogs"):
        owner._figure_dialogs.append(dlg)
    else:
        owner._figure_dialogs = [dlg]
    dlg.show()


# ======================================================================

class FeatureImportancePage(QWidget):
    """特征重要性页面（直接绘图，无临时文件）"""

    def __init__(self, monitor_panel, parent=None):
        super().__init__(parent)
        self.monitor = monitor_panel  # 监控面板引用
        self.top_n = 5  # 默认显示前5个特征
        self.canvas = None  # Matplotlib画布（已改为弹窗展示）
        self.feature_csv_path = find_latest_file(CATBOOST_RESULT_PATTERN)
        self.analysis_process: QProcess | None = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 28, 28, 32)
        layout.setSpacing(32)

        # 页面标题
        title = QLabel("各监测参数重要性排序（可配置显示数量）")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1D2129; margin-bottom: 10px;")
        layout.addWidget(title)

        # 控制区域
        control_group = QGroupBox("数据管理")
        control_layout = QGridLayout(control_group)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(15)

        # 运行CatBoost分析
        self.run_analysis_btn = QPushButton("运行重要性分析")
        self.run_analysis_btn.setStyleSheet("""
            QPushButton { background-color: #00B42A; color: white; border: none; border-radius: 6px;
                          padding: 10px 20px; font-size: 14px; font-weight: 600; }
            QPushButton:hover { background-color: #23C343; }
            QPushButton:disabled { background-color: #F5F7FA; color: #C9CDD4; border-color: #E5E6EB; }
        """)
        self.run_analysis_btn.clicked.connect(self.run_catboost_analysis)
        control_layout.addWidget(self.run_analysis_btn, 0, 0)

        # 生成图表按钮
        self.generate_btn = QPushButton("生成重要性图表")
        self.generate_btn.setStyleSheet("""
            QPushButton { background-color: #165DFF; color: white; border: none; border-radius: 6px;
                          padding: 10px 20px; font-size: 14px; font-weight: 600; }
            QPushButton:hover { background-color: #4080FF; }
        """)
        self.generate_btn.clicked.connect(self.generate_plot)
        control_layout.addWidget(self.generate_btn, 0, 1)

        # 显示特征数量设置
        control_layout.addWidget(QLabel("显示特征数量:"), 0, 2)
        self.top_n_spin = QSpinBox()
        self.top_n_spin.setRange(1, 20)
        self.top_n_spin.setValue(5)
        self.top_n_spin.setMinimumHeight(38)
        self.top_n_spin.setStyleSheet("""
            QSpinBox { border: 1px solid #E5E6EB; border-radius: 6px; 
                       padding: 10px 12px; font-size: 14px; }
            QSpinBox:focus { border-color: #165DFF; outline: none; }
        """)
        self.top_n_spin.valueChanged.connect(self.on_top_n_changed)
        control_layout.addWidget(self.top_n_spin, 0, 3)

        layout.addWidget(control_group)

        # 图表提示标签（改为弹窗展示）
        tip = QLabel("点击「生成重要性图表」后将在新窗口展示图表")
        tip.setStyleSheet("background-color: #F5F7FA; border-radius: 8px; border: 1px dashed #E5E6EB;")
        tip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tip.setMinimumHeight(36)
        layout.addWidget(tip)

    @pyqtSlot(int)
    def on_top_n_changed(self, value):
        """特征数量变化时记录日志"""
        self.top_n = value
        self.monitor.log(f"[特征重要性] 显示特征数量更新为{value}")

    def generate_plot(self):
        """直接生成图表并嵌入界面（无临时文件）"""
        try:
            self.monitor.update_progress("开始生成重要性图表", 30)
            self.monitor.log(f"[特征重要性] 筛选前{self.top_n}个重要特征")

            # 验证特征重要性CSV是否存在
            if not self.feature_csv_path:
                self.feature_csv_path = find_latest_file(CATBOOST_RESULT_PATTERN)
            if not self.feature_csv_path or not os.path.exists(self.feature_csv_path):
                raise FileNotFoundError("未找到特征重要性CSV，请先运行CatBoost分析")

            # 调用绘图函数（直接返回Figure对象）
            fig, feature_cn_list = plot_feature_importance(self.feature_csv_path, self.top_n)

            # 弹窗展示，不占主界面空间
            show_figure_in_dialog(self, fig, title="特征重要性")

            # 日志输出中文特征名
            self.monitor.log(f"✅ 重要性图表生成完成（前{self.top_n}特征）")
            feature_cn_str = ', '.join(str(s) for s in feature_cn_list if s is not None)
            self.monitor.log(f"📋 特征中文名称：{feature_cn_str}")
            self.monitor.update_progress("图表生成完成", 100)

        except Exception as e:
            self.monitor.log_error(f"生成重要性图表失败：{e}")
            QMessageBox.critical(self, "错误", f"生成图表失败：{e}")
            self.monitor.update_progress("图表生成失败", 0)

    def run_catboost_analysis(self):
        """通过子进程运行CatBoost分析脚本"""
        if self.analysis_process and self.analysis_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "任务进行中", "CatBoost分析正在运行，请稍后。")
            return

        script_path = os.path.join(BASE_DIR, "pemfc_catboost_analysis.py")
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "缺少脚本", "未找到 pemfc_catboost_analysis.py，请确认与可执行文件同目录或使用源码环境运行。")
            self.monitor.log_error("未找到 pemfc_catboost_analysis.py，已取消执行")
            return

        # 冻结态：避免再启动GUI，直接线程内 run_path；源码态仍用QProcess捕获输出
        self.run_analysis_btn.setEnabled(False)
        self.monitor.update_progress("启动CatBoost分析", 5)
        self.monitor.log("🚀 已启动CatBoost分析脚本 pemfc_catboost_analysis.py")

        if getattr(sys, "frozen", False):
            def _on_finish():
                QTimer.singleShot(0, lambda: self._on_catboost_finished(0, None))
            def _on_err(exc):
                QTimer.singleShot(0, lambda: self.monitor.log_error(f"CatBoost分析失败: {exc}"))
                QTimer.singleShot(0, lambda: self._on_catboost_finished(1, None))
            _run_script_in_thread(script_path, on_finish=_on_finish, on_error=_on_err)
        else:
            self.analysis_process = QProcess(self)
            self.analysis_process.setProgram(sys.executable)
            self.analysis_process.setArguments(_script_args(script_path))
            self.analysis_process.setWorkingDirectory(BASE_DIR)
            self.analysis_process.readyReadStandardOutput.connect(self._on_catboost_output)
            self.analysis_process.readyReadStandardError.connect(self._on_catboost_output)
            self.analysis_process.finished.connect(self._on_catboost_finished)
            self.analysis_process.start()

    def _on_catboost_output(self):
        if not self.analysis_process:
            return
        text = self.analysis_process.readAllStandardOutput().data().decode(errors="ignore")
        err_text = self.analysis_process.readAllStandardError().data().decode(errors="ignore")
        combined = f"{text}{err_text}".strip()
        if combined:
            for line in combined.splitlines():
                if line.strip():
                    self.monitor.log(f"[CatBoost] {line.strip()}")

    def _on_catboost_finished(self, exit_code, _status):
        self.run_analysis_btn.setEnabled(True)
        if exit_code == 0:
            self.monitor.update_progress("CatBoost分析完成", 80)
            self.feature_csv_path = find_latest_file(CATBOOST_RESULT_PATTERN)
            if self.feature_csv_path:
                self.monitor.log(f"📁 最新特征重要性：{os.path.basename(self.feature_csv_path)}")
                self.generate_plot()
                QMessageBox.information(self, "完成", "CatBoost分析完成，已刷新图表。")
                self.monitor.update_progress("特征图表已刷新", 100)
            else:
                self.monitor.log_error("未找到特征重要性CSV，请检查输出目录")
        else:
            self.monitor.log_error("CatBoost分析失败，查看日志输出")
            QMessageBox.critical(self, "错误", "CatBoost分析失败，请查看日志。")


class DataProcessingPage(QWidget):
    """原始数据合并可视化 + 处理后滤波对比"""

    def __init__(self, monitor_panel, parent=None):
        super().__init__(parent)
        self.monitor = monitor_panel
        self.raw_canvas = None
        self.process_canvas = None
        self.processing_process: QProcess | None = None
        self.raw_csv_path = None
        self.filtered_csv_path = None
        self.current_dataset = "FC1"
        self.setup_ui()
        self.refresh_paths()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # 页面标题
        title = QLabel("原始数据处理与可视化")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1D2129; margin-bottom: 10px;")
        layout.addWidget(title)

        # 原始数据合并+可视化
        raw_group = QGroupBox("原始数据合并与可视化")
        raw_group.setMinimumHeight(420)
        raw_layout = QGridLayout(raw_group)
        # 上移控件与图表间距，进一步压缩顶/内边距
        raw_layout.setContentsMargins(22, 8, 22, 18)
        raw_layout.setHorizontalSpacing(18)
        raw_layout.setVerticalSpacing(10)

        # 顶部控件行（居中）
        raw_controls = QHBoxLayout()
        raw_controls.setSpacing(16)
        raw_controls.setAlignment(Qt.AlignmentFlag.AlignCenter)
        raw_controls.setContentsMargins(0, 0, 0, 0)

        ds_label = QLabel("数据集:")
        ds_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #1D2129;")
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItem("请选择数据集", userData=None)
        for ds in ["FC1", "FC2"]:
            self.dataset_combo.addItem(ds, userData=ds)
        self.dataset_combo.setCurrentIndex(1)  # 默认FC1
        self.dataset_combo.setMinimumWidth(200)
        self.dataset_combo.setMinimumHeight(42)
        self.dataset_combo.setStyleSheet(
            "QComboBox { border: 1px solid #E5E6EB; border-radius: 6px; padding: 10px 14px; font-size: 14px; }"
        )
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_changed)

        self.merge_plot_btn = QPushButton("合并并可视化原始数据")
        self.merge_plot_btn.setMinimumHeight(44)
        self.merge_plot_btn.setMinimumWidth(230)
        self.merge_plot_btn.setStyleSheet("""
            QPushButton { background-color: #165DFF; color: white; border: none; border-radius: 8px;
                          padding: 12px 18px; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #3A7BFF; }
        """)
        self.merge_plot_btn.clicked.connect(self.visualize_raw_data)

        raw_controls.addStretch(1)
        raw_controls.addWidget(ds_label)
        raw_controls.addWidget(self.dataset_combo)
        raw_controls.addSpacing(12)
        raw_controls.addWidget(self.merge_plot_btn)
        raw_controls.addStretch(1)

        raw_layout.addLayout(raw_controls, 0, 0, 1, 4)

        # 提示标签直接置于分组内，紧贴控件下方，便于指示选择 FC1/FC2
        self.raw_tip_label = QLabel("选择数据集后点击“合并并可视化原始数据”")
        # 文字设为透明，保留背景框架
        self.raw_tip_label.setStyleSheet("background-color: #F5F7FA; border-radius: 8px; border: 1px dashed #E5E6EB; padding: 14px; font-size: 13px; color: transparent;")
        self.raw_tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.raw_tip_label.setMinimumHeight(140)
        raw_layout.addWidget(self.raw_tip_label, 1, 0, 1, 4)
        layout.addWidget(raw_group)
        layout.addSpacing(28)

        # 数据处理参数与脚本触发
        control_group = QGroupBox("数据处理与滤波对比")
        control_group.setMinimumHeight(360)
        control_layout = QGridLayout(control_group)
        # 进一步上移整体控件，拉近与按钮的顶端对齐
        control_layout.setContentsMargins(24, -4, 24, 20)
        control_layout.setHorizontalSpacing(20)
        control_layout.setVerticalSpacing(12)

        # 顶部左右分栏
        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        # 顶部行贴近上缘，避免控件垂直偏低
        top_row.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        top_row.setContentsMargins(0, -18, 0, 0)

        # 左列：仅数据集选择
        left_col = QGridLayout()
        left_col.setContentsMargins(0, -24, 0, 0)
        left_col.setHorizontalSpacing(10)
        left_col.setVerticalSpacing(6)

        proc_ds_label = QLabel("数据集:")
        proc_ds_label.setStyleSheet("font-size: 14px; font-weight: 600; color: #1D2129;")
        left_col.addWidget(proc_ds_label, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.proc_dataset_combo = QComboBox()
        for ds in ["FC1", "FC2"]:
            self.proc_dataset_combo.addItem(ds, userData=ds)
        self.proc_dataset_combo.setCurrentText(self.current_dataset)
        self.proc_dataset_combo.setMinimumWidth(190)
        self.proc_dataset_combo.setFixedHeight(44)  # 固定高度与右侧按钮一致
        # 限制最小/最大高度，避免样式对高度的二次放大或压缩
        self.proc_dataset_combo.setStyleSheet(
            "QComboBox { border: 1px solid #E5E6EB; border-radius: 6px; padding: 10px 14px; font-size: 14px; min-height: 44px; max-height: 44px; }"
        )
        self.proc_dataset_combo.currentTextChanged.connect(self.on_proc_dataset_changed)
        left_col.addWidget(self.proc_dataset_combo, 0, 1, alignment=Qt.AlignmentFlag.AlignTop)
        left_col.setRowMinimumHeight(0, 44)

        # 右侧：两个按钮同排并右移
        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setContentsMargins(0, -4, 16, 0)
        btn_row.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        self.run_processing_btn = QPushButton("进行数据处理")
        self.run_processing_btn.setFixedHeight(44)
        self.run_processing_btn.setMinimumWidth(210)
        self.run_processing_btn.setStyleSheet("""
            QPushButton { background-color: #00B42A; color: white; border: none; border-radius: 8px;
                          padding: 12px 16px; font-size: 14px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #23C343; }
            QPushButton:disabled { background-color: #F5F7FA; color: #C9CDD4; border-color: #E5E6EB; }
        """)
        self.run_processing_btn.clicked.connect(self.run_processing_script)

        self.process_btn = QPushButton("生成滤波对比图")
        self.process_btn.setFixedHeight(44)
        self.process_btn.setMinimumWidth(210)
        self.process_btn.setStyleSheet("""
            QPushButton { background-color: #165DFF; color: white; border: none; border-radius: 8px;
                          padding: 12px 20px; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #3A7BFF; }
            QPushButton:disabled { background-color: #F5F7FA; color: #C9CDD4; border-color: #E5E6EB; }
        """)
        self.process_btn.clicked.connect(self.process_data)

        btn_row.addStretch(1)
        btn_row.addWidget(self.run_processing_btn)
        btn_row.addSpacing(6)
        btn_row.addWidget(self.process_btn)

        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)
        right_col.addLayout(btn_row)

        top_row.addLayout(left_col)
        top_row.addSpacing(24)
        top_row.addLayout(right_col)
        top_row.addStretch(1)

        control_layout.addLayout(top_row, 0, 0, 1, 4)

        control_layout.setRowMinimumHeight(0, 0)
        control_layout.setRowStretch(0, 0)
        control_layout.setColumnStretch(0, 0)
        control_layout.setColumnStretch(1, 1)
        control_layout.setColumnStretch(2, 0)
        control_layout.setColumnStretch(3, 1)

        layout.addWidget(control_group)
        layout.addSpacing(28)

    def on_dataset_changed(self, dataset):
        if dataset not in {"FC1", "FC2"}:
            return
        self.current_dataset = dataset
        if hasattr(self, 'proc_dataset_combo'):
            self.proc_dataset_combo.blockSignals(True)
            self.proc_dataset_combo.setCurrentText(dataset)
            self.proc_dataset_combo.blockSignals(False)
        self.refresh_paths()

    def on_proc_dataset_changed(self, dataset):
        if dataset not in {"FC1", "FC2"}:
            return
        self.current_dataset = dataset
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.setCurrentText(dataset)
        self.dataset_combo.blockSignals(False)
        self.refresh_paths()

    def refresh_paths(self):
        self.raw_csv_path = self.get_latest_raw_data(self.current_dataset)
        self.filtered_csv_path = self.get_latest_processed_data(self.current_dataset)
        self.check_file_status()

    def populate_column_options(self):
        """占位：原始数据阶段使用自动列检测，不再提供手动选项"""
        return

    def populate_proc_params(self):
        pass

    def selected_time_col(self):
        return None

    def selected_voltage_col(self):
        return None

    def get_latest_raw_data(self, dataset: str):
        """获取指定数据集的最新原始CSV"""
        try:
            patterns = [
                os.path.join(RAW_VISUAL_DIR, f"{dataset}_merged_*.csv"),  # 优先使用合并后的文件
                os.path.join(RAW_DATA_DIR, f"{dataset}_Ageing_part*.csv"),
                os.path.join(_abs_path("datatest"), f"{dataset}_Ageing_part*.csv")
            ]
            return find_latest_file(patterns)
        except Exception as e:
            self.monitor.log_error(f"获取原始数据失败：{str(e)}")
            return None

    def get_latest_processed_data(self, dataset: str):
        """获取指定数据集的最新处理后CSV"""
        try:
            patterns = [
                os.path.join(PROCESSED_DATA_DIR, f"{dataset}_processed_*.csv"),
                os.path.join(PROCESSED_DATA_DIR, dataset, f"{dataset}_processed_*.csv")
            ]
            return find_latest_file(patterns)
        except Exception as e:
            self.monitor.log_error(f"获取处理后数据失败：{str(e)}")
            return None

    def merge_raw_data(self, dataset: str):
        """合并原始分片并保存到可视化目录"""
        patterns = [
            os.path.join(RAW_DATA_DIR, f"{dataset}_Ageing_part*.csv"),
            os.path.join(_abs_path("datatest"), f"{dataset}_Ageing_part*.csv")
        ]
        files = []
        for pattern in patterns:
            files.extend(sorted(glob.glob(pattern)))
        if not files:
            raise FileNotFoundError(f"未找到{dataset}原始文件，请检查data/datatest目录")

        dfs = []
        for f in files:
            try:
                dfs.append(pd.read_csv(f, encoding_errors="ignore"))
            except Exception as read_err:
                self.monitor.log_error(f"读取{os.path.basename(f)}失败：{read_err}")
        if not dfs:
            raise ValueError("原始数据读取失败")

        merged = pd.concat(dfs, ignore_index=True)
        os.makedirs(RAW_VISUAL_DIR, exist_ok=True)
        out_path = os.path.join(RAW_VISUAL_DIR, f"{dataset}_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        merged.to_csv(out_path, index=False, encoding="utf-8")
        return out_path

    def visualize_raw_data(self):
        try:
            dataset = self.dataset_combo.currentText()
            if dataset not in {"FC1", "FC2"}:
                dataset = "FC1"
                self.dataset_combo.setCurrentText(dataset)
            self.monitor.update_progress(f"开始合并{dataset}原始数据", 10)
            merged_path = self.merge_raw_data(dataset)
            self.monitor.log(f"📁 已生成合并文件：{os.path.basename(merged_path)}")

            fig = plot_raw_views(
                merged_path,
                dataset_label=dataset,
                selected_signals=None,
                time_col=None,
                voltage_col=None,
                max_signals=0,
            )

            show_figure_in_dialog(self, fig, title=f"{dataset} 原始数据视图")
            self.monitor.update_progress("原始数据可视化完成", 60)
        except Exception as e:
            self.monitor.log_error(f"原始数据可视化失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"原始数据可视化失败：{str(e)}")
            self.monitor.update_progress("原始数据可视化失败", 0)

    def run_processing_script(self):
        if self.processing_process and self.processing_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "任务进行中", "数据处理脚本正在运行，请稍后。")
            return

        script_path = os.path.join(BASE_DIR, "data_processing.py")
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "缺少脚本", "未找到 data_processing.py，请确认与可执行文件同目录或使用源码环境运行。")
            self.monitor.log_error("未找到 data_processing.py，已取消执行")
            return

        self.run_processing_btn.setEnabled(False)
        self.monitor.update_progress("启动数据处理脚本", 5)
        self.monitor.log("🚀 已启动 data_processing.py")

        if getattr(sys, "frozen", False):
            def _on_finish():
                QTimer.singleShot(0, lambda: self._on_processing_finished(0, None))
            def _on_err(exc):
                QTimer.singleShot(0, lambda: self.monitor.log_error(f"数据处理失败: {exc}"))
                QTimer.singleShot(0, lambda: self._on_processing_finished(1, None))
            _run_script_in_thread(script_path, on_finish=_on_finish, on_error=_on_err)
        else:
            self.processing_process = QProcess(self)
            self.processing_process.setProgram(sys.executable)
            self.processing_process.setArguments(_script_args(script_path))
            self.processing_process.setWorkingDirectory(BASE_DIR)
            self.processing_process.readyReadStandardOutput.connect(self._on_processing_output)
            self.processing_process.readyReadStandardError.connect(self._on_processing_output)
            self.processing_process.finished.connect(self._on_processing_finished)
            self.processing_process.start()

    def _on_processing_output(self):
        if not self.processing_process:
            return
        text = self.processing_process.readAllStandardOutput().data().decode(errors="ignore")
        err_text = self.processing_process.readAllStandardError().data().decode(errors="ignore")
        combined = f"{text}{err_text}".strip()
        if combined:
            for line in combined.splitlines():
                if line.strip():
                    self.monitor.log(f"[数据处理] {line.strip()}")

    def _on_processing_finished(self, exit_code, _status):
        self.run_processing_btn.setEnabled(True)
        if exit_code == 0:
            self.monitor.update_progress("数据处理完成", 70)
            self.refresh_paths()
            QMessageBox.information(self, "完成", "data_processing.py 执行完成，可生成滤波对比图。")
        else:
            self.monitor.log_error("数据处理脚本失败，请查看日志")
            QMessageBox.critical(self, "错误", "数据处理脚本运行失败")

    def check_file_status(self):
        """检查文件是否存在，更新按钮状态"""
        if self.raw_csv_path and self.filtered_csv_path:
            self.process_btn.setEnabled(True)
            self.monitor.log(f"📁 检测到原始数据：{os.path.basename(self.raw_csv_path)}")
            self.monitor.log(f"📁 检测到最新处理后数据：{os.path.basename(self.filtered_csv_path)}")
        else:
            self.process_btn.setEnabled(False)
            missing = []
            if not self.raw_csv_path:
                missing.append(f"{self.current_dataset} 原始数据（*_Ageing_part*.csv）")
            if not self.filtered_csv_path:
                missing.append(f"{self.current_dataset} 处理后数据（*_processed_*.csv）")
            self.monitor.log_error(f"❌ 缺少必要文件：{', '.join(missing)}")

    def process_data(self):
        """处理数据并显示对比图"""
        try:
            self.refresh_paths()
            if not self.process_btn.isEnabled():
                QMessageBox.warning(self, "缺少文件", "请先运行数据处理脚本或检查数据路径。")
                return

            if not self.filtered_csv_path:
                raise ValueError("未找到处理后CSV路径")

            self.monitor.update_progress("开始生成滤波对比图", 20)
            if not self.raw_csv_path:
                raise ValueError("未找到原始CSV路径")

            self.monitor.log(f"[数据处理] {self.current_dataset} 原始 vs 滤波对比")

            fig = plot_voltage_overlay(
                self.raw_csv_path,
                self.filtered_csv_path,
                dataset_label=self.current_dataset,
                time_col=None,
                voltage_col=None,
            )

            show_figure_in_dialog(self, fig, title=f"{self.current_dataset} 原始 vs 滤波对比")

            self.monitor.log("✅ 滤波对比图生成完成")
            self.monitor.update_progress("数据处理完成", 100)

        except Exception as e:
            self.monitor.log_error(f"数据处理失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"生成滤波对比图失败：{str(e)}")
            self.monitor.update_progress("数据处理失败", 0)


class LifePredictionPage(QWidget):
    """训练触发 + 预测对比展示"""

    def __init__(self, monitor_panel, parent=None):
        super().__init__(parent)
        self.monitor = monitor_panel
        self.prediction_csv_path = self.find_latest_prediction()
        self.pred_canvas = None
        self.train_process: QProcess | None = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title = QLabel("模型训练与预测对比")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #1D2129; margin-bottom: 10px;")
        layout.addWidget(title)

        control_group = QGroupBox("训练任务与预测绘图")
        control_layout = QGridLayout(control_group)
        control_layout.setContentsMargins(20, 20, 20, 20)
        control_layout.setSpacing(12)

        self.run_train_btn = QPushButton("运行模型训练与寿命预测")
        self.run_train_btn.setMinimumHeight(44)
        self.run_train_btn.setStyleSheet("""
            QPushButton { background-color: #00B42A; color: white; border: none; border-radius: 8px;
                          padding: 12px 18px; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #23C343; }
            QPushButton:disabled { background-color: #F5F7FA; color: #C9CDD4; border-color: #E5E6EB; }
        """)
        self.run_train_btn.clicked.connect(self.run_training)
        control_layout.addWidget(self.run_train_btn, 0, 0)

        self.param_label = QLabel(f"wavelet={TRAIN_WAVELET} | window={TRAIN_WINDOW}")
        self.param_label.setStyleSheet("color: #86909C; font-size: 12px;")
        control_layout.addWidget(self.param_label, 0, 1, 1, 2)

        self.plot_fc1_btn = QPushButton("绘制FC1预测对比")
        self.plot_fc1_btn.setMinimumHeight(44)
        self.plot_fc1_btn.setStyleSheet("""
            QPushButton { background-color: #165DFF; color: white; border: none; border-radius: 8px;
                          padding: 12px 18px; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #3A7BFF; }
        """)
        self.plot_fc1_btn.clicked.connect(lambda: self.plot_predictions("FC1", max_points=2000))
        control_layout.addWidget(self.plot_fc1_btn, 1, 0)

        self.plot_fc2_btn = QPushButton("绘制FC2预测对比")
        self.plot_fc2_btn.setMinimumHeight(44)
        self.plot_fc2_btn.setStyleSheet("""
            QPushButton { background-color: #FF7D00; color: white; border: none; border-radius: 8px;
                          padding: 12px 18px; font-size: 15px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #FF9A2E; }
        """)
        # 使用足够大的 max_points 覆盖全时间轴和未来滚动预测
        self.plot_fc2_btn.clicked.connect(lambda: self.plot_predictions("FC2", max_points=200000))
        control_layout.addWidget(self.plot_fc2_btn, 1, 1)

        self.clean_btn = QPushButton("清理临时文件")
        self.clean_btn.setMinimumHeight(44)
        self.clean_btn.setStyleSheet("""
            QPushButton { background-color: #86909C; color: white; border: none; border-radius: 8px;
                          padding: 12px 18px; font-size: 14px; font-weight: 600; letter-spacing: 0.2px; }
            QPushButton:hover { background-color: #A8ABB2; }
        """)
        self.clean_btn.clicked.connect(self.clean_generated_files)
        control_layout.addWidget(self.clean_btn, 1, 2)

        layout.addWidget(control_group)

        chart_group = QGroupBox("预测值与真实曲线")
        chart_layout = QVBoxLayout(chart_group)
        chart_layout.setContentsMargins(20, 20, 20, 20)
        chart_layout.setSpacing(15)

        self.pred_tip_label = QLabel("点击绘制按钮展示预测对比图")
        self.pred_tip_label.setStyleSheet("background-color: #F5F7FA; border-radius: 8px; border: 1px dashed #E5E6EB;")
        self.pred_tip_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_tip_label.setMinimumHeight(180)
        chart_layout.addWidget(self.pred_tip_label)

        layout.addWidget(chart_group)

    def clean_generated_files(self):
        reply = QMessageBox.question(
            self,
            "清理确认",
            "将删除处理/训练生成的CSV、NPZ、报告和图像，不会删除原始数据。是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        removed = 0
        for directory, patterns in CLEAN_TARGETS:
            if not os.path.isdir(directory):
                continue
            for pattern in patterns:
                for path in glob.glob(os.path.join(directory, pattern), recursive=True):
                    try:
                        if os.path.isfile(path):
                            os.remove(path)
                            removed += 1
                        elif os.path.isdir(path) and pattern.startswith("**/"):
                            # 仅当递归匹配到空目录时清理
                            shutil.rmtree(path, ignore_errors=True)
                    except Exception as e:
                        self.monitor.log_error(f"删除失败: {path} -> {e}")

        self.monitor.log(f"🧹 已清理生成文件: {removed} 个")
        QMessageBox.information(self, "完成", f"已清理生成文件 {removed} 个。")

    def find_latest_prediction(self):
        return find_latest_file(PREDICTION_PATTERNS)

    def run_training(self):
        if self.train_process and self.train_process.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.information(self, "任务进行中", "训练已在运行，请稍后。")
            return

        script_path = os.path.join(BASE_DIR, "train.py")
        if not os.path.exists(script_path):
            QMessageBox.warning(self, "缺少脚本", "未找到 train.py，请确认与可执行文件同目录或使用源码环境运行。")
            self.monitor.log_error("未找到 train.py，已取消执行")
            return

        self.run_train_btn.setEnabled(False)
        self.monitor.update_progress("启动训练", 5)
        self.monitor.log("🚀 已启动训练脚本 train.py")

        if getattr(sys, "frozen", False):
            def _on_finish():
                QTimer.singleShot(0, lambda: self._on_train_finished(0, None))
            def _on_err(exc):
                QTimer.singleShot(0, lambda: self.monitor.log_error(f"训练失败: {exc}"))
                QTimer.singleShot(0, lambda: self._on_train_finished(1, None))
            _run_script_in_thread(script_path, on_finish=_on_finish, on_error=_on_err)
        else:
            self.train_process = QProcess(self)
            self.train_process.setProgram(sys.executable)
            self.train_process.setArguments(_script_args(script_path))
            self.train_process.setWorkingDirectory(BASE_DIR)
            self.train_process.readyReadStandardOutput.connect(self._on_train_output)
            self.train_process.readyReadStandardError.connect(self._on_train_output)
            self.train_process.finished.connect(self._on_train_finished)
            self.train_process.start()

    def _on_train_output(self):
        if not self.train_process:
            return

        text = self.train_process.readAllStandardOutput().data().decode(errors="ignore")
        err_text = self.train_process.readAllStandardError().data().decode(errors="ignore")
        combined = f"{text}{err_text}".strip()
        if combined:
            for line in combined.splitlines():
                if line.strip():
                    self.monitor.log(f"[训练] {line.strip()}")
    def _on_train_finished(self, exit_code, _status):
        self.run_train_btn.setEnabled(True)
        if exit_code == 0:
            self.monitor.update_progress("训练完成", 90)
            self.prediction_csv_path = self.find_latest_prediction()
            QMessageBox.information(self, "完成", "训练已完成，预测结果已保存。可点击绘制按钮查看。")
        else:
            self.monitor.log_error("训练失败，请查看日志")
            QMessageBox.critical(self, "错误", "训练脚本运行失败")

    def plot_predictions(self, dataset_label: str, max_points: int):
        try:
            if not self.prediction_csv_path or not os.path.exists(self.prediction_csv_path):
                self.prediction_csv_path = self.find_latest_prediction()
            if not self.prediction_csv_path:
                raise FileNotFoundError("未找到预测结果，请先运行训练")

            fig = plot_prediction_vs_true(self.prediction_csv_path, max_points=max_points, dataset_label=dataset_label)
            show_figure_in_dialog(self, fig, title=f"{dataset_label} 预测 vs 真实")

            # 同步展示最新的评估指标表格
            metrics_path = find_latest_file(METRICS_PATTERNS)
            if metrics_path and os.path.exists(metrics_path):
                mfig = plot_metrics_table(metrics_path)
                show_figure_in_dialog(self, mfig, title="评估指标表格")

            self.monitor.log(f"✅ {dataset_label} 预测对比图生成完成")
            self.monitor.update_progress("预测图已刷新", 100)
        except Exception as e:
            self.monitor.log_error(f"生成预测对比失败：{str(e)}")
            QMessageBox.critical(self, "错误", f"生成预测对比失败：{str(e)}")