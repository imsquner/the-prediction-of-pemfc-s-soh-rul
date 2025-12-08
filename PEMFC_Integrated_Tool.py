import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QStackedWidget)
from PyQt6.QtGui import QFont
from gui.ui_components import NavigationBar, MonitorPanel
from gui.pages import FeatureImportancePage, DataProcessingPage, LifePredictionPage

class MainWindow(QMainWindow):
    """主窗口（整合导航栏、堆叠页面、监控面板）"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PEMFC 综合分析工具")
        self.setMinimumSize(1200, 800)
        self.setup_ui()

    def setup_ui(self):
        # 全局中文字体
        font = QFont("SimHei", 10)
        QApplication.setFont(font)

        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. 左侧导航栏
        self.nav_bar = NavigationBar()
        self.nav_bar.nav_changed.connect(self.switch_page)
        main_layout.addWidget(self.nav_bar)

        # 2. 右侧内容区域（堆叠页面+监控面板）
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        content_layout.setSpacing(20)

        # 堆叠窗口（3个页面）
        self.stacked_widget = QStackedWidget()
        # 创建监控面板（传递给各页面用于日志输出）
        self.monitor_panel = MonitorPanel()
        # 创建3个功能页面
        self.feature_page = FeatureImportancePage(self.monitor_panel)
        self.data_page = DataProcessingPage(self.monitor_panel)
        self.prediction_page = LifePredictionPage(self.monitor_panel)
        # 添加到堆叠窗口
        self.stacked_widget.addWidget(self.feature_page)
        self.stacked_widget.addWidget(self.data_page)
        self.stacked_widget.addWidget(self.prediction_page)

        # 添加到内容布局
        content_layout.addWidget(self.stacked_widget, 1)  # 页面占主要空间
        content_layout.addWidget(self.monitor_panel, 0)   # 监控面板固定高度

        main_layout.addWidget(content_widget, 1)  # 内容区域占主要宽度

    def switch_page(self, index):
        """切换堆叠页面"""
        self.stacked_widget.setCurrentIndex(index)
        page_names = ["参数重要性排序", "原始数据处理", "训练与预测"]
        self.monitor_panel.log(f"🔄 已切换到：{page_names[index]}")

if __name__ == "__main__":
    # 解决Matplotlib中文显示问题
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端，避免UI冲突

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())    