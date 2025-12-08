from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                             QProgressBar, QTextEdit, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class NavigationBar(QWidget):
    """左侧导航栏（确保3个界面正常切换）"""
    nav_changed = pyqtSignal(int)  # 传递选中的页面索引

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_index = 0
        self.setup_ui()

    def setup_ui(self):
        # 固定导航栏宽度（确保不消失）
        self.setFixedWidth(220)
        self.setStyleSheet("background-color: #1A2030; border-right: 1px solid #2A3248;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 20)
        layout.setSpacing(15)

        # 标题
        title = QLabel("PEMFC\n分析工具")
        title.setStyleSheet("""
            font-size: 20px; font-weight: bold; color: white;
            padding: 12px; background-color:#003366;
            border-radius: 8px; margin: 0 15px; text-align: center;
        """)
        title.setWordWrap(True)
        layout.addWidget(title)
        layout.addSpacing(20)

        # 导航按钮（3个界面）
        nav_items = [
            ("参数重要性排序", "📊", 0),
            ("原始数据处理", "🔧", 1),
            ("训练与预测", "🤖", 2)
        ]
        self.buttons = []
        for text, icon, idx in nav_items:
            btn = QPushButton(f"{icon}  {text}")
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent; border: none; border-radius: 8px;
                    color: black; font-size: 14px; font-weight: 600;
                    text-align: left; padding: 15px 20px; margin: 0 15px;
                }
                QPushButton:hover { background-color: rgba(255,255,255,0.05); }
                QPushButton[selected="true"] { background-color: #ffafaf; }
            """)
            btn.setProperty("selected", idx == 0)  # 默认选中第一个
            btn.clicked.connect(lambda _, i=idx: self.select_nav(i))
            self.buttons.append(btn)
            layout.addWidget(btn)

        layout.addStretch()

    def select_nav(self, index):
        """切换导航选中状态"""
        if index == self.current_index:
            return
        # 更新按钮样式
        for i, btn in enumerate(self.buttons):
            btn.setProperty("selected", i == index)
            btn.style().unpolish(btn)
            btn.style().polish(btn)
        self.current_index = index
        self.nav_changed.emit(index)

class MonitorPanel(QWidget):
    """系统监控面板（显示日志和进度）- 日志大小可调节"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logs = []
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        # 下移上缘，让进度条与日志整体更靠下
        layout.setContentsMargins(15, -1000, 15, 15)
        layout.setSpacing(10)

        # 标题
        title = QLabel("系统监控")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #1D2129; margin-bottom: 5px;")
        layout.addWidget(title)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #E5E6EB; border-radius: 4px; height: 20px; text-align: center; }
            QProgressBar::chunk { background-color: #165DFF; border-radius: 3px; }
        """)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        layout.addWidget(self.progress_bar)

        # 日志区域 - 取消最大高度限制，支持自由拉伸
        self.log_text = QTextEdit()
        self.log_text.setStyleSheet("""
            QTextEdit { border: 1px solid #E5E6EB; border-radius: 4px; 
                        font-size: 14px; color: #4E5969; background-color: #F5F7FA;
                        padding: 12px 14px; line-height: 170%; }
        """)
        self.log_text.setReadOnly(True)
        # 略微收缩日志栏高度，上移底端、下移顶端
        self.log_text.setMinimumHeight(200)
        # 移除最大高度限制，允许用户调节大小
        # self.log_text.setMaximumHeight(200)  # 注释掉此行
        layout.addWidget(self.log_text)

        # 分隔线
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setFrameShadow(QFrame.Shadow.Sunken)
        divider.setStyleSheet("color: #E5E6EB; margin: 6px 0;")
        layout.addWidget(divider)

        # 轻量提示标签，分隔不同流程
        self.hint_label = QLabel("提示：运行/可视化/训练的日志将按时间顺序显示，关键步骤已用表情标记。")
        self.hint_label.setStyleSheet("font-size: 12px; color: #86909C; line-height: 150%;")
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        # 初始日志
        self.log("🖥️ 系统启动完成，欢迎使用PEMFC分析工具")
        self.log("📝 请从左侧导航选择功能模块")

    def update_progress(self, message, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{message} ({value}%)")
        self.log(f"📊 {message} - {value}%")

    def log(self, message):
        """添加日志（保留最近50条）"""
        self.logs.append(message)
        if len(self.logs) > 50:
            self.logs.pop(0)
        self.log_text.setPlainText("\n".join(self.logs))
        # 滚动到底部（稳健处理 verticalScrollBar 可能为 None 的情况）
        vbar = self.log_text.verticalScrollBar()
        if vbar is not None:
            vbar.setValue(vbar.maximum())

    def log_error(self, message):
        """添加错误日志（标红）"""
        error_msg = f"❌ {message}"
        self.logs.append(error_msg)
        if len(self.logs) > 50:
            self.logs.pop(0)
        # 标红错误行
        full_text = "\n".join(self.logs)
        self.log_text.setPlainText(full_text)
        # 高亮错误行
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.select(cursor.SelectionType.LineUnderCursor)
        self.log_text.setTextCursor(cursor)
        self.log_text.setStyleSheet("""
            QTextEdit { border: 1px solid #E5E6EB; border-radius: 4px; 
                        font-size: 12px; background-color: #F5F7FA; }
            QTextEdit::cursor { background-color: #F53F3F; }
            QTextEdit::selectedText { background-color: #FFE6E6; color: #F53F3F; }
        """)
        # 恢复滚动（稳健处理 verticalScrollBar 可能为 None 的情况）
        vbar = self.log_text.verticalScrollBar()
        if vbar is not None:
            vbar.setValue(vbar.maximum())