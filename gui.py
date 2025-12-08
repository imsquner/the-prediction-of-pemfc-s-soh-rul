import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import os


class MLToolkitGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML-Toolkit - 机器学习工具集")
        self.root.geometry("1200x800")
        self.root.configure(bg=COLOR_SCHEME["bg_light"])

        # 存储数据
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None

        self.setup_ui()

    def setup_ui(self):
        """设置界面布局"""
        # 顶部导航栏
        self.setup_top_nav()

        # 主体标签页
        self.setup_main_notebook()

        # 底部状态栏
        self.setup_status_bar()

    def setup_top_nav(self):
        """设置顶部导航"""
        top_frame = tk.Frame(self.root, bg=COLOR_SCHEME["primary"], height=60)
        top_frame.pack(fill=tk.X, padx=0, pady=0)
        top_frame.pack_propagate(False)

        # 标题
        title_label = tk.Label(top_frame, text="ML-Toolkit",
                               font=FONT_CONFIG["title"],
                               fg="white", bg=COLOR_SCHEME["primary"])
        title_label.pack(side=tk.LEFT, padx=20, pady=15)

        # 导航按钮
        nav_frame = tk.Frame(top_frame, bg=COLOR_SCHEME["primary"])
        nav_frame.pack(side=tk.RIGHT, padx=20, pady=15)

        for nav_text in ["文件", "帮助"]:
            nav_btn = tk.Button(nav_frame, text=nav_text,
                                font=FONT_CONFIG["label"],
                                fg="white", bg=COLOR_SCHEME["primary"],
                                relief="flat", cursor="hand2")
            nav_btn.pack(side=tk.LEFT, padx=10)

    def setup_main_notebook(self):
        """设置主标签页"""
        style = ttk.Style()
        style.configure("TNotebook", background=COLOR_SCHEME["bg_light"])
        style.configure("TNotebook.Tab", font=FONT_CONFIG["subtitle"])

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # 标签页1：现有模型预测
        self.tab1 = tk.Frame(self.notebook, bg=COLOR_SCHEME["bg_light"])
        self.notebook.add(self.tab1, text="现有模型预测")

        # 标签页2：训练+预测
        self.tab2 = tk.Frame(self.notebook, bg=COLOR_SCHEME["bg_light"])
        self.notebook.add(self.tab2, text="训练+预测")

        self.setup_tab1()
        self.setup_tab2()

    def setup_tab1(self):
        """设置标签页1：现有模型预测"""
        # 左侧数据预览区
        left_frame = tk.Frame(self.tab1, bg=COLOR_SCHEME["bg_light"])
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # 右侧功能操作区
        right_frame = tk.Frame(self.tab1, bg=COLOR_SCHEME["bg_light"])
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        # 右侧功能模块
        self.setup_model_selection(right_frame)
        self.setup_test_upload(right_frame)
        self.setup_data_preview(right_frame)
        self.setup_prediction_ops(right_frame)
        self.setup_results_display(right_frame)

        # 左侧数据表格
        self.setup_data_table(left_frame)

    def setup_tab2(self):
        """设置标签页2：训练+预测"""
        # 在标签页1基础上扩展训练相关功能
        main_frame = tk.Frame(self.tab2, bg=COLOR_SCHEME["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 数据集上传区
        self.setup_dataset_upload(main_frame)

        # 参数配置区
        self.setup_parameter_config(main_frame)

        # 训练操作区
        self.setup_training_ops(main_frame)

        # 模型保存区
        self.setup_model_save(main_frame)

    def setup_model_selection(self, parent):
        """模型选择区"""
        frame = tk.LabelFrame(parent, text="1. 模型选择",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 模型文件选择
        select_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        select_frame.pack(fill=tk.X, padx=10, pady=10)

        self.model_path_var = tk.StringVar(value="未选择模型文件")

        tk.Button(select_frame, text="选择本地模型文件",
                  command=self.select_model_file,
                  font=FONT_CONFIG["label"],
                  bg=COLOR_SCHEME["primary"], fg="white",
                  relief="flat", cursor="hand2").pack(side=tk.LEFT)

        path_label = tk.Label(select_frame, textvariable=self.model_path_var,
                              font=FONT_CONFIG["small"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_secondary"])
        path_label.pack(side=tk.LEFT, padx=10)

        # 支持格式提示
        format_label = tk.Label(frame, text="支持格式: .pth, .h5, .pkl, .joblib",
                                font=FONT_CONFIG["small"],
                                bg=COLOR_SCHEME["bg_light"],
                                fg=COLOR_SCHEME["text_secondary"])
        format_label.pack(anchor="w", padx=10, pady=(0, 10))

    def setup_test_upload(self, parent):
        """测试集上传区"""
        frame = tk.LabelFrame(parent, text="2. 测试集上传",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 拖拽上传区域
        upload_frame = tk.Frame(frame, bg="#F0F0F0", relief="solid", bd=1, height=80)
        upload_frame.pack(fill=tk.X, padx=10, pady=10)
        upload_frame.pack_propagate(False)

        upload_label = tk.Label(upload_frame,
                                text="拖拽CSV文件至此区域 或 点击上传",
                                font=FONT_CONFIG["label"],
                                bg="#F0F0F0", fg=COLOR_SCHEME["text_secondary"])
        upload_label.pack(expand=True)

        # 绑定点击事件
        upload_frame.bind("<Button-1>", lambda e: self.upload_test_file())
        upload_label.bind("<Button-1>", lambda e: self.upload_test_file())

        # 文件信息显示
        self.file_info_var = tk.StringVar(value="未上传文件")
        info_label = tk.Label(frame, textvariable=self.file_info_var,
                              font=FONT_CONFIG["small"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_secondary"])
        info_label.pack(anchor="w", padx=10, pady=(0, 10))

        # CSV格式提示
        tip_label = tk.Label(frame,
                             text="请确保CSV包含模型所需特征列，无缺失值，编码为UTF-8",
                             font=FONT_CONFIG["small"],
                             bg=COLOR_SCHEME["bg_light"],
                             fg=COLOR_SCHEME["text_secondary"])
        tip_label.pack(anchor="w", padx=10, pady=(0, 10))

    def setup_data_preview(self, parent):
        """数据预览区控制"""
        frame = tk.LabelFrame(parent, text="3. 数据预览控制",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        tk.Button(frame, text="查看全部数据",
                  command=self.show_all_data,
                  font=FONT_CONFIG["label"],
                  bg=COLOR_SCHEME["primary"], fg="white",
                  relief="flat", cursor="hand2").pack(padx=10, pady=10)

    def setup_prediction_ops(self, parent):
        """预测操作区"""
        frame = tk.LabelFrame(parent, text="4. 预测操作",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 预测按钮
        self.predict_btn = tk.Button(frame, text="开始预测",
                                     command=self.start_prediction,
                                     font=FONT_CONFIG["label"],
                                     bg=COLOR_SCHEME["primary"], fg="white",
                                     relief="flat", cursor="hand2",
                                     state="disabled")
        self.predict_btn.pack(padx=10, pady=10)

        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(frame, variable=self.progress_var,
                                            maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.progress_bar.pack_forget()  # 初始隐藏

    def setup_results_display(self, parent):
        """结果展示区"""
        frame = tk.LabelFrame(parent, text="5. 预测结果",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.BOTH, expand=True)

        # 结果控制按钮
        control_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)

        self.export_btn = tk.Button(control_frame, text="导出CSV结果",
                                    command=self.export_results,
                                    font=FONT_CONFIG["label"],
                                    bg=COLOR_SCHEME["success"], fg="white",
                                    relief="flat", cursor="hand2",
                                    state="disabled")
        self.export_btn.pack(side=tk.LEFT)

        # 结果表格将在数据表格区域显示

    def setup_data_table(self, parent):
        """数据表格显示区域"""
        frame = tk.LabelFrame(parent, text="数据预览 / 结果展示",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.BOTH, expand=True)

        # 创建表格框架
        table_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建Treeview作为表格
        columns = ("Time_h", "soh_calculated", "soh_predicted", "soh_error")
        self.data_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=15)

        # 设置列标题
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=100)

        # 滚动条
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.data_tree.yview)
        self.data_tree.configure(yscrollcommand=scrollbar.set)

        self.data_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_dataset_upload(self, parent):
        """训练+预测的数据集上传区"""
        frame = tk.LabelFrame(parent, text="数据集上传",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 训练集和测试集并排
        upload_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        upload_frame.pack(fill=tk.X, padx=10, pady=10)

        # 训练集上传
        train_frame = tk.LabelFrame(upload_frame, text="训练集",
                                    font=FONT_CONFIG["label"],
                                    bg=COLOR_SCHEME["bg_light"])
        train_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        tk.Button(train_frame, text="上传训练集CSV",
                  command=lambda: self.upload_train_file(),
                  font=FONT_CONFIG["label"]).pack(padx=10, pady=10)

        self.train_info_var = tk.StringVar(value="未上传训练集")
        tk.Label(train_frame, textvariable=self.train_info_var,
                 font=FONT_CONFIG["small"]).pack(pady=(0, 10))

        # 测试集上传
        test_frame = tk.LabelFrame(upload_frame, text="测试集",
                                   font=FONT_CONFIG["label"],
                                   bg=COLOR_SCHEME["bg_light"])
        test_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

        tk.Button(test_frame, text="上传测试集CSV",
                  command=lambda: self.upload_test_file(),
                  font=FONT_CONFIG["label"]).pack(padx=10, pady=10)

        self.test_info_var = tk.StringVar(value="未上传测试集")
        tk.Label(test_frame, textvariable=self.test_info_var,
                 font=FONT_CONFIG["small"]).pack(pady=(0, 10))

    def setup_parameter_config(self, parent):
        """参数配置区"""
        frame = tk.LabelFrame(parent, text="模型参数配置",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 基础参数
        basic_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        basic_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(basic_frame, text="基础参数", font=FONT_CONFIG["label"]).pack(anchor="w")

        param_grid = tk.Frame(basic_frame, bg=COLOR_SCHEME["bg_light"])
        param_grid.pack(fill=tk.X, pady=5)

        # 训练轮数
        tk.Label(param_grid, text="训练轮数 (Epoch):", font=FONT_CONFIG["label"]).grid(row=0, column=0, sticky="w",
                                                                                       padx=(0, 10))
        self.epoch_var = tk.StringVar(value="10")
        epoch_entry = tk.Entry(param_grid, textvariable=self.epoch_var, width=10)
        epoch_entry.grid(row=0, column=1, padx=(0, 20))

        # 学习率
        tk.Label(param_grid, text="学习率:", font=FONT_CONFIG["label"]).grid(row=0, column=2, sticky="w", padx=(0, 10))
        self.lr_var = tk.StringVar(value="0.001")
        lr_entry = tk.Entry(param_grid, textvariable=self.lr_var, width=10)
        lr_entry.grid(row=0, column=3, padx=(0, 20))

        # 批次大小
        tk.Label(param_grid, text="批次大小:", font=FONT_CONFIG["label"]).grid(row=0, column=4, sticky="w",
                                                                               padx=(0, 10))
        self.batch_size_var = tk.StringVar(value="32")
        batch_entry = tk.Entry(param_grid, textvariable=self.batch_size_var, width=10)
        batch_entry.grid(row=0, column=5)

        # 高级参数（可折叠）
        self.advanced_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        self.advanced_frame.pack(fill=tk.X, padx=10, pady=10)

        self.advanced_visible = False
        self.toggle_advanced_btn = tk.Button(frame, text="▼ 展开高级参数",
                                             command=self.toggle_advanced_params,
                                             font=FONT_CONFIG["label"],
                                             relief="flat")
        self.toggle_advanced_btn.pack(pady=(0, 10))

        # 初始隐藏高级参数
        self.advanced_frame.pack_forget()

    def setup_training_ops(self, parent):
        """训练操作区"""
        frame = tk.LabelFrame(parent, text="训练操作",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        # 训练按钮
        self.train_btn = tk.Button(frame, text="开始训练",
                                   command=self.start_training,
                                   font=FONT_CONFIG["label"],
                                   bg=COLOR_SCHEME["primary"], fg="white",
                                   relief="flat", cursor="hand2")
        self.train_btn.pack(padx=10, pady=10)

        # 训练进度条
        self.train_progress_var = tk.DoubleVar()
        self.train_progress_bar = ttk.Progressbar(frame, variable=self.train_progress_var,
                                                  maximum=100, mode='determinate')
        self.train_progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        # 训练日志
        log_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        log_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(log_frame, text="训练日志:", font=FONT_CONFIG["label"]).pack(anchor="w")

        self.log_text = tk.Text(log_frame, height=8, font=("Consolas", 10))
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)

        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def setup_model_save(self, parent):
        """模型保存区"""
        frame = tk.LabelFrame(parent, text="模型保存",
                              font=FONT_CONFIG["subtitle"],
                              bg=COLOR_SCHEME["bg_light"],
                              fg=COLOR_SCHEME["text_primary"])
        frame.pack(fill=tk.X, pady=(0, 10))

        save_frame = tk.Frame(frame, bg=COLOR_SCHEME["bg_light"])
        save_frame.pack(fill=tk.X, padx=10, pady=10)

        # 模型命名
        tk.Label(save_frame, text="模型名称:", font=FONT_CONFIG["label"]).pack(side=tk.LEFT)

        default_name = f"trained_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.model_name_var = tk.StringVar(value=default_name)
        name_entry = tk.Entry(save_frame, textvariable=self.model_name_var, width=30)
        name_entry.pack(side=tk.LEFT, padx=10)

        # 保存按钮
        self.save_model_btn = tk.Button(save_frame, text="保存模型",
                                        command=self.save_model,
                                        font=FONT_CONFIG["label"],
                                        bg=COLOR_SCHEME["success"], fg="white",
                                        relief="flat", cursor="hand2",
                                        state="disabled")
        self.save_model_btn.pack(side=tk.LEFT, padx=10)

    def setup_status_bar(self):
        """设置底部状态栏"""
        status_frame = tk.Frame(self.root, bg=COLOR_SCHEME["text_secondary"], height=30)
        status_frame.pack(fill=tk.X, padx=0, pady=0)
        status_frame.pack_propagate(False)

        self.status_var = tk.StringVar(value="就绪")
        status_label = tk.Label(status_frame, textvariable=self.status_var,
                                font=FONT_CONFIG["small"],
                                fg="white", bg=COLOR_SCHEME["text_secondary"])
        status_label.pack(side=tk.LEFT, padx=10, pady=5)

    # 交互方法实现
    def select_model_file(self):
        """选择模型文件"""
        filetypes = [
            ("模型文件", "*.pth *.h5 *.pkl *.joblib"),
            ("PyTorch模型", "*.pth"),
            ("Keras模型", "*.h5"),
            ("Pickle文件", "*.pkl"),
            ("所有文件", "*.*")
        ]

        filename = filedialog.askopenfilename(title="选择模型文件", filetypes=filetypes)
        if filename:
            self.model_path_var.set(filename)
            self.predict_btn.config(state="normal")
            self.update_status(f"已加载模型: {os.path.basename(filename)}")

    def upload_test_file(self):
        """上传测试集文件"""
        filename = filedialog.askopenfilename(title="选择测试集CSV文件", filetypes=[("CSV文件", "*.csv")])
        if filename:
            try:
                self.test_data = pd.read_csv(filename)
                self.file_info_var.set(f"已上传: {os.path.basename(filename)} ({len(self.test_data)}行)")
                self.update_data_preview()
                self.update_status(f"已加载测试集: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("文件读取错误", f"无法读取CSV文件:\n{str(e)}")

    def upload_train_file(self):
        """上传训练集文件"""
        filename = filedialog.askopenfilename(title="选择训练集CSV文件", filetypes=[("CSV文件", "*.csv")])
        if filename:
            try:
                self.train_data = pd.read_csv(filename)
                self.train_info_var.set(f"已上传: {os.path.basename(filename)} ({len(self.train_data)}行)")
                self.update_status(f"已加载训练集: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("文件读取错误", f"无法读取CSV文件:\n{str(e)}")

    def update_data_preview(self):
        """更新数据预览"""
        if self.test_data is not None:
            # 清空现有数据
            for item in self.data_tree.get_children():
                self.data_tree.delete(item)

            # 添加新数据（前10行）
            preview_data = self.test_data.head(10)
            for _, row in preview_data.iterrows():
                self.data_tree.insert("", "end", values=tuple(row))

    def show_all_data(self):
        """显示全部数据弹窗"""
        if self.test_data is None:
            messagebox.showwarning("无数据", "请先上传测试集数据")
            return

        # 创建新窗口显示完整数据
        top = tk.Toplevel(self.root)
        top.title("完整数据预览")
        top.geometry("800x600")

        # 创建表格
        columns = list(self.test_data.columns)
        tree = ttk.Treeview(top, columns=columns, show="headings", height=25)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100)

        # 添加所有数据
        for _, row in self.test_data.iterrows():
            tree.insert("", "end", values=tuple(row))

        # 滚动条
        scrollbar = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def start_prediction(self):
        """开始预测"""
        if self.model_path_var.get() == "未选择模型文件":
            messagebox.showwarning("模型未选择", "请先选择模型文件")
            return

        if self.test_data is None:
            messagebox.showwarning("数据未上传", "请先上传测试集数据")
            return

        # 禁用按钮，显示进度条
        self.predict_btn.config(state="disabled")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(0, 10))

        # 在后台线程中执行预测
        thread = threading.Thread(target=self.run_prediction)
        thread.daemon = True
        thread.start()

    def run_prediction(self):
        """执行预测（模拟）"""
        try:
            # 模拟预测过程
            for i in range(101):
                time.sleep(0.02)  # 模拟处理时间
                self.progress_var.set(i)
                self.root.update_idletasks()

            # 模拟预测结果
            self.predictions = self.test_data.copy()
            # 这里添加实际的预测逻辑

            # 启用导出按钮
            self.export_btn.config(state="normal")
            self.update_status("预测完成")

            messagebox.showinfo("预测完成", "预测过程已完成！")

        except Exception as e:
            messagebox.showerror("预测错误", f"预测过程中发生错误:\n{str(e)}")
        finally:
            self.predict_btn.config(state="normal")

    def export_results(self):
        """导出预测结果"""
        if self.predictions is None:
            messagebox.showwarning("无结果", "请先完成预测")
            return

        filename = filedialog.asksaveasfilename(
            title="保存预测结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv")]
        )

        if filename:
            try:
                self.predictions.to_csv(filename, index=False)
                messagebox.showinfo("导出成功", f"结果已导出至: {filename}")
                self.update_status(f"结果已导出: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("导出错误", f"导出失败:\n{str(e)}")

    def toggle_advanced_params(self):
        """切换高级参数显示"""
        if self.advanced_visible:
            self.advanced_frame.pack_forget()
            self.toggle_advanced_btn.config(text="▼ 展开高级参数")
            self.advanced_visible = False
        else:
            self.advanced_frame.pack(fill=tk.X, padx=10, pady=10)
            self.toggle_advanced_btn.config(text="▲ 收起高级参数")
            self.advanced_visible = True

    def start_training(self):
        """开始训练"""
        if self.train_data is None:
            messagebox.showwarning("训练集未上传", "请先上传训练集数据")
            return

        # 验证参数
        try:
            epochs = int(self.epoch_var.get())
            if not (1 <= epochs <= 100):
                raise ValueError("训练轮数需在1-100之间")

            lr = float(self.lr_var.get())
            if not (0.0001 <= lr <= 0.1):
                raise ValueError("学习率需在0.0001-0.1之间")

            batch_size = int(self.batch_size_var.get())
            if not (8 <= batch_size <= 128):
                raise ValueError("批次大小需在8-128之间")

        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        # 禁用训练按钮，开始训练
        self.train_btn.config(state="disabled")
        self.log_text.delete(1.0, tk.END)

        # 在后台线程中执行训练
        thread = threading.Thread(target=self.run_training)
        thread.daemon = True
        thread.start()

    def run_training(self):
        """执行训练（模拟）"""
        try:
            epochs = int(self.epoch_var.get())

            for epoch in range(epochs):
                # 模拟训练过程
                time.sleep(0.5)

                # 更新进度
                progress = (epoch + 1) / epochs * 100
                self.train_progress_var.set(progress)

                # 添加日志
                loss = 0.5 * (1 - epoch / epochs) + 0.1 * np.random.random()
                acc = 0.8 * (epoch / epochs) + 0.1 * np.random.random()

                log_msg = f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - accuracy: {acc:.4f}\n"
                self.log_text.insert(tk.END, log_msg)
                self.log_text.see(tk.END)
                self.root.update_idletasks()

            # 训练完成
            self.save_model_btn.config(state="normal")
            self.update_status("训练完成")

            messagebox.showinfo("训练完成", "模型训练已完成！")

        except Exception as e:
            messagebox.showerror("训练错误", f"训练过程中发生错误:\n{str(e)}")
        finally:
            self.train_btn.config(state="normal")

    def save_model(self):
        """保存训练后的模型"""
        filename = filedialog.asksaveasfilename(
            title="保存模型",
            defaultextension=".pkl",
            filetypes=[("Pickle文件", "*.pkl"), ("所有文件", "*.*")],
            initialfile=self.model_name_var.get()
        )

        if filename:
            try:
                # 这里添加实际保存模型的代码
                # 模拟保存
                time.sleep(1)
                messagebox.showinfo("保存成功", f"模型已保存至: {filename}")
                self.update_status(f"模型已保存: {os.path.basename(filename)}")
            except Exception as e:
                messagebox.showerror("保存错误", f"模型保存失败:\n{str(e)}")

    def update_status(self, message):
        """更新状态栏"""
        self.status_var.set(message)


# 运行应用
if __name__ == "__main__":
    # 颜色和字体配置
    COLOR_SCHEME = {
        "primary": "#165DFF",
        "bg_light": "#F5F7FA",
        "success": "#00B42A",
        "error": "#F53F3F",
        "warning": "#FF7D00",
        "text_primary": "#1D2129",
        "text_secondary": "#86909C",
        "border": "#E5E6EB"
    }

    FONT_CONFIG = {
        "title": ("Microsoft YaHei", 16, "bold"),
        "subtitle": ("Microsoft YaHei", 14, "bold"),
        "label": ("Microsoft YaHei", 12),
        "small": ("Microsoft YaHei", 10)
    }

    root = tk.Tk()
    app = MLToolkitGUI(root)
    root.mainloop()