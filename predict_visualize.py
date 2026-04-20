import os
import numpy as np
import pandas as pd
import torch
import matplotlib

# Use non-interactive backend to avoid Tk/Tcl errors on headless systems
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.transforms as mtransforms
from datetime import datetime
from model import ImprovedGRUModel  # 导入统一模型定义
from data_processing import preprocess_test_data  # 导入数据预处理函数
from scipy.ndimage import uniform_filter1d  # 新增：用于后处理平滑

# 修复字体设置 - 简化字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 新增：预测后处理函数 ====================

def post_process_pred(predictions, times, stable_threshold=1000, output_range=(0.9, 1.1), window_size=20):
    """预测后处理 - 适配800h电压骤降优化"""

    processed_pred = predictions.copy()

    # 对1000h后的预测结果进行处理
    stable_mask = times >= stable_threshold

    if np.any(stable_mask):
        # 1. 截断到合理范围
        stable_pred = processed_pred[stable_mask]
        stable_pred = np.clip(stable_pred, output_range[0], output_range[1])

        # 2. 滑动平均平滑
        if len(stable_pred) >= window_size:
            stable_pred_smoothed = uniform_filter1d(stable_pred, size=window_size, mode='nearest')
        else:
            stable_pred_smoothed = stable_pred

        processed_pred[stable_mask] = stable_pred_smoothed

    print(f"后处理完成: {stable_mask.sum()}个1000h后点已处理")
    return processed_pred


# ==================== 新增：修正效果验证函数 ====================

def validate_correction_effectiveness(output_df):
    """验证修正效果 - 适配800h电压骤降优化"""
    print("\n=== SOH修正效果验证 ===")

    if 'corrected_soh' not in output_df.columns:
        print("警告：未找到修正SOH数据，无法验证修正效果")
        return

    # 分段统计
    segments = [
        ('0-800h', (0, 800)),
        ('800-1000h', (800, 1000)),
        ('1000-1200h', (1000, 1200))
    ]

    for seg_name, (start, end) in segments:
        mask = (output_df['Time_h'] >= start) & (output_df['Time_h'] < end)
        seg_data = output_df[mask]

        if len(seg_data) == 0:
            continue

        original_soh = seg_data['soh_calculated']
        predicted_soh = seg_data['soh_predicted']

        print(f"\n{seg_name}统计:")
        print(f"  原始SOH: 均值={original_soh.mean():.4f}, 方差={original_soh.std():.4f}")

        if len(predicted_soh) > 0:
            mae = np.mean(np.abs(predicted_soh - original_soh))
            rmse = np.sqrt(np.mean((predicted_soh - original_soh) ** 2))
            print(f"  预测MAE: {mae:.4f}, RMSE: {rmse:.4f}")

            # 1000-1200h特别检查
            if seg_name == '1000-1200h':
                in_range = ((predicted_soh >= 0.9) & (predicted_soh <= 1.1)).mean()
                print(f"  预测值在0.9-1.1范围内的比例: {in_range:.1%}")
                print(f"  预测值范围: {predicted_soh.min():.4f} ~ {predicted_soh.max():.4f}")


# ==================== 原有辅助函数定义 ====================

def find_stable_threshold_time(df, threshold, stable_points=5):
    """寻找稳定低于阈值的时间点（连续stable_points个点低于阈值）"""
    if df.empty:
        return None

    soh_values = df['soh_calculated'].values
    time_values = df['Time_h'].values

    # 检查连续stable_points个点是否都低于阈值
    for i in range(len(soh_values) - stable_points + 1):
        if all(soh_values[i:i + stable_points] < threshold):
            # 返回稳定段的最后一个点时间
            return time_values[i + stable_points - 1]

    return None


def find_first_threshold_time(df, threshold):
    """寻找首次低于阈值的时间点（传统方法）"""
    if df.empty:
        return None

    below_threshold = df[df['soh_calculated'] < threshold]
    if len(below_threshold) > 0:
        return below_threshold['Time_h'].iloc[0]
    return None


def print_threshold_analysis(df, thresholds, validation_split_time):
    """输出详细的阈值时间分析（传统方法 vs 优化方法）"""
    print("\n=== SOH阈值时间详细分析 ===")

    for threshold in thresholds:
        threshold_str = f"{int(threshold * 100)}%"

        # 传统方法：首次达标时间
        first_time = find_first_threshold_time(df, threshold)

        # 优化方法：稳定达标时间
        stable_time = find_stable_threshold_time(df, threshold, stable_points=5)

        if first_time and stable_time:
            improvement = stable_time - first_time

            print(f"SOH {threshold_str}:")
            print(f"  传统方法 - 首次达标: {first_time:.1f}h")
            print(f"  优化方法 - 稳定达标: {stable_time:.1f}h")
            print(f"  时间修正: +{improvement:.1f}h")

        elif first_time:
            print(f"SOH {threshold_str}:")
            print(f"  传统方法 - 首次达标: {first_time:.1f}h")
            print("  优化方法 - 未找到稳定达标点")
        else:
            print(f"SOH {threshold_str}: 未达到阈值")


def enhanced_test_set_analysis(df_test, predictions, actual_times):
    """分段时间窗口的详细测试集分析（800-1200h分段）"""
    print("\n=== 测试集详细分析 (段间统计) ===")
    time_segments = [(800, 900), (900, 1000), (1000, 1100), (1100, 1200)]
    for start, end in time_segments:
        mask = (actual_times >= start) & (actual_times < end)
        cnt = np.sum(mask)
        if cnt > 0:
            seg_pred = predictions[mask]
            # df_test may be None here; assume caller computes truth separately if needed
            print(f"{start}-{end}h: 样本数={cnt}, Pred mean={np.mean(seg_pred):.6f}, std={np.std(seg_pred):.6f}")


def generate_test_set_visualizations(df_test, predictions, file_name, out_png):
    """生成测试集可视化并保存到 out_png"""
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Test Set SOH Prediction - {file_name}')
        times = df_test['time'].values if 'time' in df_test.columns else np.arange(len(predictions))
        truth = df_test['soh_calculated'].values[
            :len(predictions)] if 'soh_calculated' in df_test.columns else np.zeros(len(predictions))

        # 时间序列对比
        axes[0, 0].plot(times, truth, 'b-', label='True')
        axes[0, 0].plot(times, predictions, 'r--', label='Pred')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Time (h)')
        axes[0, 0].set_ylabel('SOH')

        # 误差时间分布
        err = predictions - truth
        axes[0, 1].plot(times, err, 'k-')
        axes[0, 1].axhline(0, color='r')
        axes[0, 1].set_xlabel('Time (h)')
        axes[0, 1].set_ylabel('Error')

        # 误差直方图
        axes[1, 0].hist(err, bins=30, color='skyblue', edgecolor='black')
        axes[1, 0].axvline(err.mean(), color='r', linestyle='--')
        axes[1, 0].set_xlabel('Error')

        # 残差图
        axes[1, 1].scatter(predictions, err, alpha=0.5)
        axes[1, 1].axhline(0, color='r')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Residual')

        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        print(f"Saved test visualization: {out_png}")
    except Exception as e:
        print(f"generate_test_set_visualizations failed: {e}")


# ==================== 动态批处理大小计算 ====================

def calculate_dynamic_batch_size(dataset_size, device, model, sample_tensor=None, safety_factor=0.8):
    """动态计算批处理大小"""
    if torch.cuda.is_available():
        # 获取GPU内存信息
        total_memory = torch.cuda.get_device_properties(device).total_memory
        reserved_memory = torch.cuda.memory_reserved(device)
        available_memory = total_memory - reserved_memory

        # 估算单个样本所需内存
        if sample_tensor is not None:
            sample_memory = sample_tensor.element_size() * sample_tensor.nelement()
            # 考虑模型参数和中间激活的内存开销（经验系数）
            memory_per_sample = sample_memory * 3

            # 计算最大批处理大小
            max_batch_size = int((available_memory * safety_factor) / memory_per_sample)
            max_batch_size = max(1, min(max_batch_size, dataset_size, 256))  # 限制最大256

            print(f"GPU内存: {available_memory / 1024 ** 3:.2f}GB可用, 估算批处理大小: {max_batch_size}")
            return max_batch_size

    # 默认批处理大小
    if dataset_size < 100:
        return dataset_size  # 小数据集使用全量批处理
    else:
        return min(64, dataset_size)  # 默认批处理大小


# ==================== 主预测函数 ====================

def predict_fc1_full_dataset(use_additional_testset: bool = False):
    """在整个FC1数据集上进行预测并生成完整对比图表"""
    os.makedirs("predictions", exist_ok=True)

    print("=" * 60)
    print("FC1 完整数据集预测系统")
    print("=" * 60)

    # 1. 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    try:
        checkpoint = torch.load("best_gru_model_fc1.pth", map_location=device, weights_only=False)

        model = ImprovedGRUModel(
            input_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            output_dim=checkpoint['output_dim']
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("模型加载成功")

        # 获取保存的完整数据集信息
        entire_times = checkpoint['entire_times']
        validation_split_time = checkpoint.get('validation_split_time', 500.0)

        print(f"完整数据集时间范围: {entire_times.min():.1f} - {entire_times.max():.1f} h")
        print(f"验证集分割点: {validation_split_time} h")

    except FileNotFoundError:
        print("错误：未找到模型文件best_gru_model_fc1.pth")
        return
    except Exception as e:
        print(f"模型加载失败：{str(e)}")
        return

    # 2. 加载预处理数据
    try:
        processed_data = np.load("processed_data_fc1.npz", allow_pickle=True)
        X_train_scaled = processed_data['X_train_scaled']  # 完整训练集
        y_train = processed_data['y_train']  # 完整训练集标签
        train_times = processed_data['train_times']  # 完整训练集时间戳
        feature_names = processed_data['feature_names']
        target_names = processed_data['target_names']
        scaler_mean = processed_data.get('scaler_mean', None)
        scaler_scale = processed_data.get('scaler_scale', None)

        print("成功加载FC1完整训练集数据")
        print(f"训练集样本数: {len(y_train)}")
        print(f"时间范围: {train_times.min():.1f} - {train_times.max():.1f} h")
        print(f"特征变量: {feature_names}")
        print(f"目标变量: {target_names}")

    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        return

    # 3. 模型预测 - 使用动态批处理避免内存溢出
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 使用动态批处理进行预测
    batch_size = calculate_dynamic_batch_size(
        len(X_train_tensor), device, model, X_train_tensor[0] if len(X_train_tensor) > 0 else None
    )

    predictions = []
    print(f"开始批处理预测，批大小: {batch_size}")
    model.eval()

    with torch.no_grad():
        for i in range(0, len(X_train_tensor), batch_size):
            end_idx = min(i + batch_size, len(X_train_tensor))
            batch_X = X_train_tensor[i:end_idx]
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)

            # 每10个批次输出一次进度
            if (i // batch_size) % 20 == 0:
                print(f"预测进度: {end_idx}/{len(X_train_tensor)}")

            # 清理GPU缓存以释放内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    predictions = np.concatenate(predictions, axis=0)
    print(f"预测完成：样本数 {len(predictions)}")

    # 新增：应用后处理
    print("\n=== 应用预测后处理 ===")
    predictions_processed = post_process_pred(
        predictions[:, 0],
        train_times[:len(predictions)],
        stable_threshold=1000,
        output_range=(0.9, 1.1),
        window_size=20
    )

    # 调试信息：检查数组长度
    print("\n数组长度检查:")
    print(f"train_times: {len(train_times)}")
    print(f"y_train: {len(y_train)}")
    print(f"predictions: {len(predictions)}")

    # 修复：确保数组长度一致
    min_length = min(len(train_times), len(y_train), len(predictions))
    print(f"使用最小长度: {min_length}")

    # 4. 创建输出CSV - 修复长度不一致问题（仅 SOH）
    output_df = pd.DataFrame({
        'Time_h': train_times[:min_length],
        'soh_calculated': y_train[:min_length, 0],
        'soh_predicted': predictions_processed[:min_length],  # 使用后处理后的预测值
        'soh_error': predictions_processed[:min_length] - y_train[:min_length, 0]
    })

    # 保存CSV文件
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"SOH_Prediction_FC1_FullDataset_{current_date}.csv"
    output_df.to_csv(f"predictions/{output_filename}", index=False, float_format='%.6f')

    print(f"预测结果已保存至: predictions/{output_filename}")
    print(f"文件包含字段: {list(output_df.columns)}")
    print(f"数据量: {len(output_df)}行")

    # 5. 生成完整可视化结果
    generate_full_dataset_visualizations(output_df, validation_split_time)

    # 6. 生成验证集专用分析
    generate_validation_analysis(output_df, validation_split_time)

    # 7. 新增：验证修正效果
    validate_correction_effectiveness(output_df)

    # 可选：处理额外 test 文件夹下的 CSV 文件
    if use_additional_testset:
        try:
            os.makedirs('test', exist_ok=True)
            test_files = [os.path.join('test', f) for f in os.listdir('test') if f.endswith('.csv')]
            os.makedirs('predictions/test_results', exist_ok=True)

            for test_file in test_files:
                try:
                    print(f"\nProcessing additional test file: {test_file}")

                    # 使用统一的测试数据预处理函数
                    X_test_scaled_local, y_test_local, times_local, df_proc = preprocess_test_data(
                        test_file, feature_names, scaler_mean, scaler_scale, window_size=100  # 修改：窗口大小从20改为100
                    )

                    if X_test_scaled_local is None:
                        print(f"跳过 {test_file}: 预处理失败")
                        continue

                    # 预测
                    model.eval()
                    X_tensor_local = torch.FloatTensor(X_test_scaled_local).to(device)

                    # 动态批处理大小
                    batch_size_local = calculate_dynamic_batch_size(
                        len(X_tensor_local), device, model, X_tensor_local[0] if len(X_tensor_local) > 0 else None
                    )

                    preds_local = []
                    with torch.no_grad():
                        for i in range(0, len(X_tensor_local), batch_size_local):
                            p = model(X_tensor_local[i:i + batch_size_local]).cpu().numpy()
                            preds_local.append(p)

                            # 清理GPU缓存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                    preds_local = np.concatenate(preds_local, axis=0)

                    # 新增：应用后处理到测试集预测
                    preds_local_processed = post_process_pred(
                        preds_local[:, 0],
                        times_local[:len(preds_local)],
                        stable_threshold=1000,
                        output_range=(0.9, 1.1),
                        window_size=20
                    )

                    df_out = pd.DataFrame({
                        'Time_h': times_local,
                        'SOH_True': y_test_local[:len(preds_local), 0],
                        'SOH_Predicted': preds_local_processed[:len(preds_local)],
                        'SOH_Error': preds_local_processed[:len(preds_local)] - y_test_local[:len(preds_local), 0]
                    })

                    base_name = os.path.splitext(os.path.basename(test_file))[0]
                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                    csv_out = f'predictions/test_results/{base_name}_predictions_{ts}.csv'
                    png_out = f'predictions/test_results/{base_name}_predictions_{ts}.png'
                    df_out.to_csv(csv_out, index=False, float_format='%.6f')

                    # 分段分析并生成更丰富的可视化
                    try:
                        enhanced_test_set_analysis(df_proc, preds_local_processed, np.asarray(times_local))
                    except Exception as e:
                        print(f"enhanced_test_set_analysis failed: {e}")

                    try:
                        generate_test_set_visualizations(df_proc, preds_local_processed, base_name, png_out)
                    except Exception as e:
                        print(f"generate_test_set_visualizations failed: {e}")

                    print(f"Saved test predictions: {csv_out}, {png_out}")

                except Exception as e:
                    print(f"处理测试文件 {test_file} 失败: {e}")
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            print(f"额外测试集处理失败: {e}")

    return


# ==================== 可视化函数 ====================

def generate_full_dataset_visualizations(output_df, validation_split_time):
    """生成完整数据集的预测对比图表"""

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 修正1: 处理起始点差异 - 由于滑动窗口(window_size=100)，预测值从第101个点开始
    # 将实际SOH数据截取为从第101个数据点开始，与预测值对齐
    if len(output_df) > 100:
        print(f"修正起始点差异: 原始数据长度 {len(output_df)}，截取前100个数据点")
        # 从第101个点开始截取（索引100）
        output_df = output_df.iloc[100:].reset_index(drop=True)
        print(f"修正后数据长度: {len(output_df)}")
    else:
        print("警告: 数据长度不足100，无法进行起始点修正")

    # 创建多个子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FC1 SOH预测结果 - 完整数据集 (物理约束训练+后处理)', fontsize=16, fontweight='bold')

    # 1. SOH预测对比（完整时间段）
    ax1 = axes[0, 0]
    ax1.plot(output_df['Time_h'], output_df['soh_calculated'], 'b-',
             label='计算SOH', alpha=0.8, linewidth=2)
    ax1.plot(output_df['Time_h'], output_df['soh_predicted'], 'r--',
             label='预测SOH', alpha=0.8, linewidth=2)
    ax1.axvline(x=validation_split_time, color='g', linestyle=':',
                label=f'验证集分割点 ({validation_split_time}h)', alpha=0.7)
    ax1.axvline(x=800, color='purple', linestyle='--',
                label='800h电压骤降点', alpha=0.7)  # 新增：标注800h电压骤降点
    ax1.axhline(y=0.97, color='orange', linestyle=':', label='阈值 97%', alpha=0.7)
    ax1.axhline(y=0.96, color='yellow', linestyle=':', label='阈值 96%', alpha=0.7)
    ax1.axhline(y=0.95, color='red', linestyle=':', label='阈值 95%', alpha=0.7)
    ax1.axhline(y=1.1, color='gray', linestyle=':', label='上限 1.1', alpha=0.5)  # 新增：物理约束上限
    ax1.axhline(y=0.9, color='gray', linestyle=':', label='下限 0.9', alpha=0.5)  # 新增：物理约束下限

    # 添加平均误差线
    soh_error = output_df['soh_predicted'] - output_df['soh_calculated']
    avg_error = soh_error.mean()
    ax1.axhline(y=avg_error, color='purple', linestyle='-.',
                label=f'平均误差: {avg_error:.4f}', alpha=0.7)

    ax1.set_xlabel('时间 (h)')
    ax1.set_ylabel('SOH')
    ax1.set_title('SOH预测对比 (物理约束训练+后处理)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2. SOH预测误差
    ax2 = axes[0, 1]
    soh_error = output_df['soh_predicted'] - output_df['soh_calculated']
    ax2.plot(output_df['Time_h'], soh_error, 'k-', alpha=0.7, linewidth=1.5)
    ax2.axhline(y=0, color='r', linestyle='-', alpha=0.5)
    ax2.axhline(y=soh_error.mean(), color='b', linestyle='--',
                label=f'平均误差: {soh_error.mean():.6f}', alpha=0.7)
    ax2.axvline(x=validation_split_time, color='g', linestyle=':', alpha=0.7)
    ax2.axvline(x=800, color='purple', linestyle='--', alpha=0.7)  # 新增：标注800h电压骤降点
    ax2.set_xlabel('时间 (h)')
    ax2.set_ylabel('SOH预测误差')
    ax2.set_title('SOH预测误差趋势')
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3. 误差分布直方图
    ax3 = axes[1, 0]
    ax3.hist(soh_error, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(soh_error.mean(), color='red', linestyle='--',
                label=f'均值: {soh_error.mean():.6f}')
    ax3.axvline(soh_error.mean() + soh_error.std(), color='orange', linestyle=':',
                label=f'±1σ: {soh_error.std():.6f}')
    ax3.axvline(soh_error.mean() - soh_error.std(), color='orange', linestyle=':')
    ax3.set_xlabel('SOH预测误差')
    ax3.set_ylabel('频数')
    ax3.set_title('SOH预测误差分布')
    ax3.legend()
    ax3.grid(alpha=0.3)

    # 4. 预测vs真实值散点图
    ax4 = axes[1, 1]
    ax4.scatter(output_df['soh_calculated'], output_df['soh_predicted'],
                alpha=0.6, s=20, color='blue')
    min_val = min(output_df['soh_calculated'].min(), output_df['soh_predicted'].min())
    max_val = max(output_df['soh_calculated'].max(), output_df['soh_predicted'].max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='理想预测线')

    # 新增：标注物理约束边界
    ax4.axhline(y=1.1, color='gray', linestyle=':', alpha=0.5)
    ax4.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=1.1, color='gray', linestyle=':', alpha=0.5)
    ax4.axvline(x=0.9, color='gray', linestyle=':', alpha=0.5)

    ax4.set_xlabel('计算SOH')
    ax4.set_ylabel('预测SOH')
    ax4.set_title('预测值 vs 真实值 (物理约束)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'predictions/fc1_full_dataset_predictions_{current_date}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"完整数据集可视化结果已保存至: predictions/fc1_full_dataset_predictions_{current_date}.png")

    # 输出统计信息
    print("\n=== 完整数据集预测结果统计 ===")
    print(f"SOH计算范围: {output_df['soh_calculated'].min():.4f} ~ {output_df['soh_calculated'].max():.4f}")
    print(f"SOH预测范围: {output_df['soh_predicted'].min():.4f} ~ {output_df['soh_predicted'].max():.4f}")
    # 已移除多任务统计中的额外目标，仅保留 SOH 统计
    print(f"SOH预测平均绝对误差: {np.mean(np.abs(soh_error)):.6f}")
    print(f"SOH预测均方根误差: {np.sqrt(np.mean(soh_error ** 2)):.6f}")
    print(f"SOH预测平均误差: {soh_error.mean():.6f}")
    print(f"SOH预测误差标准差: {soh_error.std():.6f}")

    # 新增：物理约束合规性检查
    in_range_ratio = np.sum((output_df['soh_predicted'] >= 0.9) & (output_df['soh_predicted'] <= 1.1)) / len(output_df)
    print(f"SOH预测值在物理约束范围内比例: {in_range_ratio:.2%}")


def generate_validation_analysis(output_df, validation_split_time):
    """生成验证集专用分析图表 - 使用稳定判定逻辑"""

    # 修正起始点差异
    if len(output_df) > 100:
        output_df = output_df.iloc[100:].reset_index(drop=True)

    # 分离验证集和后续数据
    validation_mask = output_df['Time_h'] <= validation_split_time
    validation_df = output_df[validation_mask]
    later_df = output_df[~validation_mask]

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建单图表布局
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 1, height_ratios=[3, 1])

    # 主图表区域
    ax = fig.add_subplot(gs[0])

    # 1. SOH预测对比
    ax.plot(validation_df['Time_h'], validation_df['soh_calculated'], 'b-',
            label='验证集实际SOH', alpha=0.9, linewidth=2.5)
    ax.plot(validation_df['Time_h'], validation_df['soh_predicted'], 'r-',
            label='验证集预测SOH', alpha=0.9, linewidth=2.5)

    if len(later_df) > 0:
        ax.plot(later_df['Time_h'], later_df['soh_calculated'], 'b--',
                label='500h后实际SOH', alpha=0.9, linewidth=2.5)
        ax.plot(later_df['Time_h'], later_df['soh_predicted'], 'r--',
                label='500h后预测SOH', alpha=0.9, linewidth=2.5)

    # 添加验证集分割线和SOH阈值线
    ax.axvline(x=validation_split_time, color='g', linestyle=':',
               label=f'验证集分割点 ({validation_split_time}h)', alpha=0.8, linewidth=2)
    ax.axvline(x=800, color='purple', linestyle='--',
               label='800h电压骤降点', alpha=0.8, linewidth=1.5)  # 新增：标注800h电压骤降点
    ax.axhline(y=0.97, color='orange', linestyle=':', label='阈值 97%', alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.96, color='yellow', linestyle=':', label='阈值 96%', alpha=0.8, linewidth=1.5)
    ax.axhline(y=0.95, color='red', linestyle=':', label='阈值 95%', alpha=0.8, linewidth=1.5)
    ax.axhline(y=1.1, color='gray', linestyle=':', label='上限 1.1', alpha=0.5, linewidth=1)  # 新增
    ax.axhline(y=0.9, color='gray', linestyle=':', label='下限 0.9', alpha=0.5, linewidth=1)  # 新增

    # 图表美化
    ax.set_xlabel('时间 (h)', fontsize=12)
    ax.set_ylabel('SOH', fontsize=12)
    ax.set_title('FC1 SOH预测性能分析 (物理约束训练+后处理)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0.89, 1.11)  # 调整y轴范围以显示物理约束边界

    # 2. 计算表格数据 - 使用稳定判定逻辑
    table_data = []
    thresholds = [0.97, 0.96, 0.95]

    for threshold in thresholds:
        threshold_str = f"{int(threshold * 100)}%"

        # 使用稳定判定逻辑：寻找连续5个点低于阈值的位置
        stable_time = find_stable_threshold_time(later_df, threshold, stable_points=5)

        if stable_time is not None:
            time_str = f"{stable_time:.1f}"
            # 对比传统首次达标时间（用于验证优化效果）
            first_time = find_first_threshold_time(later_df, threshold)
            improvement = stable_time - first_time if first_time else 0
            if improvement > 0:
                time_str += f" (+{improvement:.1f})"
        else:
            time_str = "未达到"

        table_data.append([threshold_str, time_str])

    # 3. 创建表格
    table_ax = fig.add_subplot(gs[1])
    table_ax.axis('off')

    # 创建表格
    # 使用 transforms.Bbox.from_bounds 创建 bbox，以满足类型检查并兼容 matplotlib
    bbox = mtransforms.Bbox.from_bounds(0.2, 0, 0.6, 1)
    table = table_ax.table(
        cellText=table_data,
        colLabels=['SOH阈值', '稳定达标时间(h)'],
        cellLoc='center',
        loc='center',
        bbox=bbox
    )

    # 表格样式设置
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # 设置表头样式（基于实际列数）
    ncols = len(table_data[0]) if len(table_data) > 0 else 0
    for i in range(ncols):
        try:
            table[(0, i)].set_facecolor('#e6e6e6')
            table[(0, i)].set_text_props(weight='bold')
        except KeyError:
            # 如果表格结构意外，忽略单元格样式设置以保证鲁棒性
            continue

    # 设置表格边框
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.8)
        cell.set_edgecolor('black')

    plt.tight_layout()
    current_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'predictions/fc1_validation_analysis_{current_date}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"验证集分析图表已保存至: predictions/fc1_validation_analysis_{current_date}.png")

    # 输出详细的阈值时间分析（仅 SOH 时间点）
    print_threshold_analysis(later_df, thresholds, validation_split_time)

    # 输出统计信息 - 分别计算验证集和后续数据的性能
    print("\n=== 验证集性能统计 (0-500h) ===")
    val_error = validation_df['soh_predicted'] - validation_df['soh_calculated']
    print(f"验证集样本数: {len(validation_df)}")
    print(f"SOH计算范围: {validation_df['soh_calculated'].min():.4f} ~ {validation_df['soh_calculated'].max():.4f}")
    print(f"SOH预测范围: {validation_df['soh_predicted'].min():.4f} ~ {validation_df['soh_predicted'].max():.4f}")
    print(f"SOH预测平均绝对误差: {np.mean(np.abs(val_error)):.6f}")
    print(f"SOH预测均方根误差: {np.sqrt(np.mean(val_error ** 2)):.6f}")
    print(f"SOH预测平均误差: {val_error.mean():.6f}")
    print(f"SOH预测误差标准差: {val_error.std():.6f}")

    if len(later_df) > 0:
        print("\n=== 500h后拟合性能统计 ===")
        later_error = later_df['soh_predicted'] - later_df['soh_calculated']
        print(f"500h后样本数: {len(later_df)}")
        print(f"SOH计算范围: {later_df['soh_calculated'].min():.4f} ~ {later_df['soh_calculated'].max():.4f}")
        print(f"SOH预测范围: {later_df['soh_predicted'].min():.4f} ~ {later_df['soh_predicted'].max():.4f}")
        print(f"SOH预测平均绝对误差: {np.mean(np.abs(later_error)):.6f}")
        print(f"SOH预测均方根误差: {np.sqrt(np.mean(later_error ** 2)):.6f}")
        print(f"SOH预测平均误差: {later_error.mean():.6f}")
        print(f"SOH预测误差标准差: {later_error.std():.6f}")

        # 新增：物理约束合规性检查
        in_range_ratio = np.sum((later_df['soh_predicted'] >= 0.9) & (later_df['soh_predicted'] <= 1.1)) / len(later_df)
        print(f"SOH预测值在物理约束范围内比例: {in_range_ratio:.2%}")

        # 输出阈值时间信息（表格数据汇总，仅 SOH 稳定时间）
        print("\n=== SOH阈值时间汇总 ===")
        for i, threshold in enumerate(thresholds):
            threshold_str = table_data[i][0]
            time_str = table_data[i][1]
            print(f"SOH {threshold_str}: 稳定达标时间 = {time_str}h")

    print("\n=== 全时段性能统计 ===")
    soh_error = output_df['soh_predicted'] - output_df['soh_calculated']
    print(f"全时段样本数: {len(output_df)}")
    print(f"SOH计算范围: {output_df['soh_calculated'].min():.4f} ~ {output_df['soh_calculated'].max():.4f}")
    print(f"SOH预测范围: {output_df['soh_predicted'].min():.4f} ~ {output_df['soh_predicted'].max():.4f}")
    print(f"SOH预测平均绝对误差: {np.mean(np.abs(soh_error)):.6f}")
    print(f"SOH预测均方根误差: {np.sqrt(np.mean(soh_error ** 2)):.6f}")
    print(f"SOH预测平均误差: {soh_error.mean():.6f}")
    print(f"SOH预测误差标准差: {soh_error.std():.6f}")

    # 新增：全时段物理约束合规性检查
    in_range_ratio = np.sum((output_df['soh_predicted'] >= 0.9) & (output_df['soh_predicted'] <= 1.1)) / len(output_df)
    print(f"SOH预测值在物理约束范围内比例: {in_range_ratio:.2%}")


if __name__ == "__main__":
    try:
        predict_fc1_full_dataset()
    except Exception as e:
        print(f"预测过程失败：{str(e)}")
        import traceback

        traceback.print_exc()