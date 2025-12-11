import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 'Microsoft YaHei' 如果SimHei不可用
plt.rcParams['axes.unicode_minus'] = False

# 数据准备（从表格提取）
data = {
    '参数名称': [
        'gru_hidden_size',
        'gru_num_layers',
        'dropout_rate',
        'bidirectional',
        'sequence_length',
        'input_dim',
        'output_dim',
        'learning_rate',
        'batch_size',
        'epochs',
        'weight_decay'
    ],
    '默认值': [
        '64',
        '2',
        '0.3',
        'True',
        '50',
        '动态（基于特征数）',
        '1',
        '0.001',
        '32',
        '100',
        '0.0001'
    ],
    '作用与意义': [
        '隐藏状态向量的维度，决定模型“记忆容量”。越大，模型能捕捉更复杂模式（如电池多因素退化），但计算量增加。',
        'GRU层的堆叠数量，增加模型深度，能学习更高级时间依赖（如长期电压衰减）。',
        '随机丢弃神经元比例，防止过拟合（模型记住噪声而非模式）。',
        '是否双向GRU：正向学过去，反向学未来，提高上下文理解（如电池衰减的前后因果）。输出维度翻倍（hidden_size * 2）。',
        '输入序列的时间步长，定义“看多远历史”来预测下一步。基于论文滑动窗口。',
        '输入特征维度（如你的7个关键参数：氢气入口温度等）。决定模型输入大小。',
        '输出维度（这里是单值电压预测）。',
        '优化器步长，控制模型学习速度。',
        '每次训练的数据批量大小，影响梯度稳定性和内存。',
        '训练轮数，总迭代次数。',
        'L2正则化，防止权重过大导致过拟合。'
    ],
    '修改建议': [
        '数据复杂时增至128（提升准确性）；简单时减至32（加速训练）。如果过拟合，减小它。',
        '浅任务用1（简单预测）；深任务增至3（但小心过拟合）。你的代码用2，适合PEMFC中等序列。',
        '过拟合时增至0.4-0.5；欠拟合时减至0.1-0.2。仅在多层时有效（你的代码中if num_layers > 1）。',
        '对预测任务建议True；单向序列（如实时监控）设False以节省计算。',
        '根据数据周期调整：短序列（如噪声大）减至30；长依赖增至100。影响内存使用。',
        '自动从数据计算；若添加特征，需相应调整。',
        '固定为1，除非多目标预测。',
        '不稳定时减至0.0005；慢收敛时增至0.005。你的代码用ReduceLROnPlateau调度器自动调整。',
        '内存不足减至16；加速增至64。',
        '结合早停（你的patience=15）避免过度训练。',
        '过拟合时增至0.001。'
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 创建画布和轴
fig, ax = plt.subplots(figsize=(16, 10))  # 调整大小以适应PPT
ax.axis('off')  # 隐藏轴

# 创建表格
table = ax.table(cellText=df.values,
                 colLabels=df.columns,
                 cellLoc='center',
                 loc='center',
                 bbox=[0, 0, 1, 1])

# 调整表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)  # 调整表格缩放

# 设置列宽（根据内容长度优化）
for i, col_width in enumerate([0.15, 0.1, 0.4, 0.35]):
    for j in range(len(df) + 1):  # 包括表头
        table[(j, i)].set_width(col_width)

# 加粗表头并设置边框
for key, cell in table.get_celld().items():
    if key[0] == 0:  # 第一行是表头
        cell.set_text_props(weight='bold')
    cell.set_edgecolor('black')
    cell.set_linewidth(1)

# 自动调整布局
plt.tight_layout()

# 保存为图片（高清，便于PPT导入）
plt.savefig('gru_params_table.png', dpi=300, bbox_inches='tight')
print("图片已保存为 gru_params_table.png")