import numpy as np

# 1. 替换为你的 .npz 文件路径
file_path = "processed_results/FC1/FC1_processed_data.npz"  # 你的文件路径

# 2. 使用 np.load 加载文件（返回 NpzFile 对象，类似字典）
data_npz = np.load(file_path)

# 3. 查看所有键（通过 files 属性）
all_keys = data_npz.files
print("文件中所有的键：", all_keys)