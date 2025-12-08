# debug_data.py
import numpy as np
import sys


def main():
    data_file = "processed_results/FC1/FC1_processed_20251206_215017.npz"

    try:
        data = np.load(data_file, allow_pickle=True)
        print(f"文件: {data_file}")
        print(f"所有键: {list(data.keys())}")

        for key in data.keys():
            value = data[key]
            print(f"\n键 '{key}':")
            print(f"  类型: {type(value)}")
            if isinstance(value, np.ndarray):
                print(f"  形状: {value.shape}")
                print(f"  数据类型: {value.dtype}")
                if len(value) > 0:
                    if len(value.shape) == 1:
                        print(f"  前5个值: {value[:5]}")
                    elif len(value.shape) == 2:
                        print(f"  前5行:\n{value[:5]}")
            elif isinstance(value, (list, tuple)):
                print(f"  长度: {len(value)}")
                if len(value) > 0:
                    print(f"  前5个元素: {value[:5]}")
    except Exception as e:
        print(f"错误: {e}")


if __name__ == "__main__":
    main()