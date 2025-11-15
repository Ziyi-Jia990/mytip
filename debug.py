import pandas as pd
import torch
import numpy as np

# --- !! 修改这里的路径 !! ---
TABULAR_FILE_PATH = "/mnt/hdd/jiazy/skin-cancer/features/val_features.csv"
LENGTHS_FILE_PATH = "/mnt/hdd/jiazy/skin-cancer/features/tabular_lengths.pt"
# ---

# 加载
lengths = torch.load(LENGTHS_FILE_PATH)
df = pd.read_csv(TABULAR_FILE_PATH, header=None, dtype=np.float32)

print(f"CSV shape: {df.shape}")
print(f"Lengths: {lengths} (Total: {len(lengths)})")

# 检查
errors_found = False
for i in range(len(lengths)):
    field_len = lengths[i]
    if field_len > 1: # 这是一个类别特征
        col_data = df.iloc[:, i]
        min_val = col_data.min()
        max_val = col_data.max()
        
        # 检查值是否为整数
        if not np.all(np.equal(np.mod(col_data, 1), 0)):
            print(f"!! 错误: 第 {i} 列 (类别) 包含非整数浮点数!")
            errors_found = True

        # 检查范围
        if min_val < 0:
            print(f"!! 错误: 第 {i} 列包含负数: {min_val}")
            errors_found = True
        
        if max_val >= field_len:
            print(f"!! 错误: 第 {i} 列的值 ({max_val}) 超出范围 (应 < {field_len})")
            errors_found = True

if not errors_found:
    print("\n--- 检查通过 ---")
    print("CSV 文件内容与 tabular_lengths.pt 匹配。")
    print("这强烈表明问题是正在加载 '_TIP.csv' 文件。")