import numpy as np

# 用户可调整参数
x_shift_range = (-0.1, 0.1)  # X平移范围
x_scale_range = (0.11, 0.21)  # X缩放范围
y_shift_range = (-0.08, 0.08)  # Y平移范围
y_scale_range = (0.05, 0.15)  # Y缩放范围

# 读取原始数据，每列为一个叶型
data = np.loadtxt('kuaizhao_original.txt')

# 确保数据是二维数组（如果是单列则转置）
if len(data.shape) == 1:
    data = data.reshape(-1, 1)

# 对每个叶型（列）进行独立处理
for col_idx in range(data.shape[1]):
    # 生成随机参数
    x_shift = np.random.uniform(*x_shift_range)
    x_scale = np.random.uniform(*x_scale_range)
    y_shift = np.random.uniform(*y_shift_range)
    y_scale = np.random.uniform(*y_scale_range)

    # 处理X坐标（偶数行）
    data[::2, col_idx] = (data[::2, col_idx] + x_shift) * x_scale
    # 处理Y坐标（奇数行）
    data[1::2, col_idx] = (data[1::2, col_idx] + y_shift) * y_scale

# 保存处理后的数据
np.savetxt('kuaizhao.txt', data, fmt='%.9e', delimiter=' ')