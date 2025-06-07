import numpy as np
import matplotlib.pyplot as plt


def read_data(filename):
    """从文件中读取数据并保持原始列结构（404行×1000列）"""
    return np.loadtxt(filename)  # 移除转置以保持原始列结构


def plot_curves(data, a, k, mode='line'):
    """绘制从第a列开始的k条曲线（每列代表一个样本）
    :param mode: 'line'绘制曲线，'scatter'绘制散点图
    """
    if not (0 <= a < data.shape[1] and 0 < k <= data.shape[1] - a):
        raise ValueError("Invalid range for plotting curves.")

    plt.figure(figsize=(10, 6))

    for i in range(k):
        # 获取当前样本的列数据（404个元素）
        column_data = data[:, a + i]

        # 分割前202和后202元素
        front_part = column_data[:202]
        back_part = column_data[202:]

        # 处理后半部分：按点倒序
        back_points = back_part.reshape(-1, 2)[::-1]  # 按点倒序
        back_processed = back_points.flatten()

        # 合并数据
        combined_data = np.concatenate([front_part, back_processed])

        # 提取坐标
        x_coords = combined_data[::2]
        y_coords = combined_data[1::2]

        # 使用随机颜色
        color = np.random.rand(3, )

        # 根据模式选择绘图方式
        if mode == 'scatter':
            plt.scatter(x_coords, y_coords, color=color, label=f'Sample {a + i + 1}')
        else:  # 默认为曲线模式
            plt.plot(x_coords, y_coords, color=color, label=f'Curve {a + i + 1}')
    plt.axis('equal')  # 保持x和y轴的比例一致
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.title(f'Curves from Data File (Mode: {mode})')
    # plt.legend()
    plt.show()


# 参数设置
filename = 'reconstructed_data_cosine.txt'
a = 0  # 起始列索引
k = 1000 # 绘制样本数量
plot_mode = 'line'  # 可选'line'或'scatter'

# 执行
data = read_data(filename)
plot_curves(data, a, k, mode=plot_mode)