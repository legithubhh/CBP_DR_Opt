import numpy as np
import matplotlib.pyplot as plt

# 读取并解析数据文件
filename = 'flux1.dat'
data = []

with open(filename, 'r') as file:
    for line in file:
        # 将每一行按空格分割，并转换成浮点数列表
        numbers_str = line.split()
        numbers_float = [float(x) for x in numbers_str]
        data.append(numbers_float)

# 转换成numpy数组以便操作
data = np.array(data)

# 创建一个包含子图的画布，这里我们假设每个图表都是单独一行的数据
fig, axs = plt.subplots(len(data), figsize=(10, 8 * len(data)))

for i, row in enumerate(data):
    # 假定x和y坐标交替出现，因此我们先提取所有的x坐标，再提取所有的y坐标
    x_coords = row[0::2]  # 开始:结束:步长 - 这里是从第一个元素开始每隔一个取值（即取x坐标）
    y_coords = row[1::2]  # 从第二个元素开始每隔一个取值（即取y坐标）

    # 如果只有一个子图，则 axs 不会是一个列表而是一个单一的 AxesSubplot 对象
    if len(data) == 1:
        ax = axs
    else:
        ax = axs[i]

    # 使用散点图(scatter plot)来绘制数据点
    ax.scatter(x_coords, y_coords)

    # 添加图形属性
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title(f'Scatter Plot of Curve {i + 1} from Data File')

# 调整子图之间的距离以避免标签重叠
plt.tight_layout()

# 显示所有图表
plt.show()