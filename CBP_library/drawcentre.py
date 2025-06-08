import matplotlib.pyplot as plt

# 读取数据文件
with open('初始叶型3.txt', 'r') as f:
    lines = [line.strip() for line in f.readlines()]

# 数据完整性验证
if len(lines) != 404:
    raise ValueError(f"数据行数错误！期望404行，实际得到{len(lines)}行")
if len(lines) % 2 != 0:
    raise ValueError("数据行数必须为偶数，每个点需要x/y坐标各占一行")

# 解析坐标对
x_coords = []
y_coords = []
for i in range(0, len(lines), 2):  # 步长2遍历所有坐标对
    try:
        x = float(lines[i])
        y = float(lines[i + 1])
    except IndexError:
        raise ValueError(f"第{i + 1}行缺少对应的y坐标")
    except ValueError:
        raise ValueError(f"非数值数据存在于第{i + 1}或{i + 2}行")

    x_coords.append(x)
    y_coords.append(y)

suction_xs = x_coords[0:101]
suction_ys = y_coords[0:101]
pressure_xs = x_coords[101:202]
pressure_ys = y_coords[101:202]

# 可视化配置
plt.figure(figsize=(10, 6))
plt.title('The original leaf (202 Points)')
plt.scatter(suction_xs, suction_ys, label='Suction Side', color='blue')
plt.scatter(pressure_xs, pressure_ys, label='Pressure Side', color='red')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.axis('equal')  # 保持x和y轴的比例一致
plt.grid(True, color='gray', linestyle=':', alpha=0.5, zorder=0.1)
plt.gca().set_axisbelow(True)  # 网格线在数据下方
plt.show()