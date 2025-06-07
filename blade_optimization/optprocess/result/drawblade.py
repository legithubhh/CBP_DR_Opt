import matplotlib.pyplot as plt

# 文件路径
file_path = 'iblade2D_2.dat'

# 初始化吸力面和压力面的坐标列表
suction_side_coords = []
pressure_side_coords = []

# 读取文件并提取数据
with open(file_path, 'r') as file:
    lines = file.readlines()[21:122]  # 提取第22至122行的数据
    for line in lines:
        parts = line.split()  # 按空格分割每行的数据
        if len(parts) == 4:  # 确保每行有四个数据点
            xsuc, ysuc, xpre, ypre = map(float, parts)
            suction_side_coords.append((xsuc, ysuc))
            pressure_side_coords.append((xpre, ypre))

# 分离吸力面和压力面的x和y坐标，以便绘图
suction_xs, suction_ys = zip(*suction_side_coords)
pressure_xs, pressure_ys = zip(*pressure_side_coords)

# 绘制图形作为散点图
plt.figure(figsize=(10, 6))
plt.scatter(suction_xs, suction_ys, label='Suction Side', color='blue')
plt.scatter(pressure_xs, pressure_ys, label='Pressure Side', color='red')
plt.title('Blade Profile')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.legend()
plt.axis('equal')  # 保持x和y轴的比例一致
plt.show()