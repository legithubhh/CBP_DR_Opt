import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def plot_airfoil_profiles(data_file):
    """可视化叶型库数据"""
    try:
        # 设置中文字体（根据系统实际情况选择可用字体）
        plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # 读取数据文件
        with open(data_file, 'r') as f:
            header = f.readline().strip()  # 读取标题行
            data = np.loadtxt(f)  # 读取数值数据

        # 验证数据维度
        if data.ndim != 2:
            raise ValueError(f"数据维度错误，期望二维数组，实际维度 {data.ndim}")

        # 解析标题名称
        column_names = header.split("    ")  # 使用4空格分隔符
        num_profiles = len(column_names)

        # 准备绘图
        plt.figure(figsize=(15, 6))
        plt.title("优化结果对比", fontsize=18, pad=20)
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')  # 等比例坐标轴

        # 颜色序列
        # 生成HSV色环中的等间距颜色（适合大量类别）
        hues = np.linspace(0, 1, num_profiles, endpoint=False)  # 色相均匀分布
        saturation = 0.9  # 高饱和度
        value = 0.8  # 高亮度
        colors = np.array([
            [hue, saturation, value] for hue in hues
        ])
        colors = mcolors.hsv_to_rgb(colors)  # 转换为RGB

        # 处理每个叶型
        for col_idx in range(num_profiles):
            # 提取当前叶型数据列
            profile_data = data[:, col_idx]

            # 数据重组逻辑
            front_part = profile_data[:202]  # 前202元素
            back_part = profile_data[202:]  # 后202元素

            # 处理后段数据：按点倒序
            back_points = back_part.reshape(-1, 2)[::-1]
            back_processed = back_points.flatten()

            # 合并数据
            combined = np.concatenate([front_part, back_processed])

            # 提取坐标 (X,Y) 对
            x = combined[::2]  # 偶数索引为X
            y = combined[1::2]  # 奇数索引为Y

            # 绘制闭合曲线
            plt.plot(x, y,
                     color=colors[col_idx],
                     linewidth=1.5,
                     alpha=0.8,
                     label=column_names[col_idx])

        # 添加图例和坐标轴
        plt.legend(loc='upper left', fontsize=16)
        plt.xlabel("X Coordinate", fontsize=12)
        plt.ylabel("Y Coordinate", fontsize=12)
        plt.tick_params(axis='both',  # 同时作用于x和y轴
                        which='major',  # 调整主刻度
                        labelsize=16)  # 刻度标签字号设为14
        plt.grid(True, linestyle='--', alpha=0.5)

        # 保存并显示图像
        plt.tight_layout()
        plt.savefig("airfoil_comparison.png", dpi=600)
        plt.show()
        print("可视化完成，结果已保存至 airfoil_comparison.png")

    except Exception as e:
        print(f"可视化过程中发生错误: {str(e)}")


if __name__ == "__main__":
    input_file = "transformed_optleaf_aggregation_11.txt"
    plot_airfoil_profiles(input_file)