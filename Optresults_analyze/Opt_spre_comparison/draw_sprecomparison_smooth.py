import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['Arial']  # Windows系统黑体SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 用户可修改区域 =====================
# 需要对比的文件列表（可自由增删）
file_list = [
    "spre10PMIN.txt",
    "spre10KMIN.txt",
    # "spre_newdata.txt"  # 添加更多文件示例
]

# 对应的图例名称（与文件列表顺序一致）
legend_labels = [
    "PCA 10D Optimization Results of Blade Profile Library 10 —— 0.0275",
    "KPCA 10D Optimization Results of Blade Profile Library 10 —— 0.0270",
    # "New Case"
]

num_profiles = len(legend_labels)


def normalize_x(x):
    """将X坐标归一化到[0,1]范围"""
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)  # 线性归一化


# 光顺处理参数配置（减少平滑强度）
smooth_config = {
    "enable_smoothing": True,  # 是否启用光顺处理
    "window_ratio": 0.03,  # 减小窗口比例（减少平滑强度）
    "poly_order": 3  # 使用2阶多项式（减少平滑强度）
}

# 绘图样式配置
plot_config = {
    "figure_size": (15, 6),  # 图像尺寸（宽，高）
    "title": "Surface Static Pressure Distribution Comparison on Blade Profiles",  # 图表标题
    "xlabel": "Normalized X-Location",  # 修改X轴标签说明
    "ylabel": "Static Pressure",  # Y轴标签
    "grid_style": "--",  # 网格线样式
    "dpi": 600,  # 输出分辨率
    "equal_axis": False  # 是否等比例坐标轴
}
# ========================================================

# 初始化画布
plt.figure(figsize=plot_config["figure_size"])
plt.title(plot_config["title"], fontsize=18, pad=20)

# 颜色序列
hues = np.linspace(0, 1, num_profiles, endpoint=False)
saturation = 0.9
value = 0.8
colors = np.array([
    [hue, saturation, value] for hue in hues
])
colors = mcolors.hsv_to_rgb(colors)

# 循环读取并绘制数据
for idx, (filename, label) in enumerate(zip(file_list, legend_labels)):
    try:
        # 读取数据文件
        data = np.loadtxt(filename)
        x_original = data[:, 0]
        y = data[:, 1]

        # 归一化X坐标
        x_normalized = normalize_x(x_original)

        # 光顺处理（减少强度）
        if smooth_config["enable_smoothing"]:
            n_points = len(y)
            # 动态计算窗口大小（确保为奇数）
            window_size = max(5, min(n_points, int(n_points * smooth_config["window_ratio"])))
            if window_size % 2 == 0:
                window_size += 1

            try:
                # 应用Savitzky-Golay滤波器
                y_smooth = savgol_filter(y,
                                         window_length=window_size,
                                         polyorder=smooth_config["poly_order"])
                print(f"文件 {filename} 已光顺处理 (窗口大小={window_size})")
            except Exception as e:
                print(f"警告：{filename} 光顺失败 - {str(e)}")
                y_smooth = y
        else:
            y_smooth = y

        # 绘制曲线
        plt.plot(x_normalized, y_smooth,
                 color=colors[idx],
                 linewidth=2,
                 label=label,
                 marker='' if len(x_normalized) > 50 else 'o',
                 markersize=4 if len(x_normalized) > 50 else 6)

    except Exception as e:
        print(f"错误：文件 {filename} 读取失败 - {str(e)}")
        continue

# 坐标轴设置
ax = plt.gca()
if plot_config["equal_axis"]:
    ax.set_aspect('equal', adjustable='box')
plt.xlabel(plot_config["xlabel"], fontsize=14)
plt.ylabel(plot_config["ylabel"], fontsize=14)

# 刻度标签设置
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle=plot_config["grid_style"], alpha=0.6)

# 修改图例位置到右下角
plt.legend(
    loc='lower right',  # 图例位置修改为右下角
    fontsize=16,
    framealpha=0.9,
    prop={'weight': 'bold', 'size': 14}
)

# 保存并显示
plt.tight_layout()
plt.savefig("spre_comparison_smooth.png", dpi=plot_config["dpi"], bbox_inches='tight')
plt.show()
print("可视化完成，结果已保存至spre_comparison_smooth.png")