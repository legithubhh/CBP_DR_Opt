import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ===================== 用户可修改区域 =====================
# 需要对比的文件列表（可自由增删）
file_list = [
    "spre11P10MIN.txt",
    "spre11K10MIN.txt",
    # "spre_newdata.txt"  # 添加更多文件示例
]

# 对应的图例名称（与文件列表顺序一致）
legend_labels = [
    "叶型库11-PCA最佳优化结果",
    "叶型库11-KPCA最佳优化结果",
    # "New Case"
]

num_profiles = len(legend_labels)

def normalize_x(x):
    """将X坐标归一化到[0,1]范围"""
    x_min = np.min(x)
    x_max = np.max(x)
    return (x - x_min) / (x_max - x_min)  # 线性归一化

# 绘图样式配置
plot_config = {
    "figure_size": (15, 6),  # 图像尺寸（宽，高）
    "title": "叶型表面静压分布对比",  # 图表标题
    "xlabel": "归一化X坐标",  # 修改X轴标签说明  # 修改点1
    "ylabel": "静压值",  # Y轴标签
    "grid_style": "--",  # 网格线样式
    "dpi": 600,  # 输出分辨率
    "equal_axis": False  # 是否等比例坐标轴（叶型坐标建议开启）
}
# ========================================================

# 初始化画布
plt.figure(figsize=plot_config["figure_size"])
plt.title(plot_config["title"], fontsize=18, pad=20)

# 颜色序列
# 生成HSV色环中的等间距颜色（适合大量类别）
hues = np.linspace(0, 1, num_profiles, endpoint=False)  # 色相均匀分布
saturation = 0.9  # 高饱和度
value = 0.8  # 高亮度
colors = np.array([
    [hue, saturation, value] for hue in hues
])
colors = mcolors.hsv_to_rgb(colors)  # 转换为RGB

# 循环读取并绘制数据
for idx, (filename, label) in enumerate(zip(file_list, legend_labels)):
    try:
        # 读取数据文件
        data = np.loadtxt(filename)
        x_original = data[:, 0]
        y = data[:, 1]

        # 关键修改：归一化X坐标  # 修改点2
        x_normalized = normalize_x(x_original)

        # 绘制曲线（颜色自动分配，线宽2pt，实线）
        plt.plot(x_normalized, y,
                 color=colors[idx],
                 linewidth=2,
                 label=label,
                 marker='' if len(x_normalized) > 50 else 'o',  # 数据点少时显示标记
                 markersize=4 if len(x_normalized) > 50 else 6)

    except Exception as e:
        print(f"错误：文件 {filename} 读取失败 - {str(e)}")
        continue

# 坐标轴设置
ax = plt.gca()
if plot_config["equal_axis"]:
    ax.set_aspect('equal', adjustable='box')  # 等比例坐标轴
plt.xlabel(plot_config["xlabel"], fontsize=14)
plt.ylabel(plot_config["ylabel"], fontsize=14)

# 刻度标签设置
plt.tick_params(axis='both', which='major', labelsize=14)
plt.grid(True, linestyle=plot_config["grid_style"], alpha=0.6)

# 图例设置（自动检测最佳位置）
plt.legend(
    loc='upper right' if len(file_list) < 5 else 'best',
    fontsize=18,
    framealpha=0.9,
    prop={'weight': 'bold', 'size': 16}  # 关键修改：加粗字体
)

# 保存并显示
plt.tight_layout()
plt.savefig("spre_comparison.png", dpi=plot_config["dpi"], bbox_inches='tight')
plt.show()
print("可视化完成，结果已保存至 spre_comparison.png")