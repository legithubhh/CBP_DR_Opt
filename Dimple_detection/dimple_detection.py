from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os

# ================ 字体配置 ================
rcParams['font.sans-serif'] = ['SimHei']  # Windows系统
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 定义目标路径
target_folder = os.path.join("defect_plots", "kuaizhao_zs507")
# 创建文件夹（如果父目录不存在，自动创建）
os.makedirs(target_folder, exist_ok=True)
# ================== 用户配置区域 ==================
LEADING_EDGE_RANGE = (-0.05, 0.05)
TRAILING_EDGE_RANGE = (0.10, 0.25)
ANGLE_THRESHOLD = 90.0#90.0
SMOOTH_WINDOW = 7
OUTPUT_DIR = "defect_plots/kuaizhao_zs507"
INTERACTIVE = False  # 新增交互模式开关，True为启用交互，False为自动模式

# ================================================

def calculate_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def load_sample(data, col_idx):
    def split_regions(points, x_range):
        mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        return points[mask]

    suction = data[:202, col_idx].reshape(101, 2)
    pressure = data[202:, col_idx].reshape(101, 2)
    return {
        "suction_LE": split_regions(suction, LEADING_EDGE_RANGE),
        "suction_TE": split_regions(suction, TRAILING_EDGE_RANGE),
        "pressure_LE": split_regions(pressure, LEADING_EDGE_RANGE),
        "pressure_TE": split_regions(pressure, TRAILING_EDGE_RANGE)
    }


def analyze_region(region_points):
    if len(region_points) < 3:
        return False, None, None, []

    smoothed = savgol_filter(region_points, min(SMOOTH_WINDOW, len(region_points)), 3, axis=0)
    defect_indices = []
    for i in range(1, len(smoothed) - 1):
        angle = calculate_angle(smoothed[i - 1], smoothed[i], smoothed[i + 1])
        if angle < ANGLE_THRESHOLD:
            defect_indices.append(i)
    return len(defect_indices) > 0, smoothed[defect_indices], smoothed, defect_indices


def plot_defects(sample_id, region_name, smoothed_points, defect_indices, defects, interactive=True):
    display_indices = []
    for idx in defect_indices:
        start = max(0, idx - 10)
        end = min(len(smoothed_points) - 1, idx + 10)
        display_indices.extend(range(start, end + 1))
    display_indices = sorted(list(set(display_indices)))
    display_points = smoothed_points[display_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(display_points[:, 0], display_points[:, 1], 'b-o', markersize=4, label='局部区域', alpha=0.6)
    if len(defects) > 0:
        plt.scatter(defects[:, 0], defects[:, 1], color='red', s=50, zorder=5,
                    label=f'凹陷点（共{len(defects)}处）')
    plt.title(f"样本 {sample_id} - {region_name}\n检测阈值：{ANGLE_THRESHOLD}°")
    plt.xlabel("X 坐标")
    plt.ylabel("Y 坐标")
    plt.legend()
    plt.grid(True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_path = f"{OUTPUT_DIR}/sample_{sample_id}_{region_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')

    # 根据交互模式决定是否显示图像
    if interactive:
        plt.show()
    else:
        plt.close()
    print(f"保存图像：{save_path}")


def main():
    data = np.loadtxt('kuaizhao_zs507.txt')
    error_list = []

    for col in range(data.shape[1]):
        sample = load_sample(data, col)
        has_defect = False
        col = col + 1

        for region in ['suction_LE', 'suction_TE', 'pressure_LE', 'pressure_TE']:
            is_defect, defects, smoothed, defect_indices = analyze_region(sample[region])
            if is_defect:
                # 传递交互模式参数
                plot_defects(col, region, smoothed, defect_indices, defects, interactive=INTERACTIVE)

                # 根据交互模式进行不同处理
                if INTERACTIVE:
                    response = input(f"确认样本 {col} 的 {region} 区域存在凹陷？(y/n)").strip().lower()
                    if response == 'y':
                        has_defect = True
                else:
                    has_defect = True

        if has_defect:
            error_list.append(col)

        progress = col / data.shape[1] * 100
        print(f"\r处理进度: {progress:.1f}% | {'█' * int(progress // 2)}{' ' * (50 - int(progress // 2))}|", end='')

    if error_list:
        np.savetxt(f"error_order_kuaizhao_zs507.txt", error_list, fmt='%d')
        print(f"\n\n检测完成！发现 {len(error_list)} 个异常样本")
    else:
        print("\n\n未检测到异常样本")


if __name__ == "__main__":
    main()