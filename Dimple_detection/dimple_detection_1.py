import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm
from datetime import datetime
from scipy.signal import savgol_filter

# ================== 配置参数 ==================
LEADING_EDGE_RANGE = (-0.05, 0.05)  # 前缘检测范围
TRAILING_EDGE_RANGE = (0.10, 0.25)  # 后缘检测范围
SMOOTH_WINDOW = 5  # 平滑窗口
ANGLE_THRESHOLD = 90.0  # 凹陷判定阈值
MIN_INCREASING_POINTS = 18  # 后20点中至少18个点Y值增大
INNER_POINTS_RANGE = (50, 85)  # 拐点检测范围
SECOND_DERIV_THRESHOLD = 0.1  # 二阶导数阈值
MAX_CURVATURE_CHANGE = 1.0  # 最大曲率变化
# ==============================================

# 设置中文字体（优先尝试系统字体）
try:
    font_path = "C:/Windows/Fonts/simhei.ttf"  # Windows系统路径
    if os.path.exists(font_path):
        font_prop = fm.FontProperties(fname=font_path)
        rcParams['font.sans-serif'] = ['SimHei']
    else:  # 尝试Linux/Mac字体
        rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei']
    rcParams['axes.unicode_minus'] = False
except:
    print("字体设置失败，将使用英文显示")


def calculate_angle(p1, p2, p3):
    """计算三点间夹角"""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def load_sample(data, col_idx):
    """加载并分割叶片数据"""

    def split_regions(points, x_range):
        mask = (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1])
        return points[mask]

    suction = data[:202, col_idx].reshape(101, 2)  # 吸力面101点
    pressure = data[202:, col_idx].reshape(101, 2)  # 压力面101点

    return {
        "suction_LE": split_regions(suction, LEADING_EDGE_RANGE),
        "suction_TE": split_regions(suction, TRAILING_EDGE_RANGE),
        "pressure_LE": split_regions(pressure, LEADING_EDGE_RANGE),
        "pressure_TE": split_regions(pressure, TRAILING_EDGE_RANGE),
        "suction_full": suction,  # 完整吸力面
        "pressure_full": pressure  # 完整压力面
    }


def analyze_region(region_points):
    """分析区域是否存在凹陷"""
    if len(region_points) < 3:
        return False, None, None, []

    # 平滑处理
    smoothed = savgol_filter(region_points, min(SMOOTH_WINDOW, len(region_points)), 3, axis=0)

    # 检测凹陷点
    defect_indices = []
    for i in range(1, len(smoothed) - 1):
        angle = calculate_angle(smoothed[i - 1], smoothed[i], smoothed[i + 1])
        if angle < ANGLE_THRESHOLD:
            defect_indices.append(i)

    return len(defect_indices) > 0, smoothed[defect_indices], smoothed, defect_indices


def plot_rising_inflection_analysis(full_points, inflection_points):
    """绘制新的拐点检测分析图"""
    plt.figure(figsize=(10, 6))

    # 原始吸力面曲线
    plt.plot(full_points[:, 0], full_points[:, 1],
             'b-', linewidth=1.5, label='吸力面曲线')

    # 标记最后50个点区域
    last_50 = full_points[-50:]
    plt.plot(last_50[:, 0], last_50[:, 1],
             'g-', linewidth=1.5, alpha=0.7, label='最后50个点')

    # 标记拐点
    for i, ip in enumerate(inflection_points):
        # 绘制拐点
        plt.scatter(ip['point'][0], ip['point'][1],
                    color='red', s=100, marker='o', zorder=5, label='拐点' if i == 0 else None)

        # 绘制前6个点（递减段）
        prev_points = ip['prev_segment']
        plt.plot(prev_points[:, 0], prev_points[:, 1],
                 'm--', linewidth=1.5, alpha=0.7, label='前6点（递减）' if i == 0 else None)

        # 绘制后6个点（递增段）
        next_points = ip['next_segment']
        plt.plot(next_points[:, 0], next_points[:, 1],
                 'c--', linewidth=1.5, alpha=0.7, label='后6点（递增）' if i == 0 else None)

        # 添加信息标注
        info_text = (f"递减点数: {ip['decreasing_count']}/5\n"
                     f"递增点数: {ip['increasing_count']}/5")
        plt.annotate(info_text,
                     xy=(ip['point'][0], ip['point'][1]),
                     xytext=(10, 30),
                     textcoords='offset points',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    plt.title("吸力面拐点检测（新方法）")
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.axis('equal')  # 保持x和y轴的比例一致
    plt.legend()
    plt.grid(True, linestyle=':')

    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"rising_inflection_{timestamp}.png"
    save_path = os.path.join(os.getcwd(), filename)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()

    return save_path


def is_defective_blade(result_sd, auto_save_plot=True):
    """最终版主检测函数（基于新的升点/拐点检测）"""
    # 数据加载
    data = result_sd.T.reshape(404, 1) if result_sd.ndim == 2 else result_sd.reshape(404, 1)
    sample = load_sample(data, col_idx=0)
    full_suction = sample["suction_full"]

    # 1. 检查凹陷缺陷（保持不变）
    for region in ['suction_LE', 'suction_TE', 'pressure_LE', 'pressure_TE']:
        is_defect, *_ = analyze_region(sample[region])
        if is_defect:
            return True, f"凹陷缺陷在{region}区域", None

    # 2. 新的拐点检测方法（基于升点/拐点检测）
    # 获取吸力面最后50个点（索引51-100）
    last_50_points = full_suction[-50:]

    # 寻找升点（检查连续6个点）
    rising_points = []
    for i in range(len(last_50_points) - 5):
        segment = last_50_points[i:i + 6]  # 取6个连续点（形成5次比较）
        increasing_count = 0

        # 检查每个间隔的Y值变化（5次比较）
        for j in range(1, len(segment)):
            if segment[j][1] > segment[j - 1][1]:
                increasing_count += 1

        # 如果5次比较中有至少4次是递增的
        if increasing_count >= 4:
            # 记录升点的全局索引和位置
            global_idx = len(full_suction) - 50 + i
            rising_points.append({
                'global_idx': global_idx,
                'point': segment[0],
                'increasing_count': increasing_count,
                'next_segment': segment  # 保存后6个点用于绘图
            })

    # 在升点中寻找拐点（检查连续6个点）
    inflection_points = []
    for rp in rising_points:
        # 检查该点之前的5个点（形成6个连续点）
        global_idx = rp['global_idx']
        if global_idx < 5:
            continue  # 前面没有足够的点

        prev_segment = full_suction[global_idx - 5:global_idx + 1]  # 取6个连续点（形成5次比较）
        decreasing_count = 0

        # 检查每个间隔的Y值变化（5次比较）
        for j in range(1, len(prev_segment)):
            if prev_segment[j][1] < prev_segment[j - 1][1]:
                decreasing_count += 1

        # 如果5次比较中有至少4次是递减的
        if decreasing_count >= 4:
            inflection_points.append({
                **rp,
                'decreasing_count': decreasing_count,
                'prev_segment': prev_segment  # 保存前6个点用于绘图
            })

    # 如果找到拐点，判定为缺陷叶型
    if inflection_points:
        save_path = None
        if auto_save_plot:
            save_path = plot_rising_inflection_analysis(full_suction, inflection_points)
        return True, f"发现{len(inflection_points)}个拐点", save_path

    return False, "无缺陷", None