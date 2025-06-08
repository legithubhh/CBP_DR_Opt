from scipy.signal import savgol_filter
import numpy as np

# ================== 配置参数 ==================
LEADING_EDGE_RANGE = (-0.05, 0.05)  # 前缘检测范围
TRAILING_EDGE_RANGE = (0.10, 0.25)  # 后缘检测范围
ANGLE_THRESHOLD = 90.0  # 凹陷判定角度阈值
SMOOTH_WINDOW = 7  # 平滑窗口大小


# ==============================================

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

    suction = data[:202, col_idx].reshape(101, 2)  # 吸力面数据
    pressure = data[202:, col_idx].reshape(101, 2)  # 压力面数据

    return {
        "suction_LE": split_regions(suction, LEADING_EDGE_RANGE),
        "suction_TE": split_regions(suction, TRAILING_EDGE_RANGE),
        "pressure_LE": split_regions(pressure, LEADING_EDGE_RANGE),
        "pressure_TE": split_regions(pressure, TRAILING_EDGE_RANGE)
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


def is_defective_blade(result_sd):
    """检测单个叶型是否为缺陷叶型"""
    # 数据形状转换
    data = result_sd.T.reshape(404, 1) if result_sd.ndim == 2 else result_sd.reshape(404, 1)

    # 加载数据
    sample = load_sample(data, col_idx=0)

    # 检查四个关键区域
    for region in ['suction_LE', 'suction_TE', 'pressure_LE', 'pressure_TE']:
        is_defect, *_ = analyze_region(sample[region])
        if is_defect:
            return True
    return False