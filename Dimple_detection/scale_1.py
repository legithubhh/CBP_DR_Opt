import numpy as np


def scale_airfoil_data(airfoil_data):
    """
    处理形状为(1, 404)的叶型数据，缩放X轴长度至0.14

    参数:
    airfoil_data (np.array) -- 形状为(1, 404)的叶型数据

    返回:
    np.array -- 形状为(1, 404)的缩放后数据
    """
    # 验证输入形状
    if airfoil_data.shape != (1, 404):
        raise ValueError(f"输入数组形状应为(1, 404)，实际为{airfoil_data.shape}")

    # 将数据展平为(404,)以简化操作
    flat_data = airfoil_data.flatten()

    # 提取所有X坐标（索引0,2,4,...,402）
    x_coords = flat_data[::2]

    # 计算X轴范围
    x_min = np.min(x_coords)
    x_max = np.max(x_coords)
    x_length = x_max - x_min

    # 检查是否需缩放
    if x_length <= 0.14:
        return airfoil_data  # 直接返回原形状

    # 计算缩放比例
    scale_factor = 0.14 / x_length

    # 创建缩放后数据
    scaled_data = flat_data.copy()

    # 仅缩放X坐标，Y坐标保持不变
    scaled_data[::2] = (x_coords - x_min) * scale_factor + x_min

    # 重塑为原始形状(1, 404)
    return scaled_data.reshape(1, -1)