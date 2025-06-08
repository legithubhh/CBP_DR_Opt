import math
import random
import numpy as np
from multiprocessing import Pool, cpu_count
from decimal import Decimal, getcontext
import logging

# 配置精度控制
getcontext().prec = 12
logging.basicConfig(level=logging.INFO, format='%(message)s')


def precise_round(num):
    """高精度舍入函数"""
    return float(Decimal(num).quantize(Decimal('0.000000001')))


def read_leaf_data(filename):
    """读取叶型数据文件 (向量化版本)"""
    data = np.loadtxt(filename, dtype=np.float64)
    return data.T  # 转置为(1000, 404)的数组


def calculate_global_metrics(data):
    """计算全局基准参数"""
    # 提取所有坐标点
    all_x = np.concatenate([leaf[::2] for leaf in data])
    all_y = np.concatenate([leaf[1::2] for leaf in data])

    # 计算全局基准
    global_cx = np.mean(all_x)
    global_cy = np.mean(all_y)
    global_x_span = np.ptp(all_x)
    global_y_span = np.ptp(all_y)

    return global_cx, global_cy, global_x_span, global_y_span


def vector_rotate(points, cx, cy, angle):
    """向量化旋转计算"""
    theta = np.radians(angle)
    x = points[::2] - cx
    y = points[1::2] - cy

    # 向量化旋转计算
    rot_x = x * np.cos(theta) - y * np.sin(theta)
    rot_y = x * np.sin(theta) + y * np.cos(theta)

    # 重组坐标对
    points[::2] = rot_x + cx
    points[1::2] = rot_y + cy
    return points


def transform_leaf(args):
    """单叶型变换的向量化实现"""
    leaf_idx, leaf, cx, cy, x_span, y_span, enable_scale = args
    np.random.seed()  # 确保多进程随机性

    # 生成随机变换参数（基于全局跨度）
    # 生成旋转参数
    angle = np.random.uniform(-3, 3)
    # 生成平移参数
    dx = np.random.choice([-1, 1]) * np.random.uniform(0, 0.03 * x_span)
    dy = np.random.choice([-1, 1]) * np.random.uniform(0, 0.03 * y_span)
    x_scale, y_scale = 1.0, 1.0

    # 生成缩放参数（可选功能）
    if enable_scale:
        x_scale = np.random.uniform(0.97, 1.03)
        y_scale = x_scale

    # 深拷贝避免修改原始数据
    transformed = leaf.copy()

    # 应用缩放（基于全局中心）
    if enable_scale:
        transformed[::2] = (transformed[::2] - cx) * x_scale + cx
        transformed[1::2] = (transformed[1::2] - cy) * y_scale + cy

    # 应用旋转（基于全局中心）
    transformed = vector_rotate(transformed, cx, cy, angle)

    # 应用平移
    transformed[::2] += dx
    transformed[1::2] += dy

    # 记录变换参数
    log_message = f"叶型{leaf_idx:04d}:"
    if enable_scale:
        log_message += f" 缩放X:{x_scale:.4f} Y:{y_scale:.4f} |"
    log_message += f" 旋转{angle:.2f}度 | 平移X:{dx:.6f} Y:{dy:.6f}"
    logging.info(log_message)

    # 应用精度控制
    return np.vectorize(precise_round)(transformed)


def batch_transform(data, enable_scale):
    """多进程批量处理（使用全局参数）"""
    # 预计算全局参数
    global_cx, global_cy, global_x_span, global_y_span = calculate_global_metrics(data)

    with Pool(processes=cpu_count()) as pool:
        args = [(i, leaf, global_cx, global_cy, global_x_span, global_y_span, enable_scale)
                for i, leaf in enumerate(data)]
        results = pool.imap(transform_leaf, args, chunksize=50)
        return np.array(list(results))


def save_data(data, filename):
    """保存处理后的数据"""
    np.savetxt(filename, data.T, fmt='%.9f', delimiter=' ')


if __name__ == "__main__":
    # 读取数据
    all_leaves = read_leaf_data("kuaizhao.txt")

    # 执行变换（启用缩放）
    transformed = batch_transform(all_leaves, enable_scale=True)

    # 保存结果
    save_data(transformed, "kuaizhao_sjz_2_3.txt")
    print("处理完成！结果已保存")