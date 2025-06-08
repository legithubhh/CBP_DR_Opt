import numpy as np
import os


def load_and_sample(filename, n_samples, seed=None):
    """
    加载数据文件并随机采样指定数量的列（样本），自动排除错误叶型

    参数：
    filename : 数据文件路径
    n_samples : 需要采样的数量
    seed : 随机种子（可选）

    返回：
    采样后的numpy数组（保持原始行数）
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件 {filename} 不存在")

    data = np.loadtxt(filename)

    # 验证数据有效性
    if data.ndim != 2:
        raise ValueError(f"文件 {filename} 数据维度不正确，应为二维数组")

    total_samples = data.shape[1]

    # 构造错误文件路径
    error_dir = "Dimple_detection"
    error_filename = "error_order_" + os.path.basename(filename)
    error_path = os.path.join(error_dir, error_filename)

    error_set = set()
    # 如果错误文件存在则读取错误索引
    if os.path.exists(error_path):
        try:
            error_indices = np.loadtxt(error_path, dtype=int)
        except ValueError as e:
            raise ValueError(f"错误文件 {error_path} 格式错误，必须为纯数字文件: {e}")

        error_indices = np.atleast_1d(error_indices)

        # 验证索引有效性
        if error_indices.size > 0:
            invalid_mask = (error_indices < 0) | (error_indices >= total_samples)
            if invalid_mask.any():
                invalid_indices = error_indices[invalid_mask]
                raise ValueError(
                    f"错误文件 {error_path} 包含无效索引 {invalid_indices}，"
                    f"有效范围应为 0-{total_samples - 1}"
                )
            error_set = set(error_indices.tolist())
    else:
        print(f" 无{filename}错误叶型数据")

    # 计算有效列
    valid_columns = [col for col in range(total_samples) if col not in error_set]

    # 检查样本数量是否足够
    if len(valid_columns) < n_samples:
        raise ValueError(
            f"文件 {filename} 有效样本不足。需要 {n_samples} 个，"
            f"实际可用 {len(valid_columns)} 个（已排除 {len(error_set)} 个错误样本）"
        )

    # 设置随机种子（如果提供）
    if seed is not None:
        np.random.seed(seed)

    # 从有效列中无重复采样
    selected_cols = np.random.choice(valid_columns, n_samples, replace=False)

    return data[:, selected_cols]

def combine_and_shuffle(sr1_samples, jy_samples, zq_samples,  seed=None):
    """
    合并三个来源的样本并进行随机洗牌

    参数：
    sr_samples : 来自sr文件的样本（numpy数组）
    jy_samples : 来自jy文件的样本
    zq_samples : 来自zq文件的样本
    seed : 随机种子（可选）

    返回：
    合并并洗牌后的完整数据集
    """
    # 横向合并所有样本
    combined = np.hstack((sr1_samples, jy_samples, zq_samples))

    # 验证合并后的形状
    expected_rows = 404
    if combined.shape[0] != expected_rows:
        raise ValueError(f"合并数据行数异常，应为 {expected_rows}，实际为 {combined.shape[0]}")

    # 洗牌列顺序
    if seed is not None:
        np.random.seed(seed)

    shuffled_indices = np.random.permutation(combined.shape[1])
    return combined[:, shuffled_indices]


def save_dataset(data, output_filename):
    """
    保存最终数据集为文本文件

    参数：
    data : 要保存的numpy数组
    output_filename : 输出文件名
    """
    # 验证输出形状
    if data.shape != (404, 1000):
        raise ValueError(f"最终数据集形状异常，应为 (404, 1000)，实际为 {data.shape}")

    # 设置科学计数法格式，保留8位小数
    np.savetxt(output_filename, data, fmt="%.8f")
    print(f"成功保存数据集至 {output_filename}")

if __name__ == "__main__":
    # 设置随机种子（可修改或设置为None）
    RANDOM_SEED = None

    try:
        # 第一步：从各文件加载并采样数据（自动处理错误叶型）
        sr1_data = load_and_sample("kuaizhao_sr319528105.txt", 334, seed=RANDOM_SEED)
        # sr1_data = load_and_sample("kuaizhao_sr427_2_10.txt", 250, seed=RANDOM_SEED)
        jy_data = load_and_sample("kuaizhao_jy528105.txt", 333, seed=RANDOM_SEED)
        zq_data = load_and_sample("kuaizhao_zq528105.txt", 333, seed=RANDOM_SEED)
        # jz_data = load_and_sample("kuaizhao_jz428.txt", 200, seed=RANDOM_SEED)

        # 第二步：合并并洗牌数据
        final_dataset = combine_and_shuffle(sr1_data, jy_data, zq_data, seed=RANDOM_SEED)

        # 第三步：保存最终数据集
        save_dataset(final_dataset, "kuaizhao.txt")

    except Exception as e:
        print(f"程序执行出错: {str(e)}")