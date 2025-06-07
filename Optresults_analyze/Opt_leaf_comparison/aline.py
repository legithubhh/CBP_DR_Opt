import numpy as np


def transform_airfoils(input_file, output_file):
    """
    执行叶型变换：
    1. 平移所有叶型至首个数据点在原点(0,0)
    2. 对齐所有叶型的位置线到基准叶型
    """
    try:
        # 读取数据文件
        with open(input_file, 'r') as f:
            header = f.readline().strip()
            data = np.loadtxt(f)

        n_rows, n_columns = data.shape
        processed_data = np.zeros_like(data)
        theta0 = None  # 基准角度

        for col in range(n_columns):
            # 当前叶型数据
            raw_points = data[:, col]

            # --- 数据预处理 ---
            points = raw_points.reshape(-1, 2)  # 转换为(N,2)数组

            # --- 平移变换 ---
            first_point = points[0].copy()
            translated = points - first_point

            # --- 计算中心点 ---
            center = np.mean(translated, axis=0)

            # --- 计算角度 ---
            theta = np.arctan2(center[1], center[0])

            # --- 记录基准角度 ---
            if col == 0:
                theta0 = theta
                rotated = translated  # 基准叶型不旋转
            else:
                # --- 旋转变换 ---
                delta_theta = theta0 - theta
                cos_t = np.cos(delta_theta)
                sin_t = np.sin(delta_theta)
                rotation_matrix = np.array([[cos_t, -sin_t],
                                            [sin_t, cos_t]])
                rotated = np.dot(translated, rotation_matrix.T)

            # --- 数据后处理 ---
            processed_data[:, col] = rotated.reshape(-1)

        # --- 保存结果 ---
        np.savetxt(output_file, processed_data,
                   header=header,
                   fmt="%.18e",
                   delimiter="    ",
                   comments='')

        print(f"变换完成，结果已保存至 {output_file}")
        print(f"各叶型状态：")
        print(f"- 基准叶型角度：{np.degrees(theta0):.2f}°")
        print(f"- 共处理 {n_columns} 个叶型")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


def validate_transformation(output_file):
    """验证变换结果"""
    with open(output_file, 'r') as f:
        header = f.readline().strip()
        data = np.loadtxt(f)

    # 验证第一个点是否为原点
    for col in range(data.shape[1]):
        first_point = data[0, col], data[1, col]
        assert np.allclose(first_point, [0, 0]), f"叶型 {col + 1} 首点未对齐原点"

    # 验证基准角度一致性
    base_center = np.mean(data[:, 0].reshape(-1, 2), axis=0)
    base_angle = np.arctan2(base_center[1], base_center[0])

    for col in range(1, data.shape[1]):
        points = data[:, col].reshape(-1, 2)
        center = np.mean(points, axis=0)
        current_angle = np.arctan2(center[1], center[0])
        assert np.isclose(current_angle, base_angle, atol=1e-6), \
            f"叶型 {col + 1} 角度不一致"

    print("\n验证通过：")
    print("- 所有叶型首点位于原点")
    print(f"- 位置线角度一致 ({np.degrees(base_angle):.2f}°)")



# 使用示例
if __name__ == "__main__":
    input_path = "optleaf_aggregation.txt"
    output_path = "transformed_optleaf_aggregation.txt"
    transform_airfoils(input_path, output_path)
    # 执行验证
    validate_transformation(output_path)