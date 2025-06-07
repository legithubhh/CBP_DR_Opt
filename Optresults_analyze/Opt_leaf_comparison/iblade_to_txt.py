import numpy as np


def extract_and_save(input_file, output_file):
    """
    从iblade2D.dat文件中提取数据，按照原格式保存到iblade2D.txt中，格式同centre.txt。

    参数:
    - input_file: 输入的iblade2D.dat文件路径
    - output_file: 输出的txt文件路径
    """
    try:
        # 读取数据部分（跳过前21行文件头）
        blade_data = np.loadtxt(input_file, skiprows=21)

        # 验证数据维度是否为 (101, 4)
        if blade_data.shape != (101, 4):
            raise ValueError(f"数据格式错误，期望 (101,4)，实际维度 {blade_data.shape}")

        # 提取前两列（前202数据点）和后两列（后202数据点）
        front_points = blade_data[:, :2].flatten()  # 形状 (202,)
        rear_points = blade_data[:, 2:].flatten()  # 形状 (202,)

        # 合并数据并验证总长度
        combined_data = np.concatenate([front_points, rear_points])
        if len(combined_data) != 404:
            raise ValueError(f"数据长度错误，期望 404，实际 {len(combined_data)}")

        # 保存为与centre.txt相同格式的文本文件
        np.savetxt(output_file, combined_data, fmt="%.18e")
        print(f"数据已成功提取并保存至 {output_file}")

    except Exception as e:
        print(f"操作失败: {str(e)}")


if __name__ == "__main__":
    input_path = "iblade2D.dat"
    output_path = "iblade2D_11K10MIN.txt"
    extract_and_save(input_path, output_path)