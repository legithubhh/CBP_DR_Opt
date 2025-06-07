import numpy as np


def aggregate_leaf_files(file_list, header_names, output_file):
    """
    将指定叶型文件按给定顺序按列合并，并添加标题行

    参数:
    - file_list: 文件路径列表，列顺序与此列表顺序一致
    - header_names: 列标题列表，与file_list一一对应
    - output_file: 输出文件路径
    """
    try:
        # 验证输入参数
        if len(file_list) != len(header_names):
            raise ValueError("文件列表与标题列表长度不一致")
        if len(file_list) == 0:
            raise ValueError("文件列表不能为空")

        # 初始化数据容器
        all_data = []
        expected_length = None

        # 按给定顺序读取文件
        for idx, (file_path, col_name) in enumerate(zip(file_list, header_names)):
            # 读取数据并验证维度
            data = np.loadtxt(file_path)
            if data.ndim != 1:
                raise ValueError(f"文件 {file_path} 不是一维数据 (维度: {data.ndim})")

            # 记录第一个文件的长度作为基准
            if expected_length is None:
                expected_length = len(data)
                print(f"基准数据长度: {expected_length} (来自 {file_path})")

            # 验证数据长度一致性
            if len(data) != expected_length:
                raise ValueError(f"文件 {file_path} 长度不一致: {len(data)} vs {expected_length}")

            all_data.append(data)

        # 按列合并数据
        merged_data = np.column_stack(all_data)

        # 构建标题行（使用4空格分隔）
        header_line = "    ".join(header_names)

        # 保存文件（先写标题行，再写数据）
        with open(output_file, 'w') as f:
            f.write(header_line + "\n")
            np.savetxt(f, merged_data,
                       fmt="%.18e",  # 保持18位小数精度
                       delimiter="    ")  # 4空格分隔

        print(f"\n成功合并 {len(file_list)} 个文件")
        print(f"输出维度: {merged_data.shape} (行×列)")
        print("列标题对应关系:")
        for i, (path, name) in enumerate(zip(file_list, header_names), 1):
            print(f"列 {i}: [{name}] <-- {path}")

    except Exception as e:
        print(f"\n操作失败: {str(e)}")


if __name__ == "__main__":
    # ===== 用户配置区 =====
    input_files = [
        "iblade2D_11P10MIN.txt",
        "iblade2D_11K10MIN.txt",
        # "iblade2D_7KMIN.txt",
        # "iblade2D_7KMAX.txt",
    ]

    column_names = [
        "叶型库11-PCA10维优化结果0.0249",
        "叶型库11-KPCA10维优化结果0.0228",
        # "叶型库9-KPCA最佳优化结果0.0300",
        # "叶型库9-KPCA最差优化结果0.0312",
    ]

    output_path = "optleaf_aggregation.txt"
    # =====================

    # 执行合并操作
    aggregate_leaf_files(input_files, column_names, output_path)