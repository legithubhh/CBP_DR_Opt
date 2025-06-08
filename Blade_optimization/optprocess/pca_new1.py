import numpy as np
import subprocess
import os
import shutil
from sklearn.preprocessing import StandardScaler
from dimple_detection_1 import is_defective_blade  # 导入缺陷叶型检测函数

data = np.loadtxt('../pcaprocess_aircompressor/kuaizhao_3_5281.txt').T
# 标准化过程
scaler = StandardScaler()
X = scaler.fit_transform(data)
n_com = 10

def read_geom_ctl(filepath):
    """
    读取 geom_ctl.dat 文件并提取前5行数据。

    参数:
    - filepath: 文件的完整路径
    
    返回:
    - 一个包含5个数据的列表
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            data = [float(line.strip()) for line in lines]  # 转换为浮动数值并去除换行符
        return data
    except FileNotFoundError:
        print(f"错误：无法找到文件 {filepath}")
        return []
    except Exception as e:
        print(f"发生错误: {e}")
        return []


def rearrange_and_save(result, output_file):
    """
    将计算结果（1x404 数据）按照规定的方式写入 `iblade.dat` 文件。

    参数:
    - result: 计算得到的 1x404 数据，形状为 (1, 404)
    - output_file: `iblade.dat` 文件的路径
    """
    try:
        # 读取现有的 blade.dat 文件，保留前21行
        with open(output_file, 'r') as f:
            lines = f.readlines()

        # 保留前21行内容
        first_21_lines = lines[:21]

        # 创建一个空的矩阵，用于存储重排后的数据，形状为(101, 4)
        blade_data = np.zeros((101, 4))

        # 处理前202个数据，交替填充到第一列和第二列
        for i in range(101):
            blade_data[i, 0] = result[0, i * 2]  # x 数据放到第一列
            blade_data[i, 1] = result[0, i * 2 + 1]  # y 数据放到第二列

        # 处理后202个数据，交替填充到第三列和第四列
        for i in range(101, 202):
            blade_data[i - 101, 2] = result[0, i * 2]  # x 数据放到第三列
            blade_data[i - 101, 3] = result[0, i * 2 + 1]  # y 数据放到第四列

        # 将前21行和重排后的数据合并
        final_data = first_21_lines + [' '.join(map(str, row)) + '\n' for row in blade_data]

        # 将最终结果保存到文件
        with open(output_file, 'w') as f:
            f.writelines(final_data)

        # print(f"数据已成功保存到 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")


def matrix_multiply_and_save(vars, data_file, centre_file, output_file, backup_folder):
    """矩阵运算与缺陷检测集成函数"""
    try:
        # 数据加载与校验
        B = np.loadtxt(data_file, dtype=float)
        if B.shape != (n_com, 404):
            raise ValueError(f"数据文件应为({n_com}, 404)，实际为{B.shape}")

        TQ = np.loadtxt(centre_file, dtype=float).reshape(1, 404)
        if TQ.shape != (1, 404):
            raise ValueError(f"中心文件应为(1, 404)，实际为{TQ.shape}")

        # 矩阵运算
        vars = np.array(vars).reshape(1, n_com)
        result = np.dot(vars, B) + TQ

        # 逆标准化
        if not hasattr(scaler, 'mean_'):
            raise RuntimeError("标准化器未初始化")
        result_sd = scaler.inverse_transform(result)
        assert result_sd.shape == (1, 404), "逆变换后形状异常"

        # 缺陷检测
        is_defective = is_defective_blade(result_sd)

        # 根据检测结果决定是否保存
        if not is_defective:
            rearrange_and_save(result_sd, output_file)
            os.makedirs(backup_folder, exist_ok=True)
            shutil.copy(output_file, os.path.join(backup_folder, os.path.basename(output_file)))
        else:
            print(f"检测到缺陷，已阻止保存: {output_file}")

    except Exception as e:
        print(f"新叶型生成和检测过程处理失败: {str(e)}")
        return False  # 异常时返回失败

    return not is_defective  # 正常返回检测结果

# Step 4: 执行 MapS1.exe
def run_maps1():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在目录（pca_ada_fuben目录）
    current_script_dir = os.path.dirname(current_script_path)
    # 构造MAPS1.exe的相对路径（向上退一级到项目根目录，再进入MAPS1目录）
    maps1_path = os.path.join(current_script_dir, "..", "MAPS1", "MAPs1.exe")
    # 规范化路径（解决可能的../和斜杠问题）
    maps1_path = os.path.normpath(maps1_path)

    # 获取MapS1.exe所在目录（用于设置工作目录）
    maps1_dir = os.path.dirname(maps1_path)

    try:
        # 检查文件是否存在
        if not os.path.exists(maps1_path):
            raise FileNotFoundError(f"File {maps1_path} does not exist")

        subprocess.run(
            [maps1_path],
            check=True,
            cwd=maps1_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    except subprocess.CalledProcessError as e:
        return f"Error occurred while running MapS1.exe: {e}"
    except FileNotFoundError as e:
        return f"Error: {maps1_path} not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"


def extract_character_data(character_file):
    """
    提取 `character.dat` 文件中的第二行第二列的数据。
    
    参数:
    - character_file: `character.dat` 文件的路径
    
    返回:
    - 返回第二行第二列的数据
    """
    try:
        # 读取所有行
        with open(character_file, 'r') as f:
            lines = f.readlines()
        
        # 提取第二行第二列的数据
        second_line = lines[1].strip().split()  # 第二行数据
        value = float(second_line[1])  # 第二列的数据
        return value
    except Exception as e:
        print(f"读取 character.dat 文件时发生错误: {e}")
        return None

def main():
    # 第一步：从 geom_ctl.dat 提取数据
    geom_ctl_data = read_geom_ctl("../pca_ada_fuben/geom_ctl.dat")
    if not geom_ctl_data:
        print("读取 geom_ctl.dat 文件失败，程序终止。")
        return

    # 第二步：进行矩阵乘积运算并保存结果
    if_normal = matrix_multiply_and_save(
        vars=geom_ctl_data,
        data_file="../pcaprocess_aircompressor/flux1.dat",
        centre_file="../pcaprocess_aircompressor/centre.txt",
        output_file="../pca_ada_fuben/iblade2D.dat",
        backup_folder="../MAPS1"
    )


    if if_normal == True :
        # 第三步：调用 maps1.exe 可执行文件
        run_maps1()
        # 第四步：提取 character.dat 文件中的第二行第二列数据
        character_data = extract_character_data("../MAPS1/charact.dat")
        if character_data is None:
            print("计算结果为空，将赋1值。")
            character_data = 1.0
        # print(character_data)

        # 第五步：写出计算结果
        with open("../pca_ada_fuben/calculation_status.txt", 'w') as f:
            calculation_status = 0  # 假设 calculation_status 为 0
            f.write(f"{calculation_status:.2f}    {character_data:.5e}\n")
            # print("计算结果已保存到 calculation_status.txt")
    else:
        print("叶型错误，将赋1值。")
        character_data = 1.0
        with open("../pca_ada_fuben/calculation_status.txt", 'w') as f:
            calculation_status = 0  # 假设 calculation_status 为 0
            f.write(f"{calculation_status:.2f}    {character_data:.5e}\n")
            # print("计算结果已保存到 calculation_status.txt")


if __name__ == "__main__":
    main()
