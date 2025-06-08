import numpy as np
import subprocess
import os
import shutil
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from dimple_detection_1 import is_defective_blade  # 导入缺陷叶型检测函数

# 加载原始数据和模型参数
original_data = np.loadtxt('../kpcaprocess_aircompressor/kuaizhao_4_5143.txt').T  # (1000, 404)
kpca_result = np.loadtxt('../kpcaprocess_aircompressor/kpca_result.txt')  # (1000, 5)

# 重新训练模型（需与原始训练参数一致）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(original_data)
n_c = 5

kpca = KernelPCA(
    n_components=n_c,
    kernel='rbf',
    gamma= 10 / 404,
    fit_inverse_transform=True
)
kpca.fit(X_scaled)


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
            blade_data[i, 0] = result[0, i*2]      # x 数据放到第一列
            blade_data[i, 1] = result[0, i*2 + 1]  # y 数据放到第二列
        
        # 处理后202个数据，交替填充到第三列和第四列
        for i in range(101, 202):
            blade_data[i - 101, 2] = result[0, i*2]      # x 数据放到第三列
            blade_data[i - 101, 3] = result[0, i*2 + 1]  # y 数据放到第四列

        # 将前21行和重排后的数据合并
        final_data = first_21_lines + [' '.join(map(str, row)) + '\n' for row in blade_data]

        # 将最终结果保存到文件
        with open(output_file, 'w') as f:
            f.writelines(final_data)

        #print(f"数据已成功保存到 {output_file}")

    except Exception as e:
        print(f"发生错误: {e}")

def generate_by_weighted_average(alphas):
    """
    :param alphas: 权重列表，长度应等于从中选择的样本数
    :return: 新样本的原始空间表示
    """
    # 可选择变换alphas以优化控制参数。
    normalized_alphas = alphas

    # # 随机选择与alpha数量相同的样本
    # idx = np.random.choice(len(kpca_space), len(normalized_alphas), replace=False)
    # selected_samples = kpca_space[idx]

    # # 直接使用预选样本
    # selected_samples = kpca_space[fixed_indices]
    #
    # # 计算加权平均
    # weighted_avg_kpca = np.dot(normalized_alphas, selected_samples)

    # 逆变换到原始空间
    new_sample_scaled = kpca.inverse_transform(normalized_alphas.reshape(1, -1))
    new_sample = scaler.inverse_transform(new_sample_scaled)
    return new_sample.flatten()

def matrix_multiply_and_save(output_file, backup_folder):
    """
    参数:
    - output_file: 保存结果的文件路径
    """
    try:
        # 读取alpha值
        alphas = np.loadtxt('geom_ctl.dat').flatten()  # 假设geom_ctl.dat仅包含一列alpha值
        assert len(alphas) == n_c, "确保alpha参数个数正确"

        # 使用所有alpha值生成一个新样本
        new_sample = generate_by_weighted_average(alphas).T  # 如果每行代表一组alpha值，则选择一行
        # print(f"Generated sample with given alphas: {new_sample}")
        new_sample = new_sample.reshape(1, 404)
        if new_sample.shape != (1, 404):
            raise ValueError(f"新样本形状不匹配，应该是(1, 404)，但读取到的形状是{new_sample.shape}")

        # 缺陷检测
        is_defective = is_defective_blade(new_sample)

        # 根据检测结果决定是否保存
        if not is_defective:
            rearrange_and_save(new_sample, output_file)
            os.makedirs(backup_folder, exist_ok=True)
            shutil.copy(output_file, os.path.join(backup_folder, os.path.basename(output_file)))
        else:
            print(f"检测到缺陷，已阻止保存: {output_file}")

    except Exception as e:
        print(f"新叶型生成和检测过程处理失败: {str(e)}")
        return False  # 异常时返回失败

    return not is_defective  # 正常返回检测结果

# 执行 MapS1.exe
def run_maps1():
    # 获取当前脚本的绝对路径
    current_script_path = os.path.abspath(__file__)
    # 获取当前脚本所在目录（kpca_ada_fuben目录）
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
    # 第一步：插值法生成新叶型并保存结果
    if_normal = matrix_multiply_and_save(
        output_file="../kpca_ada_fuben/iblade2D.dat",
        backup_folder="../MAPS1"
    )
    if if_normal == True:
        # 第二步：调用 maps1.exe 可执行文件，进行流场计算
        run_maps1()

        # 第三步：提取 character.dat 文件中的第二行第二列数据
        character_data = extract_character_data("../MAPS1/charact.dat")
        if character_data is None:
            print("character.dat读取结果为空，将赋1值")
            character_data = 1.0

        # 第四步：写出计算结果
        with open("../kpca_ada_fuben/calculation_status.txt", 'w') as f:
            calculation_status = 0  # 假设 calculation_status 为 0
            f.write(f"{calculation_status:.2f}    {character_data:.5e}\n")
            #print("计算结果已保存到 calculation_status.txt")
    else:
        print("叶形错误，将赋1值")
        character_data = 1.0
        with open("../kpca_ada_fuben/calculation_status.txt", 'w') as f:
            calculation_status = 0  # 假设 calculation_status 为 0
            f.write(f"{calculation_status:.2f}    {character_data:.5e}\n")
            #print("计算结果已保存到 calculation_status.txt")

if __name__ == "__main__":
    main()
