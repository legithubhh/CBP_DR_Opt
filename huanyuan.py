import numpy as np
from sklearn.preprocessing import StandardScaler

# 标准化过程
data = np.loadtxt('kuaizhao_3_5281.txt').T  # 直接转置避免中间变量
scores = np.loadtxt('pca_result.txt')  # (n_samples, n_components)

# 标准化过程
scaler = StandardScaler()
X = scaler.fit_transform(data)

def reconstruct_data():
    # 加载保存的PCA参数
    mean_vector = np.loadtxt('centre.txt').flatten()  # (404,)
    components = np.loadtxt('flux1.dat')  # (n_components, 404)

    # 验证维度一致性
    assert components.shape[0] == scores.shape[1], "主成分数量不匹配"
    assert components.shape[1] == mean_vector.shape[0], "特征维度不匹配"

    # 执行数据重构
    reconstructed = (scores @ components + mean_vector)  # 矩阵乘法重构 .reshape(-1, 1)
    # 逆标准化过程
    X_reconstructed = scaler.inverse_transform(reconstructed)

    # 保存重构结果
    np.savetxt('reconstructed_data.txt', X_reconstructed.T, fmt='%.8e')
    print(f"重构数据已保存，维度: {X_reconstructed.shape}")
    mse = np.mean(np.abs((data- X_reconstructed)))
    print(f"\n平均绝对重构误差: {mse:.5e}")

def analyze_scores():  # 重命名函数以更准确反映功能
    print("\n主成分得分分析（特征向量系数统计）:")

    for i in range(scores.shape[1]):  # 遍历每个主成分
        comp_scores = scores[:, i]  # 第i个主成分的所有样本得分

        print(f"\n主成分 {i + 1}（共 {scores.shape[0]} 个样本的系数）:")
        print(f"最大系数值: {np.max(comp_scores):.4f}（第{np.argmax(comp_scores) + 1}个样本）")
        print(f"最小系数值: {np.min(comp_scores):.4f}（第{np.argmin(comp_scores) + 1}个样本）")
        print(f"平均绝对值:  {np.mean(np.abs(comp_scores)):.4f}")
    # 将每个主成分的最小值和最大值保存到initial.txt
    min_max_data = []
    for i in range(scores.shape[1]):
        min_val = np.min(scores[:, i])
        max_val = np.max(scores[:, i])
        min_max_data.append([min_val, max_val])

    np.savetxt('initial.txt', min_max_data, fmt='%.4f', delimiter=' ')

if __name__ == "__main__":

    analyze_scores()
    reconstruct_data()
    print("\n重构验证完成！请检查reconstructed_data.txt文件")