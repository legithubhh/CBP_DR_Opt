import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler

# 加载原始数据和模型参数
original_data = np.loadtxt('kuaizhao_4_5072.txt').T  # (1000, 404)
kpca_result = np.loadtxt('kpca_result.txt')  # (1000, 5)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(original_data)


kpca = KernelPCA(
    n_components=10,
    kernel='cosine',
    fit_inverse_transform=True  # 必须启用逆变换
)
kpca.fit(X_scaled)  # 重新训练模型

# 执行重构
X_reconstructed_scaled = kpca.inverse_transform(kpca_result)
X_reconstructed = scaler.inverse_transform(X_reconstructed_scaled)

def analyze_scores():  # 重命名函数以更准确反映功能
    scores = kpca_result  # (n_samples, n_components)
    print("\n主成分得分分析（特征向量系数统计）:")

    for i in range(scores.shape[1]):  # 遍历每个主成分
        comp_scores = scores[:, i]  # 第i个主成分的所有样本得分

        print(f"\n主成分 {i + 1}（共 {scores.shape[0]} 个样本的系数）:")
        print(f"最大系数值: {np.max(comp_scores):.4f}（第{np.argmax(comp_scores) + 1}个样本）")
        print(f"最小系数值: {np.min(comp_scores):.4f}（第{np.argmin(comp_scores) + 1}个样本）")
        print(f"平均绝对值:  {np.mean(np.abs(comp_scores)):.4f}")

    # 将每个主成分的最小值和最大值保存到initial.txt
    min_max_data = []
    for i in range(kpca_result.shape[1]):
        min_val = np.min(kpca_result[:, i])
        max_val = np.max(kpca_result[:, i])
        min_max_data.append([min_val, max_val])

    np.savetxt('initial.txt', min_max_data, fmt='%.4f', delimiter=' ')

analyze_scores()

# 计算重构误差
reconstruction_error = np.mean(np.abs(original_data - X_reconstructed))
print(f"平均绝对重构误差：{reconstruction_error:.5e}")

# 保存重构结果
np.savetxt('reconstructed_data_cosine.txt', X_reconstructed.T, fmt='%.6f')  # 保持原始维度格式
print(f"重构数据已保存，维度: {X_reconstructed.T.shape}")
