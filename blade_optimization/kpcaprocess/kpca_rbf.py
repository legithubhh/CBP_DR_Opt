import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 读取数据并转置 (1000样本 x 404特征)
data = np.loadtxt('kuaizhao_3_5283.txt')  # 原始形状 (404, 1000)
X = data.T                         # 转置后 (1000, 404)

# 数据标准化（均值为0，方差为1）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置核参数
gamma_rbf = 2.1 / 404  # gamma = 1/n_features = 1/404

# 执行核主成分分析（获取所有成分）
kpca = KernelPCA(
    n_components=None,           # 获取所有主成分
    kernel='rbf',
    gamma=gamma_rbf,
    fit_inverse_transform=True
)

# 拟合模型并转换数据
X_kpca = kpca.fit_transform(X_scaled)

# 获取主成分方差（特征值）
variance = kpca.eigenvalues_#lambdas_ eigenvalues_

# 计算总方差（所有特征值的总和）
total_variance = np.sum(variance)

# 计算方差贡献率
variance_ratios = variance / total_variance
total_variance_ratios = np.cumsum(variance_ratios)

# # 自动选择累积贡献率≥95%的维度
# desired_variance = 0.95
# selected_indices = np.where(total_variance_ratios >= desired_variance)[0]
# selected_k = selected_indices[0] + 1 if len(selected_indices) > 0 else len(variance)

selected_k = 10

sum_variance = 0

# 打印结果（截止到选定维度）
print("主成分分析结果：")
print("=====================================================")
print(f"{'PC':<5} {'方差':<15} {'贡献率':<15} {'累积贡献率':<15}")
print("-----------------------------------------------------")
for i in range(selected_k):
    var = variance[i]
    ratio = variance_ratios[i]
    cum_ratio = total_variance_ratios[i]
    sum_variance = sum_variance + variance[i]
    print(f"PC{i+1}: {var:>10.4f}    {ratio:>10.2%}    {cum_ratio:>10.2%}")
print("=====================================================")
print(f"总方差（基于所选特征值）: {sum_variance:.4f}")
print(f"总方差（基于所有特征值）: {total_variance:.4f}")

# 保存选定维度的结果
np.savetxt('kpca_result.txt', X_kpca[:, :selected_k], fmt='%.9f')

# 可视化分析（保持原有功能）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10, 4))
plt.bar(range(1, len(variance_ratios)+1), variance_ratios, alpha=0.6, label='单成分贡献率')
plt.plot(range(1, len(total_variance_ratios)+1), total_variance_ratios, 'ro-', label='累积贡献率')
plt.axhline(y=0.95, color='gray', linestyle='--', label='95%阈值')
plt.xlabel('主成分')
plt.ylabel('贡献率')
plt.legend()
plt.savefig('variance_contribution.png')
plt.show()

# 保存完整分析结果（保持原有功能）
header = "PC, Variance, Variance_Ratio"
results = np.column_stack((np.arange(1, len(variance)+1), variance, variance_ratios))
np.savetxt('kpca_variance.csv',
           results,
           header=header,
           fmt=['%d', '%.9f', '%.9f'],
           delimiter=',',
           comments='')