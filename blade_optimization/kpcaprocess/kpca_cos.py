import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号'-'显示为方块的问题

# 1. 数据读取与标准化
data = np.loadtxt('kuaizhao_4_5072.txt').T  # (1000, 404)
X_scaled = StandardScaler().fit_transform(data)

# 2. 训练 KPCA 模型时获取所有特征值
kpca = KernelPCA(
    n_components=None,  # 关键修改：计算所有特征值
    kernel='cosine',
    fit_inverse_transform=True
)
kpca.fit(X_scaled)

# 3. 手动提取前5个主成分
n_dim = 10
X_kpca = kpca.transform(X_scaled)[:, :n_dim]  # 取前5列

# 4. 计算贡献率（基于所有特征值）
all_variance = np.abs(kpca.eigenvalues_)  # 所有特征值的绝对值eigenvalues_ lambdas_
total_variance = np.sum(all_variance)  # 总方差 = 所有特征值绝对值之和
selected_variance = all_variance[:n_dim]  # 前5个特征值
variance_ratios = selected_variance / total_variance  # 正确贡献率
cumulative_ratios = np.cumsum(variance_ratios)

# 6. 打印结果（功能不变）
print("主成分分析结果：")
print("=====================================")
print(f"{'PC':<5} {'方差':<15} {'贡献率':<15} {'累积贡献率':<15}")
print("-------------------------------------")
for i, (var, ratio) in enumerate(zip(selected_variance, variance_ratios)):
    cum_ratio = cumulative_ratios[i]
    print(f"PC{i+1}: {var:>10.4f}    {ratio:>10.2%}    {cum_ratio:>10.2%}")
print("=====================================")
print(f"总贡献度（基于所有特征值）: {np.sum(variance_ratios):.4f}")
print(f"总方差（基于所有特征值）: {total_variance:.4f}")

# 7. 可视化（功能不变）
plt.figure(figsize=(10, 4))
plt.bar(range(1, len(variance_ratios)+1), variance_ratios, alpha=0.6, label='单成分贡献率')
plt.plot(range(1, len(cumulative_ratios)+1), cumulative_ratios, 'ro-', label='累积贡献率')
plt.axhline(y=0.95, color='gray', linestyle='--', label='95%阈值')
plt.xlabel('主成分')
plt.ylabel('贡献率')
plt.legend()
plt.savefig('variance_contribution.png')
plt.show()

# 8. 保存结果（功能不变）
header = "PC, Variance, Variance_Ratio"
results = np.column_stack((np.arange(1, n_dim+1), selected_variance, variance_ratios))
np.savetxt('kpca_variance.csv',
           results,
           header=header,
           fmt=['%d', '%.9f', '%.9f'],
           delimiter=',',
           comments='')
np.savetxt('kpca_result.txt', X_kpca, fmt='%.9f')