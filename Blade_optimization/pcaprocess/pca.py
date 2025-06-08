import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 若原始数据中有重复列（完全线性相关），协方差矩阵的秩会降低。例如：404维数据中实际独立维度可能只有400个
data = np.loadtxt('kuaizhao_3_5281.txt').T

# 标准化过程
scaler = StandardScaler()
X = scaler.fit_transform(data)

pca = PCA(n_components=10, svd_solver='randomized')
pca.fit(X)

# 新增主成分方差分析 ========================================
# 获取主成分方差数组
pc_variances = pca.explained_variance_

# 计算所选主成分总方差
selected_total = np.sum(pc_variances)

# 计算所有可能主成分的总方差（协方差矩阵的迹）
total_variance = np.var(X, axis=0, ddof=1).sum()  # 等价于 np.trace(pca.get_covariance())

# 打印详细方差报告
print("\n主成分方差详细报告：")
print("===========================================================")
print(f"{'PC':<5} {'方差值':<20} {'贡献率':<15} {'累计贡献率':<15}")
print("-----------------------------------------------------------")
for i, (var, ratio) in enumerate(zip(pc_variances,
                                   pca.explained_variance_ratio_)):
    cum_ratio = np.sum(pca.explained_variance_ratio_[:i+1])
    print(f"PC{i+1}: {var:.4e}    {ratio*100:.2f}%        {cum_ratio*100:.2f}%")
print("===========================================================")
print(f"前{len(pc_variances)}个主成分总方差: {selected_total:.4e}")
print(f"所有可能主成分总方差: {total_variance:.4e}")
print(f"方差解释比例: {selected_total/total_variance*100:.2f}%")
# ==========================================================

# norms = np.linalg.norm(pca.components_, axis=1)
# print("主成分的L2范数：", np.round(norms, 6))  # 输出应全为1，验证为单位化

# 保存中心点（确保列向量格式）
np.savetxt('centre.txt', pca.mean_.reshape(-1, 1), fmt='%.18e')
# 保存特征值（强制转换为列向量）
np.savetxt('lambda1.dat', pc_variances.reshape(-1, 1), fmt='%.18e')
# 保存特征向量（保持原始格式）
np.savetxt('flux1.dat', pca.components_, fmt='%.18e')
# 保存降维结果（优化内存使用）
np.savetxt('pca_result.txt', pca.transform(X), fmt='%.18e')