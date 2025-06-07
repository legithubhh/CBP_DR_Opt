import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
import csv

# 读取原始数据
original_data = np.loadtxt('kuaizhao_3_5283.txt').T  # 转置后形状(1000, 404)


def compute_mae(gamma_val):
    """计算指定gamma值的平均绝对误差"""
    gamma = gamma_val / 404.0
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(original_data)

    kpca = KernelPCA(
        n_components=10,
        kernel='rbf',
        gamma=gamma,
        fit_inverse_transform=True
    )
    X_kpca = kpca.fit_transform(X_scaled)
    X_reconstructed = scaler.inverse_transform(kpca.inverse_transform(X_kpca))

    return np.mean(np.abs(original_data - X_reconstructed))


# 初始化结果存储
all_results = []

# 阶段1：粗搜索 (步长1/404)
gamma_coarse = np.arange(0.1, 16.1, 1)  # 0-15对应0/404到15/404
print("正在进行粗搜索（步长1/404）...")
for g in gamma_coarse:
    mae = compute_mae(g)
    all_results.append((g, mae))

# 找出两个最小误差候选点
sorted_results = sorted(all_results, key=lambda x: x[1])
(g1, mae1), (g2, mae2) = sorted_results[:2]

# 确定候选点位置关系
idx1 = np.where(gamma_coarse == g1)[0][0]
idx2 = np.where(gamma_coarse == g2)[0][0]
is_adjacent = abs(idx1 - idx2) == 1

# 阶段2：细搜索 (步长0.1/404)
fine_gammas = []
if is_adjacent:
    # 相邻时搜索中间区域
    start, end = sorted([g1, g2])
    fine_gammas = np.arange(start, end + 0.1, 0.1)
else:
    # 不相邻时搜索候选点附近区域
    range1 = np.clip(np.arange(g1 - 0.9, g1 + 0.9, 0.1), 0, 15.1)
    range2 = np.clip(np.arange(g2 - 0.9, g2 + 0.9, 0.1), 0, 15.1)
    fine_gammas = np.unique(np.concatenate([range1, range2]))

print("\n正在进行细搜索（步长0.1/404）...")
for g in fine_gammas:
    if not any(np.isclose(g, exist_g) for exist_g, _ in all_results):
        mae = compute_mae(g)
        all_results.append((g, mae))

# 确定最终最优解
best_gamma, min_mae = min(all_results, key=lambda x: x[1])

# 保存结果表格
with open('gamma_optimization.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Gamma (x/404)', '实际Gamma值', '绝对平均误差'])

    for g, mae in sorted(all_results, key=lambda x: x[0]):
        actual_gamma = g / 404.0
        mark = '*' if (g, mae) == (best_gamma, min_mae) else ''
        writer.writerow([
            f"{g:.1f}/404",
            f"{actual_gamma:.10f}",
            f"{mae:.5e}{mark}"
        ])

# 控制台输出
print("\n最终优化结果：")
print(f"最佳gamma值：{best_gamma}/404 = {best_gamma / 404.0:.5e}")
print(f"最小绝对平均误差：{min_mae:.5e}")
print("完整结果已保存至 gamma_optimization.csv")