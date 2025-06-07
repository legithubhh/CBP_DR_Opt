import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from scipy.spatial import QhullError

# 读取数据文件
data = np.loadtxt('kuaizhao_zq1_5243.txt')  # 404行 x 1000列
print("原始数据维度:", data.shape)  # 应为 (404, 1000)

# 转置为样本矩阵
samples = data.T  # (1000, 404)

'''PCA降维至3D'''
pca = make_pipeline(StandardScaler(), PCA(n_components=3))
samples_3d = pca.fit_transform(samples)  # (1000, 3)

'''DBSCAN聚类'''
#eps（邻域半径）两点距离小于eps则视为彼此的邻居
#成为核心点（Core Point）所需的最小邻居数量
# 注意：参数需要根据数据分布调整，尽可能覆盖整个样本库且空白区少,使噪点数量维持在50-150。
# 如果出现体积为0的情况，优先增大eps值，再减小min_samples
# 如果出现体积过大的情况，优先增大min_samples，再减小eps值
dbscan = DBSCAN(eps=6.0, min_samples=15)  # 示例参数
labels = dbscan.fit_predict(samples_3d)

# 聚类结果验证
unique_labels = np.unique(labels)
n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
print(f"发现聚类数量: {n_clusters}")
if n_clusters == 0:
    raise ValueError("所有样本被识别为噪声，请调整eps或min_samples参数")
# 计算噪声点数量
noise_count = np.sum(labels == -1)

'''凸包体积计算'''
cluster_volumes = []
valid_clusters = []
for cluster_id in unique_labels:
    if cluster_id == -1:
        continue

    cluster_points = samples_3d[labels == cluster_id]

    # 几何验证
    if cluster_points.shape[0] < 4:
        print(f"簇 {cluster_id} 点数不足，跳过")
        continue

    try:
        # 检查点集维度
        centered = cluster_points - cluster_points.mean(axis=0)
        if np.linalg.matrix_rank(centered) < 3:
            print(f"簇 {cluster_id} 存在共面/共线现象，跳过")
            continue

        hull = ConvexHull(cluster_points)
        cluster_volumes.append(hull.volume)
        valid_clusters.append(cluster_id)
    except QhullError as e:
        print(f"簇 {cluster_id} 凸包错误: {str(e)}")

distribution_volume = sum(cluster_volumes) if cluster_volumes else 0.0

'''排序与曲线拟合'''
sorted_indices = np.argsort(samples_3d[:, 0])
sorted_samples_3d = samples_3d[sorted_indices]
sorted_labels = labels[sorted_indices]

t = np.linspace(0, 1, len(sorted_samples_3d))
degree = 3
coeffs_x = np.polyfit(t, sorted_samples_3d[:, 0], degree)
coeffs_y = np.polyfit(t, sorted_samples_3d[:, 1], degree)
coeffs_z = np.polyfit(t, sorted_samples_3d[:, 2], degree)

t_fine = np.linspace(0, 1, 100)
curve_x = np.polyval(coeffs_x, t_fine)
curve_y = np.polyval(coeffs_y, t_fine)
curve_z = np.polyval(coeffs_z, t_fine)

'''长度计算'''
start_point = np.array([curve_x[0], curve_y[0], curve_z[0]])
end_point = np.array([curve_x[-1], curve_y[-1], curve_z[-1]])
straight_length = np.linalg.norm(end_point - start_point)

curve_diff = np.diff(np.column_stack([curve_x, curve_y, curve_z]), axis=0)
curve_length = np.sum(np.linalg.norm(curve_diff, axis=1))
length_ratio = curve_length / straight_length if straight_length != 0 else np.nan

'''新增指标'''
volume_curve_ratio = distribution_volume / curve_length if curve_length != 0 else np.nan
volume_straight_ratio = distribution_volume / straight_length if straight_length != 0 else np.nan

'''3D可视化'''
fig3d = plt.figure(figsize=(10, 6))
ax3d = fig3d.add_subplot(111, projection='3d')

# 绘制样本点
ax3d.scatter(sorted_samples_3d[:, 0], sorted_samples_3d[:, 1], sorted_samples_3d[:, 2],
             c='gray', alpha=0.3, label='All Points')

# 绘制拟合曲线
ax3d.plot(curve_x, curve_y, curve_z, 'r-', lw=2, label='Fitted Curve')

# 凸包渲染
if valid_clusters:
    colors = cm.rainbow(np.linspace(0, 1, len(valid_clusters)))
    for cluster_id, color in zip(valid_clusters, colors):
        cluster_points = samples_3d[labels == cluster_id]
        try:
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                ax3d.plot_trisurf(cluster_points[simplex, 0],
                                  cluster_points[simplex, 1],
                                  cluster_points[simplex, 2],
                                  color=color, alpha=0.2, edgecolor='none')
        except Exception as e:
            print(f"簇 {cluster_id} 渲染失败: {str(e)}")
            continue

ax3d.legend()
plt.savefig('pca_curve_fit.png')
plt.show()
plt.close()

'''2D可视化'''
fig2d = plt.figure(figsize=(10, 6))
ax2d = fig2d.add_subplot(111)

# 数据分离
noise_mask = (sorted_labels == -1)
valid_cluster_mask = ~noise_mask

# 动态颜色映射
if np.any(valid_cluster_mask):
    unique_clusters = np.unique(sorted_labels[valid_cluster_mask])
    cmap = plt.get_cmap('rainbow', len(unique_clusters))

    scatter = ax2d.scatter(
        sorted_samples_3d[valid_cluster_mask, 0],
        sorted_samples_3d[valid_cluster_mask, 1],
        c=sorted_labels[valid_cluster_mask],
        cmap=cmap,
        alpha=0.7,
        edgecolors='w',
        vmin=min(unique_clusters),
        vmax=max(unique_clusters)
    )
    plt.colorbar(scatter, label='Cluster ID')
else:
    print("警告：无有效聚类可显示")

# 噪声点
ax2d.scatter(sorted_samples_3d[noise_mask, 0], sorted_samples_3d[noise_mask, 1],
             c='gray', alpha=0.2, label='Noise')

# 曲线与端点
ax2d.plot(curve_x, curve_y, 'k-', lw=2, zorder=2)
ax2d.scatter(curve_x[0], curve_y[0], c='blue', s=100, edgecolor='w', zorder=3)
ax2d.scatter(curve_x[-1], curve_y[-1], c='red', s=100, edgecolor='w', zorder=3)

ax2d.legend()
plt.savefig('pca_cluster_curve_2d.png')
plt.show()
plt.close()


'''保存结果'''

metrics = f"""
直线长度: {straight_length:.4f}
曲线长度: {curve_length:.4f}
长度比值: {length_ratio:.4f}
分布体积: {distribution_volume:.4f}
体积/曲线比: {volume_curve_ratio:.4f}
体积/直线比: {volume_straight_ratio:.4f}
噪声点数量: {noise_count}  # 新增的噪声统计

三维曲线方程:
x(t) = {np.poly1d(coeffs_x)}
y(t) = {np.poly1d(coeffs_y)}
z(t) = {np.poly1d(coeffs_z)}
"""

with open('curve_metrics.txt', 'w') as f:
    f.write(metrics)

# 打印输出新增统计
print(f"直线长度: {straight_length:.4f}")
print(f"曲线长度: {curve_length:.4f}")
print(f"长度比值: {length_ratio:.4f}")
print(f"分布体积: {distribution_volume:.4f}")
print(f"分布体积/曲线长度: {volume_curve_ratio:.4f}")
print(f"分布体积/直线长度: {volume_straight_ratio:.4f}")
print(f"噪声点数量: {noise_count}")
print("处理完成，结果已保存")