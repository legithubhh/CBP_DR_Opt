import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

'''
n维重构相似度定义：n维PCA标准化重构误差/n维KPCA（余弦核）标准化重构误差
'''

def main():
    # ==================== 在这里修改降维维度 ====================
    # 指定需要分析的降维维度列表（可添加多个维度）
    dims = [5, 10]  # 修改这里的维度值

    # 打开结果文件（覆盖模式）
    with open('reconstruct_similarity.txt', 'w') as all_file:
        # 加载原始数据
        data = np.loadtxt('kuaizhao_4_5071.txt').T

        for n_components in dims:
            print(f"\n{'=' * 50}")
            print(f"正在处理降维维度: {n_components}")
            print(f"{'=' * 50}")

            # 数据标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(data)

            # ==================== PCA处理流程 ====================
            pca = PCA(n_components=n_components, svd_solver='randomized')
            scores_pca = pca.fit_transform(X_scaled)

            # PCA重构
            reconstructed_scaled_pca = pca.inverse_transform(scores_pca)
            reconstructed_pca = scaler.inverse_transform(reconstructed_scaled_pca)
            mae_pca = np.mean(np.abs(data - reconstructed_pca))

            # ==================== KPCA处理流程 ====================
            kpca = KernelPCA(
                n_components=n_components,
                kernel='cosine',
                fit_inverse_transform=True
            )
            scores_kpca = kpca.fit_transform(X_scaled)

            # KPCA重构
            reconstructed_scaled_kpca = kpca.inverse_transform(scores_kpca)
            reconstructed_kpca = scaler.inverse_transform(reconstructed_scaled_kpca)
            mae_kpca = np.mean(np.abs(data - reconstructed_kpca))

            # ==================== 计算重构相似度 ====================
            similarity = mae_pca / mae_kpca

            # ==================== 保存结果 ====================
            # 写入到汇总文件
            all_file.write(f"{'=' * 50}\n")
            all_file.write(f"降维维度: {n_components}\n")
            all_file.write(f"{'=' * 50}\n")
            all_file.write(f"PCA重构误差(MAE): {mae_pca:.5e}\n")
            all_file.write(f"KPCA重构误差(MAE): {mae_kpca:.5e}\n")
            all_file.write(f"重构相似度(PCA/KPCA): {similarity:.3f}\n\n")

            # 控制台输出
            print(f"\n处理完成! 结果已保存到汇总文件")
            print(f"PCA重构误差(MAE): {mae_pca:.5e}")
            print(f"KPCA重构误差(MAE): {mae_kpca:.5e}")
            print(f"重构相似度(PCA/KPCA): {similarity:.3f}")

    print(f"\n所有维度处理完成! 结果已保存到 reconstruct_similarity.txt")


if __name__ == "__main__":
    main()