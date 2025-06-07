import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler


'''
当 n > 1 且 n <= 5 时：重构非线性度 = (1维PCA误差 - n维PCA误差) / (1维KPCA误差 - n维KPCA误差)
当 n > 5 时：重构非线性度 = (5维PCA误差 - n维PCA误差) / (5维KPCA误差 - n维KPCA误差)
'''


def main():
    # ==================== 在这里修改降维维度 ====================
    # 指定需要分析的降维维度列表（可添加多个维度）
    dims = [5, 10]  # 修改这里的维度值

    # 确保维度列表包含1和5作为基准
    if 1 not in dims:
        dims.append(1)
    if 5 not in dims:
        dims.append(5)

    # 加载原始数据
    data = np.loadtxt('kuaizhao_zq1_5243.txt').T

    # 存储各维度的重构误差
    pca_errors = {}
    kpca_errors = {}

    # 计算所有维度的重构误差
    for n_components in sorted(dims):  # 按维度从小到大处理
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
        pca_errors[n_components] = mae_pca

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
        kpca_errors[n_components] = mae_kpca

        print(f"PCA重构误差(MAE): {mae_pca:.5e}")
        print(f"KPCA重构误差(MAE): {mae_kpca:.5e}")

    # ==================== 计算重构非线性度 ====================
    nonlinearities = {}

    for n in sorted(dims):
        if n == 1:
            # 1维没有非线性度定义
            nonlinearities[n] = None
        elif n <= 5:
            # 当n<=5时，使用1维作为基准
            numerator = pca_errors[1] - pca_errors[n]
            denominator = kpca_errors[1] - kpca_errors[n]
            nonlinearity = numerator / denominator
            nonlinearities[n] = nonlinearity
        else:
            # 当n>5时，使用5维作为基准
            numerator = pca_errors[5] - pca_errors[n]
            denominator = kpca_errors[5] - kpca_errors[n]
            nonlinearity = numerator / denominator
            nonlinearities[n] = nonlinearity

    # ==================== 保存结果 ====================
    with open('reconstruction_nonlinearity.txt', 'w') as f:
        f.write(f"{'维度':<6}{'PCA重构误差':<20}{'KPCA重构误差':<20}{'重构非线性度':<30}{'基准维度'}\n")
        f.write("-" * 85 + "\n")

        for n in sorted(dims):
            pca_err = pca_errors[n]
            kpca_err = kpca_errors[n]
            nl = nonlinearities[n]

            if n == 1:
                f.write(f"{n:<6}{pca_err:.5e}{'':<15}{kpca_err:.5e}{'':<15}{'N/A (基准)':<30}1维\n")
            elif n <= 5:
                f.write(f"{n:<6}{pca_err:.5e}{'':<15}{kpca_err:.5e}{'':<15}{nl:.3f}{'':<24}1维\n")
            else:
                f.write(f"{n:<6}{pca_err:.5e}{'':<15}{kpca_err:.5e}{'':<15}{nl:.3f}{'':<24}5维\n")

    # 控制台输出最终结果
    print("\n" + "=" * 85)
    print("重构非线性度计算结果:")
    print("=" * 85)
    print(f"{'维度':<6}{'PCA重构误差':<20}{'KPCA重构误差':<20}{'重构非线性度':<30}{'基准维度'}")
    print("-" * 85)
    for n in sorted(dims):
        if n == 1:
            print(f"{n:<6}{pca_errors[n]:.5e}{'':<15}{kpca_errors[n]:.5e}{'':<15}{'N/A (基准)':<30}1维")
        elif n <= 5:
            print(f"{n:<6}{pca_errors[n]:.5e}{'':<15}{kpca_errors[n]:.5e}{'':<15}{nonlinearities[n]:.3f}{'':<24}1维")
        else:
            print(f"{n:<6}{pca_errors[n]:.5e}{'':<15}{kpca_errors[n]:.5e}{'':<15}{nonlinearities[n]:.3f}{'':<24}5维")

    print(f"\n所有结果已保存到 reconstruction_nonlinearity.txt")


if __name__ == "__main__":
    main()