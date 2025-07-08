import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

def projection_centroid_method():
    """投影质心法分析函数"""
    data = pd.read_csv('data/新附件2.csv', header=None)
    angles = data.iloc[0, :].values  # 第一行是角度信息
    projection_data = data.iloc[1:, :].values  # 第2行到第513行是投影数据
    centroids = []
    nan_count = 0
    
    for i in range(len(angles)):
        # 获取第i个角度对应的512维投影向量
        projection_vector = projection_data[:, i]
        # 计算质心使用公式: C_j = Σ(i * p_j(i)) / Σ(p_j(i))
        # 其中 i 是位置索引（1到512），p_j(i) 是该位置的投影值
        position_indices = np.arange(1, len(projection_vector) + 1)  # 位置索引从1开始
        numerator = np.sum(position_indices * projection_vector)
        denominator = np.sum(projection_vector)
        if denominator != 0:
            centroid = numerator / denominator
        else:
            centroid = np.nan
            nan_count += 1
        centroids.append(centroid)
    centroids = np.array(centroids)
    
    # 创建角度-质心的字典，用于合并相同角度的数据
    angle_centroid_dict = {}
    for angle, centroid in zip(angles, centroids):
        if angle in angle_centroid_dict:
            angle_centroid_dict[angle].append(centroid)
        else:
            angle_centroid_dict[angle] = [centroid]
    
    # 统计重复角度的情况
    duplicate_count = 0
    total_duplicates = 0
    for angle, centroid_list in angle_centroid_dict.items():
        if len(centroid_list) > 1:
            duplicate_count += 1
            total_duplicates += len(centroid_list)
    
    # 计算每个唯一角度对应的平均质心
    unique_angles = []
    averaged_centroids = []
    for angle in sorted(angle_centroid_dict.keys()):
        centroid_list = angle_centroid_dict[angle]
        avg_centroid = np.mean(centroid_list)
        unique_angles.append(angle)
        averaged_centroids.append(avg_centroid)
    
    # 转换为numpy数组
    unique_angles = np.array(unique_angles)
    averaged_centroids = np.array(averaged_centroids)
    
    def sine_function(x, A, phi, C):
        return A * np.sin(x + phi) + C

    angles_rad = np.deg2rad(unique_angles)

    
    A_init = (np.max(averaged_centroids) - np.min(averaged_centroids)) / 2  # 幅度初始值
    C_init = np.mean(averaged_centroids)  # 直流偏置初始值
    phi_init = 0  # 相位初始值
    
    initial_guess = [A_init, phi_init, C_init]
    
    try:
        # 执行拟合
        popt, _ = curve_fit(sine_function, angles_rad, averaged_centroids, p0=initial_guess)
        A_fitted, phi_fitted, C_fitted = popt
        # 计算拟合优度
        fitted_values = sine_function(angles_rad, *popt)
        r_squared = 1 - np.sum((averaged_centroids - fitted_values)**2) / np.sum((averaged_centroids - np.mean(averaged_centroids))**2)
        print("\n拟合结果:")
        print(f"幅度 (A): {A_fitted:.6f}")
        print(f"相位 (φ): {phi_fitted:.6f} 弧度 = {np.rad2deg(phi_fitted):.2f} 度")
        print(f"直流偏置 (C): {C_fitted:.6f}")
        print(f"R² (拟合优度): {r_squared:.6f}")
        
    except Exception as e:
        print(f"拟合过程中出现错误: {e}")
        return
    
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_angles, averaged_centroids, c='blue', s=16, label='计算的质心')
    plt.plot(unique_angles, fitted_values, 'r-', label='拟合的正弦函数', linewidth=2)
    plt.xlabel('旋转角度 (度)')
    plt.ylabel('投影质心')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('projection_centroid_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    
    return {
        'angles': unique_angles,
        'centroids': averaged_centroids,
        'fitted_parameters': popt,
        'r_squared': r_squared,
        'fitted_values': fitted_values
    }

if __name__ == "__main__":
    # 执行投影质心法分析
    results = projection_centroid_method()

