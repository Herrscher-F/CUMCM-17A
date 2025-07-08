import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

class FBPReconstruction:
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.detector_spacing = 0.2774  # mm，探测器间距
        self.num_detectors = 512  # 探测器数量
        self.image_size = 256  # 重建图像大小 256x256
        self.pixel_size = 0.3922  # mm，像素边长
        
        # 读取数据
        self.load_data()
        
    def load_data(self):
        """读取投影数据"""
        data = pd.read_csv(self.csv_path, header=None)
        self.angles = data.iloc[0, :].values  # 度
        self.angles_rad = np.deg2rad(self.angles)  # 转换为弧度
        
        # 第2-513行是投影数据（512个探测器单元）
        self.projection_data = data.iloc[1:, :].values
        
    def create_sinogram(self):
        """构建标准正弦图 p(ρ, θ)"""
        # 计算旋转中心在探测器阵列上的投影索引
        # 假设旋转中心位于探测器阵列的中心
        self.i_c = (self.num_detectors + 1) / 2  # 256.5
        # 计算每个探测器单元对应的ρ值
        detector_indices = np.arange(1, self.num_detectors + 1)
        self.rho_values = (detector_indices - self.i_c) * self.detector_spacing
        self.sinogram = self.projection_data.T  # 形状: (180, 512)

        
    def design_filter(self, n_samples):
        """设计加汉明窗的斜坡滤波器"""
        # 创建频域坐标
        freq = np.fft.fftfreq(n_samples)
        # 斜坡滤波器（|ω|）
        ramp_filter = np.abs(freq)
        # 汉明窗
        hamming_window = np.hamming(n_samples)
        # 组合滤波器
        H_win = ramp_filter * hamming_window
        return H_win
        
    def filter_projections(self):
        """对正弦图进行滤波"""
        n_angles, n_detectors = self.sinogram.shape
        self.filtered_sinogram = np.zeros_like(self.sinogram)
        # 设计滤波器
        H_win = self.design_filter(n_detectors)
        # 对每个角度的投影数据进行滤波
        for i in range(n_angles):
            # FFT到频域
            P_omega = np.fft.fft(self.sinogram[i, :])
            # 应用滤波器
            P_filtered_omega = P_omega * H_win
            # IFFT回到空间域
            p_filtered = np.real(np.fft.ifft(P_filtered_omega))
            self.filtered_sinogram[i, :] = p_filtered

    def backprojection(self):
        """反投影重建"""
        # 初始化重建图像
        self.reconstructed_image = np.zeros((self.image_size, self.image_size))
        # 计算图像坐标系
        # 图像中心对应CT旋转中心
        center_pixel = (self.image_size - 1) / 2
        x_c = center_pixel * self.pixel_size
        y_c = center_pixel * self.pixel_size
        # 角度步长（弧度）
        delta_theta_rad = np.pi / 180
        # 对每个像素进行反投影
        for i in range(self.image_size):
            for j in range(self.image_size):
                # 像素的物理坐标
                x = i * self.pixel_size
                y = j * self.pixel_size
                pixel_value = 0.0
                # 遍历所有角度
                for angle_idx in range(len(self.angles_rad)):
                    theta = self.angles_rad[angle_idx]
                    # 计算投影坐标ρ'
                    rho_prime = (x - x_c) * np.cos(theta) + (y - y_c) * np.sin(theta)
                    # 线性插值获取投影值
                    interpolated_value = self.interpolate_projection(
                        self.filtered_sinogram[angle_idx, :], rho_prime
                    )
                    # 累加到像素值
                    pixel_value += interpolated_value * delta_theta_rad
                self.reconstructed_image[i, j] = 2*pixel_value
            
            # 显示进度
            if (i + 1) % 50 == 0:
                print(f"反投影进度: {i+1}/{self.image_size}")

        
    def interpolate_projection(self, projection_line, rho_prime):
        """在滤波后的投影数据中进行线性插值"""
        # 如果超出ρ值范围，返回0
        if rho_prime < self.rho_values.min() or rho_prime > self.rho_values.max():
            return 0.0
        # 使用scipy的interp1d进行线性插值
        try:
            interpolator = interp1d(self.rho_values, projection_line, 
                                  kind='linear', fill_value=0.0, bounds_error=False)
            return interpolator(rho_prime)
        except:
            return 0.0
    
    def visualize_results(self):
        """可视化结果"""
        # 原始正弦图
        plt.figure(figsize=(10, 6))
        im1 = plt.imshow(self.sinogram, aspect='auto', cmap='gray')
        plt.xlabel('探测器索引')
        plt.ylabel('角度索引')
        plt.colorbar(im1)
        plt.tight_layout()
        plt.savefig('原始正弦图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 滤波后正弦图
        plt.figure(figsize=(10, 6))
        im2 = plt.imshow(self.filtered_sinogram, aspect='auto', cmap='gray')
        plt.xlabel('探测器索引')
        plt.ylabel('角度索引')
        plt.colorbar(im2)
        plt.tight_layout()
        plt.savefig('滤波后正弦图.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 重建图像
        plt.figure(figsize=(8, 8))
        im3 = plt.imshow(self.reconstructed_image, cmap='gray')
        plt.xlabel('像素索引')
        plt.ylabel('像素索引')
        plt.colorbar(im3)
        plt.tight_layout()
        plt.savefig('FBP重建结果.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_results(self):
        """保存重建结果"""
        # 保存重建图像为CSV文件
        np.savetxt('FBP_reconstructed_image.csv', self.reconstructed_image, delimiter=',', fmt='%.3f')
        
        # 保存一些统计信息
        stats = {
            '最小值': self.reconstructed_image.min(),
            '最大值': self.reconstructed_image.max(),
            '平均值': self.reconstructed_image.mean(),
            '标准差': self.reconstructed_image.std()
        }
        print("重建图像统计信息:")
        for key, value in stats.items():
            print(f"{key}: {value:.6f}")
            
    def run_reconstruction(self):
        """执行完整的FBP重建流程"""
        # 步骤1: 构建正弦图
        self.create_sinogram()
        # 步骤2: 滤波
        self.filter_projections()
        # 步骤3: 反投影
        self.backprojection()
        # 可视化和保存结果
        self.visualize_results()
        self.save_results()
        print("=" * 50)
        print("FBP重建完成！")

def main():
    """主函数"""
    # 数据文件路径
    csv_path = 'data/新附件3.csv'
    
    # 创建FBP重建对象并执行重建
    fbp = FBPReconstruction(csv_path)
    fbp.run_reconstruction()

if __name__ == "__main__":
    main()
