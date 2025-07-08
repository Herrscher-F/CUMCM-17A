import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
# 读取CSV文件
data = pd.read_csv('data/附件1.csv', header=None)

# 将数据转换为numpy数组
image_data = data.values

# 获取图像尺寸
height, width = image_data.shape

# 创建坐标网格，以左下角为原点
x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

# 由于CSV中的数据是以左上角为原点，需要翻转y坐标
# 将y坐标转换为以左下角为原点的坐标系
y_coords = height - 1 - y_coords

# 计算质心坐标
# 质心公式：x_c = Σ(x*I(x,y)) / Σ(I(x,y)), y_c = Σ(y*I(x,y)) / Σ(I(x,y))
total_intensity = np.sum(image_data)
centroid_x = np.sum(x_coords * image_data) / total_intensity
centroid_y = np.sum(y_coords * image_data) / total_intensity

print(f"图像尺寸: {width} x {height}")
print(f"总强度: {total_intensity}")
print(f"质心坐标 (以左下角为原点): ({centroid_x:.6f}, {centroid_y:.6f})")

