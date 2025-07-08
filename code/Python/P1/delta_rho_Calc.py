import pandas as pd
import numpy as np

data = pd.read_csv('code/data/附件2.csv', header=None)
projection_data = data.values
angles = np.arange(180) 
projections = []

for i in range(180):
    angle_projection = projection_data[:, i]  # 每一列代表一个角度的投影
    projections.append(angle_projection)

# 将投影数据二值化
threshold = 1e-6  
binary_projections = []

for projection in projections:
    # 将非零值设为1，零值保持为0
    binary_proj = (projection > threshold).astype(int)
    binary_projections.append(binary_proj)

# 识别每个角度投影中的所有分离片段
all_segments = []  # 存储所有投影片段
segment_info = []  # 存储片段信息（角度、起始位置、长度）

for angle_idx, binary_proj in enumerate(binary_projections):
    # 找到所有连续的非零区间
    diff = np.diff(np.concatenate(([0], binary_proj, [0])))
    starts = np.where(diff == 1)[0]  # 片段开始位置
    ends = np.where(diff == -1)[0]   # 片段结束位置
    
    # 计算每个片段的长度
    for start, end in zip(starts, ends):
        segment_length = end - start
        if segment_length > 0:  # 过滤掉长度为0的片段
            all_segments.append(segment_length)
            segment_info.append({
                'angle': angle_idx,
                'start': start,
                'end': end,
                'length': segment_length
            })

all_segments = np.array(all_segments)
print(f"总共识别出 {len(all_segments)} 个投影片段")
print(f"片段长度范围: {np.min(all_segments)} - {np.max(all_segments)}")

# 使用第一四分位数作为阈值，识别较小的片段
q1 = int(np.percentile(all_segments, 25))
circle_segments = all_segments[all_segments <= q1]

print(f"第一四分位数(Q1): {q1}")
print(f"圆形物体投影片段数量: {len(circle_segments)}")

# 计算圆形物体的平均投影长度
average_circle_projection = np.mean(circle_segments)
std_circle_projection = np.std(circle_segments)

# 计算探测器间距
detector_spacing = 8.0 / average_circle_projection

results = {
    '圆形物体平均投影长度': average_circle_projection,
    '探测器间距(mm)': detector_spacing
}
for key, value in results.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")










