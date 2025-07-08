import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False

# 物理参数
PIXEL_SIZE = 0.3922  # mm，像素大小
DETECTOR_SPACING = 0.2774  # mm，探测器间距

# 椭圆参数
ellipse_params = {
        'center': (0, 0),  # 椭圆中心 (mm)
        'a': 40,  # 长轴半径 (mm)
        'b': 15,  # 短轴半径 (mm)
        'rotation': 90  # 旋转角度 (度)
    }
    
# 圆参数
circle_params = {
        'center': (45, 0),  # 圆心 (mm)
        'radius': 4  # 半径 (mm)
    }

def calculate_projection_length(projection_row, threshold=1e-6):
    """计算实际角度投影长度（单位为mm）"""
    # 找到非零值的位置（使用阈值处理噪声）
    nonzero_indices = np.where(np.abs(projection_row) > threshold)[0]
    if len(nonzero_indices) == 0:
        return np.array([0])
    # 找到连续段
    segments = []
    current_segment_start = nonzero_indices[0]
    for i in range(1, len(nonzero_indices)):
        if nonzero_indices[i] - nonzero_indices[i-1] > 1:
            # 当前段结束
            segment_length = nonzero_indices[i-1] - current_segment_start + 1
            segments.append(segment_length)
            current_segment_start = nonzero_indices[i]
    # 添加最后一段
    segment_length = nonzero_indices[-1] - current_segment_start + 1
    segments.append(segment_length)
    
    # 将探测器间距单位转换为mm：段长度 * 探测器间距 (0.2774 mm)
    segments_mm = np.array(segments) * DETECTOR_SPACING
    return segments_mm


def calculate_ellipse_projection_width(ellipse_params, angle_deg):
    """计算椭圆在给定角度下的投影宽度"""
    angle_rad = np.radians(angle_deg)
    a = ellipse_params['a']  # 长轴半径 (mm)
    b = ellipse_params['b']  # 短轴半径 (mm)
    rotation_rad = np.radians(ellipse_params['rotation'])
    # 考虑椭圆的旋转
    effective_angle = angle_rad - rotation_rad
    # 椭圆投影宽度公式：2 * sqrt((a*cos(θ))² + (b*sin(θ))²)
    projection_width = 2 * np.sqrt(
        (a * np.cos(effective_angle))**2 + (b * np.sin(effective_angle))**2
    )
    return projection_width


def calculate_advanced_geometric_projection(ellipse_params, circle_params, angle_deg):
    """
    组合几何投影计算，考虑椭圆和圆的相对位置
    计算过程:
    1. 计算椭圆和圆在给定角度下的投影范围
    2. 合并重叠的投影范围
    3. 计算各段的物理长度 (mm)
    """
    angle_rad = np.radians(angle_deg)
    # 投影方向向量
    proj_direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
    # 计算椭圆的投影范围
    ellipse_center = np.array(ellipse_params['center'])
    ellipse_proj_center = np.dot(ellipse_center, proj_direction)
    ellipse_width = calculate_ellipse_projection_width(ellipse_params, angle_deg)
    ellipse_proj_range = [
        ellipse_proj_center - ellipse_width/2,
        ellipse_proj_center + ellipse_width/2
    ]
    # 计算圆的投影范围
    circle_center = np.array(circle_params['center'])
    circle_proj_center = np.dot(circle_center, proj_direction)
    circle_radius = circle_params['radius']
    circle_proj_range = [
        circle_proj_center - circle_radius,
        circle_proj_center + circle_radius
    ]
    # 合并投影范围
    all_ranges = [ellipse_proj_range, circle_proj_range]
    # 计算连续投影段
    segments = merge_overlapping_ranges(all_ranges)
    # 计算每个段的长度（物理单位mm）
    segment_lengths = [seg[1] - seg[0] for seg in segments]
    # 直接返回mm单位的投影长度
    return np.array(segment_lengths)


def merge_overlapping_ranges(ranges):
    """合并重叠的投影范围"""
    if not ranges:
        return []
    # 排序范围
    sorted_ranges = sorted(ranges, key=lambda x: x[0])
    merged = [sorted_ranges[0]]
    for current in sorted_ranges[1:]:
        last = merged[-1]
        # 如果当前范围与上一个范围重叠或相邻
        if current[0] <= last[1]:
            # 合并范围
            merged[-1] = [last[0], max(last[1], current[1])]
        else:
            # 添加新范围
            merged.append(current)
    return merged


def calculate_set_similarity(vec1, vec2):
    """计算两个向量之间的相似性"""
    if len(vec1) != len(vec2):
        return float('inf')
    if len(vec1) == 0:
        return 0.0
    # 转换为numpy数组
    v1 = np.array(vec1, dtype=float)
    v2 = np.array(vec2, dtype=float)
    v1_sorted = np.sort(v1)
    v2_sorted = np.sort(v2)
    return np.linalg.norm(v1_sorted - v2_sorted)
    

def match_projections_optimized(measured_lengths, calculated_lengths, max_candidates=100):
    """匹配投影长度"""
    matched_angles = []
    # 预处理：按投影段数分组
    measured_by_segments = {}
    for i, measured in enumerate(measured_lengths):
        num_segments = len(measured)
        if num_segments not in measured_by_segments:
            measured_by_segments[num_segments] = []
        measured_by_segments[num_segments].append((i, measured))
    calculated_by_segments = {}
    for i, calculated in enumerate(calculated_lengths):
        num_segments = len(calculated)
        if num_segments not in calculated_by_segments:
            calculated_by_segments[num_segments] = []
        calculated_by_segments[num_segments].append((i, calculated))
    for _, measured in tqdm(enumerate(measured_lengths), 
                                      total=len(measured_lengths), 
                                      desc="匹配投影"):
        num_segments = len(measured)
        best_match_idx = -1
        best_similarity = float('inf')
        # 只在相同段数的计算投影中搜索
        if num_segments in calculated_by_segments:
            candidates = calculated_by_segments[num_segments]
            # 限制候选数量以加速计算
            if len(candidates) > max_candidates:
                # 随机选择候选者或使用其他策略
                step = len(candidates) // max_candidates
                candidates = candidates[::step]
            for calc_idx, calculated in candidates:
                # 计算基于取值集合的相似性
                if len(measured) == len(calculated):
                    similarity = calculate_set_similarity(measured, calculated)
                    if similarity < best_similarity:
                        best_similarity = similarity
                        best_match_idx = calc_idx
        matched_angles.append(best_match_idx)
    return matched_angles

def save_projection_lengths_to_csv(measured_lengths, calculated_lengths, measured_filename='measured_projection_lengths.csv', calculated_filename='calculated_projection_lengths.csv'):
    # 准备实际测量的投影长度数据
    max_measured_length = max(len(lengths) for lengths in measured_lengths)
    measured_data = []
    for i, lengths in enumerate(measured_lengths):
        row = list(lengths) + [np.nan] * (max_measured_length - len(lengths))
        measured_data.append(row)
    measured_df = pd.DataFrame(measured_data)
    measured_df.to_csv(measured_filename, index=False, header=False)
    
    # 准备计算的投影长度数据
    max_calculated_length = max(len(lengths) for lengths in calculated_lengths)
    calculated_data = []
    
    for i, lengths in enumerate(calculated_lengths):
        row = list(lengths) + [np.nan] * (max_calculated_length - len(lengths))
        calculated_data.append(row)
    calculated_df = pd.DataFrame(calculated_data)
    calculated_df.to_csv(calculated_filename, index=False, header=False)

def visualize_results(matched_angles, preset_angles, save_plot=True):

    # matched_angles 已经是实际角度值，不需要再次转换
    actual_angles = matched_angles
    valid_angles = [angle for angle in actual_angles if angle != -1]
    
    if len(valid_angles) > 0:
        plt.figure(figsize=(10, 6))
        
        # 绘制角度分布
        plt.hist(valid_angles, bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('角度 (度)')
        plt.ylabel('频次')

        plt.axis('on') 
        plt.tight_layout()
        if save_plot:
            plt.savefig('angle_matching_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def save_angles_to_csv(matched_angles, filename='angles.csv'):
    # 过滤掉无效角度（-1），并排序
    valid_angles = [angle for angle in matched_angles if angle != -1]
    valid_angles.sort()  # 从小到大排序
    # 确保有180个角度
    if len(valid_angles) < 180:
        print(f"警告：只有 {len(valid_angles)} 个有效角度，少于期望的180个")
        # 用NaN填充不足的部分
        valid_angles.extend([np.nan] * (180 - len(valid_angles)))
    elif len(valid_angles) > 180:
        print(f"警告：有 {len(valid_angles)} 个有效角度，超过期望的180个，将只保留前180个")
        valid_angles = valid_angles[:180]
    # 将180个角度重塑为18x10的矩阵
    angles_matrix = np.array(valid_angles).reshape(18, 10)
    # 保存为CSV文件，指定1位小数格式
    angles_df = pd.DataFrame(angles_matrix)
    angles_df.to_csv(filename, index=False, header=False, float_format='%.1f')
    print(f"角度范围: {np.nanmin(angles_matrix):.1f}° 到 {np.nanmax(angles_matrix):.1f}°")
    
    return angles_matrix


def save_projections_with_angles(matched_angles, projection_data, filename='data/新附件2.csv'):
    # 过滤掉无效角度（-1），并获取对应的投影数据
    valid_data = []
    valid_angles_list = []
    for i, angle in enumerate(matched_angles):
        if angle != -1:
            valid_angles_list.append(angle)
            valid_data.append(projection_data[:, i])
    
    # 按角度排序
    sorted_indices = np.argsort(valid_angles_list)
    sorted_angles = [valid_angles_list[i] for i in sorted_indices]
    sorted_data = [valid_data[i] for i in sorted_indices]
    # 确保有180个角度和对应的数据
    if len(sorted_angles) < 180:
        print(f"警告：只有 {len(sorted_angles)} 个有效角度数据，少于期望的180个")
        # 填充缺失的数据
        while len(sorted_angles) < 180:
            sorted_angles.append(np.nan)
            sorted_data.append(np.full(projection_data.shape[0], np.nan))
    elif len(sorted_angles) > 180:
        print(f"警告：有 {len(sorted_angles)} 个有效角度数据，超过期望的180个，将只保留前180个")
        sorted_angles = sorted_angles[:180]
        sorted_data = sorted_data[:180]
    # 创建数据矩阵：第一行为角度，接下来512行为投影数据
    # 将角度格式化为1位小数
    formatted_angles = [f"{angle:.1f}" if not np.isnan(angle) else "NaN" for angle in sorted_angles]
    # 组合数据：角度作为第一行，投影数据作为后续行
    combined_data = np.vstack([sorted_angles, np.array(sorted_data).T])
    # 创建DataFrame并保存
    df = pd.DataFrame(combined_data)
    # 设置第一行为列名（角度）
    df.columns = formatted_angles
    # 移除第一行数据（因为已经用作列名）
    df = df.iloc[1:]
    # 保存为CSV文件
    df.to_csv(filename, index=False, float_format='%.6f')
    return df


def main():
    
    projection_data = pd.read_csv('data/附件2.csv', header=None).values
    print(f"投影数据形状: {projection_data.shape}")
    measured_projection_lengths = []
    for angle_idx in tqdm(range(projection_data.shape[1]), desc="处理实际投影"):
        projection_row = projection_data[:, angle_idx]
        length = calculate_projection_length(projection_row)
        measured_projection_lengths.append(length)

    # 角度\theta的采样
    angle_step = 0.1
    preset_angles = np.arange(-60, 120, angle_step)
    print(f"预设角度数量: {len(preset_angles)} (步长: {angle_step}度)")
    
    # 为每个角度计算投影长度
    calculated_projection_lengths = []
    for angle in tqdm(preset_angles, desc="计算几何投影"):
        lengths = calculate_advanced_geometric_projection(ellipse_params, circle_params, angle)
        calculated_projection_lengths.append(lengths)
    
    matched_indices = match_projections_optimized(
        measured_projection_lengths, 
        calculated_projection_lengths,
        max_candidates=100
    )
    
    matched_angles = [preset_angles[idx] if idx != -1 else -1 for idx in matched_indices]
    
    save_projection_lengths_to_csv(measured_projection_lengths, calculated_projection_lengths)
    # 可视化结果
    visualize_results(matched_angles, preset_angles)
    
    # 保存最终确定的180个角度为18x10的CSV文件
    save_angles_to_csv(matched_angles, 'angles.csv')
    
    # 保存角度和对应的投影数据
    save_projections_with_angles(matched_angles, projection_data, filename='data/新附件2.csv')
    
    return matched_angles

if __name__ == "__main__":
    matched_angles = main()






