import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm


def KLDIVloss(output, target, V, D, loss_cuda):
    """
    output (batch, vocab_size)
    target (batch,)
    criterion (nn.KLDIVLoss)
    V (vocab_size, k)
    D (vocab_size, k)
    """
    # (batch, k) index in vocab_size dimension
    # k-nearest neighbors for target
    indices = torch.index_select(V, 0, target)
    # (batch, k) gather along vocab_size dimension
    outputk = torch.gather(output, 1, indices)
    # (batch, k) index in vocab_size dimension
    targetk = torch.index_select(D, 0, target)
    # KLDIVcriterion
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(outputk, targetk)


def clusterLoss(q, p, loss_cuda):
    '''
    calculate the KL loss for clustering
    '''
    q, p = q.to(loss_cuda), p.to(loss_cuda)
    criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
    return criterion(q.log(), p)


#  need to rewrite
def triLoss(a, p, n, autoencoder, loss_cuda):
    """
    a (named tuple): anchor data
    p (named tuple): positive data
    n (named tuple): negative data
    """
    a_src, a_lengths, a_invp = a.src, a.lengths, a.invp
    p_src, p_lengths, p_invp = p.src, p.lengths, p.invp
    n_src, n_lengths, n_invp = n.src, n.lengths, n.invp

    a_src, a_lengths, a_invp = a_src.to(
        loss_cuda), a_lengths.to(loss_cuda), a_invp.to(loss_cuda)
    p_src, p_lengths, p_invp = p_src.to(
        loss_cuda), p_lengths.to(loss_cuda), p_invp.to(loss_cuda)
    n_src, n_lengths, n_invp = n_src.to(
        loss_cuda), n_lengths.to(loss_cuda), n_invp.to(loss_cuda)

    a_context = autoencoder.encoder_hn(a_src, a_lengths)
    p_context = autoencoder.encoder_hn(p_src, p_lengths)
    n_context = autoencoder.encoder_hn(n_src, n_lengths)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2).to(loss_cuda)

    return triplet_loss(a_context[a_invp], p_context[p_invp], n_context[n_invp])


import numpy as np

def find_closest_segment(point, simplified_trajectory):
    """找到简化轨迹中离给定点最近的线段"""
    min_distance = float('inf')
    closest_segment = None
    for i in range(len(simplified_trajectory) - 1):
        segment_start = simplified_trajectory[i]
        segment_end = simplified_trajectory[i + 1]
        distance = perpendicular_distance(point, segment_start, segment_end)
        if distance < min_distance:
            min_distance = distance
            closest_segment = (segment_start, segment_end)
    # 如果找不到最近的线段，返回一个警告信息
    if closest_segment is None:
        print(f"Warning: Could not find a closest segment for point: {point}")
    return closest_segment

def trajectory_distance(original_trajectory, simplified_trajectory):
    """计算原始轨迹和简化轨迹之间的垂直欧几里得距离"""
    total_distance = 0
    for point in original_trajectory:
        closest_segment = find_closest_segment(point, simplified_trajectory)
        if closest_segment is not None:  # 检查是否找到最近线段
            total_distance += perpendicular_distance(point, *closest_segment)
    return total_distance / len(original_trajectory)

# def synchronous_euclidean_distance(original_trajectory, simplified_trajectory):
#     """计算原始轨迹和简化轨迹之间的同步欧几里得距离"""
#     total_distance = 0
#     # 确保两个轨迹长度相同或你有方法来同步这两个轨迹
#     for point_orig, point_simpl in zip(original_trajectory, simplified_trajectory):
#         total_distance += np.linalg.norm(np.array(point_orig) - np.array(point_simpl))
#     return total_distance / len(original_trajectory)

def point_to_line_distance(point, line_start, line_end):
    """计算点到线段的垂直距离"""
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)  # 如果线段长度为0，返回点到起点的距离
    line_unit_vec = line_vec / line_len
    proj_length = np.dot(point_vec, line_unit_vec)
    proj_vec = proj_length * line_unit_vec
    closest_point = np.array(line_start) + proj_vec
    if proj_length < 0:
        closest_point = line_start
    elif proj_length > line_len:
        closest_point = line_end
    return np.linalg.norm(np.array(point) - closest_point)

def sed_error(original_trajectory, simplified_trajectory):
    """计算原始轨迹和简化轨迹之间的SED误差"""
    # 检查简化轨迹是否为空
    if len(simplified_trajectory) == 0:
        raise ValueError("简化轨迹为空，无法计算 SED 误差")

    total_error = 0
    for i, point in enumerate(original_trajectory):
        # 如果当前点在简化轨迹中，误差为0
        if point in simplified_trajectory:
            continue

        # 找到点的前驱和后继点
        # 前驱点
        if i == 0 or len(simplified_trajectory) == 1:
            pred = simplified_trajectory[0]  # 如果是第一个点，或者只有一个点，前驱点就是第一个点
        else:
            pred_index = max(0, min(len(simplified_trajectory) - 1, i - 1))
            pred = simplified_trajectory[pred_index]

        # 后继点
        if i >= len(simplified_trajectory) - 1:
            succ = simplified_trajectory[-1]  # 如果是最后一个点，后继点就是简化轨迹的最后一个点
        else:
            succ_index = max(0, min(len(simplified_trajectory) - 1, i + 1))
            succ = simplified_trajectory[succ_index]

        # 计算点到线段的垂直距离
        error = point_to_line_distance(point, pred, succ)
        total_error += error

    return total_error / len(original_trajectory)


import numpy as np

def euclidean_distance(point_a, point_b):
    """计算两点之间的欧几里得距离"""
    return np.linalg.norm(np.array(point_a) - np.array(point_b))

def dynamic_time_warping(trajectory_A, trajectory_B):
    """计算两条轨迹之间的动态时间规整 (DTW) 累积距离"""
    n, m = len(trajectory_A), len(trajectory_B)
    # 初始化累积距离矩阵
    dtw_matrix = np.full((n, m), float('inf'))
    dtw_matrix[0, 0] = 0

    # 构建累积距离矩阵
    for i in range(n):
        for j in range(m):
            cost = euclidean_distance(trajectory_A[i], trajectory_B[j])
            if i > 0:
                dtw_matrix[i, j] = min(dtw_matrix[i, j], dtw_matrix[i-1, j] + cost)
            if j > 0:
                dtw_matrix[i, j] = min(dtw_matrix[i, j], dtw_matrix[i, j-1] + cost)
            if i > 0 and j > 0:
                dtw_matrix[i, j] = min(dtw_matrix[i, j], dtw_matrix[i-1, j-1] + cost)

    # 回溯最优路径以计算平均距离 (DAD)
    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            # 选择最小累积距离方向
            options = [dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]]
            min_index = np.argmin(options)
            if min_index == 0:
                i -= 1
            elif min_index == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))

    # 计算路径上的平均距离
    total_distance = 0
    for (i, j) in path:
        total_distance += euclidean_distance(trajectory_A[i], trajectory_B[j])

    dad_distance = total_distance / len(path)
    return dad_distance




def perpendicular_distance(pt, line_start, line_end):
    """计算点到线段的垂直距离"""
    pt = np.array(pt)
    line_start = np.array(line_start)
    line_end = np.array(line_end)
    if np.all(line_start == line_end):
        return np.linalg.norm(pt - line_start)
    return np.abs(np.linalg.norm(np.cross(line_end - line_start, line_start - pt))) / np.linalg.norm(line_end - line_start)


def douglas_peucker(points, epsilon):
    """道格拉斯-普克算法实现"""
    # 找到距离最远的点
    dmax = 0.0
    index = 0
    for i in range(1, len(points) - 1):
        d = perpendicular_distance(np.array(points[i]), np.array(points[0]), np.array(points[-1]))
        if d > dmax:
            index = i
            dmax = d

    # 如果最大距离大于阈值，则递归地简化
    print(dmax)
    if dmax >= epsilon:
        # 递归地简化子轨迹
        rec_results1 = douglas_peucker(points[:index + 1], epsilon)
        rec_results2 = douglas_peucker(points[index:], epsilon)

        # 将结果组合起来
        result = rec_results1[:-1] + rec_results2
    else:
        result = [points[0], points[-1]]

    return result


def calculate_area(p1, p2, p3):
    """计算由三个点形成的三角形的面积"""
    return 0.5 * np.abs(p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p1[1] - p2[0] * p1[1] - p3[0] * p2[1] - p1[0] * p3[1])


def visvalingam_whyatt(points, num_points_to_keep):
    """Visvalingam-Whyatt算法实现"""
    points = list(points)

    if len(points) <= num_points_to_keep:
        return points

    # 计算所有点的有效面积
    areas = [np.inf] + [calculate_area(points[i - 1], points[i], points[i + 1]) for i in range(1, len(points) - 1)] + [
        np.inf]

    # 循环直到达到所需的点数
    while len(points) > num_points_to_keep:
        # 找到最小面积的点
        min_area_index = np.argmin(areas)

        # 移除这个点及其面积
        points.pop(min_area_index)
        areas.pop(min_area_index)

        # 重新计算相邻点的有效面积
        if min_area_index < len(points) - 1:
            areas[min_area_index] = calculate_area(points[min_area_index - 1], points[min_area_index],
                                                   points[min_area_index + 1])
        if min_area_index > 1:
            areas[min_area_index - 1] = calculate_area(points[min_area_index - 2], points[min_area_index - 1],
                                                       points[min_area_index])

    return points

def compare_trj(original_trj, simplified_trj):
    # 使用集合来提高查找效率
    simplified_set = set(map(tuple, simplified_trj))

    # 遍历原始轨迹，检查每个点是否被保留
    retention_list = [1 if tuple(point) in simplified_set else 0 for point in original_trj]

    return retention_list
