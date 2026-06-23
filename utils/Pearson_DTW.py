"""
计算pearson+DTW相关系数
"""

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from scipy.signal import detrend 

# 模拟数据：假设输入为20天的4维价格数据（开盘价、收盘价、最高价、最低价）
# 股票A、B数据形状：(20, 4)，市场指数CSI100数据形状：(20, 4)
np.random.seed(42)
stock_a = np.random.rand(20, 4)  # 股票A价格数据
stock_b = np.random.rand(20, 4)  # 股票B价格数据
csi100 = np.random.rand(20, 4)   # 市场指数数据


def partial_correlation(x, y, z):
    """
    计算控制变量z后的偏相关系数
    x, y, z: 一维数组（需先压缩为单维度，如取收盘价）
    """
    # 示例：取收盘价（第1列，索引0，假设数据格式为开盘价(0), 收盘价(1), 最高价(2), 最低价(3)）
    x = x[:, 1]  # 股票X收盘价
    y = y[:, 1]  # 股票Y收盘价
    z = z[:, 1]  # 市场指数收盘价
    
    # 计算简单相关系数
    r_xy = pearsonr(x, y)[0]
    r_xz = pearsonr(x, z)[0]
    r_yz = pearsonr(y, z)[0]
    
    # 偏相关系数公式
    numerator = r_xy - r_xz * r_yz
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return numerator / denominator if denominator != 0 else 0  # 避免除零

# 计算股票A与B的偏相关系数（控制CSI100）
gamma_ijh = partial_correlation(stock_a, stock_b, csi100)


def time_weighted_dtw(seq1, seq2, alpha=0.98):
    """
    时间加权DTW距离计算
    seq1, seq2: 一维数组（需先压缩为单维度，如取收盘价）
    alpha: 时间权重系数（文档中取0.98）
    """
    seq1 = seq1[:, 1]  # 取收盘价
    seq2 = seq2[:, 1]
    
    m, n = len(seq1), len(seq2)
    dtw_matrix = np.zeros((m+1, n+1))
    
    # 初始化边界条件
    dtw_matrix[0, 1:] = np.inf
    dtw_matrix[1:, 0] = np.inf
    dtw_matrix[0, 0] = 0
    
    for i in range(1, m+1):
        for j in range(1, n+1):
            # 点间距离（欧氏距离）
            dist = euclidean([seq1[i-1]], [seq2[j-1]])
            # 时间加权递推公式
            dtw_matrix[i, j] = alpha * dist + (1 - alpha) * min(
                dtw_matrix[i-1, j],    # 左
                dtw_matrix[i, j-1],    # 上
                dtw_matrix[i-1, j-1]   # 对角线
            )
    return dtw_matrix[m, n]  # 返回最小累积距离

def pearson_dtw(stock_a, stock_b,alpha1,alpha2):
# 计算股票A与B的时间加权DTW距离
    dtw_distance = time_weighted_dtw(stock_a, stock_b)
    # 参数设置（文档未明确权重，此处为经验值，需调优）
    alpha1 = 0.6  # DTW相似性权重
    alpha2 = 0.4  # 偏相关系数权重

    # 将DTW距离转换为相似性
    dtw_similarity = 1 / (1 + dtw_distance)

    # 线性组合生成联动指标
    linkage_score = alpha1 * dtw_similarity + alpha2 * gamma_ijh

    print(f"皮尔逊偏相关系数: {gamma_ijh:.4f}")
    print(f"时间加权DTW距离: {dtw_distance:.4f}")
    print(f"DTW相似性: {dtw_similarity:.4f}")
    print(f"股票联动指标: {linkage_score:.4f}")
    return linkage_score