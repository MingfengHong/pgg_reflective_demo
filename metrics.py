"""
metrics.py
指标计算函数：Gini系数、不平等度量等
"""

import numpy as np
from typing import List, Union


def gini(x: Union[List[float], np.ndarray]) -> float:
    """
    计算Gini系数
    
    参数:
        x: 收入或财富的数组
    
    返回:
        Gini系数 (0-1之间，0表示完全平等，1表示完全不平等)
    """
    x = np.array(x, dtype=float)
    
    # 处理负值：平移到非负
    if np.amin(x) < 0:
        x = x - np.amin(x)
    
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    
    # Gini = sum|xi - xj| / (2 * n^2 * mean)
    diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    return diff_sum / (2 * len(x) ** 2 * mean_x)


def coefficient_of_variation(x: Union[List[float], np.ndarray]) -> float:
    """
    计算变异系数（CV = std / mean）
    
    参数:
        x: 数值数组
    
    返回:
        变异系数
    """
    x = np.array(x, dtype=float)
    mean_x = np.mean(x)
    if mean_x == 0:
        return 0.0
    return np.std(x) / mean_x


def compliance_rate(contributions: List[float], endowments: List[float], tau: float) -> float:
    """
    计算合规率：投入超过规范阈值的比例
    
    参数:
        contributions: 各个体的贡献
        endowments: 各个体的初始禀赋
        tau: 规范阈值（比例）
    
    返回:
        合规率 (0-1)
    """
    c = np.array(contributions)
    e = np.array(endowments)
    threshold = tau * e
    return float(np.mean(c >= threshold))


def mean_absolute_deviation(x: Union[List[float], np.ndarray]) -> float:
    """
    计算平均绝对偏差
    
    参数:
        x: 数值数组
    
    返回:
        平均绝对偏差
    """
    x = np.array(x, dtype=float)
    return float(np.mean(np.abs(x - np.mean(x))))


def social_welfare(incomes: List[float], alpha: float = 0.0) -> float:
    """
    计算社会福利（可选择不同的福利函数）
    
    参数:
        incomes: 收入数组
        alpha: 不平等厌恶系数
               alpha=0: 功利主义（总和）
               alpha=1: 对数福利
               alpha->∞: Rawlsian（最小值）
    
    返回:
        社会福利值
    """
    incomes = np.array(incomes, dtype=float)
    
    if alpha == 0.0:
        return float(np.sum(incomes))
    elif alpha == 1.0:
        return float(np.sum(np.log(np.maximum(incomes, 1e-6))))
    else:
        # 广义平均（Atkinson福利函数）
        return float(np.mean(np.maximum(incomes, 1e-6) ** (1 - alpha))) ** (1 / (1 - alpha))


def deviation_rate(contributions: List[float], endowments: List[float], tau: float) -> float:
    """
    计算偏离率：投入低于规范阈值的比例
    
    参数:
        contributions: 各个体的贡献
        endowments: 各个体的初始禀赋
        tau: 规范阈值（比例）
    
    返回:
        偏离率 (0-1)
    """
    return 1.0 - compliance_rate(contributions, endowments, tau)


def average_contribution_rate(contributions: List[float], endowments: List[float]) -> float:
    """
    计算平均贡献率（相对于禀赋的比例）
    
    参数:
        contributions: 各个体的贡献
        endowments: 各个体的初始禀赋
    
    返回:
        平均贡献率 (0-1)
    """
    c = np.array(contributions)
    e = np.array(endowments)
    return float(np.mean(c / np.maximum(e, 1e-6)))

