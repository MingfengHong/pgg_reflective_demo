"""
pgg_agent.py
定义制度（Institution）与双回路智能体（PGGAgent）
"""

from dataclasses import dataclass
import math
import numpy as np
from mesa import Agent
from typing import List


@dataclass
class Institution:
    """
    可演化的制度参数
    """
    tau: float = 0.4            # 规范阈值（最低应当投入比例）
    fine_F: float = 2.0         # 对偏离者的罚金（每名惩罚者）
    punish_cost_Cp: float = 0.6 # 惩罚者每次惩罚的成本
    
    # 元规范（可选）
    meta_on: bool = False       # 是否启用元规范
    meta_F: float = 0.5         # 元规范罚金
    meta_Cp: float = 0.2        # 元规范成本
    
    # 内生更新参数
    delta_F: float = 0.2        # 罚金调整步长
    delta_tau: float = 0.02     # 阈值调整步长
    
    # 参数边界
    tau_min: float = 0.0
    tau_max: float = 0.8
    F_min: float = 0.0
    F_max: float = 5.0

    def update_from_votes(self, votes_F: List[int], votes_tau: List[int]):
        """
        基于投票更新制度参数
        
        参数:
            votes_F: 罚金投票列表 (-1/0/+1)
            votes_tau: 阈值投票列表 (-1/0/+1)
        """
        # 多数决：计算投票符号的和
        if votes_F:
            s = np.sign(np.sum(votes_F))  # -1, 0, +1
            self.fine_F = float(np.clip(
                self.fine_F + s * self.delta_F,
                self.F_min,
                self.F_max
            ))
        
        if votes_tau:
            s = np.sign(np.sum(votes_tau))
            self.tau = float(np.clip(
                self.tau + s * self.delta_tau,
                self.tau_min,
                self.tau_max
            ))


class PGGAgent(Agent):
    """
    具备双回路的公共物品博弈智能体
    
    生产回路：基于信念做出贡献决策
    反思回路：更新对他人的期望、对制度的投票
    """
    
    def __init__(self, unique_id, model, endowment=10.0):
        super().__init__(model)
        self.unique_id = unique_id  # Mesa 3.x需要手动设置
        
        # 经济状态
        self.E = endowment          # 初始禀赋
        self.c = 0.0                # 本轮贡献
        self.income = 0.0           # 本轮收入
        
        # 信念与规范心理
        self.E_i = 0.4 * self.E     # 经验性期望（对他人平均贡献的预测）
        self.theta_i = self.model.institution.tau  # 主观规范阈值（内化的应当比例）
        
        # 制裁倾向
        self.alpha_punish = 1.0     # 惩罚倾向系数
        self.alpha_meta = 0.5       # 元规范惩罚倾向
        
        # 反思参数
        self.eta = 0.3              # 经验性期望的学习率
        self.eps = 0.1 * self.E     # 预测误差触发阈值
        
        # 投票建议
        self.vote_F = 0             # 对罚金的投票 (-1/0/+1)
        self.vote_tau = 0           # 对阈值的投票 (-1/0/+1)
        
        # 簿记变量
        self._neighbors = list(self.model.G.neighbors(self.unique_id))
        self.last_contrib = 0.0     # 上一轮的贡献
        self.punish_cost_paid = 0.0 # 本轮支付的惩罚成本
        self.fines_received = 0.0   # 本轮收到的罚金
        self._had_deviant_neighbor = False  # 是否有偏离的邻居（用于元规范）

    # ==================== 阶段1：贡献（生产回路） ====================
    
    def contribute(self):
        """
        基于条件性合作的贡献决策
        
        决策逻辑：
        - 基于对他人的经验性期望 E_i
        - 基于内化的规范阈值 theta_i
        - 基于制度威慑（罚金）fine_F（关键耦合点）
        - 使用logistic函数平滑决策
        """
        inst = self.model.institution
        
        # 条件性合作者模型（含威慑信号）
        # signal = beta0 + beta1 * (E_i - target) + beta2 * (制度阈值 - 主观阈值) + beta3 * 威慑信号
        beta0, beta1, beta2, beta3 = -1.0, 0.15, 2.0, 0.8
        target = self.theta_i * self.E  # 内化的应当贡献量
        
        # 威慑信号：罚金越高，合规动机越强
        # 将 fine_F 归一化到 [0, 1] 范围（假设 F_max = 5.0）
        deterrence_signal = inst.fine_F / inst.F_max
        
        signal = (
            beta0 
            + beta1 * (self.E_i - target)
            + beta2 * (inst.tau * self.E - target)
            + beta3 * deterrence_signal  # 关键：罚金越高，signal越大，贡献越多
        )
        
        # Logistic变换得到贡献比例
        prop = 1.0 / (1.0 + math.exp(-signal))
        
        # 添加噪声
        noise = self.random.normalvariate(0, 0.03 * self.E)
        
        # 计算最终贡献（限制在[0, E]）
        self.c = float(np.clip(prop * self.E + noise, 0.0, self.E))
        self.last_contrib = self.c

    # ==================== 阶段2：惩罚 ====================
    
    def punish(self):
        """
        对偏离规范的邻居施加惩罚
        
        惩罚逻辑：
        - 一阶惩罚：对低于阈值的邻居
        - 元规范（可选）：对有偏离邻居但未惩罚的邻居
        """
        # 确保公共池收益已计算（惰性触发）
        if not self.model.pool_resolved:
            self.model.resolve_pgg_pool()
        
        inst = self.model.institution
        tau_cut = inst.tau * self.E  # 规范阈值的绝对值
        self.punish_cost_paid = 0.0
        
        # 一阶惩罚：惩罚低于阈值的邻居
        for j in self._neighbors:
            ag = self.model.agent_dict[j]
            if ag.last_contrib < tau_cut:
                # 基于偏离程度决定惩罚概率
                if self.random.random() < self._punish_prob(ag):
                    ag.fines_received += inst.fine_F
                    self.punish_cost_paid += inst.punish_cost_Cp
        
        # 元规范（可选）：惩罚那些"应该惩罚但没有惩罚"的邻居
        if inst.meta_on:
            for j in self._neighbors:
                ag = self.model.agent_dict[j]
                # 如果j有偏离的邻居但支付了零惩罚成本，则施加元罚金
                if (ag._had_deviant_neighbor and 
                    ag.punish_cost_paid == 0.0 and 
                    self.random.random() < self.alpha_meta):
                    ag.fines_received += inst.meta_F
                    self.punish_cost_paid += inst.meta_Cp
        
        # 从收入中扣除惩罚成本
        self.income -= self.punish_cost_paid

    def _punish_prob(self, other: 'PGGAgent') -> float:
        """
        计算对某个偏离者的惩罚概率
        
        参数:
            other: 其他智能体
        
        返回:
            惩罚概率 (0-1)
        """
        inst = self.model.institution
        # 偏离量：低于阈值的部分
        dev = max(0.0, inst.tau * self.E - other.last_contrib)
        # 归一化的偏离量 -> 惩罚概率
        return float(np.clip(
            self.alpha_punish * (dev / (self.E + 1e-8)),
            0.0,
            1.0
        ))

    # ==================== 阶段3：反思（元回路） ====================
    
    def reflect(self):
        """
        反思回路：更新信念、投票建议
        
        操作：
        1. 更新对他人的经验性期望 E_i（指数平滑）
        2. 检测触发条件（预测误差、偏离率、惩罚成本）
        3. 生成对制度参数的投票建议
        4. 规范内化：将主观阈值向制度阈值靠拢
        """
        # 1. 观察邻居的实际贡献
        neigh_contrib = [
            self.model.agent_dict[j].last_contrib 
            for j in self._neighbors
        ] or [self.last_contrib]
        
        avg_obs = float(np.mean(neigh_contrib))
        
        # 计算预测误差
        pred_err = abs(avg_obs - self.E_i)
        
        # 更新经验性期望（指数平滑）
        self.E_i = (1 - self.eta) * self.E_i + self.eta * avg_obs
        
        # 2. 标记是否有偏离的邻居（用于元规范）
        inst = self.model.institution
        tau_cut = inst.tau * self.E
        self._had_deviant_neighbor = any(
            self.model.agent_dict[j].last_contrib < tau_cut 
            for j in self._neighbors
        )
        
        # 3. 计算触发指标
        # 邻居中的偏离率
        dev_rate = np.mean([
            1.0 if self.model.agent_dict[j].last_contrib < tau_cut else 0.0
            for j in self._neighbors
        ]) if self._neighbors else 0.0
        
        # 是否付出了高惩罚成本
        high_cost = self.punish_cost_paid > 0.2 * self.E
        
        # 4. 生成投票建议
        self.vote_F = 0
        
        # 策略1：预测误差高或偏离率高 -> 增加罚金
        if pred_err > self.eps or dev_rate > 0.4:
            self.vote_F = +1
        
        # 策略2：成本高但偏离率低 -> 减少罚金
        if high_cost and dev_rate < 0.2:
            self.vote_F = -1
        
        # 对阈值的投票（简化版本：暂不投票）
        self.vote_tau = 0
        
        # 5. 规范内化：主观阈值向制度阈值漂移
        self.theta_i = 0.9 * self.theta_i + 0.1 * inst.tau

