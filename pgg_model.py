"""
pgg_model.py
公共物品博弈模型：支持内生制度演化与双回路智能体
"""

import numpy as np
import networkx as nx
from mesa import Model, Agent
from mesa.datacollection import DataCollector

from pgg_agent import PGGAgent, Institution
from metrics import gini


class PGGModel(Model):
    """
    公共物品博弈模型
    
    特性：
    - 三阶段激活：贡献 -> 惩罚 -> 反思
    - 内生制度更新：基于智能体投票
    - 支持多种网络拓扑
    """
    
    def __init__(
        self,
        N: int = 50,                    # 智能体数量
        endowment: float = 10.0,        # 初始禀赋
        r: float = 1.6,                 # 公共物品倍增系数
        seed: int = None,               # 随机种子
        graph_kind: str = "ws",         # 网络类型
        k: int = 6,                     # 网络参数（平均度）
        p: float = 0.1,                 # 网络参数（重连概率/边概率）
        institution: Institution = None # 制度配置
    ):
        super().__init__(seed=seed)
        
        # 模型参数
        self.N = N
        self.endowment = endowment
        self.r = r
        
        # 创建网络
        self.G = self._make_graph(graph_kind, N, k, p)
        
        # 制度
        self.institution = institution if institution else Institution()
        
        # 创建智能体 (Mesa 3.x中agents是保留属性，使用agent_dict)
        self.agent_dict = {}
        self.agent_list = []
        for i in range(N):
            agent = PGGAgent(i, self, endowment=endowment)
            self.agent_dict[i] = agent
            self.agent_list.append(agent)
        
        # 标志：公共池是否已解决
        self.pool_resolved = False
        
        # 数据收集
        self._init_datacollector()

    def _make_graph(self, kind: str, N: int, k: int, p: float) -> nx.Graph:
        """
        创建网络拓扑
        
        参数:
            kind: 网络类型 ("complete", "ws", "er")
            N: 节点数
            k: 平均度
            p: 概率参数
        
        返回:
            networkx图
        """
        if kind == "complete":
            return nx.complete_graph(N)
        elif kind == "ws":  # Watts-Strogatz小世界网络
            # k必须<N且为偶数
            k_adj = min(k, N - 1)
            if k_adj % 2 == 1:
                k_adj -= 1
            k_adj = max(2, k_adj)  # 至少为2
            return nx.watts_strogatz_graph(N, k_adj, p)
        elif kind == "er":  # Erdős-Rényi随机图
            return nx.erdos_renyi_graph(N, p or 0.1)
        elif kind == "ba":  # Barabási-Albert无标度网络
            m = max(1, min(k // 2, N - 1))
            return nx.barabasi_albert_graph(N, m)
        else:
            # 默认：小世界网络
            k_adj = min(k, N - 1)
            if k_adj % 2 == 1:
                k_adj -= 1
            k_adj = max(2, k_adj)
            return nx.watts_strogatz_graph(N, k_adj, p)

    def _init_datacollector(self):
        """
        初始化数据收集器
        
        模型级指标：
        - 平均贡献、合规率、收入、不平等
        - 制度参数轨迹
        - 惩罚成本与罚金
        """
        self.datacollector = DataCollector(
            model_reporters={
                # 贡献与合规
                "mean_contrib": lambda m: float(np.mean([
                    ag.last_contrib for ag in m.agent_dict.values()
                ])),
                "compliance_rate": lambda m: float(np.mean([
                    1.0 if ag.last_contrib >= m.institution.tau * ag.E else 0.0
                    for ag in m.agent_dict.values()
                ])),
                "contrib_rate": lambda m: float(np.mean([
                    ag.last_contrib / ag.E for ag in m.agent_dict.values()
                ])),
                
                # 收入与福利
                "mean_income": lambda m: float(np.mean([
                    ag.income for ag in m.agent_dict.values()
                ])),
                "total_income": lambda m: float(np.sum([
                    ag.income for ag in m.agent_dict.values()
                ])),
                "gini_income": lambda m: float(gini([
                    ag.income for ag in m.agent_dict.values()
                ])),
                
                # 制度参数
                "fine_F": lambda m: m.institution.fine_F,
                "tau": lambda m: m.institution.tau,
                
                # 惩罚与制裁
                "total_punish_cost": lambda m: float(np.sum([
                    ag.punish_cost_paid for ag in m.agent_dict.values()
                ])),
                "total_fines": lambda m: float(np.sum([
                    ag.fines_received for ag in m.agent_dict.values()
                ])),
                "avg_fines": lambda m: float(np.mean([
                    ag.fines_received for ag in m.agent_dict.values()
                ])),
                
                # 信念与期望
                "mean_E_i": lambda m: float(np.mean([
                    ag.E_i for ag in m.agent_dict.values()
                ])),
                "mean_theta_i": lambda m: float(np.mean([
                    ag.theta_i for ag in m.agent_dict.values()
                ])),
            },
            agent_reporters={
                # 可选：收集个体级数据
                # "contribution": "last_contrib",
                # "income": "income",
                # "E_i": "E_i",
            }
        )

    def resolve_pgg_pool(self):
        """
        解决公共物品池：计算收益并分配
        
        在每轮的"惩罚"阶段开始时调用一次
        """
        # 收集所有贡献
        contribs = [ag.c for ag in self.agent_dict.values()]
        total = float(np.sum(contribs))
        
        # 公共池回报：r * 总贡献 / N
        share = self.r * total / self.N
        
        # 分配给每个智能体
        for ag in self.agent_dict.values():
            ag.income = ag.E - ag.c + share
            ag.fines_received = 0.0     # 重置（在惩罚阶段累积）
            ag.punish_cost_paid = 0.0   # 重置
        
        self.pool_resolved = True

    def _update_institution_from_votes(self):
        """
        基于智能体投票内生更新制度参数
        """
        votes_F = [ag.vote_F for ag in self.agent_dict.values()]
        votes_tau = [ag.vote_tau for ag in self.agent_dict.values()]
        self.institution.update_from_votes(votes_F, votes_tau)

    def step(self):
        """
        执行一个时间步
        
        流程：
        1. 重置标志位
        2. 三阶段激活（contribute -> punish -> reflect）
        3. 内生制度更新
        4. 数据收集
        """
        self.pool_resolved = False
        
        # 随机打乱agent顺序（每个阶段）
        import random
        
        # 阶段1：贡献
        agents_shuffled = self.agent_list.copy()
        random.shuffle(agents_shuffled)
        for agent in agents_shuffled:
            agent.contribute()
        
        # 阶段2：惩罚
        agents_shuffled = self.agent_list.copy()
        random.shuffle(agents_shuffled)
        for agent in agents_shuffled:
            agent.punish()
        
        # 阶段3：反思
        agents_shuffled = self.agent_list.copy()
        random.shuffle(agents_shuffled)
        for agent in agents_shuffled:
            agent.reflect()
        
        # 内生制度更新
        self._update_institution_from_votes()
        
        # 收集数据
        self.datacollector.collect(self)

    def run_model(self, steps: int = 200):
        """
        运行模型指定步数
        
        参数:
            steps: 时间步数
        """
        for _ in range(steps):
            self.step()

