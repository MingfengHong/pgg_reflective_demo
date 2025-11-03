# 公共物品博弈 + 内生制度演化 Demo

> "二阶控制论"与 ABM：如何模拟"会反思"的智能体？

## 🚀 快速开始

### 方法一：命令行运行

```bash
# 1. 克隆仓库
git clone https://github.com/MingfengHong/pgg_reflective_demo.git
cd pgg_reflective_demo

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行模拟
python main.py
```

### 方法二：IDE运行

1. **打开项目**：在 PyCharm/VSCode 中打开 `pgg_reflective_demo` 文件夹
2. **安装依赖**：`pip install -r requirements.txt`
3. **运行**：直接运行 `main.py`

### 查看结果：

- ✅ **控制台输出**：统计摘要（合规率、收入、Gini等）
- ✅ **CSV文件**：`simulation_results.csv`（每步详细数据）
- ✅ **可视化图表**：`simulation_plot.png`（4个子图）📊

### 修改参数：

在 `main.py` 第29-48行修改参数，保存后重新运行。

**高压力情景参数**（展示反思回路）：
```python
r = 1.6        # 倍增系数（低回报，制造困境）
tau = 0.4      # 规范阈值（高门槛，难达到）
fine_F = 0.5   # 初始罚金（弱制度，触发自救）
```

---

## 📋 目录

- [快速开始](#快速开始)
- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [安装](#安装)
- [项目结构](#项目结构)
- [模型设计](#模型设计)
- [使用指南](#使用指南)
- [实验设计](#实验设计)
- [科学复现要点](#科学复现要点)
- [扩展方向](#扩展方向)
- [参考文献](#参考文献)
- [联系方式](#联系方式)

---

## 项目简介

本项目实现了一个基于 Mesa 的**公共物品博弈（PGG）**模型，核心创新在于：

1. **双回路智能体**：生产回路（决策）+ 反思回路（元认知、投票）
2. **内生制度演化**：制度参数（罚金、阈值）由群体投票动态调整
3. **有限 Theory of Mind**：智能体维护对他人的经验性期望并更新
4. **元规范机制**：可选的"惩罚不惩罚者"机制（Axelrod 1986）

这是一个**最小可行演示（MVD）**，旨在提供可扩展的研究型代码架构，支持进一步的理论探索和实证对齐。

---

## 核心特性

### 1. 双回路智能体（Double-Loop Agent）

每个智能体在每轮执行三个阶段：

- **贡献阶段（contribute）**：基于信念和内化规范决定对公共池的投入
- **惩罚阶段（punish）**：对低于阈值的邻居施加制裁（可选元规范）
- **反思阶段（reflect）**：
  - 更新对他人的经验性期望 \( E_i \)（指数平滑）
  - 检测触发条件（预测误差、偏离率、成本）
  - 生成对制度参数的投票建议
  - 规范内化：主观阈值向制度阈值漂移

### 2. 内生制度（Endogenous Institution）

制度包含可演化参数：

- **τ**：规范阈值（最低应当投入比例）
- **F**：对偏离者的罚金
- **C_p**：惩罚成本
- **元规范开关**：是否惩罚"不惩罚者"

每轮结束后，根据智能体投票（-1/0/+1）通过多数决调整参数，带步长和边界限制。

### 3. 网络拓扑

支持多种网络结构：

- **完全图（complete）**：所有人与所有人相连
- **小世界网络（ws）**：Watts-Strogatz，高聚类 + 短路径
- **随机图（er）**：Erdős-Rényi
- **无标度网络（ba）**：Barabási-Albert

### 4. 数据收集

自动收集以下指标：

- 贡献与合规率
- 收入与不平等（Gini 系数）
- 制度参数轨迹（F, τ）
- 惩罚成本与罚金
- 信念演化（经验性期望、主观阈值）

---

## 安装

### 环境要求

- Python 3.8+
- 依赖包见 `requirements.txt`

### 安装步骤

```bash
# 1. 克隆或下载项目
cd pgg_reflective_demo

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt
```

---

## 快速开始

### 单次运行

```bash
python run_single.py --N 50 --r 1.6 --steps 200 --seed 42 --plot
```

**参数说明**：

- `--N`：智能体数量（默认 50）
- `--r`：公共物品倍增系数（默认 1.6）
- `--steps`：模拟步数（默认 200）
- `--seed`：随机种子（默认 42）
- `--meta`：启用元规范
- `--plot`：生成可视化图表

**输出**：

- `results/simulation_data.csv`：时间序列数据
- `results/simulation_results.png`：可视化图表（如果使用 `--plot`）

### 批量实验

```bash
# 实验1：罚金步长 vs. 倍增系数
python run_batch.py --experiment fine_vs_r --steps 200 --replicates 5 --plot

# 实验2：元规范对比
python run_batch.py --experiment meta_comparison --replicates 10 --plot

# 实验3：网络拓扑
python run_batch.py --experiment network_topology --replicates 5
```

**输出**：

- `batch_results/exp1_fine_vs_r.csv`：参数扫描结果
- `batch_results/exp1_heatmap.png`：热图可视化

---

## 项目结构

```
pgg_reflective_demo/
├── pgg_model.py           # Mesa Model 类（制度、调度、数据收集）
├── pgg_agent.py           # Mesa Agent 类 + Institution（双回路智能体）
├── metrics.py             # 指标函数（Gini、合规率等）
├── run_single.py          # 单次运行脚本（CLI）
├── run_batch.py           # 批量实验脚本（参数扫描）
├── requirements.txt       # 依赖包
└── README.md              # 本文档
```

---

## 模型设计

### 公共物品博弈（PGG）

每轮：

1. 每个体从禀赋 \( E \) 中投入 \( c_i \in [0, E] \) 到公共池
2. 公共池总额 \( C = \sum_i c_i \) 乘以系数 \( r \)（通常 \( 1 < r < N \)）
3. 均分给所有人：每人获得 \( \frac{rC}{N} \)
4. 个体收益：\( \pi_i = E - c_i + \frac{rC}{N} \)

**社会困境**：个人最优策略是免费搭车（\( c_i = 0 \)），但集体最优是全投入（\( c_i = E \)）。

### 制度机制

#### 一阶规范

- 群体有一个**规范阈值** \( \tau \in [0, 1] \)，表示"应当至少投入 \( \tau E \)"
- 低于阈值者被视为**偏离者**

#### 惩罚

- 邻居可对偏离者施加罚金 \( F \)（可叠加）
- 惩罚者付出成本 \( C_p \)

#### 元规范（可选）

- 如果某智能体的邻居中有偏离者，但该智能体未惩罚，则可能被其他邻居施加**元罚金** \( F_{meta} \)
- 元惩罚者付出成本 \( C_{p,meta} \)

### 双回路智能体

#### 生产回路（Stage 1: contribute）

决策公式（条件性合作）：

```
signal = β₀ + β₁(E_i - target) + β₂(τE - target)
prop = sigmoid(signal)
c_i = prop * E + noise
```

其中：

- \( E_i \)：对他人平均贡献的经验性期望
- \( target = \theta_i \cdot E \)：内化的应当贡献量
- \( \theta_i \)：主观规范阈值

#### 反思回路（Stage 3: reflect）

1. **更新经验性期望**（指数平滑）：
   ```
   E_i ← (1 - η)E_i + η · avg_neighbor_contrib
   ```

2. **触发反思**（当满足以下条件之一）：
   - 预测误差高：\( |avg_{obs} - E_i| > \epsilon \)
   - 邻居偏离率高：\( > 40\% \)
   - 惩罚成本高：\( > 0.2E \)

3. **生成投票**：
   - `vote_F = +1`：预测误差高或偏离率高 → 增加罚金
   - `vote_F = -1`：成本高但偏离率低 → 减少罚金

4. **规范内化**：
   ```
   θ_i ← 0.9θ_i + 0.1τ
   ```

### 制度内生更新

每轮结束后：

```python
# 收集所有智能体的投票
votes_F = [agent.vote_F for agent in agents]  # 每个 ∈ {-1, 0, +1}

# 多数决
sign = sign(sum(votes_F))

# 更新罚金（带步长和边界）
F ← clip(F + sign * δ_F, F_min, F_max)
```

---

## 参数调节指南 ⚙️

### 核心参数说明

**公共物品倍增系数 r**（最关键）
- `r = 1.0-1.5`：合作难以维持（Nash均衡主导）
- `r = 1.5-2.0`：需要强制裁才能合作
- `r = 2.0-3.0`：✅ **推荐范围**，容易观察制度演化
- `r > 3.0`：自发合作，制度作用不明显

**规范阈值 τ**（贡献要求）
- `τ = 0.1-0.2`：✅ **容易达到**，适合研究制度维持
- `τ = 0.2-0.3`：中等难度，能看到投票调整
- `τ = 0.3-0.5`：较难达到，可能导致惩罚升级
- `τ > 0.5`：过高，系统可能崩溃（如您刚才的情况）

**初始罚金 F**
- `F = 0.5-1.0`：轻度惩罚
- `F = 1.0-2.0`：✅ **推荐**，平衡效果
- `F > 3.0`：重度惩罚，可能过度

### 实验场景推荐

**场景1：观察自发合作**
```python
r = 2.8          # 高回报
tau = 0.15       # 低阈值
fine_F = 1.0     # 轻惩罚
```
预期：高合规率，惩罚很少

**场景2：研究制度演化**
```python
r = 2.0          # 中等回报
tau = 0.25       # 中等阈值
fine_F = 1.5     # 中等惩罚
```
预期：罚金会上下调整，能看到动态演化

**场景3：测试制度韧性**
```python
r = 1.6          # 较低回报
tau = 0.3        # 较高阈值
fine_F = 2.0     # 较高惩罚
```
预期：合作艰难维持，惩罚成本高

**场景4：元规范实验**
```python
r = 2.2
tau = 0.2
meta_on = True   # 启用元规范
```
预期：观察"惩罚不惩罚者"的效果

### 常见问题诊断

**❌ 问题：合规率一直为0**
```
原因：阈值太高或倍增系数太低
解决：降低 tau 到 0.2 或提高 r 到 2.5
```

**❌ 问题：罚金涨到上限**
```
原因：偏离率高，系统拼命惩罚
解决：提高 r 或降低 tau
```

**❌ 问题：惩罚成本很高但无效**
```
原因：参数组合不可行
解决：重新平衡 r, tau, fine_F
```

**✅ 理想状态：**
```
合规率：60%-100%
罚金：稳定或小幅波动
惩罚成本：低或为0
收入：明显高于Nash均衡（10.0）
```

---

## 使用指南

### 修改模型参数

#### 方法1：命令行参数

```bash
python run_single.py \
  --N 100 \
  --r 1.8 \
  --endowment 20.0 \
  --tau 0.5 \
  --fine_F 3.0 \
  --punish_cost 0.8 \
  --meta \
  --graph ba \
  --steps 500
```

#### 方法2：修改代码

编辑 `run_single.py` 或直接在 Python 中：

```python
from pgg_model import PGGModel
from pgg_agent import Institution

# 自定义制度
institution = Institution(
    tau=0.5,
    fine_F=3.0,
    punish_cost_Cp=0.8,
    meta_on=True,
    delta_F=0.15,  # 罚金调整步长
    delta_tau=0.01
)

# 创建模型
model = PGGModel(
    N=100,
    r=1.8,
    seed=42,
    graph_kind='ba',
    k=4,
    institution=institution
)

# 运行
model.run_model(steps=500)

# 获取数据
df = model.datacollector.get_model_vars_dataframe()
print(df.tail())
```

### 分析结果

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv('results/simulation_data.csv', index_col=0)

# 绘制制度演化
fig, ax = plt.subplots()
ax.plot(df.index, df['fine_F'], label='罚金 F')
ax.plot(df.index, df['tau'] * 10, label='阈值 τ × 10')
ax.set_xlabel('时间步')
ax.legend()
plt.show()

# 计算长期平均值
last_50 = df.tail(50)
print(f"稳态合规率: {last_50['compliance_rate'].mean():.3f}")
print(f"稳态罚金: {last_50['fine_F'].mean():.3f}")
```

---

## 实验设计

### 实验1：罚金步长敏感性

**研究问题**：制度调整的步长如何影响收敛速度和稳定性？

**变量**：

- 自变量：`delta_F ∈ {0.05, 0.1, 0.2, 0.5}` × `r ∈ {1.2, 1.4, 1.6, 1.8, 2.0}`
- 因变量：平均贡献、合规率、罚金波动性

**预期**：

- 步长过小 → 收敛慢
- 步长过大 → 振荡/过冲

### 实验2：元规范效应

**研究问题**：元规范如何影响惩罚覆盖率与合作水平？

**对照组**：

- 控制组：`meta_on=False`
- 实验组：`meta_on=True`

**变量**：`r ∈ {1.4, 1.6, 1.8}`

**预期**（Axelrod 1986）：

- 元规范 → 惩罚覆盖率 ↑ → 偏离率 ↓ → 合作 ↑
- 但总成本可能增加

### 实验3：网络拓扑

**研究问题**：网络结构如何影响规范扩散？

**变量**：

- `graph_kind ∈ {complete, ws, er, ba}`
- 对于 WS：`p ∈ {0.05, 0.1, 0.2}`（重连概率）

**预期**：

- 完全图 → 快速全局共识
- 小世界 → 局部簇 + 长程传播
- 无标度 → hub 节点主导规范

---

## 科学复现要点

为确保结果可复现，请遵循以下原则：

### 1. 固定随机种子

```bash
python run_single.py --seed 42
```

或在代码中：

```python
model = PGGModel(seed=42, ...)
```

### 2. 记录完整参数

模型参数包括：

- 智能体数量 N
- 倍增系数 r、禀赋 E
- 初始制度参数（τ, F, C_p, δ_F, δ_τ）
- 网络类型与参数（kind, k, p）
- 智能体参数（η, ε, β₀, β₁, β₂）

建议使用配置文件（YAML/JSON）或在输出中记录：

```python
import json

config = {
    'N': 50,
    'r': 1.6,
    'seed': 42,
    # ...
}

with open('config.json', 'w') as f:
    json.dump(config, f, indent=2)
```

### 3. 版本控制

在 `requirements.txt` 中固定版本：

```
mesa==2.1.5
numpy==1.24.3
...
```

安装：

```bash
pip install -r requirements.txt
```

### 4. 保存制度版本史

```python
# 在 Model 中添加
self.institution_history = []

def step(self):
    # ...
    self.institution_history.append({
        'step': self.schedule.steps,
        'tau': self.institution.tau,
        'fine_F': self.institution.fine_F
    })
```

### 5. 批量实验的重复次数

对于参数扫描，建议至少 **5-10 次重复**，报告均值 ± 标准差。

---

## 扩展方向

从最小可行演示（MVD）到"有料"的研究模型：

### 1. 更真实的反思回路

- **主动推断**：用期望自由能（Expected Free Energy）决定是否反思
- **不确定性门控**：基于信念的熵或预测方差
- **声誉系统**：记录惩罚/不惩罚历史，影响他人对自己的期望

### 2. 多参数治理

- **联合投票**：对（F, τ, C_p）同时调整
- **门槛规则**：需 2/3 多数才能改变制度
- **渐进式制裁**：初犯轻罚，累犯重罚（Ostrom 设计原则）

### 3. 度量-目标回路（Goodhart's Law）

- 引入"绩效指标"（如贡献率）与资源分配绑定
- 允许智能体策略化回应（gaming）
- 观察指标失效与制度自我矫正

### 4. 语言与话语层

- **Rational Speech Acts（RSA）**：智能体通过话语传播规范性期望
- **框架竞争**：不同群体推广不同的"应当"定义
- **LLM 驱动的生成式智能体**：用语言模型生成自然语言推理

### 5. 递归 Theory of Mind

- **Level-k / 认知层级**：有限深度的递归推理
- **I-POMDP**：显式建模他人的信念与目标
- **对手建模**：在 MARL 中学习他人策略

### 6. 可微 ABM 与数据对齐

- **GradABM**：用自动微分对参数做梯度优化
- **变分推断（VI）**：将 ABM 参数对准真实数据
- **灵敏度分析**：全局敏感性与不确定性传播

### 7. 多层次治理

- **嵌套制度**（Ostrom）：局部群体 + 全局制度
- **联邦投票**：局部先投票，再向上层汇报
- **制度实验室**：允许子群体试行新规则，成功后扩散

### 8. 真实数据校准

- **问卷数据**：测量真实人群的经验/规范性期望
- **实验室/田野实验**：公共资源、社区治理
- **文本分析**：从社交媒体/政策文本提取规范话语

---

## 参考文献

### 二阶控制论与自创生

- Foerster, H. von (1979). *Cybernetics of Cybernetics*. University of Illinois.
- Glanville, R. (2002). Second order cybernetics. *Encyclopedia of Life Support Systems (EOLSS)*.
- Maturana, H., & Varela, F. (1980). *Autopoiesis and Cognition*. Reidel.

### 生成式社会科学与 ABM

- Epstein, J. M. (2006). *Generative Social Science*. Princeton University Press.
- Epstein, J. M., & Axtell, R. (1996). *Growing Artificial Societies: Social Science from the Bottom Up*. MIT Press.

### 规范涌现

- Axelrod, R. (1986). An evolutionary approach to norms. *American Political Science Review*, 80(4), 1095-1111.
- Bicchieri, C. (2006). *The Grammar of Society*. Cambridge University Press.
- Shoham, Y., & Tennenholtz, M. (1997). On social laws for artificial agent societies. *Artificial Intelligence*, 94(1-2), 231-256.

### Theory of Mind 与语言

- Baker, C. L., Saxe, R., & Tenenbaum, J. B. (2009). Action understanding as inverse planning. *Cognition*, 113(3), 329-349.
- Goodman, N. D., & Frank, M. C. (2016). Pragmatic language interpretation as probabilistic inference. *Trends in Cognitive Sciences*, 20(11), 818-829.

### MARL 与多智能体系统

- Albrecht, S. V., & Stone, P. (2018). Autonomous agents modelling other agents: A comprehensive survey. *Artificial Intelligence*, 258, 66-95.
- Leibo, J. Z., et al. (2021). Scalable evaluation of multi-agent reinforcement learning with Melting Pot. *ICML*.

### 可微 ABM

- Chopra, A., et al. (2023). GradABM: Gradient-based optimization for agent-based models. *AAMAS*.

### 生成式智能体

- Park, J. S., et al. (2023). Generative agents: Interactive simulacra of human behavior. *arXiv:2304.03442*.

### Goodhart's Law 与模型表演性

- MacKenzie, D. (2006). *An Engine, Not a Camera*. MIT Press. (经济学模型的表演性)
- Thomas, R., & Uminsky, D. (2020). The problem with metrics is a big problem. *arXiv:2002.08512*.

---

## 联系方式

如有问题或建议，请通过邮件联系：

📧 **Email**: hongmingfeng24@mails.ucas.ac.cn

---

## AIGC 使用声明

本项目的 README 文件由大模型生成，作者按需检查了生成的内容。部分代码由 Claude 3.5 Sonnet 进行修改。

---

**Happy Modeling! 🚀**
