"""
run_batch.py
批量实验脚本：参数扫描与敏感性分析
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from itertools import product

from pgg_model import PGGModel
from pgg_agent import Institution


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='批量运行PGG实验：参数扫描'
    )
    
    parser.add_argument('--experiment', type=str, default='fine_vs_r',
                        choices=['fine_vs_r', 'meta_comparison', 'network_topology', 'custom'],
                        help='实验类型 (默认: fine_vs_r)')
    parser.add_argument('--steps', type=int, default=200,
                        help='每次运行的步数 (默认: 200)')
    parser.add_argument('--replicates', type=int, default=5,
                        help='每个参数组合的重复次数 (默认: 5)')
    parser.add_argument('--output', type=str, default='batch_results',
                        help='输出目录 (默认: batch_results)')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    
    return parser.parse_args()


def run_single_experiment(params: dict, steps: int = 200):
    """
    运行单个实验
    
    参数:
        params: 模型参数字典
        steps: 运行步数
    
    返回:
        结果摘要字典
    """
    # 创建制度（使用优化后的默认值）
    institution = Institution(
        tau=params.get('tau', 0.2),  # 优化后的默认值
        fine_F=params.get('fine_F', 1.5),
        punish_cost_Cp=params.get('punish_cost', 0.3),
        meta_on=params.get('meta', False),
        delta_F=params.get('delta_F', 0.2)
    )
    
    # 创建模型
    model = PGGModel(
        N=params.get('N', 50),
        endowment=params.get('endowment', 10.0),
        r=params.get('r', 1.6),
        seed=params.get('seed', None),
        graph_kind=params.get('graph', 'ws'),
        k=params.get('k', 6),
        p=params.get('p', 0.1),
        institution=institution
    )
    
    # 运行模型
    model.run_model(steps)
    
    # 提取最后阶段的统计
    df = model.datacollector.get_model_vars_dataframe()
    last_n = min(50, len(df) // 4)  # 最后1/4或50步
    last_period = df.tail(last_n)
    
    # 计算摘要统计
    summary = {
        'mean_contrib': last_period['mean_contrib'].mean(),
        'contrib_rate': last_period['contrib_rate'].mean(),
        'compliance_rate': last_period['compliance_rate'].mean(),
        'mean_income': last_period['mean_income'].mean(),
        'total_income': last_period['total_income'].mean(),
        'gini_income': last_period['gini_income'].mean(),
        'final_fine_F': last_period['fine_F'].iloc[-1],
        'final_tau': last_period['tau'].iloc[-1],
        'mean_fine_F': last_period['fine_F'].mean(),
        'mean_tau': last_period['tau'].mean(),
        'total_punish_cost': last_period['total_punish_cost'].mean(),
        'total_fines': last_period['total_fines'].mean(),
    }
    
    # 添加参数信息
    summary.update(params)
    
    return summary


def experiment_fine_vs_r(output_dir: Path, steps: int = 200, replicates: int = 5):
    """
    实验1：罚金步长 × 倍增系数
    
    研究问题：不同的罚金调整步长和公共物品倍增系数如何影响合作水平？
    """
    print("\n实验1: 罚金步长 vs. 倍增系数")
    print("-" * 60)
    
    # 参数网格（使用优化后的参数范围）
    delta_F_values = [0.05, 0.1, 0.2, 0.3]
    r_values = [1.8, 2.0, 2.2, 2.5, 2.8]
    
    results = []
    
    # 遍历参数组合
    total = len(delta_F_values) * len(r_values) * replicates
    with tqdm(total=total, desc="运行实验") as pbar:
        for delta_F, r in product(delta_F_values, r_values):
            for rep in range(replicates):
                params = {
                    'N': 50,
                    'r': r,
                    'endowment': 10.0,
                    'tau': 0.2,  # 优化后的阈值
                    'fine_F': 1.5,  # 优化后的罚金
                    'punish_cost': 0.3,  # 优化后的成本
                    'meta': False,
                    'graph': 'ws',
                    'k': 6,
                    'p': 0.1,
                    'seed': 42 + rep,
                    'delta_F': delta_F,
                    'replicate': rep
                }
                
                # 手动设置delta_F（需要在Institution中）
                institution = Institution(
                    tau=params['tau'],
                    fine_F=params['fine_F'],
                    punish_cost_Cp=params['punish_cost'],
                    meta_on=params['meta'],
                    delta_F=delta_F
                )
                
                model = PGGModel(
                    N=params['N'],
                    r=r,
                    seed=params['seed'],
                    institution=institution
                )
                
                model.run_model(steps)
                
                df = model.datacollector.get_model_vars_dataframe()
                last_n = min(50, len(df) // 4)
                last_period = df.tail(last_n)
                
                summary = {
                    'delta_F': delta_F,
                    'r': r,
                    'replicate': rep,
                    'mean_contrib': last_period['mean_contrib'].mean(),
                    'compliance_rate': last_period['compliance_rate'].mean(),
                    'mean_income': last_period['mean_income'].mean(),
                    'gini_income': last_period['gini_income'].mean(),
                    'final_fine_F': last_period['fine_F'].iloc[-1],
                    'fine_volatility': last_period['fine_F'].std(),
                }
                
                results.append(summary)
                pbar.update(1)
    
    # 保存结果
    df_results = pd.DataFrame(results)
    csv_path = output_dir / 'exp1_fine_vs_r.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n结果已保存至: {csv_path}")
    
    return df_results


def experiment_meta_comparison(output_dir: Path, steps: int = 200, replicates: int = 5):
    """
    实验2：有/无元规范对比
    
    研究问题：元规范如何影响惩罚覆盖率、合作水平和效率？
    """
    print("\n实验2: 元规范对比")
    print("-" * 60)
    
    # 参数（使用优化后的范围）
    r_values = [2.0, 2.5, 3.0]
    meta_values = [False, True]
    
    results = []
    
    total = len(r_values) * len(meta_values) * replicates
    with tqdm(total=total, desc="运行实验") as pbar:
        for r, meta in product(r_values, meta_values):
            for rep in range(replicates):
                params = {
                    'N': 50,
                    'r': r,
                    'meta': meta,
                    'seed': 42 + rep,
                    'replicate': rep
                }
                
                summary = run_single_experiment(params, steps)
                results.append(summary)
                pbar.update(1)
    
    # 保存结果
    df_results = pd.DataFrame(results)
    csv_path = output_dir / 'exp2_meta_comparison.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n结果已保存至: {csv_path}")
    
    return df_results


def experiment_network_topology(output_dir: Path, steps: int = 200, replicates: int = 5):
    """
    实验3：网络拓扑影响
    
    研究问题：不同网络结构如何影响规范扩散和制度演化？
    """
    print("\n实验3: 网络拓扑")
    print("-" * 60)
    
    # 参数（使用优化后的基础参数）
    graph_types = ['complete', 'ws', 'er', 'ba']
    p_values = [0.05, 0.1, 0.2]  # 用于WS的重连概率
    
    results = []
    
    total = len(graph_types) * len(p_values) * replicates
    with tqdm(total=total, desc="运行实验") as pbar:
        for graph, p in product(graph_types, p_values):
            for rep in range(replicates):
                params = {
                    'N': 50,
                    'r': 2.5,  # 使用优化后的倍增系数
                    'tau': 0.2,  # 优化后的阈值
                    'fine_F': 1.5,  # 优化后的罚金
                    'punish_cost': 0.3,  # 优化后的成本
                    'graph': graph,
                    'p': p,
                    'seed': 42 + rep,
                    'replicate': rep
                }
                
                summary = run_single_experiment(params, steps)
                results.append(summary)
                pbar.update(1)
    
    # 保存结果
    df_results = pd.DataFrame(results)
    csv_path = output_dir / 'exp3_network_topology.csv'
    df_results.to_csv(csv_path, index=False)
    print(f"\n结果已保存至: {csv_path}")
    
    return df_results


def plot_experiment1(df: pd.DataFrame, output_dir: Path):
    """绘制实验1的结果 - 热图"""
    # 聚合重复实验
    df_agg = df.groupby(['delta_F', 'r']).agg({
        'mean_contrib': ['mean', 'std'],
        'compliance_rate': ['mean', 'std'],
        'mean_income': ['mean', 'std'],
        'final_fine_F': ['mean', 'std']
    }).reset_index()
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建热图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('实验1：罚金步长 × 倍增系数', fontsize=16, fontweight='bold')
    
    metrics = [
        ('compliance_rate', '合规率', 'RdYlGn'),  # 红黄绿配色
        ('mean_contrib', '平均贡献', 'YlGnBu'),   # 黄绿蓝配色
        ('mean_income', '平均收入', 'YlOrRd'),    # 黄橙红配色
        ('final_fine_F', '最终罚金', 'PuBu')      # 紫蓝配色
    ]
    
    for idx, (metric, title, cmap) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 创建透视表
        pivot = df_agg.pivot(
            index='delta_F',
            columns='r',
            values=(metric, 'mean')
        )
        
        # 绘制热图
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap, ax=ax, 
                    cbar_kws={'label': title}, linewidths=0.5)
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('倍增系数 r', fontsize=11)
        ax.set_ylabel('罚金步长 δF', fontsize=11)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'exp1_heatmap.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 热图已保存至: {plot_path}")
    plt.close()


def plot_experiment2(df: pd.DataFrame, output_dir: Path):
    """绘制实验2的结果"""
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('实验2：元规范对比', fontsize=16, fontweight='bold')
    
    metrics = [
        ('compliance_rate', '合规率'),
        ('mean_income', '平均收入'),
        ('total_punish_cost', '总惩罚成本'),
        ('gini_income', 'Gini系数')
    ]
    
    colors = ['#2E86AB', '#F77F00']  # 蓝色和橙色
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        sns.barplot(data=df, x='r', y=metric, hue='meta', ax=ax, palette=colors)
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('倍增系数 r', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(title='元规范', labels=['禁用', '启用'], loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'exp2_meta_comparison.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 图表已保存至: {plot_path}")
    plt.close()


def plot_experiment3(df: pd.DataFrame, output_dir: Path):
    """绘制实验3的结果 - 网络拓扑"""
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('实验3：网络拓扑影响', fontsize=16, fontweight='bold')
    
    metrics = [
        ('compliance_rate', '合规率'),
        ('mean_contrib', '平均贡献'),
        ('mean_income', '平均收入'),
        ('gini_income', 'Gini系数')
    ]
    
    # 颜色映射
    colors = {'complete': '#2E86AB', 'ws': '#06A77D', 'er': '#F77F00', 'ba': '#D62828'}
    
    for idx, (metric, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # 按网络类型分组绘图
        for graph_type in ['complete', 'ws', 'er', 'ba']:
            data = df[df['graph'] == graph_type]
            if len(data) > 0:
                # 按p值分组计算均值和标准差
                grouped = data.groupby('p')[metric].agg(['mean', 'std']).reset_index()
                ax.errorbar(grouped['p'], grouped['mean'], yerr=grouped['std'], 
                           label=graph_type, marker='o', linewidth=2, 
                           color=colors[graph_type], capsize=5, alpha=0.8)
        
        ax.set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold')
        ax.set_xlabel('重连概率 p', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.legend(title='网络类型', loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / 'exp3_network_topology.png'
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ 图表已保存至: {plot_path}")
    plt.close()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("批量实验：参数扫描")
    print("="*60)
    
    # 运行实验
    if args.experiment == 'fine_vs_r':
        df = experiment_fine_vs_r(output_dir, args.steps, args.replicates)
        if args.plot:
            plot_experiment1(df, output_dir)
    
    elif args.experiment == 'meta_comparison':
        df = experiment_meta_comparison(output_dir, args.steps, args.replicates)
        if args.plot:
            plot_experiment2(df, output_dir)
    
    elif args.experiment == 'network_topology':
        df = experiment_network_topology(output_dir, args.steps, args.replicates)
        if args.plot:
            plot_experiment3(df, output_dir)
    
    elif args.experiment == 'custom':
        print("自定义实验：请修改 run_batch.py 添加您的实验设计")
    
    print("\n完成！\n")


if __name__ == "__main__":
    main()

