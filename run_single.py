"""
run_single.py
单次运行脚本：运行一个PGG模拟并输出结果
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from pgg_model import PGGModel
from pgg_agent import Institution


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='运行公共物品博弈（PGG）+ 内生制度演化模拟'
    )
    
    # 模型参数
    parser.add_argument('--N', type=int, default=50,
                        help='智能体数量 (默认: 50)')
    parser.add_argument('--r', type=float, default=1.6,
                        help='公共物品倍增系数 (默认: 1.6)')
    parser.add_argument('--endowment', type=float, default=10.0,
                        help='初始禀赋 (默认: 10.0)')
    parser.add_argument('--steps', type=int, default=200,
                        help='模拟步数 (默认: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    # 网络参数
    parser.add_argument('--graph', type=str, default='ws',
                        choices=['complete', 'ws', 'er', 'ba'],
                        help='网络类型 (默认: ws)')
    parser.add_argument('--k', type=int, default=6,
                        help='网络平均度 (默认: 6)')
    parser.add_argument('--p', type=float, default=0.1,
                        help='网络重连/边概率 (默认: 0.1)')
    
    # 制度参数
    parser.add_argument('--tau', type=float, default=0.4,
                        help='初始规范阈值 (默认: 0.4)')
    parser.add_argument('--fine_F', type=float, default=2.0,
                        help='初始罚金 (默认: 2.0)')
    parser.add_argument('--punish_cost', type=float, default=0.6,
                        help='惩罚成本 (默认: 0.6)')
    parser.add_argument('--meta', action='store_true',
                        help='启用元规范')
    
    # 输出参数
    parser.add_argument('--output', type=str, default='results',
                        help='输出目录 (默认: results)')
    parser.add_argument('--plot', action='store_true',
                        help='生成可视化图表')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出')
    
    return parser.parse_args()


def print_summary(df: pd.DataFrame, args):
    """打印模拟摘要"""
    print("\n" + "="*60)
    print("模拟摘要")
    print("="*60)
    print(f"智能体数量: {args.N}")
    print(f"倍增系数 r: {args.r}")
    print(f"模拟步数: {args.steps}")
    print(f"网络类型: {args.graph}")
    print(f"元规范: {'启用' if args.meta else '禁用'}")
    print("-"*60)
    
    # 最后10步的统计
    last_10 = df.tail(10)
    
    print("\n最后10步平均值:")
    print(f"  平均贡献: {last_10['mean_contrib'].mean():.3f}")
    print(f"  贡献率: {last_10['contrib_rate'].mean():.3f}")
    print(f"  合规率: {last_10['compliance_rate'].mean():.3f}")
    print(f"  平均收入: {last_10['mean_income'].mean():.3f}")
    print(f"  Gini系数: {last_10['gini_income'].mean():.3f}")
    print(f"  罚金 F: {last_10['fine_F'].mean():.3f}")
    print(f"  阈值 τ: {last_10['tau'].mean():.3f}")
    print(f"  总惩罚成本: {last_10['total_punish_cost'].mean():.3f}")
    print("="*60 + "\n")


def plot_results(df: pd.DataFrame, output_dir: Path):
    """生成可视化图表"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置样式
    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建多子图
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('公共物品博弈 + 内生制度演化', fontsize=16, y=0.995)
    
    # 1. 贡献与合规
    ax = axes[0, 0]
    ax.plot(df.index, df['mean_contrib'], label='平均贡献', linewidth=2)
    ax.plot(df.index, df['compliance_rate'] * 10, label='合规率 × 10', linewidth=2, alpha=0.7)
    ax.set_xlabel('时间步')
    ax.set_ylabel('值')
    ax.set_title('(a) 贡献与合规率')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 收入与不平等
    ax = axes[0, 1]
    ax.plot(df.index, df['mean_income'], label='平均收入', linewidth=2, color='green')
    ax2 = ax.twinx()
    ax2.plot(df.index, df['gini_income'], label='Gini系数', linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('时间步')
    ax.set_ylabel('平均收入', color='green')
    ax2.set_ylabel('Gini系数', color='red')
    ax.set_title('(b) 收入与不平等')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # 3. 制度参数：罚金
    ax = axes[1, 0]
    ax.plot(df.index, df['fine_F'], label='罚金 F', linewidth=2, color='orange')
    ax.set_xlabel('时间步')
    ax.set_ylabel('罚金')
    ax.set_title('(c) 制度参数：罚金')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 制度参数：阈值
    ax = axes[1, 1]
    ax.plot(df.index, df['tau'], label='阈值 τ', linewidth=2, color='purple')
    ax.set_xlabel('时间步')
    ax.set_ylabel('阈值')
    ax.set_title('(d) 制度参数：规范阈值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. 惩罚成本与罚金
    ax = axes[2, 0]
    ax.plot(df.index, df['total_punish_cost'], label='总惩罚成本', linewidth=2)
    ax.plot(df.index, df['total_fines'], label='总罚金', linewidth=2, alpha=0.7)
    ax.set_xlabel('时间步')
    ax.set_ylabel('值')
    ax.set_title('(e) 惩罚成本与罚金')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. 信念演化
    ax = axes[2, 1]
    ax.plot(df.index, df['mean_E_i'], label='平均经验性期望', linewidth=2)
    ax.plot(df.index, df['mean_theta_i'] * 10, label='平均主观阈值 × 10', linewidth=2, alpha=0.7)
    ax.set_xlabel('时间步')
    ax.set_ylabel('值')
    ax.set_title('(f) 信念演化')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_dir / 'simulation_results.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存至: {plot_path}")
    
    # 可选：显示图表
    # plt.show()
    
    plt.close()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建制度
    institution = Institution(
        tau=args.tau,
        fine_F=args.fine_F,
        punish_cost_Cp=args.punish_cost,
        meta_on=args.meta
    )
    
    # 创建模型
    print(f"\n创建模型...")
    model = PGGModel(
        N=args.N,
        endowment=args.endowment,
        r=args.r,
        seed=args.seed,
        graph_kind=args.graph,
        k=args.k,
        p=args.p,
        institution=institution
    )
    
    # 运行模拟
    print(f"运行模拟 ({args.steps} 步)...")
    if args.verbose:
        from tqdm import tqdm
        for _ in tqdm(range(args.steps)):
            model.step()
    else:
        model.run_model(args.steps)
    
    # 获取数据
    df = model.datacollector.get_model_vars_dataframe()
    
    # 保存数据
    csv_path = output_dir / 'simulation_data.csv'
    df.to_csv(csv_path, index=True)
    print(f"\n数据已保存至: {csv_path}")
    
    # 打印摘要
    print_summary(df, args)
    
    # 生成图表
    if args.plot:
        plot_results(df, output_dir)
    
    print("完成！\n")


if __name__ == "__main__":
    main()

