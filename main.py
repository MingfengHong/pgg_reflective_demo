"""
main.py
PyCharm IDE 运行入口 - 自动生成可视化图表 📊

使用方法：
1. 在PyCharm中打开此文件
2. 右键点击 -> Run 'main'
3. 或者点击编辑器右上角的绿色运行按钮

运行后会生成：
✓ 控制台输出：统计摘要
✓ simulation_results.csv：详细数据
✓ simulation_plot.png：可视化图表（4个子图）
✓ 图表窗口：实时显示（可关闭）

修改参数：
- 在下方"配置参数"区域修改
- 保存后重新运行即可
"""

from pgg_model import PGGModel
from pgg_agent import Institution
import pandas as pd


def main():
    """主函数 - 可以直接在PyCharm中运行"""
    
    # ==================== 配置参数 ====================
    # 在这里修改参数，然后直接运行即可
    
    # 模型参数
    N = 50              # 智能体数量
    r = 2.5             # 公共物品倍增系数（提高以激励合作）
    endowment = 10.0    # 初始禀赋
    steps = 100         # 模拟步数
    seed = 42           # 随机种子
    
    # 网络参数
    graph_kind = 'ws'   # 网络类型: 'ws', 'complete', 'er', 'ba'
    k = 6               # 平均度
    p = 0.1             # 重连概率
    
    # 制度参数
    tau = 0.2           # 初始规范阈值（降低，更容易达到）
    fine_F = 1.5        # 初始罚金（降低初始惩罚）
    punish_cost = 0.3   # 惩罚成本（降低惩罚成本）
    meta_on = False     # 是否启用元规范
    
    # ==================== 运行模拟 ====================
    
    print("\n" + "="*60)
    print("公共物品博弈 + 内生制度演化")
    print("="*60)
    
    # 创建制度
    institution = Institution(
        tau=tau,
        fine_F=fine_F,
        punish_cost_Cp=punish_cost,
        meta_on=meta_on
    )
    
    # 创建模型
    print(f"\n创建模型...")
    print(f"  智能体数量: {N}")
    print(f"  倍增系数 r: {r}")
    print(f"  网络类型: {graph_kind}")
    print(f"  初始规范阈值 τ: {tau}")
    print(f"  初始罚金 F: {fine_F}")
    print(f"  元规范: {'启用' if meta_on else '禁用'}")
    
    model = PGGModel(
        N=N,
        endowment=endowment,
        r=r,
        seed=seed,
        graph_kind=graph_kind,
        k=k,
        p=p,
        institution=institution
    )
    
    # 运行模拟
    print(f"\n运行模拟 ({steps} 步)...")
    for step in range(steps):
        model.step()
        
        # 每20步打印一次进度
        if (step + 1) % 20 == 0:
            print(f"  进度: {step + 1}/{steps}")
    
    print("✓ 模拟完成！")
    
    # ==================== 分析结果 ====================
    
    # 获取数据
    df = model.datacollector.get_model_vars_dataframe()
    
    # 打印摘要
    print("\n" + "-"*60)
    print("结果摘要")
    print("-"*60)
    
    # 初始阶段（前10步）
    first_10 = df.head(10)
    print("\n【初始阶段】前10步平均值:")
    print(f"  平均贡献: {first_10['mean_contrib'].mean():.3f}")
    print(f"  贡献率: {first_10['contrib_rate'].mean():.3f}")
    print(f"  合规率: {first_10['compliance_rate'].mean():.3f}")
    print(f"  平均收入: {first_10['mean_income'].mean():.3f}")
    print(f"  Gini系数: {first_10['gini_income'].mean():.3f}")
    
    # 稳态阶段（后20步）
    last_20 = df.tail(20)
    print("\n【稳态阶段】最后20步平均值:")
    print(f"  平均贡献: {last_20['mean_contrib'].mean():.3f}")
    print(f"  贡献率: {last_20['contrib_rate'].mean():.3f}")
    print(f"  合规率: {last_20['compliance_rate'].mean():.3f}")
    print(f"  平均收入: {last_20['mean_income'].mean():.3f}")
    print(f"  Gini系数: {last_20['gini_income'].mean():.3f}")
    
    # 制度演化
    print("\n【制度演化】:")
    print(f"  罚金 F:  {df['fine_F'].iloc[0]:.3f} → {df['fine_F'].iloc[-1]:.3f}")
    print(f"  阈值 τ:  {df['tau'].iloc[0]:.3f} → {df['tau'].iloc[-1]:.3f}")
    
    # 惩罚统计
    print("\n【惩罚机制】最后20步平均:")
    print(f"  总惩罚成本: {last_20['total_punish_cost'].mean():.3f}")
    print(f"  总罚金: {last_20['total_fines'].mean():.3f}")
    
    print("\n" + "="*60)
    
    # ==================== 保存数据 ====================
    
    # 可选：保存为CSV
    save_csv = True  # 改为False则不保存
    
    if save_csv:
        output_file = "simulation_results.csv"
        df.to_csv(output_file, index=True, encoding='utf-8-sig')
        print(f"\n✓ 数据已保存至: {output_file}")
    
    # ==================== 可视化 ====================
    
    # 是否生成图表（改为False则不生成）
    generate_plot = True
    
    # 是否显示图表窗口（改为False则只保存不显示，适合批量运行）
    show_plot = True
    
    if generate_plot:
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 配置中文字体
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
            matplotlib.rcParams['axes.unicode_minus'] = False
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('公共物品博弈模拟结果', fontsize=14, fontweight='bold')
            
            # 1. 贡献与合规
            ax = axes[0, 0]
            ax.plot(df.index, df['mean_contrib'], label='平均贡献', linewidth=2, color='#2E86AB')
            ax.set_xlabel('时间步')
            ax.set_ylabel('贡献量')
            ax.set_title('(a) 平均贡献演化')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 2. 合规率
            ax = axes[0, 1]
            ax.plot(df.index, df['compliance_rate'], label='合规率', color='#06A77D', linewidth=2)
            ax.set_xlabel('时间步')
            ax.set_ylabel('合规率')
            ax.set_title('(b) 合规率演化')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 3. 制度参数：罚金
            ax = axes[1, 0]
            ax.plot(df.index, df['fine_F'], label='罚金 F', color='#F77F00', linewidth=2)
            ax.set_xlabel('时间步')
            ax.set_ylabel('罚金')
            ax.set_title('(c) 罚金演化')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 4. 收入与不平等
            ax = axes[1, 1]
            line1 = ax.plot(df.index, df['mean_income'], label='平均收入', linewidth=2, color='#2E86AB')
            ax2 = ax.twinx()
            line2 = ax2.plot(df.index, df['gini_income'], label='Gini系数', color='#D62828', linewidth=2, alpha=0.8)
            ax.set_xlabel('时间步')
            ax.set_ylabel('平均收入', color='#2E86AB')
            ax2.set_ylabel('Gini系数', color='#D62828')
            ax.set_title('(d) 收入与不平等')
            
            # 合并图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图表
            plot_file = 'simulation_plot.png'
            plt.savefig(plot_file, dpi=200, bbox_inches='tight')
            print(f"\n✓ 图表已保存至: {plot_file}")
            
            # 显示图表（可选）
            if show_plot:
                plt.show()
            else:
                plt.close()
            
        except ImportError as e:
            print(f"\n⚠ 无法生成图表：缺少matplotlib库")
            print(f"  安装方法：pip install matplotlib")
        except Exception as e:
            print(f"\n⚠ 生成图表时出错：{e}")
    
    print("\n运行完成！\n")
    
    return df


if __name__ == "__main__":
    # 这是程序的入口点
    # 在PyCharm中右键点击此文件，选择 "Run 'main'" 即可运行
    
    results = main()
    
    # 如果需要，可以在这里进一步分析结果
    # 例如：
    # print(results.describe())
    # print(results.tail(10))

