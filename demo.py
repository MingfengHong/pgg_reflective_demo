"""
demo.py
快速演示脚本：运行一个简单的模拟并打印结果
"""

from pgg_model import PGGModel
from pgg_agent import Institution


def main():
    """运行一个简单的演示"""
    print("\n" + "="*60)
    print("公共物品博弈 + 内生制度演化 - 快速演示")
    print("="*60)
    
    # 创建制度（使用默认参数）
    institution = Institution(
        tau=0.4,
        fine_F=2.0,
        punish_cost_Cp=0.6,
        meta_on=False  # 禁用元规范
    )
    
    # 创建模型
    print("\n创建模型...")
    print("  智能体数量: 50")
    print("  倍增系数 r: 1.6")
    print("  初始规范阈值 τ: 0.4")
    print("  初始罚金 F: 2.0")
    
    model = PGGModel(
        N=50,
        endowment=10.0,
        r=1.6,
        seed=42,
        graph_kind='ws',  # 小世界网络
        k=6,
        p=0.1,
        institution=institution
    )
    
    # 运行模拟
    steps = 100
    print(f"\n运行模拟 ({steps} 步)...")
    
    for step in range(steps):
        model.step()
        
        # 每20步打印一次进度
        if (step + 1) % 20 == 0:
            print(f"  步骤 {step + 1}/{steps}")
    
    # 获取结果
    df = model.datacollector.get_model_vars_dataframe()
    
    # 打印摘要
    print("\n" + "-"*60)
    print("模拟结果摘要")
    print("-"*60)
    
    # 初始10步
    first_10 = df.head(10)
    print("\n初始10步平均值:")
    print(f"  平均贡献: {first_10['mean_contrib'].mean():.3f}")
    print(f"  合规率: {first_10['compliance_rate'].mean():.3f}")
    print(f"  平均收入: {first_10['mean_income'].mean():.3f}")
    print(f"  罚金 F: {first_10['fine_F'].mean():.3f}")
    
    # 最后10步
    last_10 = df.tail(10)
    print("\n最后10步平均值:")
    print(f"  平均贡献: {last_10['mean_contrib'].mean():.3f}")
    print(f"  合规率: {last_10['compliance_rate'].mean():.3f}")
    print(f"  平均收入: {last_10['mean_income'].mean():.3f}")
    print(f"  罚金 F: {last_10['fine_F'].mean():.3f}")
    print(f"  阈值 τ: {last_10['tau'].mean():.3f}")
    
    # 制度演化
    print("\n制度参数演化:")
    print(f"  初始罚金 F: {df['fine_F'].iloc[0]:.3f}")
    print(f"  最终罚金 F: {df['fine_F'].iloc[-1]:.3f}")
    print(f"  初始阈值 τ: {df['tau'].iloc[0]:.3f}")
    print(f"  最终阈值 τ: {df['tau'].iloc[-1]:.3f}")
    
    print("\n" + "="*60)
    print("演示完成！")
    print("="*60 + "\n")
    
    # 简单的可视化提示
    print("提示：")
    print("  - 运行 'python run_single.py --plot' 生成可视化图表")
    print("  - 运行 'python run_batch.py --experiment meta_comparison --plot' 进行批量实验")
    print("  - 查看 README.md 了解更多使用方法\n")


if __name__ == "__main__":
    main()

