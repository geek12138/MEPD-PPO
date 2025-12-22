import torch
from main_PPO_sci import main as run_main
import asyncio
import argparse

if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Train expert model')
    
    # 专家模型参数设置
    parser.add_argument('--expert_name', type=str, default='expert1', help='Expert model name')
    parser.add_argument('--r_value', type=float, default=5.5, help='r value for training')
    parser.add_argument('--epochs', type=int, default=10000, help='Training epochs')
    parser.add_argument('--save_dir', type=str, default='experts', help='Directory to save experts')
    
    # 解析参数
    args = parser.parse_args()
    
    # 设置训练参数
    train_args = argparse.Namespace(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        epochs=args.epochs,
        runs=1,  # 只训练1个模型
        L_num=100,  # 网格大小
        alpha=1e-2,  # 学习率
        gamma=0.99,  # 折扣因子
        clip_epsilon=0.2,  # PPO裁剪参数
        question=1,  # 初始策略设置
        ppo_epochs=1,  # PPO内部更新轮数
        batch_size=1,  # 批量大小
        gae_lambda=0.95,  # GAE参数
        seed=42,  # 随机种子
        output_path=args.save_dir,  # 输出目录
        delta=0.5,  # 价值损失系数
        rho=0.01,  # 熵奖励系数
        r_values=[args.r_value]  # 训练使用的r值
    )
    
    # 运行训练
    asyncio.run(run_main(train_args))
    
    print(f"Expert model '{args.expert_name}' trained and saved in {args.save_dir}")