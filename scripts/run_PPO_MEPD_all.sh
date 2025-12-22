#!/bin/bash

# 定义所有pth文件的数组
pth_files=(
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.1_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.2_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.3_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.4_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.5_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.6_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.7_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.8_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r5.9_final.pth
    data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/model_r6.0_final.pth
)

# 获取初始文件数量
num_files=${#pth_files[@]}

# 循环，每次减少一个前面的文件
for ((i=0; i<$num_files; i++)); do
    echo "Running iteration with ${#pth_files[@]} expert models"
    
    # 构建命令
    cmd="python main_PPO_MEPD.py -epochs 1000 -runs 1 -L_num 200 -question 2 -seed 42 -alpha 1e-2 -tau 10 -eta 0.1 -expert_paths"
    
    # 添加当前剩余的所有pth文件
    for file in "${pth_files[@]}"; do
        cmd+=" $file"
    done
    
    # 执行命令
    eval $cmd
    
    # 如果不是最后一次迭代，则删除第一个元素
    if [ ${#pth_files[@]} -gt 1 ]; then
        pth_files=("${pth_files[@]:1}")  # 删除数组第一个元素
    fi
done