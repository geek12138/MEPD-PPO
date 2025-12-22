expert_root=data/PPO_expert_2025_07_18_104025_q2_e_100_L_200_a_0.01_g_0.99_ce_0.2_gl_0.95_p_1_b_1_delta_0.5_rho_0.001_seed_42/checkpoint/

for i in 0
do
python main_PPO_MEPD.py -epochs 1000 -runs 1 \
    -L_num 200 -question 2 -seed 42 \
    -alpha 1e-2 -tau 5 -eta 0.1 -beta 100 \
    -expert_paths \
    ${expert_root}model_r5.5_final.pth \
    ${expert_root}model_r5.6_final.pth \
    ${expert_root}model_r5.7_final.pth \
    ${expert_root}model_r5.8_final.pth \
    ${expert_root}model_r5.9_final.pth \
    ${expert_root}model_r6.0_final.pth
done
