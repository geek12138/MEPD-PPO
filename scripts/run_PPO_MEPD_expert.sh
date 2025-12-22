for i in 1
do
python main_PPO_MEPD.py -epochs 100 -runs 1 \
    -L_num 200 -question 2 -seed 42 \
    -alpha 1e-3
done
