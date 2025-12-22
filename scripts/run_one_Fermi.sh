for i in 1
do
CUDA_VISIBLE_DEVICES=1 python main_Fermi.py -epochs 10000 -runs 1 \
    -L_num 200 -question 2 -seed 42
done
