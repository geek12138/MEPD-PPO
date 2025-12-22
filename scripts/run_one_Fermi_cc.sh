for i in 1
do
python main_Fermi_cc_v4.py -epochs 10000 -runs 1 \
    -L_num 200 -question 1 -is_cc -seed 44 \
    -alpha 0.2 -lam 0.2 -delta 0.7 -beta 0.9
done
