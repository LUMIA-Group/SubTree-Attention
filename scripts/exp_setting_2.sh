python main.py --dataset pubmed --metric acc --num_heads 4 --lr 0.01 \
--dropout 0.2 --weight_decay 0.0005 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2

python main.py --dataset computers --metric acc --num_heads 4 --lr 0.01 \
--dropout 0.4 --weight_decay 0.0001 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2

python main.py --dataset corafull --metric acc --num_heads 4 --lr 0.001 --multi_concat \
--dropout 0.4 --weight_decay 0.0001 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2

python main.py --dataset cs --metric acc --num_heads 6 --lr 0.001 --multi_concat \
--dropout 0.6 --weight_decay 0.00005 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2

python main.py --dataset photo --metric acc --num_heads 8 --lr 0.01 --multi_concat \
--dropout 0.4 --weight_decay 0.00005 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2

python main.py --dataset physics --metric acc --num_heads 8 --lr 0.01 --multi_concat \
--dropout 0.4 --weight_decay 0.00005 --K 3 --runs 10 --device 0 --seed 3047 --exp_setting setting_2