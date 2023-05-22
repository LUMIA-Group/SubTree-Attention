python main.py --dataset citeseer --metric acc --num_heads 2 --lr 0.01 --multi_concat \
--dropout 0.6 --weight_decay 0.0001 --K 3 --runs 5 --device 0 --seed 42 --exp_setting setting_1

python main.py --dataset cora --metric acc --num_heads 4 --lr 0.01 \
--dropout 0.6 --weight_decay 0.0001 --K 10 --runs 5 --device 0 --seed 42 --exp_setting setting_1

python main.py --dataset deezer-europe --metric rocauc --num_heads 2 --lr 0.01 \
--dropout 0.0 --weight_decay 0.005 --K 3 --runs 5 --device 0 --seed 42 --exp_setting setting_1

python main.py --dataset film --metric acc --num_heads 4 --lr 0.001 \
--dropout 0.0 --weight_decay 0.005 --K 3 --runs 5 --device 0 --seed 42 --exp_setting setting_1