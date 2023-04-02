# node classification on small datasets
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 3000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 3000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 \
--weight_decay  5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 1000 --device 0

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 1000 --device 0


# node classification on large datasets
python main-batch.py --dataset ogbn-proteins --metric rocauc --method pfgnn --lr 0.01 \
--weight_decay 0. --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --batch_size 100 --eval_step 9 --epochs 1000 --device 0

python main-batch.py --dataset ogbn-proteins --metric rocauc --method nodeformer --lr 1e-2 \
--weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.1 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 10000 \
--runs 5 --epochs 1000 --eval_step 9 --device 1

python main-batch.py --dataset amazon2m --rand_split --metric acc --method nodeformer  --lr 1e-2 \
--weight_decay 0. --num_layers 3 --hidden_channels 64 --num_heads 1 --rb_order 1 --rb_trans identity \
--lamda 0.01 --M 50 --K 5 --use_bn --use_residual --use_gumbel --use_act --use_jk --batch_size 100000 \
--runs 5 --epochs 1000 --eval_step 9 --device 2