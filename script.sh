# node classification on small datasets
# single head
# without pe
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 \
--weight_decay  5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 \
--weight_decay 5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0

# with pe
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay  5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0

# multi heads
# with pe
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --multi_concat --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --multi_concat --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --alpha 0.1 --runs 1 --epochs 2000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --num_heads 2 --multi_concat --pe --pe_dim 3 \
--weight_decay  5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --multi_concat --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 5 --alpha 0.5 --runs 1 --epochs 2000 --device 0