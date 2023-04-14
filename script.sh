# node classification on small datasets

# random split
# single head
# without pe
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-4 --hidden_channels 32 --K 10 --runs 1 --epochs 2000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 10 --runs 1 --epochs 2000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay  5e-3 --hidden_channels 32 --K 5 --runs 1 --epochs 2000 --device 0

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 --pe --pe_dim 3 \
--weight_decay 5e-3 --hidden_channels 32 --K 5 --runs 1 --epochs 2000 --device 0



# multi heads
# with pe , wo softmax
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --pe --pe_dim 3 \
--multi_concat --weight_decay 5e-3 --hidden_channels 32 --K 10 --runs 1 --epochs 1000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 4 --ind_gamma --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-3 --hidden_channels 128 --K 10 --runs 1 --epochs 1000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --pe --pe_dim 5 \
--multi_concat --weight_decay 5e-3 --hidden_channels 32 --K 5 --runs 1 --epochs 1000 --device 0 

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --pe --pe_dim 3 \
--multi_concat --weight_decay 5e-4 --hidden_channels 64 --K 10 --runs 1 --epochs 1000 --device 0


# multi heads
# with pe and softmax
python main.py --dataset cora --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 5e-3 --hidden_channels 32 --K 10 --runs 1 --epochs 2000 --device 0

python main.py --dataset citeseer --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 4 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-3 --hidden_channels 128 --K 10 --runs 1 --epochs 1000 --device 0

python main.py --dataset deezer-europe --rand_split --metric rocauc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --gamma_softmax --pe --pe_dim 5 \
--multi_concat --weight_decay 5e-3 --hidden_channels 32 --K 5 --runs 1 --epochs 1000 --device 0 

python main.py --dataset film --rand_split --metric acc --method pfgnn --lr 0.001 --num_heads 2 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 5e-4 --hidden_channels 64 --K 10 --runs 1 --epochs 1000 --device 0



