python main.py --dataset corafull --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2

python main.py --dataset pubmed --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2

python main.py --dataset cs --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2

python main.py --dataset physics --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2

python main.py --dataset computers --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2

python main.py --dataset photo --rand_split --metric acc --method stagnn --lr 0.01 --num_heads 1 --ind_gamma --gamma_softmax --pe --pe_dim 3 \
--multi_concat --weight_decay 1e-05 --dropout 0. --hidden_channels 32 --K 3 --runs 3 --epochs 1000 --device 0 --exp_setting setting_2



