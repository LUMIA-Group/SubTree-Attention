import sys
import os
sys.path.append(os.path.realpath('.'))

import random
from dataset import create_split_idx_lst

random.seed(42)

for dataset in ['cora','citeseer', 'film', 'deezer-europe']:
    # Create split_idx_lst
    exp_setting = 'nodeformer'
    create_split_idx_lst(exp_setting, dataset)