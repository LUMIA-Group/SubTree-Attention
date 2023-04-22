import sys
import os
sys.path.append(os.path.realpath('.'))

import random
from dataset import create_split_idx_lst

random.seed(2022)

for dataset in ['cora', 'citeseer', 'pubmed', 'chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin']:
    # Create split_idx_lst
    create_split_idx_lst(dataset)