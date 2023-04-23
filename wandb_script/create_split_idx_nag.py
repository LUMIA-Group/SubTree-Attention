import sys
import os
sys.path.append(os.path.realpath('.'))

import random
from dataset import create_split_idx_lst

random.seed(3407)

for dataset in ['pubmed','corafull', 'cs', 'physics','computers','photo']:
    # Create split_idx_lst
    exp_setting = 'nagphormer'
    create_split_idx_lst(exp_setting, dataset)