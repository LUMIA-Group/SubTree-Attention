import torch
import random
import numpy as np
import yaml
from argparse import Namespace
import os
import sys

from dataset import load_dataset
from data_utils import load_fixed_splits
from sweep import get_configs_from_file

def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

assert len(sys.argv)==2

dict_yaml = yaml.load(open(f'yamls/{sys.argv[1]}.yaml').read(), Loader=yaml.Loader)['params_config']
dict_yaml = {k:v[0] for k,v in dict_yaml.items()}

args = Namespace(**dict_yaml)

fixSeed(args.seed)

dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

# get the splits for all runs
assert args.rand_split or args.rand_split_class

if args.rand_split:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.num_runs)]
elif args.rand_split_class:
    split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                     for _ in range(args.num_runs)]

rand_split_path = '{}rand_split/{}'.format(args.data_dir, args.dataset) if args.rand_split else '{}rand_split_class/{}'.format(args.data_dir, args.dataset)
target_path = os.path.join(rand_split_path,f'{args.num_runs}run_{args.seed}seed_split_idx_lst.pt')
if not os.path.exists(target_path):
    if not os.path.exists(rand_split_path):
        os.makedirs(rand_split_path)
    torch.save(split_idx_lst,target_path)