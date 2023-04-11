import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import json
import numpy as np
import argparse
from torch_geometric.utils import to_undirected

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits
from pfgnn import PFGT, MHPFGT
from eval import evaluate, eval_acc, eval_rocauc, eval_f1


# Seed
def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Parser - add_argument
parser = argparse.ArgumentParser(description='PFGNN')

parser.add_argument('--method', '-m', type=str, default='pfgnn')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--sub_dataset', type=str, default='')
parser.add_argument('--data_dir', type=str, default='../data/')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--eval_step', type=int, default=1)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--train_prop', type=float, default=.5,
                    help='training label proportion')
parser.add_argument('--valid_prop', type=float, default=.25,
                    help='validation label proportion')
parser.add_argument('--protocol', type=str, default='semi',
                    help='protocol for cora datasets with fixed splits, semi or supervised')
parser.add_argument('--rand_split', action='store_true',
                    help='use random splits')
parser.add_argument('--rand_split_class', action='store_true',
                    help='use random splits with a fixed number of labeled nodes for each class')
parser.add_argument('--label_num_per_class', type=int,
                    default=20, help='labeled nodes randomly selected')
parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc', 'f1'],
                    help='evaluation metric')
parser.add_argument('--save_model', action='store_true',
                    help='whether to save model')
parser.add_argument('--model_dir', type=str, default='../model/')
parser.add_argument('--exp_setting', type=str, default='nodeformer')

# hyper-parameter for model arch and training
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-3)

# hyper-parameter for PFGNN
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--pe', action='store_true')
parser.add_argument('--pe_dim', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--multi_concat', action='store_true')
parser.add_argument('--ind_gamma', action='store_true')
parser.add_argument('--gamma_softmax', action='store_true')

# hyper-parameter for gnn baseline
parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')


# Parser - parse args
args = parser.parse_args()
print(args)

# Fix seed
fixSeed(args.seed)

# Select device
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")

# Load data and preprocess
dataset = load_dataset(args.data_dir, args.dataset, args.exp_setting, args.pe, args.pe_dim, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
if (args.exp_setting == 'nodeformer'):
    if args.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.runs)]
    elif args.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=args.label_num_per_class)
                        for _ in range(args.runs)]
    elif args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
        split_idx_lst = [dataset.load_fixed_splits()
                        for _ in range(args.runs)]
    else:
        split_idx_lst = load_fixed_splits(
            args.data_dir, dataset, name=args.dataset, protocol=args.protocol)
elif (args.exp_setting == 'nagphormer'):
    split_idx_lst = [dataset.split_idx]


# Get num_nodes and num_edges
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]

# Infer the number of classes for non one-hot and one-hot labels and the dimension of input features
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# Print basic infomation of the dataset
print()
print(f"dataset {args.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")
print()
print(f"exp_setting {args.exp_setting}")

# Whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

# Transfer input to selected device
dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

# Load model
assert args.method == 'pfgnn'
assert args.num_heads > 0
if (args.num_heads == 1):
    model = PFGT(num_features=d, num_classes=c, hidden_channels=args.hidden_channels,
                dropout=args.dropout, K=args.K, alpha=args.alpha).to(device)
else:
    model = MHPFGT(num_features=d, num_classes=c, hidden_channels=args.hidden_channels,
                dropout=args.dropout, K=args.K, alpha=args.alpha, num_heads=args.num_heads, ind_gamma=args.ind_gamma, gamma_softmax=args.gamma_softmax, multi_concat=args.multi_concat).to(device)

### Loss function (Single-class, Multi-class) ###
if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

# Initialize logger
logger = Logger(args.runs, args)

# Model info
model.train()
print()
print('MODEL:', model)

# Training loop
for run in range(args.runs):
    if (args.exp_setting == 'nodeformer'):
        if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
    elif (args.exp_setting == 'nagphormer'):
        print('using nagphormer exp setting !')
        split_idx = split_idx_lst[0]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('-inf')

    time_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        out = model(dataset)
    

        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            if dataset.label.shape[1] == 1:
                true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            else:
                true_label = dataset.label
            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            loss = criterion(out[train_idx], true_label.squeeze(1)[
                train_idx].to(torch.float))
        else:
            out = F.log_softmax(out, dim=1)
            loss = criterion(
                out[train_idx], dataset.label.squeeze(1)[train_idx])
            
        loss.backward()
        optimizer.step()
        
        # Epoch-wise result
        if epoch % args.eval_step == 0:
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args.dataset)
            logger.add_result(run, result[:-1])

            if result[1] > best_val:
                best_val = result[1]
                if args.save_model:
                    torch.save(model.state_dict(), args.model_dir + f'{args.dataset}-{args.method}.pkl')

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%')
            
    time_end = time.time()
    print(f'Run: {run + 1:02d}, ' f'Time: {time_end - time_start:.4f}s')
    
    # Run-wise result
    logger.print_statistics(run)

# All runs overall result
results = logger.print_statistics()