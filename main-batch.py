import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import argparse
from torch_geometric.utils import to_undirected, subgraph

from logger import Logger
from dataset import load_dataset
from data_utils import load_fixed_splits
from pfgnn import PFGT
from eval import evaluate_cpu, eval_acc, eval_rocauc, eval_f1
from dataset import NCDataset


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
parser.add_argument('--batch_size', type=int, default=10000)


# hyper-parameter for model arch and training
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=5e-3)

# hyper-parameter for PFGNN
parser.add_argument('--K', type=int, default=3)
parser.add_argument('--alpha', type=float, default=0.1)


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
dataset = load_dataset(args.data_dir, args.dataset, args.sub_dataset)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)


# get the splits for all runs
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


# Get num_nodes and num_edges
n = dataset.graph['num_nodes']

# Infer the number of classes for non one-hot and one-hot labels and the dimension of input features
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# Whether or not to symmetrize
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

# Print basic infomation of the dataset
edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
print()
print(f"dataset {args.dataset} | num nodes {n} | num edges {edge_index.size(1)} | num classes {c} | num node feats {d}")

# Load model
assert args.method == 'pfgnn'
model = PFGT(num_features=d, num_classes=c, hidden_channels=args.hidden_channels,
             dropout=args.dropout, K=args.K, alpha=args.alpha).to(device)

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


if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
    if dataset.label.shape[1] == 1:
        true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
    else:
        true_label = dataset.label


# Training loop
for run in range(args.runs):
    if args.dataset in ['cora', 'citeseer', 'pubmed'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.5)
    best_val = float('-inf')
    num_batch = train_idx.size(0) // args.batch_size + 1

    for epoch in range(args.epochs):
        model.to(device)
        model.train()

        if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
            true_label = true_label.to(device)
        else:
            dataset.label = dataset.label.to(device)


        idx = torch.randperm(train_idx.size(0))
        for i in range(num_batch):
            idx_i = train_idx[idx[i*args.batch_size:(i+1)*args.batch_size]]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)

            dataset_i = NCDataset(args.dataset)
            dataset_i.graph['edge_index'], dataset_i.graph['node_feat'] = edge_index_i.to(device), x_i

            optimizer.zero_grad()
            out_i = model(dataset_i)

            if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                loss = criterion(out_i, true_label.squeeze(1)[idx_i].to(torch.float))
            else:
                out_i = F.log_softmax(out_i, dim=1)
                loss = criterion(out_i, dataset.label.squeeze(1)[idx_i])
            loss.backward()
            optimizer.step()
            if args.dataset == 'ogbn-proteins':
                scheduler.step()


        # Epoch-wise result
        if epoch % 1 == 0:
            result = evaluate_cpu(model, dataset, split_idx, eval_func, criterion, args)
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
    
    # Run-wise result
    logger.print_statistics(run)

# All runs overall result
results = logger.print_statistics()