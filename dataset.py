import csv
import json
import yaml
from argparse import Namespace
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy
import scipy.io
from sklearn.preprocessing import label_binarize
import torch_geometric.transforms as T

from data_utils import rand_train_test_idx_502525, even_quantile_labels, to_sparse_tensor, dataset_drive_url, rand_train_test_idx_602020, laplacian_positional_encoding

from torch_geometric.datasets import Planetoid, Amazon, Coauthor, CitationFull
from torch_geometric.utils import degree
import os
from os import path

from google_drive_downloader import GoogleDriveDownloader as gdd

import networkx as nx
import scipy.sparse as sp

from ogb.nodeproppred import NodePropPredDataset


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx_502525(
                self.label, train_prop=.5, valid_prop=.25, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'ANSGT':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx_602020(self.label, train_prop=.6, valid_prop=.2, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'setting_2':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx_602020(self.label, train_prop=.6, valid_prop=.2, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_dataset(data_dir, dataname, exp_setting, pe, pe_dim, sub_dataname=''):
    # print(f'experiment settings: {exp_setting}')
    assert exp_setting in ('setting_1', 'setting_2')
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(data_dir, dataname)
    elif dataname in ('chameleon', 'cornell', 'film', 'squirrel', 'texas', 'wisconsin'):
        dataset = load_geom_gcn_dataset(data_dir, dataname)
    elif dataname in ('corafull'):
        dataset = load_citation_full_dataset(data_dir, dataname)
    elif dataname in ('computers', 'photo'):
        dataset = load_Amazon_dataset(data_dir, dataname)
    elif dataname in ('cs', 'physics'):
        dataset = load_Coauthor_dataset(data_dir, dataname)
    elif dataname == 'deezer-europe':
        dataset = load_deezer_dataset(data_dir)
    elif dataname == 'ogbn-proteins':
        dataset = load_proteins_dataset(data_dir)
    elif dataname in ('ogbn-arxiv', 'ogbn-products'):
        dataset = load_ogb_dataset(data_dir, dataname)
    elif dataname == 'amazon2m':
        dataset = load_amazon2m_dataset(data_dir)
    elif dataname == 'pokec':
        dataset = load_pokec_mat(data_dir)
    else:
        raise ValueError('Invalid dataname')
    
    if (pe):
        print(f'use positional encoding with dim {pe_dim}')
        lpe = laplacian_positional_encoding(dataset, pe_dim) 
        node_feat = torch.cat((dataset.graph['node_feat'], lpe), dim=1)
        dataset.graph['node_feat'] = node_feat

    return dataset

def load_citation_full_dataset(data_dir, name):
    if name == 'corafull': name = 'cora'
    transform = T.NormalizeFeatures()
    torch_dataset = CitationFull(root=f'{data_dir}Citation_Full',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_Amazon_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Amazon(root=f'{data_dir}Amazon',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset

def load_Coauthor_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Coauthor(root=f'{data_dir}Coauthor',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset


def load_proteins_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-proteins', root=f'{data_dir}/ogb')
    dataset = NCDataset('ogbn-proteins')
    def protein_orig_split(**kwargs):
        split_idx = ogb_dataset.get_idx_split()
        return {'train': torch.as_tensor(split_idx['train']),
                'valid': torch.as_tensor(split_idx['valid']),
                'test': torch.as_tensor(split_idx['test'])}
    dataset.load_fixed_splits = protein_orig_split
    dataset.graph, dataset.label = ogb_dataset.graph, ogb_dataset.labels

    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['edge_feat'] = torch.as_tensor(dataset.graph['edge_feat'])
    dataset.label = torch.as_tensor(dataset.label)

    edge_index_ = to_sparse_tensor(dataset.graph['edge_index'],
                                   dataset.graph['edge_feat'], dataset.graph['num_nodes'])
    dataset.graph['node_feat'] = edge_index_.mean(dim=1)
    dataset.graph['edge_feat'] = None
    return dataset

def load_ogb_dataset(data_dir, name):
    dataset = NCDataset(name)
    ogb_dataset = NodePropPredDataset(name=name, root=f'{data_dir}/ogb')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])

    def ogb_idx_to_tensor():
        split_idx = ogb_dataset.get_idx_split()
        tensor_split_idx = {key: torch.as_tensor(
            split_idx[key]) for key in split_idx}
        return tensor_split_idx

    dataset.load_fixed_splits = ogb_idx_to_tensor
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)
    return dataset

def load_amazon2m_dataset(data_dir):
    ogb_dataset = NodePropPredDataset(name='ogbn-products', root=f'{data_dir}/ogb')
    dataset = NCDataset('amazon2m')
    dataset.graph = ogb_dataset.graph
    dataset.graph['edge_index'] = torch.as_tensor(dataset.graph['edge_index'])
    dataset.graph['node_feat'] = torch.as_tensor(dataset.graph['node_feat'])
    dataset.label = torch.as_tensor(ogb_dataset.labels).reshape(-1, 1)

    def load_fixed_splits(train_prop=0.5, val_prop=0.25):
        dir = f'{data_dir}ogb/ogbn_products/split/random_0.5_0.25'
        tensor_split_idx = {}
        if os.path.exists(dir):
            tensor_split_idx['train'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_train.txt'), dtype=torch.long)
            tensor_split_idx['valid'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_valid.txt'), dtype=torch.long)
            tensor_split_idx['test'] = torch.as_tensor(np.loadtxt(dir + '/amazon2m_test.txt'), dtype=torch.long)
        else:
            os.makedirs(dir)
            tensor_split_idx['train'], tensor_split_idx['valid'], tensor_split_idx['test'] \
                = rand_train_test_idx_502525(dataset.label, train_prop=train_prop, valid_prop=val_prop)
            np.savetxt(dir + '/amazon2m_train.txt', tensor_split_idx['train'], fmt='%d')
            np.savetxt(dir + '/amazon2m_valid.txt', tensor_split_idx['valid'], fmt='%d')
            np.savetxt(dir + '/amazon2m_test.txt', tensor_split_idx['test'], fmt='%d')
        return tensor_split_idx
    dataset.load_fixed_splits = load_fixed_splits
    return dataset

def load_pokec_mat(data_dir):
    """ requires pokec.mat """
    if not path.exists(f'{data_dir}pokec.mat'):
        gdd.download_file_from_google_drive(
            file_id=dataset_drive_url['pokec'], \
            dest_path=f'{data_dir}pokec.mat', showsize=True)

    fulldata = scipy.io.loadmat(f'{data_dir}pokec.mat')

    dataset = NCDataset('pokec')
    edge_index = torch.tensor(fulldata['edge_index'], dtype=torch.long)
    node_feat = torch.tensor(fulldata['node_feat']).float()
    num_nodes = int(fulldata['num_nodes'])
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}

    label = fulldata['label'].flatten()
    dataset.label = torch.tensor(label, dtype=torch.long)

    return dataset


def load_planetoid_dataset(data_dir, name):
    transform = T.NormalizeFeatures()
    torch_dataset = Planetoid(root=f'{data_dir}Planetoid',
                              name=name, transform=transform)
    # torch_dataset = Planetoid(root=f'{DATAPATH}Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset



def load_deezer_dataset(data_dir):
    filename = 'deezer-europe'
    dataset = NCDataset(filename)
    deezer = scipy.io.loadmat(f'{data_dir}deezer/deezer-europe.mat')

    A, label, features = deezer['A'], deezer['label'], deezer['features']
    edge_index = torch.tensor(A.nonzero(), dtype=torch.long)
    node_feat = torch.tensor(features.todense(), dtype=torch.float)
    label = torch.tensor(label, dtype=torch.long).squeeze()
    num_nodes = label.shape[0]

    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': node_feat,
                     'num_nodes': num_nodes}
    dataset.label = label
    return dataset


def load_geom_gcn_dataset(data_dir, name):
    graph_adjacency_list_file_path = f'{data_dir}geom-gcn/{name}/out1_graph_edges.txt'
    graph_node_features_and_labels_file_path = f'{data_dir}geom-gcn/{name}/out1_node_feature_label.txt'

    G = nx.DiGraph()
    graph_node_features_dict = {}
    graph_labels_dict = {}

    if name == 'film':
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                feature_blank = np.zeros(932, dtype=np.uint8)
                feature_blank[np.array(line[1].split(','), dtype=np.uint16)] = 1
                graph_node_features_dict[int(line[0])] = feature_blank
                graph_labels_dict[int(line[0])] = int(line[2])
    else:
        with open(graph_node_features_and_labels_file_path) as graph_node_features_and_labels_file:
            graph_node_features_and_labels_file.readline()
            for line in graph_node_features_and_labels_file:
                line = line.rstrip().split('\t')
                assert (len(line) == 3)
                assert (int(line[0]) not in graph_node_features_dict and int(line[0]) not in graph_labels_dict)
                graph_node_features_dict[int(line[0])] = np.array(line[1].split(','), dtype=np.uint8)
                graph_labels_dict[int(line[0])] = int(line[2])

    with open(graph_adjacency_list_file_path) as graph_adjacency_list_file:
        graph_adjacency_list_file.readline()
        for line in graph_adjacency_list_file:
            line = line.rstrip().split('\t')
            assert (len(line) == 2)
            if int(line[0]) not in G:
                G.add_node(int(line[0]), features=graph_node_features_dict[int(line[0])],
                           label=graph_labels_dict[int(line[0])])
            if int(line[1]) not in G:
                G.add_node(int(line[1]), features=graph_node_features_dict[int(line[1])],
                           label=graph_labels_dict[int(line[1])])
            G.add_edge(int(line[0]), int(line[1]))

    adj = nx.adjacency_matrix(G, sorted(G.nodes()))
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = adj.tocoo().astype(np.float32)
    features = np.array(
        [features for _, features in sorted(G.nodes(data='features'), key=lambda x: x[0])])
    labels = np.array(
        [label for _, label in sorted(G.nodes(data='label'), key=lambda x: x[0])])

    def preprocess_features(feat):
        """Row-normalize feature matrix and convert to tuple representation"""
        rowsum = np.array(feat.sum(1))
        rowsum = (rowsum == 0) * 1 + rowsum
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        feat = r_mat_inv.dot(feat)
        return feat

    features = preprocess_features(features)

    edge_index = torch.from_numpy(
        np.vstack((adj.row, adj.col)).astype(np.int64))
    node_feat = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    num_nodes = node_feat.shape[0]

    dataset = NCDataset(name)

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = labels

    return dataset


def create_split_idx_lst(exp_setting, yaml_file):

    dict_yaml = yaml.load(open(f'best_params_yamls/{exp_setting}/{yaml_file}.yaml').read(), Loader=yaml.Loader)['params_config']
    dict_yaml = {k:v[0] for k,v in dict_yaml.items()}

    args = Namespace(**dict_yaml)
    args.pe = False

    dataset = load_dataset(args.data_dir, args.dataset, args.exp_setting, args.pe, args.pe_dim, args.sub_dataset)

    # get the splits for all runs
    assert args.rand_split or args.rand_split_class

    if (args.exp_setting == 'setting_1'):
        print(f'using setting_1 split for {args.dataset}')
        if args.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products', 'amazon2m']:
            split_idx_lst = [dataset.load_fixed_splits()
                            for _ in range(args.num_runs)]
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                        for _ in range(args.num_runs)]
    elif (args.exp_setting == 'setting_2'):
        print(f'using setting_2 split for {args.dataset}')
        split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop, split_type='setting_2')
                        for _ in range(args.num_runs)]

    rand_split_path = '{}splits/{}/rand_split/{}'.format(args.data_dir, args.exp_setting, args.dataset)
    target_path = os.path.join(rand_split_path,f'{args.num_runs}run_{args.seed}seed_split_idx_lst.pt')
    if not os.path.exists(target_path):
        if not os.path.exists(rand_split_path):
            os.makedirs(rand_split_path)
        torch.save(split_idx_lst,target_path)




