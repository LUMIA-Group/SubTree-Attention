import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.realpath('.'))

import random
import numpy as np
import wandb
import json
import hashlib
import traceback
import time
from torch_geometric.utils import to_undirected

from dataset import load_dataset
from data_utils import load_fixed_splits
from stagnn import STAGNN, MSTAGNN
from eval import evaluate, eval_acc, eval_rocauc, eval_f1


# generate hash tag for one set of hyper parameters


def get_hash(dict_in, hash_keys, ignore_keys):
    dict_in = {k: v for k, v in dict_in.items() if k in hash_keys}
    dict_in = {k: v for k, v in dict_in.items() if k not in ignore_keys}
    hash_out = hashlib.blake2b(json.dumps(
        dict_in, sort_keys=True).encode(), digest_size=4).hexdigest()
    return str(hash_out)

# fix random seed


def fixSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# running and evaluating, return ID of this run


def runner(wandb_base, sweep_id, gpu_index, code_fullname, save_model):
    dir_name = 'remote'

    wandb.init(dir=dir_name, reinit=True, group=sweep_id)

    try:
        wandb.use_artifact(code_fullname, type='code')

        params_hash = get_hash(
            wandb.config, wandb.config['hash_keys'], wandb.config['ignore_keys'])
        wandb.config.update({'params_hash': params_hash},
                            allow_val_change=True)
        wandb.config.update({'gpu_index': gpu_index}, allow_val_change=True)

        params = dict(wandb.config)
        print("This trial's parameters: %s" % (params))

        if save_model == True:
            checkpoint_path = os.path.join(wandb.run.dir, 'checkpoint')
            os.makedirs(checkpoint_path)

        device = torch.device("cuda:" + str(gpu_index)
                              ) if torch.cuda.is_available() else torch.device("cpu")

        seed = params['seed']
        # Fix seed
        fixSeed(seed)

        # Select device
        device = torch.device("cuda:" + str(gpu_index)
                              ) if torch.cuda.is_available() else torch.device("cpu")

        print('Using device:', device)

        # Load data and preprocess
        dataset = load_dataset(
            params['data_dir'], params['dataset'], params['exp_setting'], params['pe'], params['pe_dim'], params['sub_dataset'])

        if len(dataset.label.shape) == 1:
            dataset.label = dataset.label.unsqueeze(1)
        dataset.label = dataset.label.to(device)

        rand_split_path = '{}splits/{}/rand_split/{}'.format(params['data_dir'], params['exp_setting'], params['dataset'])

        # get the splits for all runs
        if (params['exp_setting'] == 'setting_1'):
            if params['rand_split']:
                target_rand_split_path = os.path.join(rand_split_path,f'{params["num_runs"]}run_{params["seed"]}seed_split_idx_lst.pt')
                assert os.path.exists(target_rand_split_path)
                split_idx_lst = torch.load(target_rand_split_path)
        elif (params['exp_setting'] == 'setting_2'):
            print('using setting_2 exp setting !')
            target_rand_split_path = os.path.join(rand_split_path,f'{params["num_runs"]}run_{params["seed"]}seed_split_idx_lst.pt')
            assert os.path.exists(target_rand_split_path)
            split_idx_lst = torch.load(target_rand_split_path)

        # Get num_nodes and num_edges
        n = dataset.graph['num_nodes']
        e = dataset.graph['edge_index'].shape[1]

        # Infer the number of classes for non one-hot and one-hot labels and the dimension of input features
        c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
        d = dataset.graph['node_feat'].shape[1]

        # Whether or not to symmetrize
        if not params['directed']:
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

        # Transfer input to selected device
        dataset.graph['edge_index'], dataset.graph['node_feat'] = \
            dataset.graph['edge_index'].to(
                device), dataset.graph['node_feat'].to(device)
        
        # Load model
        assert params['method'] == 'stagnn'
        assert params['num_heads'] > 0
        if (params['num_heads'] == 1):
            model = STAGNN(num_features=d, num_classes=c, hidden_channels=params['hidden_channels'],
                        dropout=params['dropout'], K=params['K'], global_attn=params['global_attn']).to(device)
        else:
            model = MSTAGNN(num_features=d, num_classes=c, hidden_channels=params['hidden_channels'],
                        dropout=params['dropout'], K=params['K'], num_heads=params['num_heads'],
                        ind_gamma=params['ind_gamma'], gamma_softmax=params['gamma_softmax'], 
                        multi_concat=params['multi_concat'], global_attn=params['global_attn']).to(device)


        ### Loss function (Single-class, Multi-class) ###
        if params['dataset'] in ('deezer-europe'):
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.NLLLoss()

        ### Performance metric (Acc, AUC, F1) ###
        if params['metric'] == 'rocauc':
            eval_func = eval_rocauc
        elif params['metric'] == 'f1':
            eval_func = eval_f1
        else:
            eval_func = eval_acc

        # Model info
        model.train()
        print()
        print('MODEL:', model)
        print()
        print(f"exp_setting {params['exp_setting']}")

        run = params['runs']

        if (params['exp_setting'] == 'setting_1'):
            if params['dataset'] in ['cora', 'citeseer', 'pubmed']:
                split_idx = split_idx_lst[0]
            else:
                split_idx = split_idx_lst[run]
        elif (params['exp_setting'] == 'setting_2'):
            split_idx = split_idx_lst[run]

        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        no_decay_params = [model.headwise, model.hopwise, model.teleport] if (params["num_heads"]>1 and params["gamma_softmax"] and params["ind_gamma"]) else [model.hopwise, model.teleport]
        decay_params = [p for p in model.parameters() if id(p) not in (id(param) for param in no_decay_params)]    
        param_groups = [
        {"params": no_decay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": params["weight_decay"]}
        ]
        optimizer = torch.optim.Adam(param_groups, lr=params["lr"])

        best_val = float('-inf')

        patience = params['patience']
        patience_counter = 0

        time_start = time.time()

        best_test_metric = 0

        for epoch in range(params['epochs']):
            model.train()
            optimizer.zero_grad()
            
            out = model(dataset)

            if params['dataset'] in ('deezer-europe'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                train_loss = criterion(out[train_idx], true_label.squeeze(1)[
                    train_idx].to(torch.float))
                # train_metric = eval_func(
                #     dataset.label[train_idx], out[train_idx])
            else:
                out = F.log_softmax(out, dim=1)
                train_loss = criterion(
                    out[train_idx], dataset.label.squeeze(1)[train_idx])
                # train_metric = eval_func(
                #     dataset.label.squeeze(1)[train_idx], out[train_idx])
    
            train_loss.backward()
            optimizer.step()
            
            if epoch % params['eval_step'] == 0:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, params['dataset'])

                if result[1] > best_val:
                    best_val = result[1]
                    patience_counter = 0
                    best_test_metric = result[2]
                    if params['save_model']:
                        if not (os.path.exists(params['model_dir'])):
                            os.makedirs(params['model_dir'])
                        torch.save(model.state_dict(), params['model_dir'] + f'{params["dataset"]}-{params["method"]}.pkl')
                else:
                    patience_counter += 1 


                # print(f'Epoch: {epoch:02d}, '
                #     f'Loss: {train_loss:.4f}, '
                #     f'Train: {100 * result[0]:.2f}%, '
                #     f'Valid: {100 * result[1]:.2f}%, '
                #     f'Test: {100 * result[2]:.2f}%')

            if epoch % params['log_freq'] == 0:
                wandb.log({'metric/train': result[0], 'metric/val': result[1], 'metric/test': result[2], 'loss/train': train_loss, 'loss/val': result[3], 'loss/test': result[4]})

            if patience_counter == patience:
                print('Early stopping!')
                break

        time_end = time.time()
        print(f'Run: {run + 1:02d}, ' f'Time: {time_end - time_start:.4f}s')
        print('Final metric is [%s]' % (best_test_metric))
        wandb.run.summary["metric/final"] = best_test_metric

    except Exception as e:
        # exit gracefully, so wandb logs the problem
        print(traceback.print_exc(), file=sys.stderr)
        exit(1)

    wandb.finish()

    return str(wandb.run.id)
