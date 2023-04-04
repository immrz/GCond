from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import get_dataset, Transd2Ind
import torch.nn.functional as F
from gcond_agent_transduct import GCond
from train_full import GCondFullData
from utils_graphsaint import DataGraphSAINT
from utils_fairness import GroupedTrans, BiSensAttrTrans, DegreeGroupedTrans
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--no_norm_feat', dest='normalize_features', action='store_false')
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='saved_ours')
parser.add_argument('--one_step', default=False, action='store_true')
parser.add_argument('--load_exist', default=False, action='store_true')
parser.add_argument('--label_number', type=int, default=500)

# inner model
parser.add_argument('--inner_model', default='gcn', type=str, choices=['gcn', 'appnp', 'gcn_pokec'])
parser.add_argument('--inner_hidden', type=int, default=256)
parser.add_argument('--inner_nlayers', type=int, default=3)

# fair arguments
parser.add_argument('--group_method', type=str, default=None, choices=['agg', 'geo', 'sens', 'degree'])
parser.add_argument('--group_num', type=int, default=5)

# train with full data
parser.add_argument('--full_data', default=False, action='store_true')
parser.add_argument('--full_data_epoch', type=int, default=1000)
parser.add_argument('--full_data_lr', type=float, default=0.01)
parser.add_argument('--full_data_wd', type=float, default=5e-4)

# use DEMD
parser.add_argument('--demd_lambda', default=-1, type=float)
parser.add_argument('--demd_bins', type=int, default=10)

# use wandb logging
parser.add_argument('--wandb', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
parser.add_argument('--wandb_group', type=str, default=None)

args = parser.parse_args()

# torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

# init wandb
wandb_config_keys = ['dataset', 'epochs', 'nlayers', 'hidden', 'lr_adj', 'lr_feat', 'lr_model',
                     'weight_decay', 'dropout', 'normalize_features', 'reduction_rate', 'seed',
                     'alpha', 'sgc', 'inner', 'outer', 'inner_model', 'group_method', 'group_num',
                     'full_data', 'full_data_epoch', 'full_data_lr', 'full_data_wd',
                     'label_number', 'inner_hidden', 'inner_nlayers', 'one_step', 'load_exist',
                     'demd_lambda', 'demd_bins']
wandb_config = {k: getattr(args, k) for k in wandb_config_keys}
wandb_group = args.wandb_group or ('Full Data' if args.full_data else 'Condensed')
wandb.init(mode=args.wandb,
           project='FairGCond',
           group=wandb_group,
           config=wandb_config)

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    data = DataGraphSAINT(args.dataset)
    data_full = data.data_full
else:
    data_full = get_dataset(args, args.dataset, normalize_features=args.normalize_features)
    if args.group_method is None:
        data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)
    elif args.group_method == 'sens':
        assert args.dataset.startswith('pokec')
        data = BiSensAttrTrans(data_full, args.keep_ratio)
    elif args.group_method == 'degree':
        assert args.dataset in ['cora', 'citeseer']
        data = DegreeGroupedTrans(data_full, args.keep_ratio)
    else:
        data = GroupedTrans(data_full, args.keep_ratio, args.group_method, group_num=args.group_num)

if args.full_data:
    agent = GCondFullData(data, args, device='cuda')
    agent.train()
    exit(0)

agent = GCond(data, args, device='cuda')

if not args.load_exist:
    agent.train()
else:
    agent.test_with_val(load_exist=True)
