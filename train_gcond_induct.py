from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from gcond_agent_induct import GCond
from utils_graphsaint import DataGraphSAINT, GroupedGraphSaint
import wandb
from train_full import GCondFullDataInductive


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--dis_metric', type=str, default='ours')
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--nlayers', type=int, default=3)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--lr_adj', type=float, default=0.01)
parser.add_argument('--lr_feat', type=float, default=0.01)
parser.add_argument('--lr_model', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--keep_ratio', type=float, default=1.0)
parser.add_argument('--reduction_rate', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--alpha', type=float, default=0, help='regularization term.')
parser.add_argument('--debug', type=int, default=0)
parser.add_argument('--sgc', type=int, default=1)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--outer', type=int, default=20)
parser.add_argument('--option', type=int, default=0)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--label_rate', type=float, default=1)
parser.add_argument('--one_step', type=int, default=0)

# only evaluate saved model
parser.add_argument('--save_dir', type=str, default='saved_ours')
parser.add_argument('--load_exist', default=False, action='store_true')

# fair arguments
parser.add_argument('--group_method', type=str, default=None, choices=['degree'])

# train with full data
parser.add_argument('--full_data', default=False, action='store_true')
parser.add_argument('--full_data_epoch', type=int, default=1000)
parser.add_argument('--full_data_lr', type=float, default=0.01)
parser.add_argument('--full_data_wd', type=float, default=5e-4)

# use wandb logging
parser.add_argument('--wandb', type=str, default='disabled', choices=['online', 'offline', 'disabled'])
parser.add_argument('--wandb_group', type=str, default=None)

args = parser.parse_args()

torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

print(args)

wandb.init(mode='disabled')

data_graphsaint = ['flickr', 'reddit', 'ogbn-arxiv']
if args.dataset in data_graphsaint:
    # data = DataGraphSAINT(args.dataset)
    if args.group_method is None:
        data = DataGraphSAINT(args.dataset, label_rate=args.label_rate)
    else:
        data = GroupedGraphSaint(args.dataset, args.group_method, label_rate=args.label_rate)
    data_full = data.data_full
else:
    data_full = get_dataset(args.dataset, args.normalize_features)
    data = Transd2Ind(data_full, keep_ratio=args.keep_ratio)

if args.full_data:
    agent = GCondFullDataInductive(data, args, device='cuda')
    agent.train()
    exit(0)

agent = GCond(data, args, device='cuda')

if not args.load_exist:
    agent.train()
else:
    agent.test_with_val(load_exist=True)
