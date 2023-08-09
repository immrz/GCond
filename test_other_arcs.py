import sys
from deeprobust.graph.data import Dataset
import numpy as np
import random
import time
import argparse
import torch
from utils import *
import torch.nn.functional as F
from tester_other_arcs import Evaluator
from utils_graphsaint import DataGraphSAINT
from utils_attr_bias import BiClassBiAttrTrans
from utils_degree_bias import DegreeGroupedTrans
from utils_graphsaint import DegreeGroupedGraphSaintTrans


parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--keep_ratio', type=float, default=1)
parser.add_argument('--reduction_rate', type=float, default=1)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--mlp', type=int, default=0)
parser.add_argument('--inner', type=int, default=0)
parser.add_argument('--epsilon', type=float, default=-1)
parser.add_argument('--nruns', type=int, default=20)
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--suffix', type=str, required=True)
parser.add_argument('--save_as_csv', type=int, default=1)
args = parser.parse_args()

# torch.cuda.set_device(args.gpu_id)

# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset in ['cora', 'citeseer']:
    args.epsilon = 0.05
else:
    args.epsilon = 0.01

print(args)

if args.dataset == 'ogbn-arxiv':
    data = DegreeGroupedGraphSaintTrans('ogbn-arxiv', thres=5.5)
elif args.dataset == 'cora':
    data_full = get_dataset(args, args.dataset, normalize_features=True)
    data = DegreeGroupedTrans(data_full, keep_ratio=1, thres=3)
elif args.dataset == 'bail':
    data = BiClassBiAttrTrans('bail', label_number=1000)
elif args.dataset == 'credit':
    data = BiClassBiAttrTrans('credit')
else:
    raise ValueError

agent = Evaluator(data, args, device='cuda')
agent.train()
