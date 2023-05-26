import scipy.sparse as sp
import numpy as np
import sys
import json
import os
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import InMemoryDataset, Data
import torch
from itertools import repeat
from torch_geometric.data import NeighborSampler
import torch.nn.functional as F
from deeprobust.graph.utils import accuracy
from collections import Counter
from utils_degree_bias import groupby_degree, delta_std_parity


class DataGraphSAINT:
    '''datasets used in GraphSAINT paper'''

    def __init__(self, dataset, **kwargs):
        dataset_str='data/'+dataset+'/'
        adj_full = sp.load_npz(dataset_str+'adj_full.npz')
        self.nnodes = adj_full.shape[0]
        if dataset == 'ogbn-arxiv':
            adj_full = adj_full + adj_full.T
            adj_full[adj_full > 1] = 1
            self.transductive = True  # this dataset is used for transductive setting
        else:
            self.transductive = False

        role = json.load(open(dataset_str+'role.json','r'))
        idx_train = role['tr']
        idx_test = role['te']
        idx_val = role['va']

        if 'label_rate' in kwargs:
            label_rate = kwargs['label_rate']
            if label_rate < 1:
                idx_train = idx_train[:int(label_rate*len(idx_train))]

        self.adj_train = adj_full[np.ix_(idx_train, idx_train)]
        self.adj_val = adj_full[np.ix_(idx_val, idx_val)]
        self.adj_test = adj_full[np.ix_(idx_test, idx_test)]

        feat = np.load(dataset_str+'feats.npy')
        # ---- normalize feat ----
        feat_train = feat[idx_train]
        scaler = StandardScaler()
        scaler.fit(feat_train)
        feat = scaler.transform(feat)

        self.feat_train = feat[idx_train]
        self.feat_val = feat[idx_val]
        self.feat_test = feat[idx_test]

        class_map = json.load(open(dataset_str + 'class_map.json','r'))
        labels = self.process_labels(class_map)

        self.labels_train = labels[idx_train]
        self.labels_val = labels[idx_val]
        self.labels_test = labels[idx_test]

        self.data_full = GraphData(adj_full, feat, labels, idx_train, idx_val, idx_test)
        self.class_dict = None
        self.class_dict2 = None

        self.adj_full = adj_full
        self.feat_full = feat
        self.labels_full = labels
        self.idx_train = np.array(idx_train)
        self.idx_val = np.array(idx_val)
        self.idx_test = np.array(idx_test)
        self.samplers = None

    def process_labels(self, class_map):
        """
        setup vertex property map for output classests
        """
        num_vertices = self.nnodes
        if isinstance(list(class_map.values())[0], list):
            num_classes = len(list(class_map.values())[0])
            self.nclass = num_classes
            class_arr = np.zeros((num_vertices, num_classes))
            for k,v in class_map.items():
                class_arr[int(k)] = v
        else:
            class_arr = np.zeros(num_vertices, dtype=np.int)
            for k, v in class_map.items():
                class_arr[int(k)] = v
            class_arr = class_arr - class_arr.min()
            self.nclass = max(class_arr) + 1
        return class_arr

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.labels_train == i)
        idx = np.arange(len(self.labels_train))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if args.nlayers == 1:
            sizes = [30]
        if args.nlayers == 2:
            if args.dataset in ['reddit', 'flickr']:
                if args.option == 0:
                    sizes = [15, 8]
                if args.option == 1:
                    sizes = [20, 10]
                if args.option == 2:
                    sizes = [25, 10]
            else:
                sizes = [10, 5]

        if self.class_dict2 is None:
            print(sizes)
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    idx_train = np.array(self.idx_train)
                    idx = idx_train[self.labels_train == i]
                else:
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if self.samplers is None:
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                if len(node_idx) == 0:
                    continue

                self.samplers.append(NeighborSampler(adj,
                                    node_idx=node_idx,
                                    sizes=sizes, batch_size=num,
                                    num_workers=8, return_e_id=False,
                                    num_nodes=adj.size(0),
                                    shuffle=True))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out

    def compute_test_metric(self, model_output):
        """
        This is dataset is used for inductive learning, so model_output only contains the outputs
        corresponding to the test nodes, i.e., its shape is (num_test, h).
        """
        labels_test = torch.LongTensor(self.labels_test).cuda()
        if not self.transductive:
            loss_test = F.nll_loss(model_output, labels_test)
            acc_test = accuracy(model_output, labels_test)
        else:
            loss_test = F.nll_loss(model_output[self.idx_test], labels_test)
            acc_test = accuracy(model_output[self.idx_test], labels_test)
        return {'loss': loss_test.item(), 'acc': acc_test.item()}


class GroupedGraphSaint(DataGraphSAINT):
    def __init__(self, dataset, group_method, **kwargs):
        super().__init__(dataset, **kwargs)
        assert group_method == 'degree', "Only degree grouping is supported now."

        # use adj_test instead of adj_full[idx_test] since this is an inductive dataset
        # 10.5 is a good number leading to 10 groups with nearly uniform sizes
        self.test_gid = torch.LongTensor(
            groupby_degree(self.adj_test, thres=10.5)
        )

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)
        group_acc = []

        # compute group-wise accuracy
        labels = torch.LongTensor(self.labels_test).cuda()
        preds = model_output.max(1)[1].type_as(labels)
        correct = torch.eq(labels, preds).double()

        for i in range(self.test_gid.max().item() + 1):
            group_mask = (self.test_gid == i)
            group_size = group_mask.sum().item()
            assert group_size > 0
            group_acc.append(correct[group_mask].sum().item() / group_size)

        res['delta'] = max(group_acc) - min(group_acc)
        res['std'] = np.std(group_acc)
        res['group_acc'] = group_acc
        return res


class DegreeGroupedGraphSaintTrans(DataGraphSAINT):
    """Transductive GraphSaint dataset, nodes grouped by degree. For Ogbn-arxiv.
    """
    def __init__(self, dataset, thres, **kwargs):
        super().__init__(dataset, **kwargs)
        self.train_gid = torch.LongTensor(
            groupby_degree(adj=self.adj_full[self.idx_train], thres=thres)
        )
        self.test_gid = torch.LongTensor(
            groupby_degree(adj=self.adj_full[self.idx_test], thres=thres)
        )

        # scatter the train_gid to the full graph; -1 for non-train nodes, and non-negative integers for train nodes
        self.all_gid = -torch.ones(self.adj_full.shape[0], dtype=torch.long)
        self.all_gid[self.idx_train] = self.train_gid

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)
        labels = torch.LongTensor(self.labels_test).cuda()
        preds = model_output[self.idx_test].max(1)[1].type_as(labels)
        fair_res = delta_std_parity(labels, preds, self.test_gid)
        res.update(fair_res)
        return res


class GraphData:

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


class Data2Pyg:

    def __init__(self, data, device='cuda', transform=None, **kwargs):
        self.data_train = Dpr2Pyg(data.data_train, transform=transform)[0].to(device)
        self.data_val = Dpr2Pyg(data.data_val, transform=transform)[0].to(device)
        self.data_test = Dpr2Pyg(data.data_test, transform=transform)[0].to(device)
        self.nclass = data.nclass
        self.nfeat = data.nfeat
        self.class_dict = None

    def retrieve_class(self, c, num=256):
        if self.class_dict is None:
            self.class_dict = {}
            for i in range(self.nclass):
                self.class_dict['class_%s'%i] = (self.data_train.y == i).cpu().numpy()
        idx = np.arange(len(self.data_train.y))
        idx = idx[self.class_dict['class_%s'%c]]
        return np.random.permutation(idx)[:num]


class Dpr2Pyg(InMemoryDataset):

    def __init__(self, dpr_data, transform=None, **kwargs):
        root = 'data/' # dummy root; does not mean anything
        self.dpr_data = dpr_data
        super(Dpr2Pyg, self).__init__(root, transform)
        pyg_data = self.process()
        self.data, self.slices = self.collate([pyg_data])
        self.transform = transform

    def process(self):
        dpr_data = self.dpr_data
        edge_index = torch.LongTensor(dpr_data.adj.nonzero())
        # by default, the features in pyg data is dense
        if sp.issparse(dpr_data.features):
            x = torch.FloatTensor(dpr_data.features.todense()).float()
        else:
            x = torch.FloatTensor(dpr_data.features).float()
        y = torch.LongTensor(dpr_data.labels)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.train_mask = None
        data.val_mask = None
        data.test_mask = None
        return data


    def get(self, idx):
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[self.data.__cat_dim__(key, item)] = slice(slices[idx],
                                                   slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def _download(self):
        pass


