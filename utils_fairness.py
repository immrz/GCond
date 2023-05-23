from utils import Transd2Ind
import networkx as nx
import numpy as np
import torch
from collections import Counter
from torch_geometric.loader import NeighborSampler


def groupby_degree(adj: np.ndarray, thres: float, verbose=True):
    """Group nodes by degree.
    Parameters:
        adj: adjacency matrix, shape (N, M).
        thres: approximate number of groups; this is a floating point number,
            and the actual number of groups may be smaller.
    Returns:
        group_ids: ndarray shaped (N, ), i-th element is the group id of the i-th node.
    """
    # degree: length is N (number of nodes)
    degree = np.asarray(np.sum(adj, axis=1))
    degree = degree.squeeze().astype(np.int32)

    # degree -> number of nodes with that degree; length is D (number of distinct degrees)
    d2n = Counter(degree)

    # sort by degree, ascending
    d2n = list(sorted(d2n.items(), key=lambda _p: _p[0]))

    # separate degree and number lists
    d, n = list(zip(*d2n))

    cur_groupsize = 0  # number of nodes in the current group
    cut_position = [0]  # index in d2n to cut it into groups
    for i, (cur_degree, cur_number) in enumerate(d2n):
        # the last one, cut at D
        if i == len(d2n) - 1:
            cut_position.append(i + 1)
            continue
        cur_groupsize += cur_number

        # if current group contains enough nodes, stop and put a cut point here
        if cur_groupsize >= degree.shape[0] / thres:
            # cut at i+1, which is excluded
            cut_position.append(i + 1)
            cur_groupsize = 0

    # now the cut_position is [0, c1, c2, ..., ck] (where k is the number of groups)
    # ci and c{i+1} index into d2n:
    # d2n[ci], d2n[ci]+1, ..., d2n[c{i+1}]-1 are degrees and numbers belonging to the i-th group

    group_sizes = [sum(n[cut_position[i]:cut_position[i + 1]]) for i in range(len(cut_position) - 1)]

    if verbose:
        # min_degree->max_degree: n nodes
        for i in range(len(cut_position) - 1):
            min_deg = d[cut_position[i]]
            max_deg = d[cut_position[i + 1] - 1]
            print(f'{min_deg}->{max_deg}:\t\t{group_sizes[i]} nodes')

    group_ids = -np.ones(degree.shape[0], dtype=np.int32)
    for i in range(len(cut_position) - 1):
        start = cut_position[i]
        end = cut_position[i + 1] - 1
        group_mask = (degree >= d[start]) & (degree <= d[end])
        group_ids[group_mask] = i

    # check that every node is assigned to a group
    assert np.all(group_ids >= 0), 'Some nodes are not assigned to any group'
    return group_ids


def delta_std_parity(labels: torch.LongTensor, preds: torch.LongTensor, group_ids: torch.LongTensor):
    """Computes fairness metrics: delta, std and SP. Here SP is computed in a multi-class multi-group setting.
    """
    res = {}
    group_acc = []

    ng = group_ids.max().item() + 1
    ny = labels.max().item() + 1
    correct = torch.eq(labels, preds).float()

    for i in range(ng):
        group_mask = (group_ids == i)
        group_size = group_mask.sum().item()
        assert group_size > 0
        group_acc.append(correct[group_mask].sum().item() / group_size)

    res['delta'] = max(group_acc) - min(group_acc)
    res['std'] = np.std(group_acc)
    res['group_acc'] = group_acc

    # compute statistical parity
    average_parities = []
    for i in range(ng):
        group_mask = (group_ids == i)
        max_diff = -1
        for y in range(ny):
            pisy = torch.eq(preds, y).float()
            diff = torch.abs(pisy.mean() - pisy[group_mask].mean()).item()
            max_diff = max(max_diff, diff)
        assert max_diff >= 0
        average_parities.append(max_diff)
    res['parity'] = sum(average_parities) / ng
    return res


class DegreeGroupedTrans(Transd2Ind):
    """Transductive dataset, nodes grouped by degrees. For Cora.
    """
    def __init__(self, dpr_data, keep_ratio, thres):
        super().__init__(dpr_data, keep_ratio)

        # group training nodes by degree
        self.train_gid = torch.LongTensor(
            groupby_degree(adj=self.adj_full[self.idx_train], thres=thres)
        )

        # group testing nodes by degree
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

    def retrieve_class_sampler(self, c, adj, transductive, num=256, args=None):
        if self.class_dict2 is None:
            self.class_dict2 = {}
            for i in range(self.nclass):
                if transductive:
                    # idx is the index in all nodes, and adj is adj_full
                    idx = self.idx_train[self.labels_train == i]
                else:
                    # idx is the index in the training nodes, and adj is adj_train
                    idx = np.arange(len(self.labels_train))[self.labels_train==i]
                self.class_dict2[i] = idx

        if args.nlayers == 1:
            sizes = [15]
        if args.nlayers == 2:
            sizes = [10, 5]
            # sizes = [-1, -1]
        if args.nlayers == 3:
            sizes = [15, 10, 5]
        if args.nlayers == 4:
            sizes = [15, 10, 5, 5]
        if args.nlayers == 5:
            sizes = [15, 10, 5, 5, 5]

        if self.samplers is None:
            # each sampler for each class
            self.samplers = []
            for i in range(self.nclass):
                node_idx = torch.LongTensor(self.class_dict2[i])
                self.samplers.append(NeighborSampler(
                    adj,
                    node_idx=node_idx,
                    sizes=sizes,
                    batch_size=num,  # this is a useless parameter
                    num_workers=12,
                    return_e_id=False,
                    num_nodes=adj.size(0),
                    shuffle=True
                ))
        batch = np.random.permutation(self.class_dict2[c])[:num]
        out = self.samplers[c].sample(batch)
        return out
