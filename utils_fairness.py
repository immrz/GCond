from utils import Transd2Ind
import networkx as nx
import numpy as np
import torch
from collections import Counter
from torch_geometric.loader import NeighborSampler


class GroupedTrans(Transd2Ind):
    """
    Transductive graph dataset where the test nodes are divided into subgroups
    for fairness evaluation.
    """
    def __init__(self, dpr_data, keep_ratio, group_method, group_num=5):
        super().__init__(dpr_data, keep_ratio)
        assert group_method in ['agg', 'geo']

        nx_graph = nx.Graph(self.adj_full)
        if group_method == 'agg':
            self.test_groups = get_agg_feature_distance_community(self.idx_train,
                                                                  self.idx_test,
                                                                  nx_graph,
                                                                  self.feat_full,
                                                                  group_num=group_num)
        else:
            self.test_groups = get_geodesic_distance_community(self.idx_train,
                                                               self.idx_test,
                                                               nx_graph,
                                                               group_num=group_num)

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)
        group_acc = []

        # compute group-wise accuracy
        labels = torch.LongTensor(self.labels_test).cuda()
        preds = model_output[self.idx_test].max(1)[1].type_as(labels)
        correct = torch.eq(labels, preds).double()
        for group in self.test_groups:
            assert len(group) > 0
            group_acc.append(correct[group].sum().item() / len(group))
        delta = max(group_acc) - min(group_acc)
        res['delta'] = delta
        res['group_acc'] = group_acc
        return res


def get_agg_feature_distance_community(train_idx: np.ndarray,
                                       test_idx: np.ndarray,
                                       nx_graph: nx.Graph,
                                       feature: np.ndarray,
                                       group_num: int = 5):
    A = nx.adj_matrix(nx_graph).A
    A = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A, axis=1))
    D_1 = np.linalg.inv(D)

    test_num = test_idx.shape[0]
    agg = np.matmul(np.matmul(np.matmul(np.matmul(D_1, A), D_1), A), feature)  # (N, D) agg feature
    agg_distance = [float('inf') for _ in range(test_num)]  # idx_of_test_node_id -> distance
    for i in range(test_num):
        k = test_idx[i]  # this is the test_node_id
        for j in train_idx:
            agg_distance[i] = min(agg_distance[i],
                                  np.linalg.norm(agg[k] - agg[j]))  # minimize over j
    sort_res = [x[0] for x in sorted(enumerate(agg_distance), key=lambda x: x[1])]  # ascending by distance
    group_size = len(sort_res) // group_num + 1
    return [sort_res[i:i+group_size] for i in range(0, len(sort_res), group_size)]


def get_geodesic_distance_community(train_idx: np.ndarray,
                                    test_idx: np.ndarray,
                                    nx_graph: nx.Graph,
                                    group_num: int = 5):
    geodesic_matrix = nx.shortest_path_length(nx_graph)
    geodesic_matrix = dict(geodesic_matrix)  # {src -> {dest -> distance}}

    test_num = test_idx.shape[0]
    geo_distance = [float('inf') for _ in range(test_num)]  # idx_of_test_node_id -> distance
    for i in range(test_num):
        k = test_idx[i]  # this is the test_node_id
        for j in train_idx:
            if j not in geodesic_matrix[k]:  # not connected
                continue
            geo_distance[i] = min(geo_distance[i],
                                  geodesic_matrix[k][j])
    sort_res = [x[0] for x in sorted(enumerate(geo_distance), key=lambda x: x[1])]  # ascending by distance
    group_size = len(sort_res) // group_num + 1
    return [sort_res[i:i+group_size] for i in range(0, len(sort_res), group_size)]


class BiSensAttrTrans(Transd2Ind):
    def __init__(self, dpr_data, keep_ratio):
        super().__init__(dpr_data, keep_ratio)
        assert hasattr(dpr_data, 'sens')
        self.sens = dpr_data.sens.numpy()
        self.mask_s0 = self.sens[self.idx_test] == 0
        self.mask_s1 = self.sens[self.idx_test] == 1
        self.mask_s0_y1 = np.bitwise_and(self.mask_s0, self.labels_test == 1)
        self.mask_s1_y1 = np.bitwise_and(self.mask_s1, self.labels_test == 1)

        assert self.mask_s0.shape[0] > 0
        assert self.mask_s1.shape[0] > 0
        assert self.mask_s0_y1.shape[0] > 0
        assert self.mask_s1_y1.shape[0] > 0

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)

        if model_output.shape[1] == 1:
            # sigmoid is used for binary classification
            preds = (model_output[self.idx_test] >= 0).int().squeeze()
        else:
            # softmax is used
            preds = model_output[self.idx_test].max(1)[1]
        labels = torch.LongTensor(self.labels_test).cuda()
        correct = torch.eq(preds, labels).double()
        acc_s0 = correct[self.mask_s0].mean().item()
        acc_s1 = correct[self.mask_s1].mean().item()

        preds = preds.double()
        parity = abs(preds[self.mask_s0].mean().item() -
                     preds[self.mask_s1].mean().item())
        equality = abs(preds[self.mask_s0_y1].mean().item() -
                       preds[self.mask_s1_y1].mean().item())
        res['acc_s0'] = acc_s0
        res['acc_s1'] = acc_s1
        res['parity'] = parity
        res['equality'] = equality
        return res


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


class DegreeGroupedTrans(Transd2Ind):
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

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)
        group_acc = []

        # compute group-wise accuracy
        labels = torch.LongTensor(self.labels_test).cuda()
        preds = model_output[self.idx_test].max(1)[1].type_as(labels)
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
