from utils import Transd2Ind
import networkx as nx
import numpy as np
import torch


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
        res['group_accuracy'] = group_acc
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
