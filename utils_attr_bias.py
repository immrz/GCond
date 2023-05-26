import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from utils import Transd2Ind
from typing import NamedTuple
import torch


class _DataWrapper(NamedTuple):
    features: np.ndarray
    labels: np.ndarray
    adj: sp.coo_matrix
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray
    sens: np.ndarray


def _feature_norm(features):
    min_values = features.min(axis=0)
    max_values = features.max(axis=0)
    return 2*(features - min_values)/(max_values-min_values) - 1


def _load_credit(path='./data/credit', sens_attr="Age", predict_attr="NoDefaultNextMonth", label_number=6000):
    idx_features_labels = pd.read_csv(f'{path}/credit.csv')
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    edges_unordered = np.genfromtxt(f'{path}/credit_edges.txt').astype('int')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = np.array(features.todense())
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    norm_features = _feature_norm(features)
    norm_features[:, 1] = features[:, 1]  # 1 is the index of Age
    features = norm_features

    data = _DataWrapper(features, labels, adj, idx_train, idx_val, idx_test, sens)

    return data


def _load_bail(sens_attr="WHITE", predict_attr="RECID", path="./data/bail", label_number=100):
    print(f"Reading Bail dataset, label_number={label_number}.")
    idx_features_labels = pd.read_csv(f'{path}/bail.csv')
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    edges_unordered = np.genfromtxt(f'{path}/bail_edges.txt').astype('int')

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = np.array(features.todense())
    # features = torch.FloatTensor(np.array(features.todense()))
    # labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels==0)[0]
    label_idx_1 = np.where(labels==1)[0]
    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number//2)], label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number//2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))], label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    # sens = torch.FloatTensor(sens)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)

    norm_features = _feature_norm(features)
    norm_features[:, 0] = features[:, 0]  # 0 is the index of WHITE
    features = norm_features

    data = _DataWrapper(features, labels, adj, idx_train, idx_val, idx_test, sens)
    return data


def _load_pokec(dataset, sens_attr='region', predict_attr='I_am_working_in_field', seed=20, path="./data/pokec/", label_number=500):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    # build graph
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    idx_map = {j: i for i, j in enumerate(idx)}  # user_id -> order
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)  # shape (E, 2)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = np.array(features.todense())

    import random
    random.seed(seed)
    label_idx = np.where(labels>=0)[0]
    random.shuffle(label_idx)

    idx_train = label_idx[:min(int(0.5 * len(label_idx)), label_number)]  # at most half of all data
    idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]  # a quarter of all data
    idx_test = label_idx[int(0.75 * len(label_idx)):]

    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])  # this is meaningless since all elements are already non-negative
    idx_test = np.asarray(list(sens_idx & set(idx_test)))

    # binarize labels, map those greater than 1 to 1
    labels[labels > 1] = 1
    sens[sens > 0] = 1

    data = _DataWrapper(features, labels, adj, idx_train, idx_val, idx_test, sens)
    return data


class BiClassBiAttrTrans(Transd2Ind):
    def __init__(self, dataset, data_root='./data', label_number=-1):
        kwargs = {}
        if label_number > 0:
            kwargs['label_number'] = label_number

        if dataset == 'credit':
            data = _load_credit(path=f'{data_root}/credit')
        elif dataset == 'bail':
            data = _load_bail(path=f'{data_root}/bail', **kwargs)
        else:
            assert dataset.startswith('pokec')
            data = _load_pokec('region_job' if dataset == 'pokec_z' else 'region_job_2', path=f'{data_root}/pokec')

        super().__init__(dpr_data=data, keep_ratio=1)
        self.sens = data.sens.astype('int')

        self.train_gid = torch.LongTensor(self.sens[self.idx_train]).cuda()
        self.test_gid = torch.LongTensor(self.sens[self.idx_test]).cuda()

        # scatter the train_gid to the full graph; -1 for non-train nodes, and non-negative integers for train nodes
        self.all_gid = -torch.ones(self.sens.shape[0], dtype=torch.long).cuda()
        self.all_gid[self.idx_train] = self.train_gid

        # print dataset statistics
        print(f"Training Set:\nN: {len(self.idx_train)}\t\tY=0: {(self.labels_train==0).sum()}\t\tY=1: {(self.labels_train==1).sum()}")
        print(f"Y=0 & S=0: {(self.train_gid[self.labels_train==0]==0).sum().item()}", end='\t\t')
        print(f"Y=0 & S=1: {(self.train_gid[self.labels_train==0]==1).sum().item()}", end='\t\t')
        print(f"Y=1 & S=0: {(self.train_gid[self.labels_train==1]==0).sum().item()}", end='\t\t')
        print(f"Y=1 & S=1: {(self.train_gid[self.labels_train==1]==1).sum().item()}")

        print(f"Test Set:\nN: {len(self.idx_test)}\t\tY=0: {(self.labels_test==0).sum()}\t\tY=1: {(self.labels_test==1).sum()}")
        print(f"Y=0 & S=0: {(self.test_gid[self.labels_test==0]==0).sum().item()}", end='\t\t')
        print(f"Y=0 & S=1: {(self.test_gid[self.labels_test==0]==1).sum().item()}", end='\t\t')
        print(f"Y=1 & S=0: {(self.test_gid[self.labels_test==1]==0).sum().item()}", end='\t\t')
        print(f"Y=1 & S=1: {(self.test_gid[self.labels_test==1]==1).sum().item()}")

    def compute_test_metric(self, model_output):
        res = super().compute_test_metric(model_output)
        labels = torch.LongTensor(self.labels_test).cuda()
        preds = model_output[self.idx_test].max(1)[1].type_as(labels)

        idx_s0 = (self.test_gid == 0)
        idx_s1 = (self.test_gid == 1)
        idx_s0_y1 = idx_s0 & (labels == 1)
        idx_s1_y1 = idx_s1 & (labels == 1)
        parity = (preds[idx_s0].sum() / idx_s0.sum() - preds[idx_s1].sum() / idx_s1.sum()).abs().item()
        equality = (preds[idx_s0_y1].sum() / idx_s0_y1.sum() - preds[idx_s1_y1].sum() / idx_s1_y1.sum()).abs().item()

        correct = torch.eq(preds, labels).double()
        acc_s0 = (correct[idx_s0].sum() / idx_s0.sum()).item()
        acc_s1 = (correct[idx_s1].sum() / idx_s1.sum()).item()

        res['parity'] = parity
        res['equality'] = equality
        res['delta'] = abs(acc_s0 - acc_s1)
        res['std'] = res['delta'] / 2
        res['group_acc'] = [acc_s0, acc_s1]

        return res
