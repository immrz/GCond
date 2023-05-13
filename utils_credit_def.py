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


def _load_credit(sens_attr="Age", predict_attr="NoDefaultNextMonth", label_number=6000):
    idx_features_labels = pd.read_csv('./data/credit/credit.csv')
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)
    header.remove('Single')

    edges_unordered = np.genfromtxt('./data/credit/credit_edges.txt').astype('int')

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


class BiClassBiAttrTrans(Transd2Ind):
    def __init__(self):
        data = _load_credit()
        super().__init__(dpr_data=data, keep_ratio=1)
        self.sens = data.sens.astype('int')

        self.train_gid = torch.LongTensor(self.sens[self.idx_train]).cuda()
        self.test_gid = torch.LongTensor(self.sens[self.idx_test]).cuda()

        # scatter the train_gid to the full graph; -1 for non-train nodes, and non-negative integers for train nodes
        self.all_gid = -torch.ones(self.sens.shape[0], dtype=torch.long).cuda()
        self.all_gid[self.idx_train] = self.train_gid

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


def _fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()
