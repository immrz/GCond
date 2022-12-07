from models.gcn import GraphConvolution
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import deeprobust.graph.utils as utils
from copy import deepcopy
import wandb


class GCNForBiCls(nn.Module):
    """
    GCN for binary classification dataset (like pokec). An implementation of E. Dai, WSDM 2021.
    """
    def __init__(self, nfeat, nhid, dropout=0.5, lr=0.01, weight_decay=5e-4, **kwargs):
        super().__init__()
        self.nfeat = nfeat
        self.lr = lr
        self.wd = weight_decay
        self.device = 'cuda:0'

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nhid)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(nhid, 1)
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, g, x):
        """
        g: adjacency matrix
        x: node features
        """
        x = F.relu(self.gc1(input=x, adj=g))
        x = self.dropout(x)
        x = self.gc2(input=x, adj=g)
        return x

    def fit_with_val2(self, features, adj, labels, data, train_iters=200, verbose=False, **kwargs):
        # move to gpu
        if not isinstance(adj, torch.Tensor):
            features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)
        else:
            features = features.to(self.device)
            adj = adj.to(self.device)
            labels = labels.to(self.device)

        # normalize adj
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        # reshape labels to (N, 1)
        labels = labels.unsqueeze(1)

        # training
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        best_acc_val = 0
        weights = None
        idx_train, idx_val = data.idx_train, data.idx_val

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(g=adj_norm, x=features)
            logits = self.fc(output)
            loss_train = self.loss(logits[idx_train], labels[idx_train].float())
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            # validate
            self.eval()
            with torch.no_grad():
                output = self.forward(g=adj_norm, x=features)
                logits = self.fc(output)
            loss_val = self.loss(logits[idx_val], labels[idx_val].float())
            pred_val = (logits[idx_val] >= 0).int()
            acc_val = (pred_val == labels[idx_val]).double().mean()

            # logging
            wandb.log({'loss_train': loss_train.item(),
                       'loss_val': loss_val.item(),
                       'acc_val': acc_val.item()})

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)

    @torch.no_grad()
    def predict(self, features, adj):
        self.eval()
        if not isinstance(adj, torch.Tensor):
            features, adj = utils.to_tensor(features, adj, device=self.device)
        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)
        return self.fc(self.forward(g=adj_norm, x=features))
