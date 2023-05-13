import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter
import torch.nn.functional as F
from utils import match_loss, regularization, row_normalize_tensor
import deeprobust.graph.utils as utils
from copy import deepcopy
import numpy as np
from tqdm import tqdm
from models.gcn import GCN
from models.myappnp1 import APPNP1
from models.sgc import SGC
from models.sgc_multi import SGC as SGC1
from models.parametrized_adj import PGE
import scipy.sparse as sp
from torch_sparse import SparseTensor
import wandb
from demd import DEMDLayer


class GCond:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device
        self.global_step = 0
        self.demd_lambda = args.demd_lambda
        self.demd_bins = args.demd_bins

        # n = data.nclass * args.nsamples
        if args.reduction_rate <= 1:
            n = int(data.feat_train.shape[0] * args.reduction_rate)
        else:  # if r > 1, consider r as the number of nodes in the synthetic graph
            n = int(args.reduction_rate)
        # from collections import Counter; print(Counter(data.labels_train))

        d = data.feat_train.shape[1]
        self.nnodes_syn = n
        self.feat_syn = nn.Parameter(torch.FloatTensor(n, d).to(device))
        self.pge = PGE(nfeat=d, nnodes=n, device=device, args=args).to(device)

        self.labels_syn = torch.LongTensor(self.generate_labels_syn(data)).to(device)

        self.reset_parameters()
        self.optimizer_feat = torch.optim.Adam([self.feat_syn], lr=args.lr_feat)
        self.optimizer_pge = torch.optim.Adam(self.pge.parameters(), lr=args.lr_adj)
        print('adj_syn:', (n,n), 'feat_syn:', self.feat_syn.shape)

        # if demd_lambda is nonnegative, create demd_layer
        self.demd_layer = DEMDLayer(discretization=self.demd_bins)

    def reset_parameters(self):
        self.feat_syn.data.copy_(torch.randn(self.feat_syn.size()))

    def generate_labels_syn(self, data):
        from collections import Counter
        counter = Counter(data.labels_train)
        num_class_dict = {}
        n = len(data.labels_train)

        sorted_counter = sorted(counter.items(), key=lambda x: x[1])
        sum_ = 0
        labels_syn = []
        self.syn_class_indices = {}
        for ix, (c, num) in enumerate(sorted_counter):
            if ix == len(sorted_counter) - 1:
                # num_class_dict[c] = int(n * self.args.reduction_rate) - sum_
                num_class_dict[c] = self.nnodes_syn - sum_
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]
            else:
                # num_class_dict[c] = max(int(num * self.args.reduction_rate), 1)
                num_class_dict[c] = max(int(num * self.nnodes_syn / n), 1)
                sum_ += num_class_dict[c]
                self.syn_class_indices[c] = [len(labels_syn), len(labels_syn) + num_class_dict[c]]
                labels_syn += [c] * num_class_dict[c]

        self.num_class_dict = num_class_dict
        return labels_syn

    def test_with_val(self, verbose=True, load_exist=False):
        res = {}

        data, device, args = self.data, self.device, self.args

        if not load_exist:
            feat_syn, pge, labels_syn = self.feat_syn.detach(), \
                                    self.pge, self.labels_syn
            adj_syn = pge.inference(feat_syn)
        else:
            feat_syn = torch.load(f'{args.save_dir}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt',
                                  map_location='cuda:0')
            adj_syn = torch.load(f'{args.save_dir}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt',
                                 map_location='cuda:0')
            pge, labels_syn = self.pge, self.labels_syn

        # with_bn = True if args.dataset in ['ogbn-arxiv'] else False
        model_class = {'gcn': GCN, 'appnp': APPNP1}[args.inner_model]
        # dropout = 0.5  # NOTE: uncommenting this would cause changes to non-GCN results
        dropout = 0.5 if args.inner_model == 'gcn' else 0.
        model = model_class(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=dropout,
                            weight_decay=5e-4, nlayers=2,
                            nclass=data.nclass, device=device).to(device)

        if self.args.dataset in ['ogbn-arxiv']:
            model = GCN(nfeat=feat_syn.shape[1], nhid=self.args.hidden, dropout=0.5,
                        weight_decay=0e-4, nlayers=2, with_bn=False,
                        nclass=data.nclass, device=device).to(device)

        if self.args.save:
            if not os.path.isdir(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(adj_syn, f'{args.save_dir}/adj_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')
            torch.save(feat_syn, f'{args.save_dir}/feat_{args.dataset}_{args.reduction_rate}_{args.seed}.pt')

        if self.args.lr_adj == 0:
            n = len(labels_syn)
            adj_syn = torch.zeros((n, n))

        model.fit_with_val(feat_syn, adj_syn, labels_syn, data,
                           train_iters=600, normalize=True, verbose=False)

        model.eval()
        labels_test = torch.LongTensor(data.labels_test).cuda()

        labels_train = torch.LongTensor(data.labels_train).cuda()
        output = model.predict(data.feat_train, data.adj_train)
        loss_train = F.nll_loss(output, labels_train)
        acc_train = utils.accuracy(output, labels_train)
        if verbose:
            print("Train set results:",
                  "loss={:.4f}".format(loss_train.item()),
                  "accuracy={:.4f}".format(acc_train.item()))
        # res.append(acc_train.item())
        res['acc_train'] = acc_train.item()

        # Full graph
        output = model.predict(data.feat_full, data.adj_full)
        test_res = self.data.compute_test_metric(output)
        if verbose:
            test_msg = "Test set results: "
            for k, v in test_res.items():
                if isinstance(v, float):
                    # if scalar, add a suffix _test so that the average will be computed afterwards
                    res[k + '_test'] = v
                    v = f'{v:.4f}'
                elif isinstance(v, (list, tuple)):
                    # if iterable, convert to string and just print
                    v = '[' + ','.join([f'{vi:.4f}' for vi in v]) + ']'
                test_msg += f'{k}={v} '
            print(test_msg)
        return res

    def train(self, verbose=True):
        args = self.args
        data = self.data
        feat_syn, pge, labels_syn = self.feat_syn, self.pge, self.labels_syn
        features, adj, labels = data.feat_full, data.adj_full, data.labels_full
        idx_train = data.idx_train

        syn_class_indices = self.syn_class_indices

        features, adj, labels = utils.to_tensor(features, adj, labels, device=self.device)

        # initialize feat_syn with original node features of the same classes
        feat_sub, adj_sub = self.get_sub_adj_feat(features)
        self.feat_syn.data.copy_(feat_sub)

        if utils.is_sparse_tensor(adj):
            adj_norm = utils.normalize_adj_tensor(adj, sparse=True)
        else:
            adj_norm = utils.normalize_adj_tensor(adj)

        adj = adj_norm
        adj = SparseTensor(row=adj._indices()[0],
                           col=adj._indices()[1],
                           value=adj._values(),
                           sparse_sizes=adj.size()).t()

        outer_loop, inner_loop = get_loops(args)
        self.global_step = 0

        try:
            samp_neig_it = PreSampNeighbIter(self.args.dataset)
            print("Using pre-sampled neighbors for faster computation.")
        except FileNotFoundError:
            print("Pre-sampled neighbors not found. Sample the neighbors with retrieve_class_sampler method instead.")
            samp_neig_it = None

        for it in range(args.epochs+1):
            loss_avg = 0  # loss over one epoch of training condensed graph

            if args.dataset in ['ogbn-arxiv']:
                model = SGC1(nfeat=feat_syn.shape[1],
                             nhid=self.args.hidden,
                             dropout=0.0,
                             with_bn=False,
                             weight_decay=0e-4,
                             nlayers=2,
                             nclass=data.nclass,
                             device=self.device).to(self.device)
            else:
                if args.sgc == 1:
                    model = SGC(nfeat=data.feat_train.shape[1],
                                nhid=args.hidden,
                                nclass=data.nclass,
                                dropout=args.dropout,
                                nlayers=args.nlayers,
                                with_bn=False,
                                device=self.device).to(self.device)
                else:
                    model = GCN(nfeat=data.feat_train.shape[1],
                                nhid=args.hidden,
                                nclass=data.nclass,
                                dropout=args.dropout,
                                nlayers=args.nlayers,
                                device=self.device).to(self.device)

            model.initialize()  # first loop - sample initial parameters for the condensation model

            model_parameters = list(model.parameters())

            optimizer_model = torch.optim.Adam(model_parameters, lr=args.lr_model)
            model.train()

            for ol in range(outer_loop):  # second loop - train for this many steps
                adj_syn = pge(self.feat_syn)
                adj_syn_norm = utils.normalize_adj_tensor(adj_syn, sparse=False)
                feat_syn_norm = feat_syn

                BN_flag = False
                for module in model.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    model.train() # for updating the mu, sigma of BatchNorm
                    output_real = model.forward(features, adj_norm)
                    for module in model.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer

                loss = torch.tensor(0.0).to(self.device)  # loss over all classes
                loss_ot_accum = 0.0  # OT loss over all classes
                optimizer_model.zero_grad()

                if samp_neig_it is not None:
                    try:
                        samp_neig = next(samp_neig_it)
                    except StopIteration:
                        print("Pre-sampled neighbors exhausted. Reinitializing ...")
                        samp_neig_it = PreSampNeighbIter(self.args.dataset)
                        samp_neig = next(samp_neig_it)
                else:
                    samp_neig = None

                outputs, batch_gids = [], []
                for c in range(data.nclass):  # third loop - process each class separately
                    if samp_neig is not None:
                        batch_size, n_id, adjs = samp_neig[c]
                    else:
                        batch_size, n_id, adjs = data.retrieve_class_sampler(c, adj, transductive=True, args=args)
                    if args.nlayers == 1:
                        adjs = [adjs]

                    adjs = [adj.to(self.device) for adj in adjs]
                    output = model.forward_sampler(features[n_id], adjs)
                    loss_real = F.nll_loss(output, labels[n_id[:batch_size]])

                    # compute OT loss
                    if self.demd_lambda >= 0:
                        # train_gid = self.data.train_gid
                        train_gid = self.data.all_gid  # above is wrong, `n_id` indexes into the full graph
                        batch_gid = train_gid[n_id[:batch_size]]
                        assert batch_gid.min() >= 0
                        outputs.append(output)
                        batch_gids.append(batch_gid)
                        # loss_ot = self.demd_lambda * self.demd_layer(output, batch_gid)
                        # loss_ot.backward(retain_graph=True)  # accumulate model gradients
                        # loss_ot_accum += loss_ot.item() / self.demd_lambda

                    gw_real = torch.autograd.grad(loss_real, model_parameters, retain_graph=True)
                    gw_real = list((_.detach().clone() for _ in gw_real))
                    output_syn = model.forward(feat_syn, adj_syn_norm)

                    ind = syn_class_indices[c]
                    loss_syn = F.nll_loss(
                            output_syn[ind[0]: ind[1]],
                            labels_syn[ind[0]: ind[1]])
                    gw_syn = torch.autograd.grad(loss_syn, model_parameters, create_graph=True)
                    coeff = self.num_class_dict[c] / max(self.num_class_dict.values())
                    loss += coeff * match_loss(gw_syn, gw_real, args, device=self.device)

                loss_avg += loss.item()
                wandb.log({'loss_grad_match': loss.item()}, step=self.global_step)
                self.global_step += 1

                # TODO: regularize
                if args.alpha > 0:
                    loss_reg = args.alpha * regularization(adj_syn, utils.tensor2onehot(labels_syn))
                else:
                    loss_reg = torch.tensor(0)

                loss = loss + loss_reg

                # update sythetic graph
                self.optimizer_feat.zero_grad()
                self.optimizer_pge.zero_grad()
                loss.backward()
                if it % 50 < 10:
                    self.optimizer_pge.step()
                else:
                    self.optimizer_feat.step()

                # update model
                if self.demd_lambda >= 0:
                    outputs = torch.cat(outputs, dim=0)
                    batch_gids = torch.cat(batch_gids, dim=0).cuda()
                    loss_ot = self.demd_layer(outputs, batch_gids)
                    wandb.log({'loss_ot': loss_ot.item()}, step=self.global_step)
                    (loss_ot * self.demd_lambda).backward()
                    optimizer_model.step()

                if args.debug and ol % 5 ==0:
                    print('Gradient matching loss:', loss.item())

                if ol == outer_loop - 1:
                    # print('loss_reg:', loss_reg.item())
                    # print('Gradient matching loss:', loss.item())
                    break

                feat_syn_inner = feat_syn.detach()
                adj_syn_inner = pge.inference(feat_syn_inner)
                adj_syn_inner_norm = utils.normalize_adj_tensor(adj_syn_inner, sparse=False)
                feat_syn_inner_norm = feat_syn_inner
                for j in range(inner_loop):
                    optimizer_model.zero_grad()
                    output_syn_inner = model.forward(feat_syn_inner_norm, adj_syn_inner_norm)
                    loss_syn_inner = F.nll_loss(output_syn_inner, labels_syn)
                    loss_syn_inner.backward()
                    # print(loss_syn_inner.item())
                    optimizer_model.step() # update gnn param

            loss_avg /= (data.nclass*outer_loop)
            if it % 50 == 0:
                print('Epoch {}, loss_avg: {}'.format(it, loss_avg))

            # eval_epochs = [400, 600, 800, 1000, 1200, 1600, 2000, 3000, 4000, 5000]
            eval_epochs = list(range(199, args.epochs, 200))
            if verbose and it in eval_epochs:
                res = []
                runs = 1 if args.dataset in ['ogbn-arxiv'] else 3
                for i in range(runs):
                    res.append(self.test_with_val())

                # res = np.array(res)
                res = {k: np.array([_r[k] for _r in res]) for k in res[0].keys()}
                # print('Train/Test Mean Accuracy:', repr([res.mean(0), res.std(0)]))
                print(f"Train Mean Accuracy: ({res['acc_train'].mean():.4f}, {res['acc_train'].std():.4f})", end=', ')
                print(f"Test Mean Accuracy: ({res['acc_test'].mean():.4f}, {res['acc_test'].std():.4f})")
                for k in res.keys():
                    if k.endswith('_test'):
                        wandb.log({k: res[k].mean()}, step=self.global_step)

    def get_sub_adj_feat(self, features):
        data = self.data
        args = self.args
        idx_selected = []

        from collections import Counter
        counter = Counter(self.labels_syn.cpu().numpy())

        for c in range(data.nclass):
            tmp = data.retrieve_class(c, num=counter[c])
            tmp = list(tmp)
            idx_selected = idx_selected + tmp
        idx_selected = np.array(idx_selected).reshape(-1)
        features = features[self.data.idx_train][idx_selected]

        # adj_knn = torch.zeros((data.nclass*args.nsamples, data.nclass*args.nsamples)).to(self.device)
        # for i in range(data.nclass):
        #     idx = np.arange(i*args.nsamples, i*args.nsamples+args.nsamples)
        #     adj_knn[np.ix_(idx, idx)] = 1

        from sklearn.metrics.pairwise import cosine_similarity
        # features[features!=0] = 1
        k = 2
        sims = cosine_similarity(features.cpu().numpy())
        sims[(np.arange(len(sims)), np.arange(len(sims)))] = 0
        for i in range(len(sims)):
            indices_argsort = np.argsort(sims[i])
            sims[i, indices_argsort[: -k]] = 0
        adj_knn = torch.FloatTensor(sims).to(self.device)
        return features, adj_knn


def get_loops(args):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if args.one_step:
        if args.dataset in ['ogbn-arxiv', 'credit']:
            return 5, 0
        return 1, 0
    if args.dataset in ['ogbn-arxiv']:
        return args.outer, args.inner
    if args.dataset in ['cora']:
        return 20, 15 # sgc
    if args.dataset in ['citeseer']:
        return 20, 15
    if args.dataset in ['physics']:
        return 20, 10
    else:
        return 20, 10


class PreSampNeighbIter:
    def __init__(self, dataset):
        if os.path.isfile(f'data/{dataset}/neighbors/disabled'):
            raise FileNotFoundError
        self.i = 0   # file index
        self.j = -1  # list index
        self.n = 10  # number of files
        self.prefix = f'data/{dataset}/neighbors/{dataset}_nbsamp_'
        self.data = torch.load(f'{self.prefix}{self.i}.pt')

    def __next__(self):
        if self.j+1 < len(self.data):
            self.j += 1
            return self.data[self.j]
        elif self.i+1 < self.n:
            self.i += 1
            self.j = 0
            self.data = torch.load(f'{self.prefix}{self.i}.pt')
            return self.data[self.j]
        else:
            raise StopIteration
