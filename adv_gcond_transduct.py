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
from gcond_agent_transduct import GCond, get_loops, PreSampNeighbIter


class AdvGCond(GCond):

    def __init__(self, data, args, device='cuda', **kwargs):
        super().__init__(data, args, device=device, **kwargs)
        self.adv_lambda = args.adv_lambda

    def train(self, verbose=True):
        print(f"Training with adversarial learning: lambda={self.adv_lambda}.")
        args = self.args
        data = self.data
        ngroup = data.all_gid.max() + 1
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

            adv = nn.Linear(data.nclass, ngroup)  # the dimension of hidden layers is nclass for SGC
            adv.to(self.device)
            optimizer_adv = torch.optim.Adam(adv.parameters(), lr=args.lr_model)

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

                    # store outputs for later adversarial training
                    # TODO: change `output` to the embedding before log_softmax
                    train_gid = self.data.all_gid
                    batch_gid = train_gid[n_id[:batch_size]]
                    assert batch_gid.min() >= 0
                    outputs.append(output)
                    batch_gids.append(batch_gid)

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

                # adversarial training
                outputs = torch.cat(outputs, dim=0)
                batch_gids = torch.cat(batch_gids, dim=0).cuda()
                # fix discriminator, update generator
                adv.requires_grad_(False)
                g_pred = adv(outputs)
                loss_adv = F.cross_entropy(g_pred, batch_gids)
                (-self.adv_lambda * loss_adv).backward()
                optimizer_model.step()
                # update discriminator
                adv.requires_grad_(True)
                optimizer_adv.zero_grad()
                g_pred = adv(outputs.detach())
                loss_adv = F.cross_entropy(g_pred, batch_gids)
                loss_adv.backward()
                optimizer_adv.step()

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
