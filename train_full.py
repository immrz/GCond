import os
import torch
from models.gcn import GCN
from models.gcn_pokec import GCNForBiCls
from models.myappnp1 import APPNP1
import wandb


class GCondFullData:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

    def train(self):
        data, device, args = self.data, self.device, self.args
        model_class = {'gcn': GCN, 'appnp': APPNP1, 'gcn_pokec': GCNForBiCls}[args.inner_model]
        # dropout = 0.5  # NOTE: uncommenting this would cause changes to non-GCN results
        dropout = 0.5 if args.inner_model == 'gcn' else 0.
        model = model_class(nfeat=data.feat_train.shape[1],
                            nhid=args.hidden,
                            dropout=dropout,
                            weight_decay=args.full_data_wd,
                            lr=args.full_data_lr,
                            nlayers=args.nlayers,
                            nclass=data.nclass,
                            device=device).to(device)

        model.fit_with_val2(data.feat_full,
                            data.adj_full,
                            data.labels_full,
                            data,
                            train_iters=args.full_data_epoch,
                            verbose=True)

        if self.args.save:
            if not os.path.isdir(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            torch.save(model.state_dict(), f'{self.args.save_dir}/model_{self.args.dataset}_full_{self.args.seed}.pt')
        model.eval()
        output = model.predict(data.feat_full, data.adj_full)
        test_res = self.data.compute_test_metric(output)

        # logging
        # wandb.run.summary['test acc'] = test_res['accuracy']
        test_msg = "Test set results: "
        for k, v in test_res.items():
            if isinstance(v, float):
                wandb.run.summary[k + '_test'] = v
                v = f'{v:.4f}'
            elif isinstance(v, (list, tuple)):
                v = '[' + ','.join([f'{vi:.4f}' for vi in v]) + ']'
            test_msg += f'{k}={v} '
        print(test_msg)

    def test(self):
        data, device, args = self.data, self.device, self.args
        model_class = {'gcn': GCN, 'appnp': APPNP1, 'gcn_pokec': GCNForBiCls}[args.inner_model]
        dropout = 0.5 if args.inner_model == 'gcn' else 0.
        model = model_class(nfeat=data.feat_train.shape[1],
                            nhid=args.hidden,
                            dropout=dropout,
                            weight_decay=args.full_data_wd,
                            lr=args.full_data_lr,
                            nlayers=args.nlayers,
                            nclass=data.nclass,
                            device=device).to(device)
        model.load_state_dict(torch.load(f'{self.args.save_dir}/model_{self.args.dataset}_full_{self.args.seed}.pt'))
        model.eval()
        output = model.predict(data.feat_full, data.adj_full)
        test_res = self.data.compute_test_metric(output)

        test_msg = "Test set results: "
        for k, v in test_res.items():
            if isinstance(v, float):
                v = f'{v:.4f}'
            elif isinstance(v, (list, tuple)):
                v = '[' + ','.join([f'{vi:.4f}' for vi in v]) + ']'
            test_msg += f'{k}={v} '
        print(test_msg)


class GCondFullDataInductive:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

    def train(self):
        data, device, args = self.data, self.device, self.args
        model_class = GCN
        dropout = 0.5 if self.args.dataset in ['reddit'] else 0
        model = model_class(nfeat=data.feat_train.shape[1],
                            nhid=args.hidden,
                            dropout=dropout,
                            weight_decay=args.full_data_wd,
                            lr=args.full_data_lr,
                            nlayers=args.nlayers,
                            nclass=data.nclass,
                            device=device).to(device)

        model.fit_with_val(data.feat_train,
                           data.adj_train,
                           data.labels_train,
                           data,
                           train_iters=args.full_data_epoch,
                           verbose=True,
                           normalize=True,
                           noval=True)

        if self.args.save:
            if not os.path.isdir(self.args.save_dir):
                os.makedirs(self.args.save_dir)
            torch.save(model.state_dict(), f'{self.args.save_dir}/full_{self.args.seed}.pt')

        model.eval()
        output = model.predict(data.feat_test, data.adj_test)
        test_res = self.data.compute_test_metric(output)

        # logging
        # wandb.run.summary['test acc'] = test_res['accuracy']
        test_msg = "Test set results: "
        for k, v in test_res.items():
            if isinstance(v, float):
                wandb.run.summary[k + '_test'] = v
                v = f'{v:.4f}'
            elif isinstance(v, (list, tuple)):
                v = '[' + ','.join([f'{vi:.4f}' for vi in v]) + ']'
            test_msg += f'{k}={v} '
        print(test_msg)
