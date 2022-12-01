from models.gcn import GCN
from models.myappnp1 import APPNP1


class GCondFullData:

    def __init__(self, data, args, device='cuda', **kwargs):
        self.data = data
        self.args = args
        self.device = device

    def train(self):
        data, device, args = self.data, self.device, self.args
        model_class = {'gcn': GCN, 'appnp': APPNP1}[args.inner_model]
        # dropout = 0.5  # NOTE: uncommenting this would cause changes to non-GCN results
        dropout = 0.5 if args.inner_model == 'gcn' else 0.
        model = model_class(nfeat=data.feat_train.shape[1], nhid=self.args.hidden, dropout=dropout,
                            weight_decay=5e-4, nlayers=2,
                            nclass=data.nclass, device=device).to(device)

        model.fit_with_val2(data.feat_full,
                            data.adj_full,
                            data.labels_full,
                            data,
                            train_iters=1000,
                            verbose=True)

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
