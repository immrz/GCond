import torch
import numpy as np


def OBJ(i):
    # return 0 if max(i) == min(i) else 1
    return max(i) - min(i)


class dEMDLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        d, n = x.shape

        AA = x.clone()

        xx = {}
        dual = torch.zeros(d, n).double().to(x.device)
        idx = [0, ] * d
        obj = 0

        while all([i < n for i in idx]):

            vals = [AA[i, j] for i, j in zip(range(d), idx)]

            minval = min(vals).clone()
            ind = vals.index(minval)
            xx[tuple(idx)] = minval
            obj += (OBJ(idx)) * minval
            for i, j in zip(range(d), idx): AA[i, j] -= minval
            oldidx = np.copy(idx)
            idx[ind] += 1
            if idx[ind] < n:
                dual[ind, idx[ind]] += OBJ(idx) - OBJ(oldidx) + dual[ind, idx[ind] - 1]

        # TODO: maybe unncessary/better way?
        for _, i in enumerate(idx):
            try:
                dual[_][i:] = dual[_][i]  # [x0, x1, ..., xi, 0, 0, 0, ...] => [x0, x1, ..., xi, xi, xi, xi, ...]
            except:
                pass

        # dualobj =  sum([_.dot(_d) for _, _d in zip(x, dual)])

        ctx.save_for_backward(dual)

        return obj

    @staticmethod
    def backward(ctx, grad_output):
        dual, = ctx.saved_tensors
        return grad_output * dual


dEMDLossFunc = dEMDLoss.apply
