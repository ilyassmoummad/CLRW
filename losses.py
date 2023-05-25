import torch
from torch import nn
from torch.nn import functional as F
from args import args

import math
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class RandomWalkLoss(nn.Module):
    def __init__(self):
        super(RandomWalkLoss, self).__init__()
        
        self.rw_target = torch.zeros((2*args.bs, 2*args.bs)).scatter_(0, torch.arange(2*args.bs).roll(args.bs).unsqueeze(0), 1).to(args.device) #cuda()

    def forward(self, z1_features, z2_features, labels=None):
        device = args.device #(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        z = torch.cat((z1_features, z2_features), dim=0)

        bs = z1_features.shape[0]
        z = F.normalize(z, p=2., dim=1)

        sim = torch.mm(z, z.T) / args.tau

        sim = torch.exp(sim)

        sim = sim - torch.diag(torch.diag(sim))

        degree = torch.diag(sim.sum(dim=1))

        #rw = torch.inverse(degree) @ sim
        #laplacian = degree - sim

        rw = torch.diag(1./(sim.sum(dim=1))) @ sim

        loss = (off_diagonal(rw)-off_diagonal(self.rw_target)).pow(2).sum()

        return loss
    