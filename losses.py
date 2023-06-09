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
    def __init__(self, temperature=0.4):
        super(RandomWalkLoss, self).__init__()

        self.temperature = temperature
        self.rw_target = torch.zeros((2*args.bs, 2*args.bs)).scatter_(0, torch.arange(2*args.bs).roll(args.bs).unsqueeze(0), 1).to(args.device) #cuda()

    def forward(self, z1_features, z2_features, labels=None):
        device = args.device #(torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        z = torch.cat((z1_features, z2_features), dim=0)

        bs = z1_features.shape[0]
        z = F.normalize(z, p=2., dim=1)

        sim = torch.mm(z, z.T) / self.temperature

        sim = torch.exp(sim)

        sim = sim - torch.diag(torch.diag(sim))

        degree = torch.diag(sim.sum(dim=1))

        #rw = torch.inverse(degree) @ sim
        #laplacian = degree - sim

        rw = torch.diag(1./(sim.sum(dim=1))) @ sim

        loss = (off_diagonal(rw)-off_diagonal(self.rw_target)).pow(2).sum()

        return loss
    
class SupConLoss(nn.Module): # inspired by : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.5, device="cuda:0"):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, projection1, projection2, labels=None):

        projection1, projection2 = F.normalize(projection1), F.normalize(projection2)
        features = torch.cat([projection1.unsqueeze(1), projection2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()
        
        return loss