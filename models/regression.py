#!/usr/bin/env python
import torch
import torch.nn.functional as F
from models.init_nn import SubModule
    
class Regression(SubModule):
    def __init__(self,
                 max_disparity=192,
                 top_k=6):
        super(Regression, self).__init__()
        self.D = int(max_disparity)
        self.top_k = top_k
        self.ind_init = False

    def forward(self, cost, res=None):
        corr, disp = self.topkpool(cost, self.top_k)
        
        if res is not None:
            corr = F.softmax(corr*torch.gather(res, 1, disp),dim=1)
        else:
            corr = F.softmax(corr, 1)
        disp = torch.sum(corr * disp, 1, keepdim=False)
        return disp

    def topkpool(self, cost, k):
        if k == 1:
            _, ind = cost.sort(2, True)
            pool_ind_ = ind[:, :, :k]
            b, _, _, h, w = pool_ind_.shape
            pool_ind = pool_ind_.new_zeros((b, 1, 3, h, w))
            pool_ind[:, :, 1:2] = pool_ind_
            pool_ind[:, :, 0:1] = torch.max(
                pool_ind_-1, pool_ind_.new_zeros(pool_ind_.shape))
            pool_ind[:, :, 2:] = torch.min(
                pool_ind_+1, self.D*pool_ind_.new_ones(pool_ind_.shape))
            cv = torch.gather(cost, 2, pool_ind)

            disp = pool_ind

        else:
            _, ind = cost.sort(1, True)
            pool_ind = ind[:, :k]
            cv = torch.gather(cost, 1, pool_ind)
            disp = pool_ind

        return cv, disp