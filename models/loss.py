import torch
import torch.nn as nn
import torch.nn.functional as F

from models.disp2prob import LaplaceDisp2Prob
from torch_scatter import scatter_mean

def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], reduction='mean'))
    return sum(all_losses)

def cost_loss(costs,disp_gt,spixel_viz,mask,vars):
    all_losses = []
    weights = [0.5,0.5,0.7,1.0]
    eps = torch.tensor(1e-20).cuda()
    B,H,W = disp_gt.shape
    mask = mask.view(B,H*W)
    label_sp = spixel_viz.long().view(-1,B,H*W) + torch.arange(B).view(B,1).cuda() * 512
    
    for estProb, weight, var in zip(costs, weights, vars):
        ce = 0
        estProb = estProb.view(-1,B,H*W) + eps
        gtProb = LaplaceDisp2Prob(192, disp_gt, variance=var, start_disp=0, dilation=1).getProb().view(-1,B,H*W)
        
        pred = torch.exp(scatter_mean(estProb[:,mask].log(),label_sp[:,mask],dim=-1))
        gt = torch.exp(scatter_mean(gtProb[:,mask].log(),label_sp[:,mask],dim=-1))
        ce = -((gt * pred.log())).sum(dim=0, keepdim=True).mean()

        var = var.view(B,H*W)
        nll = (-1.0 * var.log() * mask).mean()

        all_losses.append(weight*(ce+nll))

    return sum(all_losses)

#for unimodal
def cost_loss_uni(costs,disp_gt,mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    mask = mask.detach_().type_as(disp_gt)
    mask_disp_gt = disp_gt.clone() * mask
    gtProb = LaplaceDisp2Prob(192, mask_disp_gt, variance=1.0, start_disp=0, dilation=1).getProb()
    for cost_est, weight in zip(costs, weights):
        costProb = F.log_softmax(cost_est,dim=1)
        ce_gt = -((gtProb * costProb) * mask.unsqueeze(1).float()).sum(dim=1, keepdim=True).mean()
        all_losses.append(weight * ce_gt)
    return sum(all_losses)

#L1
# def compute_slic_loss(reconstr_fea,LABXY_featL_full):
    
#     # m_w = 30
#     m_w = 0.5
#     sp_w = 1
#     patch_sz = 8
#     b = reconstr_fea.size()[0]
        
#     loss_map_L = (reconstr_fea - LABXY_featL_full)
#     col_err = torch.norm(loss_map_L[:, :-2, :, :], p=1, dim=1).mean().unsqueeze(0)
#     pos_err = torch.norm(loss_map_L[:, -2:, :, :], p=2, dim=1).mean().unsqueeze(0)
                
#     loss_col = col_err.sum()
#     loss_pos = m_w / patch_sz * pos_err 

#     spixle_loss = sp_w * (loss_col + loss_pos) / b
    
#     del loss_map_L,reconstr_fea,LABXY_featL_full
    
#     return spixle_loss

def compute_slic_loss(reconstr_fea,LABXY_featL_full,mask):
    
    # m_w = 30
    m_w = 0.005
    sp_w = 1
    patch_sz = 16
    b = reconstr_fea.size()[0]
    loss_map_L = (reconstr_fea - LABXY_featL_full) * mask.unsqueeze(1)
    col_err = torch.norm(loss_map_L[:, :-2, :, :], p=1, dim=1).mean().unsqueeze(0)
    pos_err = torch.norm(loss_map_L[:, -2:, :, :], p=2, dim=1).mean().unsqueeze(0)
                
    loss_col = col_err.sum()
    loss_pos = m_w / patch_sz * pos_err 

    spixle_loss = sp_w * (loss_col + loss_pos) / b
    
    del loss_map_L,reconstr_fea,LABXY_featL_full
    
    return spixle_loss
#L2
def compute_slic_loss_color(reconstr_fea,LABXY_featL_full):
    
    # m_w = 30
    m_w = 5
    sp_w = 1
    patch_sz = 16
    b = reconstr_fea.size()[0]
        
    loss_map_L = (reconstr_fea - LABXY_featL_full)
    col_err = torch.norm(loss_map_L[:, :-2, :, :], p=2, dim=1).mean().unsqueeze(0)
    pos_err = torch.norm(loss_map_L[:, -2:, :, :], p=2, dim=1).mean().unsqueeze(0)
                
    loss_col = col_err.sum()
    loss_pos = m_w / patch_sz * pos_err 

    spixle_loss = sp_w * (loss_col + loss_pos) / b
    
    del loss_map_L,reconstr_fea,LABXY_featL_full
    
    return spixle_loss
