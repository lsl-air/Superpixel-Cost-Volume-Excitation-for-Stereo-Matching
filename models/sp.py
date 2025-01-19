from __future__ import print_function
import torch
import torch.nn as nn
from .superpixel.Spixel_single_layer import SpixelNet
from .superpixel.train_util import *

def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

class SS(nn.Module):
    def __init__(self,patchsz):
        super(SS,self).__init__()
        self.patchsz = patchsz
        self.spixel = SpixelNet()
        
        self.conv0 = nn.Sequential(convbn(256,128,1,1,0,1),nn.ReLU(inplace=True))  
        self.conv1 = nn.Sequential(convbn(128,64,1,1,0,1),nn.ReLU(inplace=True)) 
        self.conv2 = nn.Sequential(convbn(64,64,1,1,0,1),nn.ReLU(inplace=True))
        self.smooth = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)                  

    def upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y
    
    def ref_to_LABXYfeat(self,ref,XY_feat):
        img_lab = rgb2Lab_torch(ref)
        LABXY_feat_full = build_LABXY_feat(img_lab, XY_feat)
        return LABXY_feat_full
    
    def forward(self,left):
        b,c,h,w = left.shape
        
        prob_l,_16x,_8x,_4x = self.spixel(left)
        # guide = self.smooth(self.upsample_add(self.conv1(self.upsample_add(self.conv0(_16x),_8x)),_4x))
        
        spixlId,xy_feat = init_spixel_grid(b,h,w, self.patchsz)
        curr_spixel_map_l = update_spixl_map(spixlId,prob_l)
        
        return xy_feat,curr_spixel_map_l,prob_l
        # return xy_feat,guide,curr_spixel_map_l,prob_l



class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv, self).__init__()

        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = nn.LeakyReLU()(x)#, inplace=True)
        return x

class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, 32, kernel_size=1, stride=1, padding=0),
            # BasicConv(32, 32, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(32, cv_chan, 1))

    def forward(self, cv, feat):
        '''
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)
        cv = torch.sigmoid(feat_att)*cv
        return cv