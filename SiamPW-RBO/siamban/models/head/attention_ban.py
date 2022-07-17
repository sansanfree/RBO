from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from siamban.core.xcorr import xcorr_fast, xcorr_depthwise

class BAN(nn.Module):
    def __init__(self):
        super(BAN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError



 
      

class CARHead(torch.nn.Module):
    def __init__(self, in_channels,out_channels,cls_out_num_classes):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(CARHead, self).__init__()
        # TODO: Implement the sigmoid version first.
     
        self.fi = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        cls_tower = []
        reg_tower = []
        for i in range(3):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=0
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            reg_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=0
                )
            )
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU())
     
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('reg_tower', nn.Sequential(*reg_tower))
        
        self.cls_logits = nn.Conv2d(
            out_channels, cls_out_num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            out_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        

        # initialization
        for modules in [self.cls_tower, self.reg_tower,
                        self.cls_logits, self.bbox_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        x=self.fi(x)
        cls_tower = self.cls_tower(x)
      #  print("cls tower shape",cls_tower.shape)
        logits = self.cls_logits(cls_tower)
        reg_tower=self.reg_tower(x)
       # print("reg tower shape",reg_tower.shape)
        bbox_reg = self.bbox_pred(reg_tower)
     
        return logits, bbox_reg


def non_local_xcorr(fm, fq):
        B, C, h, w = fm.shape
       
        B, C, H, W = fq.shape
        #print("fm shape:",fm.shape)
        #print("fq shape:",fq.shape)
        fm0 = fm.clone()
        fq0 = fq.clone()
        
        fm = fm.contiguous().view(B, C, h * w)  # B, C, hw
        
        fm = fm.permute(0, 2, 1)  # B, hw, C
       
        fq = fq.contiguous().view(B, C, H * W)  # B, C, HW
        
       # print("fm shape:",fm.shape)
       # print("fq shape:",fq.shape)
        similar = torch.matmul(fm, fq) / math.sqrt(C)  # B, hw, HW
       # print("w shape:",similar.shape)
       
        similar = torch.softmax(similar, dim=1)   # B, hw, HW
       
        fm1 = fm0.view(B, C, h*w)  # B, C, hw
        mem_info = torch.matmul(fm1, similar)  # (B, C, hw) x (B, hw, HW) = (B, C, HW)
        mem_info = mem_info.view(B, C, H, W)

        y = torch.cat([mem_info, fq0], dim=1)
        return y






class NONLOCALBAN(BAN):
    def __init__(self, in_channels=256, out_channels=256, cls_out_channels=2):
        super(NONLOCALBAN, self).__init__()
        self.head = CARHead(in_channels, out_channels, cls_out_channels)
     
       
    def forward(self, z_f, x_f):
        features=non_local_xcorr(z_f,x_f)
        cls,reg= self.head(features)
       
        return cls,reg


class MultiBAN(BAN):
    def __init__(self, in_channels, cls_out_channels, weighted=False):
        super(MultiBAN, self).__init__()
        self.weighted = weighted
        for i in range(len(in_channels)):
            self.add_module('box'+str(i+2), NONLOCALBAN(in_channels[i], in_channels[i], cls_out_channels))
        if self.weighted:
            self.cls_weight = nn.Parameter(torch.ones(len(in_channels)))
            self.loc_weight = nn.Parameter(torch.ones(len(in_channels)))
         
        self.loc_scale = nn.Parameter(torch.ones(len(in_channels)))

    def forward(self, z_fs, x_fs):
        cls = []
        loc = []
      
       # print("len features",len(x_fs),len(adjacent_zfs))
        for idx, (z_f, x_f) in enumerate(zip(z_fs, x_fs), start=2):
            box = getattr(self, 'box'+str(idx))
            c, l= box(z_f, x_f)
            cls.append(c)
            loc.append(torch.exp(l*self.loc_scale[idx-2]))
            

        if self.weighted:
            cls_weight = F.softmax(self.cls_weight, 0)
            loc_weight = F.softmax(self.loc_weight, 0)
          

        def avg(lst):
            return sum(lst) / len(lst)

        def weighted_avg(lst, weight):
            s = 0
          
            for i in range(len(weight)):
                s += lst[i] * weight[i]
            return s

        if self.weighted:
            return weighted_avg(cls, cls_weight), weighted_avg(loc, loc_weight)
        else:
            return avg(cls), avg(loc)
