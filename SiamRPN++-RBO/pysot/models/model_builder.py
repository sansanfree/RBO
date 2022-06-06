# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss,rank_cls_loss,rank_loc_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.utils.anchor import Anchors


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE, #resnet
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST: #True
            self.neck = get_neck(cfg.ADJUST.TYPE,#AdjustAllLayer
                                 **cfg.ADJUST.KWARGS)

        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE, #'MultiRPN'
                                     **cfg.RPN.KWARGS)
        
        # build mask head
        if cfg.MASK.MASK:  #False
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

        if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)
        self.rank_cls_loss=rank_cls_loss()
        self.rank_loc_loss=rank_loc_loss()

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def convert_bbox(self,delta, anchors):
        batch_size=delta.shape[0]
        delta = delta.view(batch_size, 4, -1)
        anchors=anchors.view(4,-1).permute(1,0).contiguous()
        output_boxes=torch.zeros(batch_size,4,delta.shape[2])
        for i in range (batch_size):
           output_boxes[i][0, :] = delta[i][0, :] * anchors[:, 2] + anchors[:, 0]
           output_boxes[i][1, :] = delta[i][1, :] * anchors[:, 3] + anchors[:, 1]
           output_boxes[i][2, :] = torch.exp(delta[i][2, :]) * anchors[:, 2]
           output_boxes[i][3, :] = torch.exp(delta[i][3, :]) * anchors[:, 3]
        return output_boxes


    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls,loc = self.rpn_head(self.zf, xf)

        return {
                'cls': cls,
                'loc': loc,
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        label_target = data['label_target'].cuda()
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                               cfg.ANCHOR.RATIOS,
                               cfg.ANCHOR.SCALES)

        anchors.generate_all_anchors(im_c=cfg.TRAIN.SEARCH_SIZE//2,
                                          size=cfg.TRAIN.OUTPUT_SIZE)
        anchors = anchors.all_anchors[1]
   
        anchors_tensor=torch.from_numpy(anchors).cuda()
       


        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc= self.rpn_head(zf, xf)
        pred_bboxes = self.convert_bbox(loc, anchors_tensor).cuda()

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)
        CR_loss=self.rank_cls_loss(cls,label_cls)
        IGR_loss_1,IGR_loss_2=self.rank_loc_loss(cls,label_cls,pred_bboxes,label_target)
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss +cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss+cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_1+cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_2
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['CR_loss'] = cfg.TRAIN.RANK_CLS_WEIGHT*CR_loss
        outputs['IGR_loss_1'] =cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_1
        outputs['IGR_loss_2'] = cfg.TRAIN.RANK_IGR_WEIGHT*IGR_loss_2
        return outputs
