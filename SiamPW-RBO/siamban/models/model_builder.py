# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torch.nn as nn
import torch.nn.functional as F

from siamban.core.config import cfg
from siamban.models.loss import select_cross_entropy_loss, select_iou_loss,rank_cls_loss,rank_loc_loss
from siamban.models.backbone import get_backbone
from siamban.models.head import get_ban_head
from siamban.models.neck import get_neck
from siamban.utils.point import Point

class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)
        self.rank_cls_loss=rank_cls_loss()
        self.rank_loc_loss=rank_loc_loss()
     
     

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf=zf
      
    

    def convert_bbox(self,delta, points):
        batch_size=delta.shape[0]
        delta = delta.view(batch_size, 4, -1) #delta shape before: [batch_size,4,25,25]
        points=points.view(2,-1) #points shape before: [2,25,25]
        output_boxes=torch.zeros(batch_size,4,delta.shape[2])
        for i in range (batch_size):
           output_boxes[i][0, :] = points[0,:] - delta[i][0, :]
           output_boxes[i][1, :] = points[1,:] - delta[i][1, :]
           output_boxes[i][2, :] = points[0,:] + delta[i][2, :]
           output_boxes[i][3, :] = points[1,:] + delta[i][3, :]
        return output_boxes


    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls,loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }


    def log_softmax(self, cls):
        if cfg.BAN.BAN:
            cls = cls.permute(0, 2, 3, 1).contiguous()
            cls = F.log_softmax(cls, dim=3)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_target = data['search_bbox'].cuda()
        zf = self.backbone(template)
        xf = self.backbone(search)
       
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        cls, loc = self.head(zf, xf)
        # get loss
        self.points = Point(cfg.POINT.STRIDE, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.SEARCH_SIZE//2)
        points = self.points.points
        point_tensor=torch.from_numpy(points).cuda()
        pred_bboxes = self.convert_bbox(loc, point_tensor).cuda()


        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = select_iou_loss(loc, label_loc, label_cls)
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
