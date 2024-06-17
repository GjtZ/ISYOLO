#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F

# from yolox.utils import bboxes_iou      # 原始的计算iou会出现nan，我们重写该函数，过滤nan为0

import math

from yolox.models.losses import IOUloss
from yolox.models.network_blocks import BaseConv, DWConv

def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):             # 要不会触发cuda trigger
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError
    # print(bboxes_b,'asassss')
    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = torch.max(
            (bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = torch.min(
            (bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    area_i=torch.where(torch.isnan(area_i),torch.full_like(area_i, 0.0),area_i) # 过滤nan
    # area_i=torch.nan_to_num(area_i, nan=0.0)            # 过滤nan
    # print(br-tl )
    # print(torch.prod(br - tl, 2))
    return area_i / (area_a[:, None] + area_b - area_i )

class TALHead(nn.Module):
    def __init__(
        self,
        num_classes,
        width=1.0,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        act="silu",
        depthwise=False,
        gamma=1.5,
        ignore_thr=0.2,
        ignore_value=0.2,
        appear_factor = 1.5                   # ## DPM
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): wheather apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.gamma = gamma
        self.ignore_thr = ignore_thr
        self.ignore_value = ignore_value

        self.appear_factor = appear_factor        
       
        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)
        self.expanded_strides = [None] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        self.h_w=[]
        # print(self.use_l1)
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)

                self.h_w.append( (output.shape[-2:]))   # my

                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
              
            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
       
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets

        mixup = labels[0].shape[2] > 5
        if mixup:
            label_cut = labels[0][..., :5]
            support_label = labels[1][..., :5]
        else:
            label_cut = labels[0]
            support_label = labels[1]
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects
        support_nlabel = (support_label.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ious_targets = []
        # obj weight
        reg_pos_weights=[]

        appear_targets=[]                           

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            support_num_gt = int(support_nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                appear_target=outputs.new_ones((total_num_anchors,1))     # factor h
         
                
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ious_target = outputs.new_zeros((0, 1))
                # obj weight
                reg_pos_weight=outputs.new_zeros((0,1))

            else:
                gt_bboxes_per_image = labels[0][batch_idx, :num_gt, 1:5]
                support_gt_bboxes_per_image = labels[1][batch_idx, :support_num_gt, 1:5]
                gt_classes = labels[0][batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)

                # calculate pos weight for objectness
                matched_matirx_= F.one_hot(
                    matched_gt_inds, num_gt          
                ).T                                 
                matched_matirx_mask= matched_matirx_ >=1      
                gt_fg_class=F.one_hot(gt_matched_classes.unsqueeze(0). repeat(num_gt,1).view(-1).to(torch.int64),self.num_classes).view(num_gt,num_fg_img,self.num_classes)         
                gt_fg_class_matrix=torch.zeros_like(gt_fg_class)
                gt_fg_class_matrix[matched_matirx_mask] = gt_fg_class[matched_matirx_mask]      # [n,m_fg,8]   
                

                iou_positive=gt_fg_class_matrix * pred_ious_this_matching.unsqueeze(-1)        #  
                iou_positive_loss=-torch.log(iou_positive+1e-12)                  # iou loss
                loc_conf=torch.exp(-iou_positive_loss*5)                               # location quality estimates

                
                cls_conf=cls_preds[batch_idx].view(-1, self.num_classes)[fg_mask].sigmoid()* obj_preds[batch_idx].view(-1, 1)[fg_mask].sigmoid()     # [batch, n_anchors_all, 8]-->[1,m_fg,8]
                joint_conf=loc_conf*cls_conf.unsqueeze(0)   # [n,m_fg,8]         # Joint confidence

                cls_target_one=F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes     # one_hot label
                ) 
                # Normalization method: sum or max
                pos_fg_weight= (torch.exp(5*joint_conf) * joint_conf ) / torch.sum(torch.exp(5*joint_conf) * joint_conf ,dim=(1,2),keepdim=True).clamp(min=1e-12) #[n,m_fg,8] 
                # pos_fg_weight= (torch.exp(5*joint_conf) * joint_conf ) / torch.amax(torch.exp(5*joint_conf) * joint_conf ,dim=(1,2),keepdim=True).clamp(min=1e-12) #[n,m_fg,8] 

                pos_fg_weight=pos_fg_weight.sum(0)         # [m_fg,8]
                reg_pos_weight=pos_fg_weight[cls_target_one>0].unsqueeze(1)  #[m_fg,1]

                # print('此图片的gt框：',num_gt)
                # print('simota的分配正样本数量：',fg_mask.sum())
                obj_target = fg_mask.unsqueeze(-1)

                """
                #-------------------------------------------------------------------------------
                # 做sim-OTA分配正样本的可视化，运行vis_important.py文件时，取消下面代码的注释
                # To do a visualization of the positive sample of the sim-OTA allocation, uncomment the code below when running the vis_important.py file
                f0='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494080628232.jpg'
                name=f0.split('/')[-1].split('.')[0]
                save_path=f'./vis_important_w/{name}/'


                pre=0
                for i,(level_h,level_w) in enumerate(self.h_w):
                    now_num=level_h*level_w
                    cur_level=obj_target[pre:pre+now_num,:].reshape(level_h,level_w,1).squeeze(2)
                    pre+=now_num
                    cur_img = F.interpolate(imgs[:,0:3,:,:], size=(level_h,level_w), mode='bilinear')[batch_idx]# 512/16
                    cur_img=(cur_img.permute(1,2,0))
                    import cv2
                    import numpy as np
                    heatmap = cv2.applyColorMap(np.uint8(255*cur_level.to('cpu')),cv2.COLORMAP_JET)
                    heatmap =np.float32(heatmap)/255
                    cam = 200*1.5*heatmap + 1* np.float32(cur_img.cpu())[:,:,::-1]
                    cam= cam/np.max(cam)
                    cv2.imwrite(save_path+"ota{}.png".format(i),np.uint8(255 * cam))

                pos_fg_weight= (torch.exp(5*joint_conf) * joint_conf ) / torch.amax(torch.exp(5*joint_conf) * joint_conf ,dim=(1,2),keepdim=True).clamp(min=1e-12) #[n,m_fg,8] 
                # pos_fg_weight= (torch.exp(5*joint_conf) * joint_conf ) / torch.sum(torch.exp(5*joint_conf) * joint_conf ,dim=(1,2),keepdim=True).clamp(min=1e-12) #[n,m_fg,8] 

                pos_fg_weight=pos_fg_weight.sum(0)         # [m_fg,8]
                reg_pos_weight=pos_fg_weight[cls_target_one>0].unsqueeze(1)  #[m_fg,1]
                
                pre=0
                obj_pos_weight=torch.zeros_like(obj_target.to(torch.float32))          # [M_fg , 1]
                obj_pos_weight=obj_pos_weight.squeeze(1)            # [M_fg]
                obj_pos_weight[fg_mask]= reg_pos_weight.squeeze(1)
                obj_pos_weight=obj_pos_weight.unsqueeze(1) .detach()    

                for i,(level_h,level_w) in enumerate(self.h_w):
                    now_num=level_h*level_w
                    cur_level=obj_pos_weight[pre:pre+now_num,:].reshape(level_h,level_w,1).squeeze(2)
                    cur_level=cur_level/torch.max(cur_level)
                    pre+=now_num
                    cur_img = F.interpolate(imgs[:,0:3,:,:], size=(level_h,level_w), mode='bilinear')[batch_idx].permute(1,2,0) # 512/16
                    import cv2
                    import numpy as np          
                    heatmap = cv2.applyColorMap(np.uint8(255*cur_level.to('cpu')),cv2.COLORMAP_JET)
                    heatmap =np.float32(heatmap)/255         
                    cam =255*1.5* heatmap +np.float32(cur_img.cpu())[:,:,::-1]
                    cam= cam/np.max(cam)
                    cv2.imwrite(save_path+"pos{}.png".format(i),np.uint8(255 * cam))
                #-------------------------------------------------------------------------------------------
                """


                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

                ####
                if support_num_gt == 0:
                    ious = torch.ones((num_gt, 1)).cuda()
                    ious_target = ious[matched_gt_inds]
                else:
                    pair_iou_between_current_and_support = bboxes_iou(gt_bboxes_per_image,
                                                                  support_gt_bboxes_per_image, False)

                    ious, support_id = torch.max(pair_iou_between_current_and_support, dim=1)
                    filter_id = (ious < self.ignore_thr)
                    ious[filter_id] = self.ignore_value


                    ious_target = ious[matched_gt_inds].unsqueeze(1)

                # -------------------------------------------------------------------------------------------------------------------------------------------
                # DPM:Disappearing factor h
                appear_target = outputs.new_ones((total_num_anchors,1))  
                pair_iou_between_support_and_current = bboxes_iou(support_gt_bboxes_per_image,
                                                                gt_bboxes_per_image, False)  # iou            [n_t-1, n]
                if pair_iou_between_support_and_current!=[] and support_num_gt != 0:
                    ious_appear, current_id = torch.max(pair_iou_between_support_and_current, dim=1)  # [n_t-1]
                    appear_num_gt=(ious_appear<=0).sum()
                    if appear_num_gt!=0:
                        # print('消失的物体:',appear_num_gt)
                        appear_boxes=support_gt_bboxes_per_image[ious_appear<=0]   # Disappear the object's box [n_a,4]
                        if len(appear_boxes.size())==1:
                            appear_boxes=appear_boxes.unsqueeze(0)                                    # [n_a, 4]
                        
                        appear_fg_mask, appear_is_in_boxes_and_center = self.get_in_boxes_info( # noqa
                            appear_boxes,                                           
                            expanded_strides,                                                
                            x_shifts,                                                       
                            y_shifts,
                            total_num_anchors,                                              
                            appear_num_gt,                                                          
                        )
                        # print(appear_is_in_boxes_and_center.sum(0).sum(),'----------')     
                        appear_fg_mask_copy=appear_fg_mask.clone()
                        appear_fg_mask[appear_fg_mask_copy]=appear_is_in_boxes_and_center.sum(0).bool()     #  [h1*w1+h2*w2+h3*w3].bool   
                        appear_target[appear_fg_mask]= (torch.ones_like(appear_target[appear_fg_mask]) *self.appear_factor)
                    
            ious_targets.append(ious_target)

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)

            appear_targets.append(appear_target) #

            if self.use_l1:
                l1_targets.append(l1_target)

            # obj weight
            reg_pos_weights.append(reg_pos_weight)

        ious_targets = torch.cat(ious_targets, 0)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        # obj weight
        reg_pos_weights=torch.cat(reg_pos_weights,0).squeeze(1)    # [M_fg]

        appear_targets=torch.cat(appear_targets,0)      #AAL

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        #
        ious_targets = ious_targets.squeeze(1)
        gamma = self.gamma
        weight = 1 / (ious_targets ** gamma + 1e-8)

        iou_loss = self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        iou_loss_weight = (weight * iou_loss.sum()) / (weight * iou_loss).sum()
        iou_loss_weight = iou_loss_weight.detach()

        if self.use_l1:
            l1_loss = self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            l1_weight = weight.unsqueeze(1).repeat(1, 4)
            l1_weight = (l1_weight * l1_loss.sum()) / (l1_weight * l1_loss).sum()
            l1_weight = l1_weight.detach()


        num_fg = max(num_fg, 1)
        loss_iou = (
            iou_loss_weight * self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
        ).sum() / num_fg

        # obj weight 
        obj_pos_weights=torch.ones_like(obj_targets.to(torch.float32))        
        # reg_pos_weights=  (reg_pos_weights*fg_masks.sum())/(reg_pos_weights.sum())
        obj_loss = (           
           self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)  
        ).squeeze(1)   # [H*W]                                           

        reg_pos_weights=(reg_pos_weights*obj_loss[fg_masks].sum() )/ (reg_pos_weights*obj_loss[fg_masks]).sum()   

        obj_pos_weights=obj_pos_weights.squeeze(1)           
        obj_pos_weights[fg_masks]= reg_pos_weights
        obj_pos_weights=obj_pos_weights.unsqueeze(1)
        obj_pos_weights.detach()

# Normalization of disappearing weights       
        obj_loss = ( self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets) )          
        appear_weight= (appear_targets*obj_loss.sum())/ (appear_targets*obj_loss).sum()
        appear_weight=appear_weight.detach()

        loss_obj = (
            appear_weight *obj_pos_weights*self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
        ).sum() / num_fg
        
        loss_cls = (
            self.bcewithlog_loss(
                cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
            )
        ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                l1_weight * self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
            ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        """
        import cv2
        import random
        import numpy as np
        img = (imgs[batch_idx].cpu().numpy().transpose(1,2,0))
        img = img.astype(np.uint8)
        img = np.clip(img.copy(),0,255)
        coords = np.stack([(x_shifts*expanded_strides)[0][fg_mask].cpu().numpy(),(y_shifts*expanded_strides)[0][fg_mask].cpu().numpy()], 1)
        coords[:,0] = (x_shifts*expanded_strides+0.5*expanded_strides)[0][fg_mask].cpu().numpy()
        coords[:,1] = (y_shifts*expanded_strides+0.5*expanded_strides)[0][fg_mask].cpu().numpy()
        for coord in coords:
            cv2.circle(img, (int(coord[0]), int(coord[1])), 3, (255,0,0), -1)
        for bbox in gt_bboxes_per_image:
            cv2.rectangle(img, (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[3]/2)),(int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2)),  (0,255,0),2)
        cv2.imwrite('/data/debug_vis/'+str(random.randint(0,1000))+'.png', img[:,:,::-1])
        """

        """
        # Visualize simOTA
        # f0='data/Argoverse-1.1/tracking/val/7d37fc6b-1028-3f6f-b980-adb5fa73021e/ring_front_center/ring_front_center_315968385043071320.jpg'
        # f0='data/Argoverse-1.1/tracking/val/f1008c18-e76e-3c24-adcc-da9858fac145/ring_front_center/ring_front_center_315973412617897824.jpg'
        # f0='data/Argoverse-1.1/tracking/val/da734d26-8229-383f-b685-8086e58d1e05/ring_front_center/ring_front_center_315967919692039472.jpg'
        # f0='data/Argoverse-1.1/tracking/val/cb762bb1-7ce1-3ba5-b53d-13c159b532c8/ring_front_center/ring_front_center_315967331713239720.jpg'
        # f0='data/Argoverse-1.1/tracking/val/cb0cba51-dfaf-34e9-a0c2-d931404c3dd8/ring_front_center/ring_front_center_315972700940697960.jpg' 
        # f0='data/Argoverse-1.1/tracking/val/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9/ring_front_center/ring_front_center_315973074977221128.jpg'
        # f0='data/Argoverse-1.1/tracking/val/b1ca08f1-24b0-3c39-ba4e-d5a92868462c/ring_front_center/ring_front_center_315980040416005288.jpg'
        # f0='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968252735036520.jpg'
        # f0='data/Argoverse-1.1/tracking/val/033669d3-3d6b-3d3d-bd93-7985d86653ea/ring_front_center/ring_front_center_315968255432336368.jpg'
        f0='data/Argoverse-1.1/tracking/train/02cf0ce1-699a-373b-86c0-eb6fd5f4697a/ring_front_center/ring_front_center_315968494080628232.jpg'
        name=f0.split('/')[-1].split('.')[0]
        save_path=f'./vis_important_w/{name}/'
        import cv2
        import random
        import numpy as np
        img = (imgs[batch_idx][0:3,:,:].cpu().numpy().transpose(1,2,0))
        img = img.astype(np.uint8)
        img = np.clip(img.copy(),0,255)
        coords = np.stack([(x_shifts*expanded_strides)[0][fg_mask].cpu().numpy(),(y_shifts*expanded_strides)[0][fg_mask].cpu().numpy()], 1)
        coords[:,0] = (x_shifts*expanded_strides+0.5*expanded_strides)[0][fg_mask].cpu().numpy()
        coords[:,1] = (y_shifts*expanded_strides+0.5*expanded_strides)[0][fg_mask].cpu().numpy()
        for coord in coords:
            cv2.circle(img, (int(coord[0]), int(coord[1])), 3, (255,0,0), -1)
        for bbox in gt_bboxes_per_image:
            cv2.rectangle(img, (int(bbox[0]-bbox[2]/2),int(bbox[1]-bbox[3]/2)),(int(bbox[0]+bbox[2]/2),int(bbox[1]+bbox[3]/2)),  (0,255,0),2)
        # cv2.imwrite('./'+str(random.randint(0,2))+'.png', img[:,:,::-1])
        cv2.imwrite(save_path+'SIM-OTA'+'.png', img[:,:,::-1])
#------------------------------------------------------------------------------------------------------------
        """

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            cost_min, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
