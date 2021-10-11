# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        if cfg.WSVL.OFFLINE_OD: # use offline object detector
            self.roi_heads = build_roi_heads(cfg, cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS)
        else: # use the default object detector
            self.backbone = build_backbone(cfg)
            self.rpn = build_rpn(cfg, self.backbone.out_channels)
            self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None, logger=None, det_feats=None, det_dists=None, det_boxes=None,\
        det_tag_ids=None, det_norm_pos=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.cfg.WSVL.OFFLINE_OD: # offline detector, skip detector
            features = None
            proposals = None
            proposal_losses = None
        else: # online detector
            images = to_image_list(images)  # ImageList
            features = self.backbone(images.tensors) #  2-D feature map at different levels
            proposals, proposal_losses = self.rpn(images, features, targets)  # proposals[i].bbox shaped [#box,4]

        if self.roi_heads:
            if self.cfg.WSVL.OFFLINE_OD: # offline detector
                if self.cfg.WSVL.USE_UNITER: # use uniter as relation predictor
                    x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, \
                                                det_feats=det_feats, det_dists=det_dists, det_boxes=det_boxes, \
                                                det_tag_ids=det_tag_ids, det_norm_pos=det_norm_pos)  
                else:
                    x, result, detector_losses = self.roi_heads(features, proposals, targets, logger, \
                                                det_feats=det_feats, det_dists=det_dists, det_boxes=det_boxes)
            else: # online detector
                x, result, detector_losses = self.roi_heads(features, proposals, targets, logger)
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            if not self.cfg.MODEL.RELATION_ON:
                # During the relationship training stage, the rpn_head should be fixed, and no loss. 
                losses.update(proposal_losses)
            return losses

        return result
