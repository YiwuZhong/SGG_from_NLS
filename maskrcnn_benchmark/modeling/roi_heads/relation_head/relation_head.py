# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn

from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from ..attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from ..box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
from .roi_relation_predictors import make_roi_relation_predictor
from .inference import make_roi_relation_post_processor
from .loss import make_roi_relation_loss_evaluator
from .sampling import make_roi_relation_samp_processor

class ROIRelationHead(torch.nn.Module):
    """
    Generic Relation Head class.
    """

    def __init__(self, cfg, in_channels):
        super(ROIRelationHead, self).__init__()
        self.cfg = cfg.clone()
        # same structure with box head, but different parameters
        # these param will be trained in a slow learning rate, while the parameters of box head will be fixed
        # Note: there is another such extractor in union_feature_extractor
        self.union_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        if cfg.MODEL.ATTRIBUTE_ON:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True)
            feat_dim = self.box_feature_extractor.out_channels * 2
        else:
            self.box_feature_extractor = make_roi_box_feature_extractor(cfg, in_channels)
            feat_dim = self.box_feature_extractor.out_channels
        self.predictor = make_roi_relation_predictor(cfg, feat_dim)
        self.post_processor = make_roi_relation_post_processor(cfg)
        self.loss_evaluator = make_roi_relation_loss_evaluator(cfg)
        self.samp_processor = make_roi_relation_samp_processor(cfg)

        # parameters
        self.use_union_box = self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        
        if self.cfg.WSVL.HARD_DICT_MATCH is not None: # use class dictionary to match gt objects
            self.dict_match = True
        else:
            self.dict_match = False

    def forward(self, features, proposals, targets=None, logger=None, det_feats=None, det_dists=None,\
        det_tag_ids=None, det_norm_pos=None):
        """
        Relation head includes 4 steps:
        1. determine subject-object pairs
        2. obtain region features
        3. predict subject-relation-object labels
        4. model testing (post-processing) or model training (loss calculation)
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        # 1. determine subject-object pairs
        if self.training:
            # relation subsamples and assign ground truth label during training
            with torch.no_grad():
                if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.gtbox_relsample(proposals, targets)
                else:
                    # sampling pairs only depending on 'labels' field of proposals and targets
                    proposals, rel_labels, rel_pair_idxs, rel_binarys = self.samp_processor.detect_relsample(proposals, targets,\
                        use_bert=self.cfg.WSVL.USE_UNITER, dict_match=self.dict_match)
        else:
            rel_labels, rel_binarys = None, None
            rel_pair_idxs = self.samp_processor.prepare_test_pairs(proposals[0].bbox.device, proposals)

        # 2. use box_head to extract features that will be fed to the later predictor processing
        if self.cfg.WSVL.OFFLINE_OD:  # offline object detector detection features
            roi_features = torch.cat(det_feats)
        else:  # online object detector
            roi_features = self.box_feature_extractor(features, proposals) # [#box in batch,4096]

        if self.cfg.MODEL.ATTRIBUTE_ON:
            att_features = self.att_feature_extractor(features, proposals)
            roi_features = torch.cat((roi_features, att_features), dim=-1)

        if self.use_union_box:
            union_features = self.union_feature_extractor(features, proposals, rel_pair_idxs)
        else:
            union_features = None
        
        # 3. relation predictor that converts the features into predictions
        if self.cfg.WSVL.USE_UNITER:  # use uniter as relation predictor
            if not self.training and not self.cfg.WSVL.OFFLINE_OD: # online detector during testing, split a batch into several group to avoid OOM
                sub_logits, obj_logits, relation_logits, add_losses = self.group_inference(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger, \
                                                                            det_dists=det_dists, det_tag_ids=det_tag_ids, det_norm_pos=det_norm_pos)
            else:
                uniter_outputs = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger, \
                                                                            det_dists=det_dists, det_tag_ids=det_tag_ids, det_norm_pos=det_norm_pos)
                sub_logits, obj_logits, relation_logits, add_losses = uniter_outputs
        else:  # use default relation predictor
            refine_logits, relation_logits, add_losses = self.predictor(proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger, \
                                                                        det_dists=det_dists)

        # 4. model testing
        if not self.training:
            if self.cfg.WSVL.USE_UNITER:  # use uniter as relation predictor
                result = self.post_processor((relation_logits, sub_logits, obj_logits), rel_pair_idxs, proposals, offline_od=self.cfg.WSVL.OFFLINE_OD)
            else:  # use default relation predictor
                result = self.post_processor((relation_logits, refine_logits), rel_pair_idxs, proposals, offline_od=self.cfg.WSVL.OFFLINE_OD)
            return roi_features, result, {}
        
        # 4. model training: calculate loss
        if self.cfg.WSVL.USE_UNITER: # use uniter as relation predictor 
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, (sub_logits, obj_logits), rel_pair_idxs)
        else:  # use default relation predictor
            loss_relation, loss_refine = self.loss_evaluator(proposals, rel_labels, relation_logits, refine_logits)
        
        if self.cfg.MODEL.ATTRIBUTE_ON and isinstance(loss_refine, (list, tuple)):
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine[0], loss_refine_att=loss_refine[1])
        else:
            output_losses = dict(loss_rel=loss_relation, loss_refine_obj=loss_refine)

        output_losses.update(add_losses)

        return roi_features, proposals, output_losses
    
    def group_inference(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger, \
                        det_dists=None, det_tag_ids=None, det_norm_pos=None, a_group=1000):
        """
        OOM due to large #boxes when use online detector, have to break pairs into several pieces, each GPU run on single image at a time
        """
        rel_pair_idxs = rel_pair_idxs[0]  # only one image
        if union_features is not None:
            assert rel_pair_idxs.size(0) == union_features.size(0)
        num_groups = int(len(rel_pair_idxs) / a_group)  # only one image

        all_sub_logits = []
        all_obj_logits = []
        all_relation_logits = []                
        for grp_i in range(num_groups+1):
            if grp_i == num_groups:  # the last group
                this_rel_pair_idxs = [rel_pair_idxs[grp_i * a_group :]]
                this_union_features = union_features[grp_i * a_group :, :] if union_features is not None else None
            else:
                this_rel_pair_idxs = [rel_pair_idxs[grp_i * a_group : (grp_i + 1) * a_group]]
                this_union_features = union_features[grp_i * a_group : (grp_i + 1) * a_group, :] if union_features is not None else None
            uniter_outputs = self.predictor(proposals, this_rel_pair_idxs, rel_labels, rel_binarys, roi_features, this_union_features, logger, \
                                            det_dists=det_dists, det_tag_ids=det_tag_ids, det_norm_pos=det_norm_pos)
            sub_logits, obj_logits, relation_logits, add_losses = uniter_outputs
            all_sub_logits.append(sub_logits)
            all_obj_logits.append(obj_logits)
            all_relation_logits.append(relation_logits)
        sub_logits = torch.cat(all_sub_logits)
        obj_logits = torch.cat(all_obj_logits)
        all_relation_logits = [rel_img[0] for rel_img in all_relation_logits]
        relation_logits = (torch.cat(all_relation_logits),)

        return sub_logits, obj_logits, relation_logits, None, None

def build_roi_relation_head(cfg, in_channels):
    """
    Constructs a new relation head.
    By default, uses ROIRelationHead, but if it turns out not to be enough, just register a new class
    and make it a parameter in the config
    """
    return ROIRelationHead(cfg, in_channels)
