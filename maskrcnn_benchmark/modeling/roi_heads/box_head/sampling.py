# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch.nn import functional as F
import numpy as np
import numpy.random as npr

from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.balanced_positive_negative_sampler import (
    BalancedPositiveNegativeSampler
)
from maskrcnn_benchmark.modeling.utils import cat


class FastRCNNSampling(object):
    """
    Sampling RoIs
    """
    def __init__(
        self,
        proposal_matcher,
        fg_bg_sampler,
        box_coder,
    ):
        """
        Arguments:
            proposal_matcher (Matcher)
            fg_bg_sampler (BalancedPositiveNegativeSampler)
            box_coder (BoxCoder)
        """
        self.proposal_matcher = proposal_matcher
        self.fg_bg_sampler = fg_bg_sampler
        self.box_coder = box_coder

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Fast RCNN only need "labels" field for selecting the targets
        target = target.copy_with_fields(["labels", "attributes"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        labels = []
        attributes = []
        regression_targets = []
        matched_idxs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs_per_image = matched_targets.get_field("matched_idxs")
            
            labels_per_image = matched_targets.get_field("labels")
            attris_per_image = matched_targets.get_field("attributes")
            labels_per_image = labels_per_image.to(dtype=torch.int64)
            attris_per_image = attris_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs_per_image == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0
            attris_per_image[bg_inds,:] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs_per_image == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            # compute regression targets
            regression_targets_per_image = self.box_coder.encode(
                matched_targets.bbox, proposals_per_image.bbox
            )

            labels.append(labels_per_image)
            attributes.append(attris_per_image)
            regression_targets.append(regression_targets_per_image)
            matched_idxs.append(matched_idxs_per_image)

        return labels, attributes, regression_targets, matched_idxs

    def subsample(self, proposals, targets):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """

        labels, attributes, regression_targets, matched_idxs = self.prepare_targets(proposals, targets)

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)

        proposals = list(proposals)
        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, attributes_per_image, regression_targets_per_image, matched_idxs_per_image, proposals_per_image in zip(
            labels, attributes, regression_targets, matched_idxs, proposals
        ):
            proposals_per_image.add_field("labels", labels_per_image)
            proposals_per_image.add_field("attributes", attributes_per_image)
            proposals_per_image.add_field("regression_targets", regression_targets_per_image)
            proposals_per_image.add_field("matched_idxs", matched_idxs_per_image)

        # distributed sampled proposals, that were obtained on all feature maps
        # concatenated via the fg_bg_sampler, into individual feature map levels
        for img_idx, (pos_inds_img, neg_inds_img) in enumerate(zip(sampled_pos_inds, sampled_neg_inds)):
            img_sampled_inds = torch.nonzero(pos_inds_img | neg_inds_img).squeeze(1)
            proposals_per_image = proposals[img_idx][img_sampled_inds]
            proposals[img_idx] = proposals_per_image

        return proposals

    def assign_label_to_proposals(self, proposals, targets):
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            match_quality_matrix = boxlist_iou(target, proposal)
            # proposal.bbox.shape[0]; -1 is below low threshold; -2 is between thresholds; the others are matched gt indices
            matched_idxs = self.proposal_matcher(match_quality_matrix) 
            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "attributes"])  # only copy "labels" and "attributes" to extra_fields (dict)
            matched_targets = target[matched_idxs.clamp(min=0)]  # index items like List
            
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)

            labels_per_image[matched_idxs < 0] = 0  # background
            attris_per_image[matched_idxs < 0, :] = 0  # background
            proposals[img_idx].add_field("labels", labels_per_image)
            proposals[img_idx].add_field("attributes", attris_per_image)
        return proposals

    def assign_label_to_proposals_by_dict(self, proposals, targets, cls_dict):
        """
        Instead of using box location to match gt objects, use a dictionary to assign gt object labels
        """
        device = proposals[0].bbox.device
        for img_idx, (target, proposal) in enumerate(zip(targets, proposals)):
            # detected cls --> vg cls --> check whether exist in gt cls --> if so, randomly select the gt cls, and then random one gt object
            det_dist = proposal.extra_fields['det_dist'] # the dist after softmax
            det_cls = torch.argmax(det_dist, dim=1).cpu().numpy()
            gt_cls = target.get_field("labels").cpu().numpy()
            dict_matched_idxs = []
            for i, det_c in enumerate(det_cls):
                # for each detector cls, there might be multiple corresponding vg cls
                vg_cls = cls_dict[det_c]
                cls_cand = [vg_c for vg_c in vg_cls if vg_c in gt_cls]  

                if len(cls_cand) == 0:  # no matched gt cls
                    dict_matched_idxs.append(-99)
                else:  # there are gt cls that can be matched to detected objects, then randomly select one
                    selected_cls = cls_cand[npr.permutation(np.arange(len(cls_cand)))[0]]
                    # there are multiple gt objects that have same gt cls, then randomly select one,
                    # though it's class-level selection in this function (not instance-level selection)
                    obj_cand = [gt_i for gt_i, gt_c in enumerate(gt_cls) if gt_c == selected_cls]
                    selected_obj = obj_cand[npr.permutation(np.arange(len(obj_cand)))[0]]
                    dict_matched_idxs.append(selected_obj)
            dict_matched_idxs = torch.tensor(dict_matched_idxs, dtype=torch.long).to(device)

            ################# the following is the same as assign_label_to_proposals #################
            #match_quality_matrix = boxlist_iou(target, proposal)
            # proposal.bbox.shape[0]; -1 is below low threshold; -2 is between thresholds; the others are matched gt indices
            #matched_idxs = self.proposal_matcher(match_quality_matrix) 
            matched_idxs = dict_matched_idxs

            # Fast RCNN only need "labels" field for selecting the targets
            target = target.copy_with_fields(["labels", "attributes"])  # only copy "labels" and "attributes" to extra_fields (dict)
            matched_targets = target[matched_idxs.clamp(min=0)]  # index items like List
            
            labels_per_image = matched_targets.get_field("labels").to(dtype=torch.int64)
            attris_per_image = matched_targets.get_field("attributes").to(dtype=torch.int64)

            labels_per_image[matched_idxs < 0] = 0  # background
            attris_per_image[matched_idxs < 0, :] = 0  # background
            proposals[img_idx].add_field("labels", labels_per_image)
            proposals[img_idx].add_field("attributes", attris_per_image)
        return proposals


def make_roi_box_samp_processor(cfg):
    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_sampler = BalancedPositiveNegativeSampler(
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE, cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION
    )

    samp_processor = FastRCNNSampling(
        matcher,
        fg_bg_sampler,
        box_coder,
    )

    return samp_processor
