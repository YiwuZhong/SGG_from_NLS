# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms
from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist
from maskrcnn_benchmark.modeling.box_coder import BoxCoder
from .utils_relation import obj_prediction_nms

class PostProcessor(nn.Module):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
        cap2vg_dict=None,
    ):
        super(PostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        if cap2vg_dict is not None:
            self.cap2vg_dict = np.load(cap2vg_dict, allow_pickle=True).tolist()
            self.cap2vg_obj = self.cap2vg_dict['object_cls']
            self.cap2vg_predicate = self.cap2vg_dict['predicate_cls']   
            # get the indices of the categories which don't exist in VG, ignore their logits
            self.inds_ign_obj = [k for k in self.cap2vg_obj if len(self.cap2vg_obj[k]) == 0]
            self.inds_ign_obj = torch.from_numpy(np.array(sorted(self.inds_ign_obj))).long()
            self.inds_ign_rel = [k for k in self.cap2vg_predicate if len(self.cap2vg_predicate[k]) == 0]
            self.inds_ign_rel = torch.from_numpy(np.array(sorted(self.inds_ign_rel))).long()    
            # get the matrix for mapping the probability of predicted categories into the VG categories
            self.merge_mtx_obj = torch.zeros(len(self.cap2vg_obj), 151)
            for cap_ind in self.cap2vg_obj:
                if len(self.cap2vg_obj[cap_ind]) != 0:
                    for vg_ind in self.cap2vg_obj[cap_ind][0:1]:
                        self.merge_mtx_obj[cap_ind, vg_ind] = 1
            self.merge_mtx_rel = torch.zeros(len(self.cap2vg_predicate), 51)
            for cap_ind in self.cap2vg_predicate:
                if len(self.cap2vg_predicate[cap_ind]) != 0:
                    for vg_ind in self.cap2vg_predicate[cap_ind][0:1]:
                        self.merge_mtx_rel[cap_ind, vg_ind] = 1
        else:
            self.cap2vg_dict = None

    def forward(self, x, rel_pair_idxs, boxes, offline_od=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, refine_logits = x
        
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes
        )):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)

            obj_class_prob = F.softmax(obj_logit, -1)
            if self.cap2vg_dict is not None: # ignore the categories that don't exist in VG
                obj_class_prob = torch.matmul(obj_class_prob, self.merge_mtx_obj.type(obj_class_prob.dtype).to(obj_class_prob.device))
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                if not offline_od: # online object detector
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres) # shape [#box]
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]  # shape [#box]  
                else: # offline object detector
                    # assume boxes_per_cls is the same for all classes since boxes from offline OD are after nms 
                    # obj_pred would be quite similar to the result of obj_class_prob.argmax(1)
                    boxes_per_cls = box.bbox.unsqueeze(1).expand(-1, obj_class_prob.shape[1], -1).contiguous()
                    obj_pred = obj_prediction_nms(boxes_per_cls, obj_logit, self.later_nms_pred_thres,\
                               cap2vg=(self.inds_ign_obj,self.merge_mtx_obj) if self.cap2vg_dict is not None else None) # shape [#box]
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]  # shape [#box]  
            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                if not offline_od: # online object detector, get the box regressed towards regressed_box_idxs
                    boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
                else: # offline object detector, just use the detected box
                    boxlist = BoxList(box.bbox, box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            if self.cap2vg_dict is not None: # ignore the categories that don't exist in VG
                rel_class_prob = torch.matmul(rel_class_prob, self.merge_mtx_rel.type(rel_class_prob.dtype).to(rel_class_prob.device))
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1

            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx] #rel_class_prob[sorting_idx] if self.cap2vg_dict is None else rel_scores[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class) or (#rel, )
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(boxlist)
        return results


class UniterPostProcessor(nn.Module):
    """
    Different from original PostProcessor which takes detected objects and predicate as inputs, 
    this takes subject, object and predicate as inputs. Instead, convert the subject & object back
    to the detected object by averaging over the logits.
    Further, for the evaluation of the model trained by caption triplets, the predicted caption 
    categories will be converted into VG standard categories.
    
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        attribute_on,
        use_gt_box=False,
        later_nms_pred_thres=0.3,
        cap2vg_dict=None,
    ):
        super(UniterPostProcessor, self).__init__()
        self.attribute_on = attribute_on
        self.use_gt_box = use_gt_box
        self.later_nms_pred_thres = later_nms_pred_thres
        self.sigmoid = nn.Sigmoid()
        if cap2vg_dict is not None:
            self.cap2vg_dict = np.load(cap2vg_dict, allow_pickle=True).tolist()
            self.cap2vg_obj = self.cap2vg_dict['object_cls']
            self.cap2vg_predicate = self.cap2vg_dict['predicate_cls']
            # get the indices of the categories which don't exist in VG, ignore their logits
            self.inds_ign_obj = [k for k in self.cap2vg_obj if len(self.cap2vg_obj[k]) == 0]
            self.inds_ign_obj = torch.from_numpy(np.array(sorted(self.inds_ign_obj))).long()
            self.inds_ign_rel = [k for k in self.cap2vg_predicate if len(self.cap2vg_predicate[k]) == 0]
            self.inds_ign_rel = torch.from_numpy(np.array(sorted(self.inds_ign_rel))).long()       
            # get the matrix for mapping the probability of predicted categories into the VG categories
            self.merge_mtx_obj = torch.zeros(len(self.cap2vg_obj), 151)
            for cap_ind in self.cap2vg_obj:
                if len(self.cap2vg_obj[cap_ind]) != 0:
                    for vg_ind in self.cap2vg_obj[cap_ind][0:1]:
                        self.merge_mtx_obj[cap_ind, vg_ind] = 1
            self.merge_mtx_rel = torch.zeros(len(self.cap2vg_predicate), 51)
            for cap_ind in self.cap2vg_predicate:
                if len(self.cap2vg_predicate[cap_ind]) != 0:
                    for vg_ind in self.cap2vg_predicate[cap_ind][0:1]:
                        self.merge_mtx_rel[cap_ind, vg_ind] = 1
        else:
            self.cap2vg_dict = None

    def forward(self, x, rel_pair_idxs, boxes, offline_od=False):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the relation logits
                and finetuned object logits from the relation model.
            rel_pair_idxs （list[tensor]): subject and object indice of each relation,
                the size of tensor is (num_rel, 2)
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        relation_logits, sub_logits, obj_logits = x

        # convert the subject & object logtis back to detected objects by averaging
        num_rels_per_img = [r.shape[0] for r in rel_pair_idxs]
        sub_logits = sub_logits.split(num_rels_per_img, dim=0)
        obj_logits = obj_logits.split(num_rels_per_img, dim=0)  
        refine_logits = []

        # for each detected object, get the predicted logits from its counterparts (sub&obj) from corresponding triplets
        for bs_i, (rel_pair, box) in enumerate(zip(rel_pair_idxs,boxes)): # per image
            sub_ind = rel_pair[:,0]
            obj_ind = rel_pair[:,1]
            this_img = []

            for i in range(box.bbox.size(0)): # per detected region
                this_region = []
                inv_sub_ind = torch.nonzero(sub_ind == i).view(-1)
                this_region.append(sub_logits[bs_i][inv_sub_ind])
                inv_obj_ind = torch.nonzero(obj_ind == i).view(-1)
                this_region.append(obj_logits[bs_i][inv_obj_ind])
                # concat the counterparts of this region, and then average the logits
                this_region = torch.cat(this_region, dim=0)
                this_region = torch.mean(this_region, dim=0)
                this_img.append(this_region.view(1, -1))
            this_img = torch.cat(this_img, dim=0)
            
            refine_logits.append(this_img)
        
        #####################################################
        # the following code is from PostProcessor.forward()
        #####################################################
        if self.attribute_on:
            if isinstance(refine_logits[0], (list, tuple)):
                finetune_obj_logits, finetune_att_logits = refine_logits
            else:
                # just use attribute feature, do not actually predict attribute
                self.attribute_on = False
                finetune_obj_logits = refine_logits
        else:
            finetune_obj_logits = refine_logits

        results = []
        for i, (rel_logit, obj_logit, rel_pair_idx, box) in enumerate(zip(
            relation_logits, finetune_obj_logits, rel_pair_idxs, boxes)):
            if self.attribute_on:
                att_logit = finetune_att_logits[i]
                att_prob = torch.sigmoid(att_logit)

            obj_class_prob = F.softmax(obj_logit, -1)
            if self.cap2vg_dict is not None: # ignore the categories that don't exist in VG
                obj_class_prob = torch.matmul(obj_class_prob, self.merge_mtx_obj.type(obj_class_prob.dtype).to(obj_class_prob.device))
            obj_class_prob[:, 0] = 0  # set background score to 0
            num_obj_bbox = obj_class_prob.shape[0]
            num_obj_class = obj_class_prob.shape[1]

            if self.use_gt_box:
                obj_scores, obj_pred = obj_class_prob[:, 1:].max(dim=1)
                obj_pred = obj_pred + 1
            else:
                # NOTE: by kaihua, apply late nms for object prediction
                if not offline_od: # online object detector
                    obj_pred = obj_prediction_nms(box.get_field('boxes_per_cls'), obj_logit, self.later_nms_pred_thres) # shape [#box]
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]  # shape [#box]  
                else: # offline object detector
                    # assume boxes_per_cls is the same for all classes since boxes from offline OD are after nms 
                    # obj_pred would be quite similar to the result of obj_class_prob.argmax(1)
                    boxes_per_cls = box.bbox.unsqueeze(1).expand(-1, obj_class_prob.shape[1], -1).contiguous()
                    obj_pred = obj_prediction_nms(boxes_per_cls, obj_logit, self.later_nms_pred_thres,\
                                cap2vg=(self.inds_ign_obj,self.merge_mtx_obj) if self.cap2vg_dict is not None else None) # shape [#box]
                    obj_score_ind = torch.arange(num_obj_bbox, device=obj_logit.device) * num_obj_class + obj_pred
                    obj_scores = obj_class_prob.view(-1)[obj_score_ind]  # shape [#box]                      
            
            assert obj_scores.shape[0] == num_obj_bbox
            obj_class = obj_pred

            if self.use_gt_box:
                boxlist = box
            else:
                # mode==sgdet
                # apply regression based on finetuned object class
                device = obj_class.device
                batch_size = obj_class.shape[0]
                regressed_box_idxs = obj_class
                if not offline_od: # online object detector, get the box regressed towards regressed_box_idxs
                    boxlist = BoxList(box.get_field('boxes_per_cls')[torch.arange(batch_size, device=device), regressed_box_idxs], box.size, 'xyxy')
                else: # offline object detector, just use the detected box
                    boxlist = BoxList(box.bbox, box.size, 'xyxy')
            boxlist.add_field('pred_labels', obj_class) # (#obj, )
            boxlist.add_field('pred_scores', obj_scores) # (#obj, )

            if self.attribute_on:
                boxlist.add_field('pred_attributes', att_prob)
            
            # sorting triples according to score production
            obj_scores0 = obj_scores[rel_pair_idx[:, 0]]
            obj_scores1 = obj_scores[rel_pair_idx[:, 1]]
            rel_class_prob = F.softmax(rel_logit, -1)
            if self.cap2vg_dict is not None: # ignore the categories that don't exist in VG
                rel_class_prob = torch.matmul(rel_class_prob, self.merge_mtx_rel.type(rel_class_prob.dtype).to(rel_class_prob.device))
            rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
            rel_class = rel_class + 1

            triple_scores = rel_scores * obj_scores0 * obj_scores1
            _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)
            rel_pair_idx = rel_pair_idx[sorting_idx]
            rel_class_prob = rel_class_prob[sorting_idx] #rel_class_prob[sorting_idx] if self.cap2vg_dict is None else rel_scores[sorting_idx]
            rel_labels = rel_class[sorting_idx]

            boxlist.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
            boxlist.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class) or (#rel, )
            boxlist.add_field('pred_rel_labels', rel_labels) # (#rel, )
            # should have fields : rel_pair_idxs, pred_rel_class_prob, pred_rel_labels, pred_labels, pred_scores
            results.append(boxlist)
        return results

    
def make_roi_relation_post_processor(cfg):
    attribute_on = cfg.MODEL.ATTRIBUTE_ON
    use_gt_box = cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX
    later_nms_pred_thres = cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
    
    if cfg.WSVL.USE_UNITER:  # use uniter as relation predictor
        postprocessor = UniterPostProcessor(
            attribute_on,
            use_gt_box,
            later_nms_pred_thres,
            cap2vg_dict=cfg.WSVL.CAP_VG_DICT if cfg.WSVL.CAP_VG_DICT is not None and cfg.WSVL.USE_CAP_TRIP else None,
        )
    else:  # use default relation predictor
        postprocessor = PostProcessor(
            attribute_on,
            use_gt_box,
            later_nms_pred_thres,
            cap2vg_dict=cfg.WSVL.CAP_VG_DICT if cfg.WSVL.CAP_VG_DICT is not None and cfg.WSVL.USE_CAP_TRIP else None,
        )
            
    return postprocessor
