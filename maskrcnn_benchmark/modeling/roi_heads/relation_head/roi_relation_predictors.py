# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.layers import smooth_l1_loss, kl_div_loss, entropy_loss, Label_Smoothing_Regression
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.data import get_dataset_statistics
from .model_msg_passing import IMPContext
from .model_vtranse import VTransEFeature
from .model_vctree import VCTreeLSTMContext
from .model_motifs import LSTMContext, FrequencyBias
from .model_motifs_with_attribute import AttributeLSTMContext
from .model_transformer import TransformerContext
from .model_uniter import UniterPreTrainedModel, UniterModel
from .utils_relation import layer_init, get_box_info, get_box_pair_info
from .utils_uniter import get_gather_index, pad_tensors
from .utils_motifs import obj_edge_vectors


@registry.ROI_RELATION_PREDICTOR.register("UniterPredictor")
class UniterPredictor(nn.Module):
    def __init__(self, config, in_channels, uniter_hidden_size=768):
        super(UniterPredictor, self).__init__()
        # common init
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        assert in_channels is not None
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # whether use caption triplet as supervision during training
        self.use_cap_trip = config.WSVL.USE_CAP_TRIP

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        if self.use_cap_trip:
            self.num_obj_cls = len(obj_classes)
            self.num_att_cls = len(att_classes)
            self.num_rel_cls = len(rel_classes)
        else:
            assert self.num_obj_cls==len(obj_classes)
            assert self.num_att_cls==len(att_classes)
            assert self.num_rel_cls==len(rel_classes)

        # Transformer encoder
        self.uniter = UniterModel.from_pretrained(img_dim=config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM,\
                                                offline_od=config.WSVL.OFFLINE_OD)

        # classifier
        len_obj_classes = len(obj_classes)
        len_rel_classes = len(rel_classes)
        encoder_output_dim = uniter_hidden_size
        self.out_oobj = nn.Sequential(*[
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(encoder_output_dim, len_obj_classes)
        ])
        self.out_rel = nn.Sequential(*[
            nn.Linear(encoder_output_dim, encoder_output_dim),
            nn.ReLU(inplace=True), 
            nn.Linear(encoder_output_dim, len_rel_classes)
        ])
        fc_layers = [self.out_oobj[0], self.out_oobj[2], self.out_rel[0], self.out_rel[2]]
        for fc_l in fc_layers:
            layer_init(fc_l, xavier=True)   
        
        # detection tag embedding 
        self.embed_dim = config.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        with torch.no_grad():
            if config.WSVL.OFFLINE_OD: # offline detector
                category_file = "datasets/vg/oid_v4_vocab.txt"
            else: # online detector
                category_file = "datasets/vg/vg_vocab.txt"
            with open(category_file, 'r') as f:
                detobj_classes = []
                for item in f:
                    detobj_classes.append(str(item.split(",")[0].strip()))
            offline_obj_embed_vecs = obj_edge_vectors(detobj_classes, wv_dir=config.GLOVE_DIR, wv_dim=self.embed_dim, offline_od=True)
            self.obj_embed = nn.Embedding(offline_obj_embed_vecs.shape[0], self.embed_dim)
            self.obj_embed.weight.copy_(offline_obj_embed_vecs, non_blocking=True)
        self.obj_embed_proj = nn.Linear(self.embed_dim, encoder_output_dim)#len(obj_classes))
        self.rel_subj_proj = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.rel_oobj_proj = nn.Linear(encoder_output_dim, encoder_output_dim)
        layer_init(self.obj_embed_proj, xavier=True)
        layer_init(self.rel_subj_proj, xavier=True)
        layer_init(self.rel_oobj_proj, xavier=True)

        self.glove_emb_layer = True # if true, the BERT layer only include a small number of tokens with glove as initialization
        self.mask_id = self.obj_embed.weight.size(0) + 2
        self.sep_id = self.obj_embed.weight.size(0) + 1
        self.cls_id = self.obj_embed.weight.size(0)
        self.subj_text_proj = nn.Linear(encoder_output_dim, encoder_output_dim)
        self.oobj_text_proj = nn.Linear(encoder_output_dim, encoder_output_dim)
        layer_init(self.subj_text_proj, xavier=True)
        layer_init(self.oobj_text_proj, xavier=True)
        
        # frequency bias
        if self.use_bias and not self.use_cap_trip:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

        # union features for relation prediction
        if self.use_vision:  
            self.union_proj = nn.Linear(config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM, encoder_output_dim)
            layer_init(self.union_proj, xavier=True)
        
        # PredCls: instead of using predicted object labels, use the given GT labels
        if config.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            self.use_gt_obj_label = True
        else:
            self.use_gt_obj_label = False

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, \
                det_dists=None, det_tag_ids=None, det_norm_pos=None):
        """
        Inputs:
            proposals (list[BoxList]): proposal boxes. Note: it has been post-processed (regression, nms) in sgdet mode.
            rel_pair_idxs (list[Tensor]): object pair indices. Each Tensor shaped as [#object pairs in an image, 2].
            rel_labels (list[Tensor]): predicate labels for sampled object pairs. Each Tensor shaped as [#object pairs in an image].
            rel_binarys (list[Tensor]): binary labels for sampled object pairs. Each Tensor shaped as [#bbox in an image, #bbox in an image].
            roi_features (Tensor): region visual features. Shaped as [#bbox in current batch, feature dimension] 
            det_norm_pos (list[Tensor]): region position features. Each Tensor shaped as [#bbox in an image, 7]
            det_tag_ids (list[list[Tensor]]): region tag id in BERT embeddings. Each Tensor has variable length.
            det_dists (list[Tensor]): region detection distribution. Each Tensor shaped as [#bbox in an image, #detection categories] 
            union_features: None
        Returns:
            subj_dists/oobj_dists (list[Tensor]): logits of subject/object label distribution
            rel_dists (list[Tensor]): logits of relation label distribution
            add_loss: emtpy dict to keep the same interface
        """
        # prepare detection results
        if det_dists is None and det_tag_ids is None and det_norm_pos is None:  # online detector
            det_norm_pos, det_dists, det_tag_ids = self.prepare_online_od_feat(proposals)
            det_tag_ids = rel_pair_idxs
        det_norm_pos = torch.cat(det_norm_pos)
        
        # predict triplets
        subj_dists, oobj_dists, rel_dists = self.predict_triplet(proposals, rel_pair_idxs, roi_features, \
            det_tag_ids, det_norm_pos, det_dists, union_features)
        add_loss = {}
        return subj_dists, oobj_dists, rel_dists, add_loss

    def predict_triplet(self, proposals, rel_pair_idxs, roi_features, det_tag_ids, det_norm_pos, det_dists, union_features):
        """
        Inputs include: all region features, tag tokens of a triplet (subject-MASK-object)
        Each feedforward only encodes and predict a single triplet.
        The final batch size is #all triplet
        """
        # prepare the inputs for encoder
        input_ids, position_ids, img_feat, img_pos_feat, attn_masks, gather_index, label_inds, det_subj_dists, det_oobj_dists, img_type_ids = \
            self.prepare_inputs(proposals, rel_pair_idxs, roi_features, det_tag_ids, det_norm_pos, det_dists)

        # encoder encoes subject, object and relation
        subj_dists, oobj_dists, rel_dists = \
            self.encode_predict(input_ids, position_ids, img_feat, img_pos_feat, attn_masks, gather_index, label_inds, det_subj_dists, \
                det_oobj_dists, img_type_ids, union_features)
        
        # PredCls: instead of using predicted object labels, use the given GT labels
        if self.use_gt_obj_label:
            subj_dists = det_subj_dists
            oobj_dists = det_oobj_dists    

        # refine the rel logits with frequency bias
        if self.use_bias:
            rel_dists = rel_dists + self.bias_module(proposals, rel_pair_idxs, subj_dists, oobj_dists)

        num_rels_per_img = [r.shape[0] for r in rel_pair_idxs]
        rel_dists = rel_dists.split(num_rels_per_img, dim=0)

        return subj_dists, oobj_dists, rel_dists

    def encode_predict(self, input_ids, position_ids, img_feat, img_pos_feat, attn_masks, gather_index, label_inds, \
        det_subj_dists, det_oobj_dists, img_type_ids, union_features):
        """
        Encode and predict the input features using the Transformer-based encoder
        """
        # uniter as encoder, output compact sequence
        sequence_output = self.uniter(input_ids, position_ids,
                                      img_feat, img_pos_feat,
                                      attn_masks, gather_index,
                                      output_all_encoded_layers=False,
                                      img_type_ids=img_type_ids)
        
        # index the features of subject-object-predicate
        batch_ind = torch.arange(sequence_output.size(0), dtype=torch.long).to(sequence_output.device)
        subj_rep = sequence_output[batch_ind, label_inds[:,0], :]
        oobj_rep = sequence_output[batch_ind, label_inds[:,1], :]
        rel_rep = sequence_output[batch_ind, label_inds[:,2], :]
        subj_t_rep = sequence_output[batch_ind, label_inds[:,2]-1, :]
        oobj_t_rep = sequence_output[batch_ind, label_inds[:,2]+1, :]

        # classify the features, late fusion with detection tag embedding (soft distribution)
        subj_emb = det_subj_dists @ self.obj_embed.weight
        oobj_emb = det_oobj_dists @ self.obj_embed.weight
        rel_rep = rel_rep + self.subj_text_proj(subj_t_rep) + self.oobj_text_proj(oobj_t_rep) \
                          + self.rel_subj_proj(subj_rep) + self.rel_oobj_proj(oobj_rep)
        subj_rep = subj_rep + self.obj_embed_proj(subj_emb)
        oobj_rep = oobj_rep + self.obj_embed_proj(oobj_emb)
        subj_dists = self.out_oobj(subj_rep)
        oobj_dists = self.out_oobj(oobj_rep)

        if self.use_vision:
            rel_dists = self.out_rel(rel_rep + self.union_proj(union_features))
        else:
            rel_dists = self.out_rel(rel_rep)

        return subj_dists, oobj_dists, rel_dists

    def prepare_inputs(self, proposals, rel_pair_idxs, roi_features, det_tag_ids, det_norm_pos, det_dists):
        """
        Prepare textual and visual inputs for Transformer-based encoder (refer to mlm_collate in Uniter)
        """
        device = rel_pair_idxs[0].device
        # 1. text token preparation: 
        # construct the tag sequece as [CLS, sub_wds, MASK, obj_wds, SEP, 0, ..., 0], based on rel_pair_idxs
        mask_token = torch.tensor([self.mask_id], dtype=torch.long).to(device)
        sep_token = torch.tensor([self.sep_id], dtype=torch.long).to(device)
        
        input_ids = []  # the tag sequece as [CLS, sub_wds, MASK, obj_wds, SEP, 0, ..., 0]
        label_inds = [] # the index in the compact V+L sequence, these tokens will be attached losses
        attn_masks = [] # only (#text + #bbox) are valid for each image
        txt_lens = []  # len([CLS, ..., SEP]), #text sequence
        num_bbs = []  # #bbox for each image
        
        # concate CLS-subj-predicate-obj-SEP together
        for img_i, (rel_pair, det_tag) in enumerate(zip(rel_pair_idxs, det_tag_ids)):  # per image, batch-wise processing
            # text ids (without CLS token)
            num_pair = rel_pair.size(0)

            # Glove embedding
            det_dist = det_dists[img_i]
            det_glove_tag = torch.max(det_dist[:,1:], dim=1)[1]+1
            det_glove_tag = list(det_glove_tag.split([1 for split_i in range(det_glove_tag.size(0))], dim=0))
            det_tag = det_glove_tag
            
            sub_ids = [det_tag[rel_pair[pair_i, 0].item()] for pair_i in range(num_pair)]
            obj_ids = [det_tag[rel_pair[pair_i, 1].item()] for pair_i in range(num_pair)]
            input_id = [torch.cat([sub, mask_token, obj, sep_token]) for (sub,obj) in zip(sub_ids, obj_ids)]          
            input_ids.extend(input_id)

            # [subj visual token index, obj visual token index, predicate index] (counting CLS token in)
            label_inds.extend([[1+input_id[pair_i].size(0)+rel_pair[pair_i, 0].item(), 1+input_id[pair_i].size(0)+rel_pair[pair_i, 1].item(), \
                                1+sub_ids[pair_i].size(0)] for pair_i in range(num_pair)])

            # valid L+V token masks (counting CLS token in)
            num_bb = proposals[img_i].bbox.size(0)
            num_bbs.extend([num_bb for _ in range(num_pair)])
            attn_masks.extend([torch.ones(1+input_trip.size(0)+num_bb, dtype=torch.long) for input_trip in input_id])  
        
        # padding sequence with same length
        txt_lens = [input_id.size(0) + 1 for input_id in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)  # pad to same size with 0
        cls_tokens = torch.zeros((input_ids.size(0), 1), dtype=torch.long).fill_(self.cls_id).to(device)
        input_ids = torch.cat((cls_tokens, input_ids), dim=1) # prepend the CLS token to input_ids 
        input_ids = input_ids.to(device)
        position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long).unsqueeze(0).to(device) # position emb indices for text
        
        label_inds = torch.tensor(label_inds, dtype=torch.long).to(device)  # the indices will be used for feature indexing
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)  # valid L+V token masks
        attn_masks = attn_masks.to(device)
        
        bs, max_tl = input_ids.size()
        out_size = attn_masks.size(1)
        gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size) # index the embedings into compact form
        gather_index = gather_index.to(device)

        # 2. visual region token preparation:
        # split region feature into a batch-wise list
        num_objs_per_img = [len(b) for b in proposals]
        img_feat = roi_features.split(num_objs_per_img, dim=0)
        img_pos_feat = det_norm_pos.split(num_objs_per_img, dim=0)

        # repeat region features to align with #object_pair / tag triplets
        repeat_ind = [btch_i for btch_i in range(len(rel_pair_idxs)) for _ in range(rel_pair_idxs[btch_i].size(0)) ]
        img_feat = [img_feat[r_ind] for r_ind in repeat_ind]
        img_feat = pad_tensors(img_feat, num_bbs)
        img_feat = img_feat.to(device)
        img_pos_feat = [img_pos_feat[r_ind] for r_ind in repeat_ind]
        img_pos_feat = pad_tensors(img_pos_feat, num_bbs)
        img_pos_feat = img_pos_feat.to(device)

        # detection information
        det_subj_dists = torch.cat([det_dists[bs_i][rel_pair_idxs[bs_i][:,0]] for bs_i in range(len(proposals))], dim=0)
        det_oobj_dists = torch.cat([det_dists[bs_i][rel_pair_idxs[bs_i][:,1]] for bs_i in range(len(proposals))], dim=0)
        
        # img token type id (subject, object or other general regioins)
        img_type_ids = torch.ones_like(img_feat[:, :, 0].long())  # general regions
        batch_ind = torch.arange(img_feat.size(0), dtype=torch.long).to(device)
        concat_rel_idxs = torch.cat(rel_pair_idxs, dim=0)
        img_type_ids[batch_ind, concat_rel_idxs[:,0]] = 2  # subject token
        img_type_ids[batch_ind, concat_rel_idxs[:,1]] = 3  # object token

        return input_ids, position_ids, img_feat, img_pos_feat, attn_masks, gather_index, label_inds, \
            det_subj_dists, det_oobj_dists, img_type_ids

    def bias_module(self, proposals, rel_pair_idxs, subj_dists, oobj_dists):
        """
        Given the logits of object pairs, output the rel logits based on frequency bias
        """
        subj_pred = subj_dists[:, 1:].max(1)[1] + 1
        oobj_pred = oobj_dists[:, 1:].max(1)[1] + 1
        # during training, use the gt label to index frequncy bias       
        if self.training:
            subj_labels = torch.cat([proposals[bs_i].get_field("labels")[rel_pair_idxs[bs_i][:,0]] for bs_i in range(len(proposals))], dim=0)
            oobj_labels = torch.cat([proposals[bs_i].get_field("labels")[rel_pair_idxs[bs_i][:,1]] for bs_i in range(len(proposals))], dim=0)
            s_label_ind = torch.nonzero(subj_labels != 0)
            o_label_ind = torch.nonzero(oobj_labels != 0)
            if s_label_ind.dim() > 0:
                s_label_ind = s_label_ind.squeeze(1)
                subj_pred[s_label_ind] = subj_labels[s_label_ind]
            if o_label_ind.dim() > 0:
                o_label_ind = o_label_ind.squeeze(1)
                oobj_pred[o_label_ind] = oobj_labels[o_label_ind]   
         
        # during testing, directly use the prediction for frequency bias indexing
        pair_pred = torch.cat((subj_pred.view(-1,1), oobj_pred.view(-1,1)), dim=1)
        rel_dists_bias = self.freq_bias.index_with_labels(pair_pred.long())
        
        return rel_dists_bias

    def prepare_online_od_feat(self, proposals):
        """Given a list of BoxList, return the normalized box features
        """
        dists_list = [F.softmax(prop.extra_fields['predict_logits'], -1) for prop in proposals]
        tag_list = None  # since we use glove embedding, then tag list won't be used in prepare_inputs() function

        device = proposals[0].bbox.device
        data_type = proposals[0].bbox.dtype
        pos_list = []
        for i, box_list in enumerate(proposals):
            det_box = box_list.bbox
            # normalize the box coords, [x1, y1, x2, y2, w, h, w ∗ h] normalized by image_w & image_h
            det_norm_pos = torch.zeros((det_box.size(0), 7)).type(data_type).to(device) 
            det_norm_pos[:, [0,2]] = det_box[:, [0,2]] / box_list.size[0]  # x, width
            det_norm_pos[:, [1,3]] = det_box[:, [1,3]] / box_list.size[1]  # y, height
            det_norm_pos[:, 4] = det_norm_pos[:, 2] - det_norm_pos[:, 0]  # box width
            det_norm_pos[:, 5] = det_norm_pos[:, 3] - det_norm_pos[:, 1]  # box height
            det_norm_pos[:, 6] = det_norm_pos[:, 4] * det_norm_pos[:, 5]  # box area
            pos_list.append(det_norm_pos)
        return pos_list, dists_list, tag_list


@registry.ROI_RELATION_PREDICTOR.register("TransformerPredictor")
class TransformerPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(TransformerPredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        # load parameters
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # module construct
        self.context_layer = TransformerContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.hidden_dim * 2, self.num_rel_cls)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.rel_compress, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, \
                det_dists=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx = self.context_layer(roi_features, proposals, logger, det_dists=det_dists)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        # from object level feature to pairwise relation level feature
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append(torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1))
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1))
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        ctx_gate = self.post_cat(prod_rep)

        # use union box and mask convolution
        if self.use_vision:
            if self.union_single_not_match:
                visual_rep = ctx_gate * self.up_dim(union_features)
            else:
                visual_rep = ctx_gate * union_features
        else:
            visual_rep = ctx_gate
        rel_dists = self.rel_compress(visual_rep) + self.ctx_compress(prod_rep)
                
        # use frequence bias
        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred)

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("IMPPredictor")
class IMPPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(IMPPredictor, self).__init__()
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.use_bias = False

        assert in_channels is not None

        self.context_layer = IMPContext(config, self.num_obj_cls, self.num_rel_cls, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        # freq 
        if self.use_bias:
            statistics = get_dataset_statistics(config)
            self.freq_bias = FrequencyBias(config, statistics)


    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        # encode context infomation
        obj_dists, rel_dists = self.context_layer(roi_features, proposals, union_features, rel_pair_idxs, logger)

        num_objs = [len(b) for b in proposals]
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        assert len(num_rels) == len(num_objs)

        if self.use_bias:
            obj_preds = obj_dists.max(-1)[1]
            obj_preds = obj_preds.split(num_objs, dim=0)

            pair_preds = []
            for pair_idx, obj_pred in zip(rel_pair_idxs, obj_preds):
                pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_pred = cat(pair_preds, dim=0)

            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("MotifPredictor")
class MotifPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(MotifPredictor, self).__init__()
        # whether use caption triplet as supervision during training
        self.use_cap_trip = config.WSVL.USE_CAP_TRIP

        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels
        self.use_vision = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION
        self.use_bias = config.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        if self.use_cap_trip:
            self.num_obj_cls = len(obj_classes)
            self.num_att_cls = len(att_classes)
            self.num_rel_cls = len(rel_classes)
        else:
            assert self.num_obj_cls==len(obj_classes)
            assert self.num_att_cls==len(att_classes)
            assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if self.attribute_on:
            self.context_layer = AttributeLSTMContext(config, obj_classes, att_classes, rel_classes, in_channels)
        else:
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)
        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        layer_init(self.rel_compress, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        if self.use_bias:
            # convey statistics into FrequencyBias to avoid loading again
            self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None, \
                det_dists=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        if self.attribute_on:
            obj_dists, obj_preds, att_dists, edge_ctx = self.context_layer(roi_features, proposals, logger)
        else:
            obj_dists, obj_preds, edge_ctx, _ = self.context_layer(roi_features, proposals, logger, det_dists=det_dists)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        if self.use_vision:
            if self.union_single_not_match:
                prod_rep = prod_rep * self.up_dim(union_features)
            else:
                prod_rep = prod_rep * union_features

        rel_dists = self.rel_compress(prod_rep)

        if self.use_bias:
            rel_dists = rel_dists + self.freq_bias.index_with_labels(pair_pred.long())

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.attribute_on:
            att_dists = att_dists.split(num_objs, dim=0)
            return (obj_dists, att_dists), rel_dists, add_losses
        else:
            return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("VCTreePredictor")
class VCTreePredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(VCTreePredictor, self).__init__()
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_att_cls = config.MODEL.ROI_ATTRIBUTE_HEAD.NUM_ATTRIBUTES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes, att_classes = statistics['obj_classes'], statistics['rel_classes'], statistics['att_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_att_cls==len(att_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.post_cat = nn.Linear(self.hidden_dim * 2, self.pooling_dim)

        # learned-mixin
        #self.uni_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.frq_gate = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #self.uni_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        #layer_init(self.uni_gate, xavier=True)
        #layer_init(self.frq_gate, xavier=True)
        layer_init(self.ctx_compress, xavier=True)
        #layer_init(self.uni_compress, xavier=True)

        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        layer_init(self.post_cat, xavier=True)
        
        if self.pooling_dim != config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM:
            self.union_single_not_match = True
            self.up_dim = nn.Linear(config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.pooling_dim)
            layer_init(self.up_dim, xavier=True)
        else:
            self.union_single_not_match = False

        self.freq_bias = FrequencyBias(config, statistics)

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """

        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger)

        # post decode
        edge_rep = F.relu(self.post_emb(edge_ctx))
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.hidden_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.hidden_dim)

        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        assert len(num_rels) == len(num_objs)

        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        
        prod_reps = []
        pair_preds = []
        for pair_idx, head_rep, tail_rep, obj_pred in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds):
            prod_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
        prod_rep = cat(prod_reps, dim=0)
        pair_pred = cat(pair_preds, dim=0)

        prod_rep = self.post_cat(prod_rep)

        # learned-mixin Gate
        #uni_gate = torch.tanh(self.uni_gate(self.drop(prod_rep)))
        #frq_gate = torch.tanh(self.frq_gate(self.drop(prod_rep)))

        if self.union_single_not_match:
            union_features = self.up_dim(union_features)

        ctx_dists = self.ctx_compress(prod_rep * union_features)
        #uni_dists = self.uni_compress(self.drop(union_features))
        frq_dists = self.freq_bias.index_with_labels(pair_pred.long())

        rel_dists = ctx_dists + frq_dists
        #rel_dists = ctx_dists + uni_gate * uni_dists + frq_gate * frq_dists

        obj_dists = obj_dists.split(num_objs, dim=0)
        rel_dists = rel_dists.split(num_rels, dim=0)

        # we use obj_preds instead of pred from obj_dists
        # because in decoder_rnn, preds has been through a nms stage
        add_losses = {}

        if self.training:
            binary_loss = []
            for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                bi_gt = (bi_gt > 0).float()
                binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
            add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

        return obj_dists, rel_dists, add_losses


@registry.ROI_RELATION_PREDICTOR.register("CausalAnalysisPredictor")
class CausalAnalysisPredictor(nn.Module):
    def __init__(self, config, in_channels):
        super(CausalAnalysisPredictor, self).__init__()
        self.cfg = config
        self.attribute_on = config.MODEL.ATTRIBUTE_ON
        self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
        self.num_obj_cls = config.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.num_rel_cls = config.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.fusion_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.FUSION_TYPE
        self.separate_spatial = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        self.use_vtranse = config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse"
        self.effect_type = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_TYPE
        
        assert in_channels is not None
        num_inputs = in_channels

        # load class dict
        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        assert self.num_obj_cls==len(obj_classes)
        assert self.num_rel_cls==len(rel_classes)
        # init contextual lstm encoding
        if config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "motifs":
            self.context_layer = LSTMContext(config, obj_classes, rel_classes, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vctree":
            self.context_layer = VCTreeLSTMContext(config, obj_classes, rel_classes, statistics, in_channels)
        elif config.MODEL.ROI_RELATION_HEAD.CAUSAL.CONTEXT_LAYER == "vtranse":
            self.context_layer = VTransEFeature(config, obj_classes, rel_classes, in_channels)
        else:
            print('ERROR: Invalid Context Layer')

        # post decoding
        self.hidden_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        
        if self.use_vtranse:
            self.edge_dim = self.pooling_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.pooling_dim * 2)
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls, bias=False)
        else:
            self.edge_dim = self.hidden_dim
            self.post_emb = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
            self.post_cat = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.pooling_dim),
                                            nn.ReLU(inplace=True),])
            self.ctx_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)
        self.vis_compress = nn.Linear(self.pooling_dim, self.num_rel_cls)

        if self.fusion_type == 'gate':
            self.ctx_gate_fc = nn.Linear(self.pooling_dim, self.num_rel_cls)
            layer_init(self.ctx_gate_fc, xavier=True)
        
        # initialize layer parameters 
        layer_init(self.post_emb, 10.0 * (1.0 / self.hidden_dim) ** 0.5, normal=True)
        if not self.use_vtranse:
            layer_init(self.post_cat[0], xavier=True)
            layer_init(self.ctx_compress, xavier=True)
        layer_init(self.vis_compress, xavier=True)
        
        assert self.pooling_dim == config.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM

        # convey statistics into FrequencyBias to avoid loading again
        self.freq_bias = FrequencyBias(config, statistics)

        # add spatial emb for visual feature
        if self.spatial_for_vision:
            self.spt_emb = nn.Sequential(*[nn.Linear(32, self.hidden_dim), 
                                            nn.ReLU(inplace=True),
                                            nn.Linear(self.hidden_dim, self.pooling_dim),
                                            nn.ReLU(inplace=True)
                                        ])
            layer_init(self.spt_emb[0], xavier=True)
            layer_init(self.spt_emb[2], xavier=True)

        self.label_smooth_loss = Label_Smoothing_Regression(e=1.0)

        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        self.average_ratio = 0.0005

        self.register_buffer("untreated_spt", torch.zeros(32))
        self.register_buffer("untreated_conv_spt", torch.zeros(self.pooling_dim))
        self.register_buffer("avg_post_ctx", torch.zeros(self.pooling_dim))
        self.register_buffer("untreated_feat", torch.zeros(self.pooling_dim))

        
    def pair_feature_generate(self, roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=False):
        # encode context infomation
        obj_dists, obj_preds, edge_ctx, binary_preds = self.context_layer(roi_features, proposals, rel_pair_idxs, logger, ctx_average=ctx_average)
        obj_dist_prob = F.softmax(obj_dists, dim=-1)

        # post decode
        edge_rep = self.post_emb(edge_ctx)
        edge_rep = edge_rep.view(edge_rep.size(0), 2, self.edge_dim)
        head_rep = edge_rep[:, 0].contiguous().view(-1, self.edge_dim)
        tail_rep = edge_rep[:, 1].contiguous().view(-1, self.edge_dim)
        # split
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        obj_preds = obj_preds.split(num_objs, dim=0)
        obj_prob_list = obj_dist_prob.split(num_objs, dim=0)
        obj_dist_list = obj_dists.split(num_objs, dim=0)
        ctx_reps = []
        pair_preds = []
        pair_obj_probs = []
        pair_bboxs_info = []
        for pair_idx, head_rep, tail_rep, obj_pred, obj_box, obj_prob in zip(rel_pair_idxs, head_reps, tail_reps, obj_preds, obj_boxs, obj_prob_list):
            if self.use_vtranse:
                ctx_reps.append( head_rep[pair_idx[:,0]] - tail_rep[pair_idx[:,1]] )
            else:
                ctx_reps.append( torch.cat((head_rep[pair_idx[:,0]], tail_rep[pair_idx[:,1]]), dim=-1) )
            pair_preds.append( torch.stack((obj_pred[pair_idx[:,0]], obj_pred[pair_idx[:,1]]), dim=1) )
            pair_obj_probs.append( torch.stack((obj_prob[pair_idx[:,0]], obj_prob[pair_idx[:,1]]), dim=2) )
            pair_bboxs_info.append( get_box_pair_info(obj_box[pair_idx[:,0]], obj_box[pair_idx[:,1]]) )
        pair_obj_probs = cat(pair_obj_probs, dim=0)
        pair_bbox = cat(pair_bboxs_info, dim=0)
        pair_pred = cat(pair_preds, dim=0)
        ctx_rep = cat(ctx_reps, dim=0)
        if self.use_vtranse:
            post_ctx_rep = ctx_rep
        else:
            post_ctx_rep = self.post_cat(ctx_rep)

        return post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list
        
        

    def forward(self, proposals, rel_pair_idxs, rel_labels, rel_binarys, roi_features, union_features, logger=None):
        """
        Returns:
            obj_dists (list[Tensor]): logits of object label distribution
            rel_dists (list[Tensor])
            rel_pair_idxs (list[Tensor]): (num_rel, 2) index of subject and object
            union_features (Tensor): (batch_num_rel, context_pooling_dim): visual union feature of each pair
        """
        num_rels = [r.shape[0] for r in rel_pair_idxs]
        num_objs = [len(b) for b in proposals]
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in proposals]

        assert len(num_rels) == len(num_objs)

        post_ctx_rep, pair_pred, pair_bbox, pair_obj_probs, binary_preds, obj_dist_prob, edge_rep, obj_dist_list = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger)

        if (not self.training) and self.effect_analysis:
            with torch.no_grad():
                avg_post_ctx_rep, _, _, avg_pair_obj_prob, _, _, _, _ = self.pair_feature_generate(roi_features, proposals, rel_pair_idxs, num_objs, obj_boxs, logger, ctx_average=True)

        if self.separate_spatial:
            union_features, spatial_conv_feats = union_features
            post_ctx_rep = post_ctx_rep * spatial_conv_feats
        
        if self.spatial_for_vision:
            post_ctx_rep = post_ctx_rep * self.spt_emb(pair_bbox)

        rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_pred, use_label_dist=False)
        rel_dist_list = rel_dists.split(num_rels, dim=0)

        add_losses = {}
        # additional loss
        if self.training:
            rel_labels = cat(rel_labels, dim=0)

            # binary loss for VCTree
            if binary_preds is not None:
                binary_loss = []
                for bi_gt, bi_pred in zip(rel_binarys, binary_preds):
                    bi_gt = (bi_gt > 0).float()
                    binary_loss.append(F.binary_cross_entropy_with_logits(bi_pred, bi_gt))
                add_losses["binary_loss"] = sum(binary_loss) / len(binary_loss)

            # branch constraint: make sure each branch can predict independently
            add_losses['auxiliary_ctx'] = F.cross_entropy(self.ctx_compress(post_ctx_rep), rel_labels)
            if not (self.fusion_type == 'gate'):
                add_losses['auxiliary_vis'] = F.cross_entropy(self.vis_compress(union_features), rel_labels)
                add_losses['auxiliary_frq'] = F.cross_entropy(self.freq_bias.index_with_labels(pair_pred.long()), rel_labels)

            # untreated average feature
            if self.spatial_for_vision:
                self.untreated_spt = self.moving_average(self.untreated_spt, pair_bbox)
            if self.separate_spatial:
                self.untreated_conv_spt = self.moving_average(self.untreated_conv_spt, spatial_conv_feats)
            self.avg_post_ctx = self.moving_average(self.avg_post_ctx, post_ctx_rep)
            self.untreated_feat = self.moving_average(self.untreated_feat, union_features)

        elif self.effect_analysis:
            with torch.no_grad():
                # untreated spatial
                if self.spatial_for_vision:
                    avg_spt_rep = self.spt_emb(self.untreated_spt.clone().detach().view(1, -1))
                # untreated context
                avg_ctx_rep = avg_post_ctx_rep * avg_spt_rep if self.spatial_for_vision else avg_post_ctx_rep  
                avg_ctx_rep = avg_ctx_rep * self.untreated_conv_spt.clone().detach().view(1, -1) if self.separate_spatial else avg_ctx_rep
                # untreated visual
                avg_vis_rep = self.untreated_feat.clone().detach().view(1, -1)
                # untreated category dist
                avg_frq_rep = avg_pair_obj_prob

            if self.effect_type == 'TDE':   # TDE of CTX
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs)
            elif self.effect_type == 'NIE': # NIE of FRQ
                rel_dists = self.calculate_logits(union_features, avg_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            elif self.effect_type == 'TE':  # Total Effect
                rel_dists = self.calculate_logits(union_features, post_ctx_rep, pair_obj_probs) - self.calculate_logits(union_features, avg_ctx_rep, avg_frq_rep)
            else:
                assert self.effect_type == 'none'
                pass
            rel_dist_list = rel_dists.split(num_rels, dim=0)

        return obj_dist_list, rel_dist_list, add_losses

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def calculate_logits(self, vis_rep, ctx_rep, frq_rep, use_label_dist=True, mean_ctx=False):
        if use_label_dist:
            frq_dists = self.freq_bias.index_with_probability(frq_rep)
        else:
            frq_dists = self.freq_bias.index_with_labels(frq_rep.long())

        if mean_ctx:
            ctx_rep = ctx_rep.mean(-1).unsqueeze(-1)
        vis_dists = self.vis_compress(vis_rep)
        ctx_dists = self.ctx_compress(ctx_rep)

        if self.fusion_type == 'gate':
            ctx_gate_dists = self.ctx_gate_fc(ctx_rep)
            union_dists = ctx_dists * torch.sigmoid(vis_dists + frq_dists + ctx_gate_dists)
            #union_dists = (ctx_dists.exp() * torch.sigmoid(vis_dists + frq_dists + ctx_constraint) + 1e-9).log()    # improve on zero-shot, but low mean recall and TDE recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists * frq_dists)                                          # best conventional Recall results
            #union_dists = (ctx_dists.exp() + vis_dists.exp() + frq_dists.exp() + 1e-9).log()                        # good zero-shot Recall
            #union_dists = ctx_dists * torch.max(torch.sigmoid(vis_dists), torch.sigmoid(frq_dists))                 # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid(vis_dists) * torch.sigmoid(frq_dists)                           # balanced recall and mean recall
            #union_dists = ctx_dists * (torch.sigmoid(vis_dists) + torch.sigmoid(frq_dists)) / 2.0                   # good zero-shot Recall
            #union_dists = ctx_dists * torch.sigmoid((vis_dists.exp() + frq_dists.exp() + 1e-9).log())               # good zero-shot Recall, bad for all of the rest
            
        elif self.fusion_type == 'sum':
            union_dists = vis_dists + ctx_dists + frq_dists
        else:
            print('invalid fusion type')

        return union_dists

    def binary_ce_loss(self, logits, gt):
        batch_size, num_cat = logits.shape
        answer = torch.zeros((batch_size, num_cat), device=gt.device).float()
        answer[torch.arange(batch_size, device=gt.device), gt.long()] = 1.0
        return F.binary_cross_entropy_with_logits(logits, answer) * num_cat

    def fusion(self, x, y):
        return F.relu(x + y) - (x - y) ** 2


def make_roi_relation_predictor(cfg, in_channels):
    func = registry.ROI_RELATION_PREDICTOR[cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR]
    return func(cfg, in_channels)
