import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import copy

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

BOX_SCALE = 1024  # Scale at which we have the boxes

class VGDataset(torch.utils.data.Dataset):

    def __init__(self, split, img_dir, roidb_file, dict_file, image_file, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='',
                offline_OD=False, use_uniter=False, offline_OD_type=None,
                use_cap_trip=False, caption_label_file=None,):
        """
        Torch dataset for VisualGenome
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        assert split in {'train', 'val', 'test'}
        self.flip_aug = flip_aug
        self.split = split  # split in {'train', 'val', 'test'}
        self.img_dir = img_dir  # 'datasets/vg/VG_100K'
        self.dict_file = dict_file  # 'datasets/vg/VG-SGG-dicts-with-attri.json'
        self.roidb_file = roidb_file  # 'datasets/vg/VG-SGG-with-attri.h5'
        self.image_file = image_file  # 'datasets/vg/image_data.json'
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms # not None
        self.use_cap_trip = use_cap_trip  # whether use caption triplets as supervision for training model

        if self.split == 'train' and self.use_cap_trip:  # only use caption triplets during training
            self.cap_labels = np.load(caption_label_file, allow_pickle=True).tolist()
            self.ind_to_classes = self.cap_labels['ind_to_classes']
            self.ind_to_predicates = self.cap_labels['ind_to_predicates']
            self.cap_img_info = self.cap_labels['img_info']  # Dict
            self.cap_imgid_list = self.cap_labels['img_id_list']
            self.gt_classes = self.cap_labels['gt_classes']
            self.relationships = self.cap_labels['relationships']
            self.img_info = [self.cap_img_info[img_id] for img_id in self.cap_imgid_list]
            self.filenames = [img_dict['file_name'] for img_dict in self.img_info]
            self.split_mask = np.zeros((len(self.cap_imgid_list))) == 0
        else:
            self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(dict_file) # contiguous 151, 51 containing __background__
        self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        if self.custom_eval:
            self.get_custom_imgs(custom_path)
        else:
            if self.split == 'train' and self.use_cap_trip:  # only use caption triplets during training 
                self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
                self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
                self.gt_classes = [self.gt_classes[i] for i in np.where(self.split_mask)[0]]
                self.relationships = [self.relationships[i] for i in np.where(self.split_mask)[0]]
            else:                
                self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
                    self.roidb_file, self.split, num_im, num_val_im=num_val_im,
                    filter_empty_rels=filter_empty_rels,
                    filter_non_overlap=self.filter_non_overlap,
                )

                self.filenames, self.img_info = load_image_filenames(img_dir, image_file) # length equals to split_mask
                self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
                self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]
                if len(self.filenames) < 500: # during testing, save the file names in order, for visualization
                    np.save(os.path.join("visualization","custom_filenames.npy"), self.filenames)
        
        # detection results from offline detector
        self.offline_OD = offline_OD 
        offline_OD_type = offline_OD_type if self.offline_OD else "online"
        if offline_OD_type == "OID": # OID detector
            if not self.use_cap_trip or self.split != 'train' or 'vg' in os.path.basename(caption_label_file):
                self.det_path = os.path.join(os.path.dirname(self.dict_file), 'VG_detection_results_oid')
            elif 'COCO' in os.path.basename(caption_label_file): 
                self.det_path = os.path.join(os.path.dirname(self.dict_file), 'COCO_detection_results_oid')
            elif 'cc' in os.path.basename(caption_label_file):
                self.det_path = os.path.join(os.path.dirname(self.dict_file), 'cc_detection_results_oid')
            elif 'mix' in os.path.basename(caption_label_file):
                self.det_path = {}
                self.det_path['COCO'] = os.path.join(os.path.dirname(self.dict_file), 'COCO_detection_results_oid')
                self.det_path['cc'] = os.path.join(os.path.dirname(self.dict_file), 'cc_detection_results_oid')
                self.dataset_name = self.cap_labels['dataset_name_list']
                self.dataset_name = [self.dataset_name[i] for i in np.where(self.split_mask)[0]]
        else:
            self.det_path = 'none'

        if type(self.det_path) is str:  # single dataset
            self.det_feat_path = os.path.join(self.det_path, '_att')
            self.det_dist_path = os.path.join(self.det_path, '_dists')
            if not (self.det_path == 'none'):
                self.det_box = np.load(os.path.join(self.det_path, 'box_dict.npy'), allow_pickle=True, encoding='latin1').tolist()
        else:  # multiple datasets merged
            self.det_feat_path = {}
            self.det_dist_path = {}
            self.det_box = {}
            for d_n in self.det_path:
                self.det_feat_path[d_n] = os.path.join(self.det_path[d_n], '_att')
                self.det_dist_path[d_n] = os.path.join(self.det_path[d_n], '_dists')
                if not (self.det_path == 'none'):
                    self.det_box[d_n] = np.load(os.path.join(self.det_path[d_n], 'box_dict.npy'), allow_pickle=True, encoding='latin1').tolist()
        
        # load detection label id when using UNITER as relation predictor
        self.use_uniter = use_uniter
        if self.use_uniter:
            if offline_OD_type == "OID":
                det_id_file = 'oid_v4_det_ids.npy'
            else:
                det_id_file = 'none'
            if det_id_file != 'none':
                self.det_tag_ids = np.load(os.path.join(os.path.dirname(self.dict_file), det_id_file), allow_pickle=True, encoding='latin1').tolist()


    def __getitem__(self, index):
        if not self.offline_OD: # use online object detector 
            if self.custom_eval:
                img = Image.open(self.custom_files[index]).convert("RGB")
                target = torch.LongTensor([-1])
                if self.transforms is not None:
                    img, target = self.transforms(img, target)
                return img, target, index
            
            img = Image.open(self.filenames[index]).convert("RGB")
            if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
                print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']), ' ', '='*20)

            flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
            
            target = self.get_groundtruth(index, flip_img)

            if flip_img:
                img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

            if self.transforms is not None:
                img, target = self.transforms(img, target) # img: from PIL.Image to tensor

            return img, target, index # tensor, BoxList, int
        else:  # use offline object detector results
            if not self.use_uniter: # use default predictor
                if self.split == 'train' and self.use_cap_trip:  # only use caption triplets during training
                    target = self.get_groundtruth_from_cap_triplets(index, flip_img=False)
                else:
                    target = self.get_groundtruth(index, flip_img=False)
                img = index
                
                img_id = self.filenames[index].split('/')[-1].split('.')[0]
                if type(self.det_path) is str:  # single dataset
                    det_feat = np.load(os.path.join(self.det_feat_path, img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_feat = torch.as_tensor(det_feat, dtype=torch.float32)
                    det_dist = np.load(os.path.join(self.det_dist_path, img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_dist = torch.as_tensor(det_dist, dtype=torch.float32)
                    det_box = self.det_box[img_id]
                    det_box = BoxList(det_box, target.size, 'xyxy') # xyxy
                    det_box.add_field('det_dist', det_dist)
                else:  # multiple datasets merged
                    d_n = self.dataset_name[index]
                    det_feat = np.load(os.path.join(self.det_feat_path[d_n], img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_feat = torch.as_tensor(det_feat, dtype=torch.float32)
                    det_dist = np.load(os.path.join(self.det_dist_path[d_n], img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_dist = torch.as_tensor(det_dist, dtype=torch.float32)
                    det_box = self.det_box[d_n][img_id]
                    det_box = BoxList(det_box, target.size, 'xyxy') # xyxy
                    det_box.add_field('det_dist', det_dist)

                return det_feat, det_dist, det_box, target, index, img
            else: # use UNITER as relation predictor
                if self.split == 'train' and self.use_cap_trip:  # only use caption triplets during training
                    target = self.get_groundtruth_from_cap_triplets(index, flip_img=False)
                else:
                    target = self.get_groundtruth(index, flip_img=False)
                img = index
                
                # prepare the inputs for uniter
                img_id = self.filenames[index].split('/')[-1].split('.')[0]

                if type(self.det_path) is str:  # single dataset
                    # region feature
                    det_feat = np.load(os.path.join(self.det_feat_path, img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_feat = torch.as_tensor(det_feat, dtype=torch.float32)
                    # region distribution and tag ids in BERT vocabulary
                    det_dist = np.load(os.path.join(self.det_dist_path, img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_pred = np.argmax(det_dist[:,1:], axis=1) + 1
                    det_tag_ids = [torch.tensor(self.det_tag_ids[pred]) for pred in det_pred]
                    det_dist = torch.as_tensor(det_dist, dtype=torch.float32)
                    # region box and normalized position feature
                    det_box = self.det_box[img_id]
                else:   # multiple datasets merged
                    d_n = self.dataset_name[index]
                    # region feature
                    det_feat = np.load(os.path.join(self.det_feat_path[d_n], img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_feat = torch.as_tensor(det_feat, dtype=torch.float32)
                    # region distribution and tag ids in BERT vocabulary
                    det_dist = np.load(os.path.join(self.det_dist_path[d_n], img_id+'.npz'), allow_pickle=True, encoding='latin1')['feat']
                    det_pred = np.argmax(det_dist[:,1:], axis=1) + 1
                    det_tag_ids = [torch.tensor(self.det_tag_ids[pred]) for pred in det_pred]
                    det_dist = torch.as_tensor(det_dist, dtype=torch.float32)
                    # region box and normalized position feature
                    det_box = self.det_box[d_n][img_id]

                det_norm_pos = np.zeros((det_box.shape[0], 7)) # [x1, y1, x2, y2, w, h, w ∗ h] normalized by image_w & image_h
                det_norm_pos[:, [0,2]] = det_box[:, [0,2]] / target.size[0]  # x, width
                det_norm_pos[:, [1,3]] = det_box[:, [1,3]] / target.size[1]  # y, height
                det_norm_pos[:, 4] = det_norm_pos[:, 2] - det_norm_pos[:, 0]  # box width
                det_norm_pos[:, 5] = det_norm_pos[:, 3] - det_norm_pos[:, 1]  # box height
                det_norm_pos[:, 6] = det_norm_pos[:, 4] * det_norm_pos[:, 5]  # box area
                det_norm_pos = torch.as_tensor(det_norm_pos, dtype=torch.float32)
                det_box = BoxList(det_box, target.size, 'xyxy') # xyxy
                det_box.add_field('det_dist', det_dist)

                return det_feat, det_dist, det_box, target, index, img, det_tag_ids, det_norm_pos

    def get_statistics(self):
        if self.use_cap_trip:  # use caption triplets during training
            result = {
                'fg_matrix': None,
                'pred_dist': None,
                'obj_classes': self.ind_to_classes,
                'rel_classes': self.ind_to_predicates,
                'att_classes': self.ind_to_classes, # use self.ind_to_classes to keep the slot
            }
            return result
        else: 
            fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.dict_file,
                                                    image_file=self.image_file, must_overlap=True)
            eps = 1e-3
            bg_matrix += 1
            fg_matrix[:, :, 0] = bg_matrix
            pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

            result = {
                'fg_matrix': torch.from_numpy(fg_matrix),
                'pred_dist': torch.from_numpy(pred_dist).float(),
                'obj_classes': self.ind_to_classes,
                'rel_classes': self.ind_to_predicates,
                'att_classes': self.ind_to_attributes,
            }
            return result

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        for file_name in os.listdir(path):
            self.custom_files.append(os.path.join(path, file_name))
            img = Image.open(os.path.join(path, file_name)).convert("RGB")
            self.img_info.append({'width':int(img.width), 'height':int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False, flip_img=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        if flip_img:
            new_xmin = w - box[:,2]
            new_xmax = w - box[:,0]
            box[:,0] = new_xmin
            box[:,2] = new_xmax
        target = BoxList(box, (w, h), 'xyxy') # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy() # np array, (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            target = target.clip_to_image(remove_empty=True)
            return target
    
    def get_groundtruth_from_cap_triplets(self, index, evaluation=False, flip_img=False):
        img_info = self.img_info[index]
        w, h = img_info['width'], img_info['height']
        # important: to keep the BoxList class consistent
        box = np.zeros((self.gt_classes[index].shape[0],2))
        box = torch.from_numpy(box).reshape(-1, 2)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy', pseudo_bbox=True) # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index].astype(int)))
        target.add_field("attributes", torch.from_numpy(self.gt_classes[index].astype(int)).reshape(-1,1))  # not used

        relation = self.relationships[index].copy().astype(int) # np array, (num_rel, 3)
        # filter out the duplicate predictes with same sub-obj concept pair, to give more chance to the less frequent predicates
        if self.filter_duplicate_rels: 
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)
        
        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            # for the same sub-obj concept pair (not region pair), if there are multiple possible relations,
            # we only keep one relation in this iteration
            if relation_map[int(relation[i,0]), int(relation[i,1])] > 0: 
                if (random.random() > 0.5):
                    relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])
            else:
                relation_map[int(relation[i,0]), int(relation[i,1])] = int(relation[i,2])

        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation: # during evaluation, we still use VG human annotation
            target = target.clip_to_image(remove_empty=False)  
            target.add_field("relation_tuple", torch.LongTensor(relation)) # for evaluation
            return target
        else:
            #target = target.clip_to_image(remove_empty=True)
            return target
    
    def record_stats(self):
        if self.split == "train":
            all_nouns = []
            all_predicates = []
            for img_i,img_rel in enumerate(self.relationships):
                this_cls = self.gt_classes[img_i]
                all_nouns.extend(list(this_cls[img_rel[:,0]]))
                all_nouns.extend(list(this_cls[img_rel[:,1]]))
                all_predicates.extend(list(img_rel[:,2]))
            nouns, n_cnt = np.unique(all_nouns, return_counts=True)
            nouns = [(nn,nn_c) for nn,nn_c in zip(nouns, n_cnt)] + [(0,0)]
            nouns = sorted(nouns, key=lambda x: x[1], reverse=True)
            predicates, p_cnt = np.unique(all_predicates, return_counts=True)
            predicates = [(pp,pp_c) for pp,pp_c in zip(predicates, p_cnt)] + [(0,0)]
            predicates = sorted(predicates, key=lambda x: x[1], reverse=True)
            with open("objects_vg_vocab_counts.txt",'w') as f:
                for i in range(len(self.ind_to_classes)):
                    f.write(str(self.ind_to_classes[nouns[i][0]])+','+str(nouns[i][1])+'\n')
            with open("predicates_vg_vocab_counts.txt",'w') as f:
                for i in range(len(self.ind_to_predicates)):
                    f.write(str(self.ind_to_predicates[predicates[i][0]])+','+str(predicates[i][1])+'\n')

    def __len__(self):
        if self.custom_eval:
            return len(self.custom_files)
        return len(self.filenames)


def get_VG_statistics(img_dir, roidb_file, dict_file, image_file, must_overlap=True):
    train_data = VGDataset(split='train', img_dir=img_dir, roidb_file=roidb_file, 
                        dict_file=dict_file, image_file=image_file, num_val_im=5000, 
                        filter_duplicate_rels=False)
    num_obj_classes = len(train_data.ind_to_classes)
    num_rel_classes = len(train_data.ind_to_predicates)
    fg_matrix = np.zeros((num_obj_classes, num_obj_classes, num_rel_classes), dtype=np.int64)
    bg_matrix = np.zeros((num_obj_classes, num_obj_classes), dtype=np.int64)

    for ex_ind in tqdm(range(len(train_data))):
        gt_classes = train_data.gt_classes[ex_ind].copy()
        gt_relations = train_data.relationships[ex_ind].copy()
        gt_boxes = train_data.gt_boxes[ex_ind].copy()

        # For the foreground, we'll just look at everything
        o1o2 = gt_classes[gt_relations[:, :2]]
        for (o1, o2), gtr in zip(o1o2, gt_relations[:,2]):
            fg_matrix[o1, o2, gtr] += 1
        # For the background, get all of the things that overlap.
        o1o2_total = gt_classes[np.array(
            box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
        for (o1, o2) in o1o2_total:
            bg_matrix[o1, o2] += 1

    return fg_matrix, bg_matrix
    

def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations. 
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes

def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    #print('boxes1: ', boxes1.shape)
    #print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:,:,:2], boxes2.reshape([1, num_box2, -1])[:,:,:2]) # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:,:,2:], boxes2.reshape([1, num_box2, -1])[:,:,2:]) # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter

def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:  
        json.dump(data, outfile)

def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return: 
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return: 
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground 
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of 
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag
    # Data efficiency experiment: only use subset of VG to train model
    # if split_flag == 0: 
    #     kept_ind = np.random.permutation(split_mask.nonzero()[0])[:32000]
    #     split_mask = np.zeros_like(data_split).astype(bool)
    #     split_mask[kept_ind] = True
    # Only test on a few images for visualization
    # split_mask[76000:] = False # split_mask[75700:] = False  # split_mask[75670:] = False  # 

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start : i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start : i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start : i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start : i_rel_end + 1]
            obj_idx = _relations[i_rel_start : i_rel_end + 1] - i_obj_start # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates)) # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
