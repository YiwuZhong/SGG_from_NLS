# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list
import torch

class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if len(transposed_batch) == 3: # use online object detector
            images = to_image_list(transposed_batch[0], self.size_divisible) # ImageList
            targets = transposed_batch[1]
            img_ids = transposed_batch[2]
            return images, targets, img_ids # ImageList, tuple of BoxList, tuple of ints
        else: # use offline object detector detection results
            if isinstance(transposed_batch[5][0], int): # pseudo image
                images = torch.zeros([1,1,1])
            else:
                images = to_image_list(transposed_batch[5], self.size_divisible)
            targets = transposed_batch[3] # tuple of BoxList 
            img_ids = transposed_batch[4] # tuple of ints 
            
            det_feats = transposed_batch[0] # tuple of tensors
            det_dists = transposed_batch[1] # tuple of tensors
            det_boxes = transposed_batch[2] # tuple of BoxList    
            
            if len(transposed_batch) > 6: # use uniter
                det_tag_ids = transposed_batch[6] # tuple of List, the elements have variable length
                det_norm_pos = transposed_batch[7]
                return det_feats, det_dists, det_boxes, targets, img_ids, images, det_tag_ids, det_norm_pos  
            else:
                return det_feats, det_dists, det_boxes, targets, img_ids, images     


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))

