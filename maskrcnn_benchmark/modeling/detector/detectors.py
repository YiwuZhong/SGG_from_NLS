# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import GeneralizedRCNN


_DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}


def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
    
    # if not cfg.WSVL.OFFLINE_OD:  # online object detector
    #     from .generalized_rcnn import GeneralizedRCNN
    #     _DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
    #     meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    #     return meta_arch(cfg)
    # else: # offline object detector detection results
    #     from .generalized_rcnn_offline_od import GeneralizedRCNN
    #     _DETECTION_META_ARCHITECTURES = {"GeneralizedRCNN": GeneralizedRCNN}
    #     meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    #     return meta_arch(cfg)

