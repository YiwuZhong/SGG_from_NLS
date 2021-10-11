export CUDA_VISIBLE_DEVICES=0,1

MODEL_TYPE=$1

##############################################################################################

if [ $MODEL_TYPE == "Language_CC-COCO_Uniter" ]
then
    # Conceptual Caption & COCO Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_CC-COCO_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/mix-cc-COCO/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/mix-cc-COCO/mix_triplet_labels.npy \
    WSVL.LOSS_CLS_WEIGHTS ./datasets/vg/caption_labels/mix-cc-COCO/mix2vg_weights.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/mix-cc-COCO/mix2VG_word_map.npy \
    WSVL.SKIP_TRAIN False
fi 

if [ $MODEL_TYPE == "Language_CC_Uniter" ]
then
    # Conceptual Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_CC_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/cc-149-65/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/cc-149-65/cc_triplet_labels.npy \
    WSVL.LOSS_CLS_WEIGHTS ./datasets/vg/caption_labels/cc-149-65/cc2vg_weights.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/cc-149-65/cc2VG_word_map.npy \
    WSVL.SKIP_TRAIN False 
fi

if [ $MODEL_TYPE == "Language_COCO_Uniter" ]
then
    # COCO Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_COCO_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/COCO-144-57/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/COCO-144-57/COCO_triplet_labels.npy \
    WSVL.LOSS_CLS_WEIGHTS ./datasets/vg/caption_labels/COCO-144-57/COCO2vg_weights.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/COCO-144-57/COCO2VG_word_map.npy \
    WSVL.SKIP_TRAIN False 
fi

if [ $MODEL_TYPE == "Language_VG_Uniter" ]
then
    # Visual Genome Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_VG_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/vg-149-53/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/vg-149-53/vg_triplet_labels.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/vg-149-53/vg2VG_word_map.npy \
    WSVL.SKIP_TRAIN False 
fi

if [ $MODEL_TYPE == "Language_OpensetCOCO_Uniter" ]
then
    # Openset COCO Caption supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.0001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_OpensetCOCO_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/COCO-4274-678/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/COCO-4274-678/COCO_triplet_labels.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/COCO-4274-678/COCO2VG_word_map.npy \
    WSVL.SKIP_TRAIN False 
fi

if [ $MODEL_TYPE == "Language_CC-COCO_MotifNet" ]
then
    # Conceptual Caption & COCO Caption supervision + offline detector + MotifNet as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 16 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 32 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Language_CC-COCO_MotifNet \
    TEST.IMS_PER_BATCH 2  \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_CAP_TRIP True \
    WSVL.HARD_DICT_MATCH ./datasets/vg/caption_labels/mix-cc-COCO/oid_word_map_synset_cap-trip.npy \
    WSVL.CAP_TRIP_LABEL ./datasets/vg/caption_labels/mix-cc-COCO/mix_triplet_labels.npy \
    WSVL.LOSS_CLS_WEIGHTS ./datasets/vg/caption_labels/mix-cc-COCO/mix2vg_weights.npy \
    WSVL.CAP_VG_DICT ./datasets/vg/caption_labels/mix-cc-COCO/mix2VG_word_map.npy \
    WSVL.SKIP_TRAIN False
fi

##############################################################################################

if [ $MODEL_TYPE == "Weakly_Uniter" ]
then
    # Unlocalized scene graph supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 128 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Weakly_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP False \
    WSVL.HARD_DICT_MATCH ./datasets/vg/oid_word_map_synset.npy \
    WSVL.SKIP_TRAIN False 
fi

##############################################################################################

if [ $MODEL_TYPE == "Sup_Uniter" ]
then
    # Localized scene graph supervision + offline detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 36 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS False \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION False \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 1536 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 128 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 0 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.STEPS 10000,16000 \
    SOLVER.MAX_ITER 16000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Sup_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD True \
    WSVL.OFFLINE_OD_TYPE OID \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP False \
    WSVL.SKIP_TRAIN False 
fi

##############################################################################################

if [ $MODEL_TYPE == "Sup_OnlineDetector_Uniter" ]
then
    # Localized scene graph supervision + online detector + Uniter as relation predictor
    python -m torch.distributed.launch --master_port 10025 \
    --nproc_per_node=2 tools/relation_train_net.py \
    --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" \
    MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth \
    MODEL.ROI_HEADS.DETECTIONS_PER_IMG 80 \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR UniterPredictor \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS True \
    MODEL.ROI_RELATION_HEAD.PREDICT_USE_VISION True \
    MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM 4096 \
    MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE 36 \
    GLOVE_DIR ./glove \
    DATALOADER.NUM_WORKERS 4 \
    DTYPE "float16" \
    SOLVER.BASE_LR 0.001 \
    SOLVER.SCHEDULE.TYPE WarmupMultiStepLR \
    SOLVER.IMS_PER_BATCH 4 \
    TEST.IMS_PER_BATCH 2 \
    SOLVER.STEPS 22500,36000 \
    SOLVER.MAX_ITER 36000 \
    SOLVER.VAL_PERIOD 6000 \
    SOLVER.CHECKPOINT_PERIOD 16000 \
    SOLVER.PRE_VAL False \
    OUTPUT_DIR ./checkpoints/Sup_OnlineDetector_Uniter \
    TEST.IMS_PER_BATCH 2 \
    WSVL.OFFLINE_OD False \
    WSVL.USE_UNITER True \
    WSVL.USE_CAP_TRIP False \
    WSVL.SKIP_TRAIN False 
fi