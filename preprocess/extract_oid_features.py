# References:
# https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb
# https://gist.github.com/markdtw/02ece6b90e75832bd44787c03a664e8d?permalink_comment_id=3444976#gistcomment-3444976
# https://stackoverflow.com/questions/49170336/tf-object-detection-api-extract-feature-vector-for-each-detection-bbox

import os
from os import listdir
from os.path import isfile, join
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import time
import cv2

def load_graph(graph, ckpt_path):
    with graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(ckpt_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

if __name__ == '__main__':
    """
    run_inference_for_multiple_images
    """
    ckpt_path = './faster_rcnn_inception_resnet_v2_atrous_oid_v4_2018_12_12_SGG_36/frozen_inference_graph.pb'
    res_path = './_att'
    img_path = '/media/user/4T/VL-BERT-master/data/conceptual-captions/cc_train_image'
    
    for output_dir in ['./_att','./_box','./_score','./_dists']: #,'./_pool']:
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

    # check the results that obtained previously
    got_file = {f.split('.')[0]: 1 for f in listdir(res_path) if isfile(join(res_path, f))}
    print("Had got {} results".format(len(got_file)))
    
    # check all images
    onlyfiles = [f for f in listdir(img_path) if isfile(join(img_path, f))]
    image_names = [(img_path+'/'+f,f.split('.')[0]) for f in onlyfiles if f.split('.')[0] not in got_file]
    image_names = image_names
    print("Will detect {} images".format(len(image_names)))
    
    # load model (a frozen graph)
    graph = tf.Graph()
    load_graph(graph, ckpt_path)
    box_cnt = []
    detect_fail = []

    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.compat.v1.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',\
                        'detection_features']: #, 'SecondStageBoxPredictor/AvgPool']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

            start_t = time.time()
            for img_i, img_item in enumerate(image_names):
                if img_i >= 100 and img_i % 100 == 0:
                    print('Image {}: Used {} seconds for 100 images'.format(img_i, time.time() - start_t))
                    print("Each image has {} boxes in average!".format(np.mean(box_cnt)))
                    start_t = time.time()

                # Load image, using PIL according to object_detection_tutorial.ipynb on the top of this file
                # cv2 reads in BGR but PIL.Image reads in RGB
                img_name, image_id = img_item
                try:
                    image = Image.open(img_name)
                except:
                    print("\nImage {} is corrupted\n".format(img_name.split('/')[-1]))
                    continue
                image = np.asarray(image)
                if image.ndim == 2:  # convert 1-channel image to 3 channels
                    image = Image.open(img_name).convert("RGB")
                    image = np.asarray(image)
                elif image.ndim == 3 and image.shape[2] != 3:  # some CC images have [*,*,4] shape
                    image = Image.open(img_name).convert("RGB")
                    image = np.asarray(image)
                image = image[np.newaxis,:,:,:]  # input should be [batch,h,w,3]

                # Run inference
                try:
                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image}) # image should be [batch,h,w,3]
                except:
                    print("\nError occurs for image {}".format(img_name.split('/')[-1]))
                    print(image.shape)
                    detect_fail.append(img_name.split('/')[-1])
                    continue

                # all outputs are float32 numpy arrays, so convert types as appropriate
                valid_det_num = int(output_dict['num_detections'][0])
                box_cnt.append(valid_det_num) 
                if valid_det_num == 0:
                    print("\nImage {} has zero detection!\n".format(img_name.split('/')[-1]))
                    detect_fail.append(img_name.split('/')[-1])
                    continue
                
                cls_prob = np.zeros((valid_det_num, 601+1)) # pad background in index 0
                det_cls = output_dict['detection_classes'][0][:valid_det_num].astype(np.int64)
                cls_prob[np.arange(valid_det_num), det_cls] = 1  # index 0 is reserved for background

                boxes = output_dict['detection_boxes'][0][:valid_det_num] # normalized (y1, x1, y2, x2)
                converted_boxes = np.zeros((boxes.shape[0],4)) # convert to unnormalized (x1, y1, x2, y2)
                converted_boxes[:, [0,2]] = boxes[:, [1,3]] * image.shape[2]  # image width
                converted_boxes[:, [1,3]] = boxes[:, [0,2]] * image.shape[1]  # image height

                det_score = output_dict['detection_scores'][0][:valid_det_num]

                feature_2d = output_dict['detection_features'][0][:valid_det_num] # [#obj, 8, 8, 1536]
                avg_feat = np.mean(feature_2d.reshape((valid_det_num, 64, 1536)), axis=1)
                
                # save results
                np.save(os.path.join('./'+'_box', str(image_id)), converted_boxes.astype('float32'))
                np.save(os.path.join('./'+'_score', str(image_id)), det_score.astype('float32'))
                np.savez_compressed(os.path.join('./'+'_dists', str(image_id)), feat=cls_prob.astype('float32'))
                np.savez_compressed(os.path.join('./'+'_att', str(image_id)), feat=avg_feat.astype('float32'))

    np.save("detect_fail.npy", detect_fail)
    print("Each image has {} boxes in average!".format(np.mean(box_cnt)))

