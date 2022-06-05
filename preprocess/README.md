# Scripts for data preprocessing

This section is under construction. The following information is provided for the researchers who requested it.


`extract_oid_features.py`: This is the script used for region feature extraction with [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models) (faster_rcnn_inception_resnet_v2_atrous_oidv4). After cloning Tensorflow repo, this script was put in the folder `/research/oid` and the way to run it is: `python extract_oid_features.py`. 


For more information, please refer to [Tensorflow 1 Detection tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb), [the post for enabling proto with region feature output](https://gist.github.com/markdtw/02ece6b90e75832bd44787c03a664e8d?permalink_comment_id=3444976#gistcomment-3444976), and [previous posts about region feature extraction](https://stackoverflow.com/questions/49170336/tf-object-detection-api-extract-feature-vector-for-each-detection-bbox).

