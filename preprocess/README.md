# Scripts for data preprocessing

This section is under construction. The following information is provided for the researchers who are interested in the preprocessing.

## Region feature extraction

`extract_oid_features.py`: This is the script used for region feature extraction with [TensorFlow 1 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md#open-images-trained-models) (faster_rcnn_inception_resnet_v2_atrous_oidv4). After cloning Tensorflow repo, this script was put in the folder `/research/oid` and the way to run it is: `python extract_oid_features.py`. 


For more information, please refer to [Tensorflow 1 Detection tutorial](https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb), [the post for enabling proto with region feature output](https://gist.github.com/markdtw/02ece6b90e75832bd44787c03a664e8d?permalink_comment_id=3444976#gistcomment-3444976), and [previous posts about region feature extraction](https://stackoverflow.com/questions/49170336/tf-object-detection-api-extract-feature-vector-for-each-detection-bbox).


## Pseudo label creation

We provide the key functions for pseudo label creation. The users can use these functions as references as needed.

We used [SceneGraphParser](https://github.com/vacancy/SceneGraphParser) to parse nouns and predicates from image captions. This parser is based on Spacy library and manually-designed parsing rules.

`link_det_cap.py`: Given the detection labels (e.g., `objects_oidv4_vocab.txt`) and the parsed nouns from captions (e.g., `parsed_nouns.txt`), this file tries to link each detected category into one of the parsed nouns using WordNet, thereby creating pseudo label for training.

`link_cap_vg.py`: Given the parsed nouns/predicates from captions (e.g., `parsed_nouns.txt`, `parsed_predicates.txt`) and the evaluation categories in Visual Genome (VG) (e.g., `objects_vg_vocab.txt`, `predicates_vg_vocab.txt`), this file tries to link parsed concept into VG categories using WordNet and VG metadata (e.g., `predicate_alias.txt`), thereby creating mapping between model prediction and target categories for evaluation.



