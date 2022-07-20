## DATASET
The following is adapted from [Scene-Graph-Benchmark](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/DATASET.md), [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md) and [neural-motifs](https://github.com/rowanz/neural-motifs).

### Download:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `datasets/vg/VG_100K`. If you want to use other directory, please link it in `DATASETS['VG_stanford_filtered']['img_dir']` of `maskrcnn_benchmark/config/paths_catelog.py`. 
2. Download the [scene graphs labels](https://1drv.ms/u/s!AmRLLNf6bzcir8xf9oC3eNWlVMTRDw?e=63t7Ed) and extract them to `datasets/vg/VG-SGG-with-attri.h5`, or you can edit the path in `DATASETS['VG_stanford_filtered_with_attribute']['roidb_file']` of `maskrcnn_benchmark/config/paths_catalog.py`.
3. Download the [detection results](https://drive.google.com/drive/folders/1SdMXwXpdTZdxYOZl0OcPqqGd2p4DhePt?usp=sharing) of 3 datasets, including: Conceptual Caption, COCO Caption and Visual Genome. After downloading, you can run `cat cc_detection_results.zip.part* > cc_detection_results.zip` to merge several partitions into one zip file and unzip it to folder `datasets/vg/`.

### Folder structure:
After downloading the above files, you should have following hierarchy in folder `datasets/vg/`:

```
├── VG_100K
├── cc_detection_results_oid
├── COCO_detection_results_oid
├── VG_detection_results_oid
└── VG-SGG-with-attri.h5
```

### Preprocessing scripts

We provide scripts for data preprocessing, such as extracting the detection results from images and creating pseudo labels based on detection results and parsed concepts from image captions. More detail can be found in the [folder preprocess](https://github.com/YiwuZhong/SGG_from_NLS/tree/main/preprocess).

