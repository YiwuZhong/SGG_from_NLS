# Scene Graph Generation from Natural Language Supervision
This repository includes the Pytorch code for our paper "[Learning to Generate Scene Graph from Natural Language Supervision](https://arxiv.org/abs/2109.02227)" accepted in ICCV 2021. Code will be released soon. Stay tuned!

<p align="center">
  <img src="figures/overview.png" alt="overview figure" width="50%" height="50%">
</p>
Top (our setting): Our goal is learning to generate localized scene graphs from image-text pairs. Once trained, our model takes an image and its detected objects as inputs and outputs the image scene graph. Bottom (our results): A comparison of results from our method and state-of-the-art (SOTA) with varying levels of supervision.


## Contents

1. [Overview](#Overview)
2. [Qualitative Results](#Qualitative-Results)
3. [Reference](#Reference)
<!-- 4. [Metrics and Results for our Toolkit](METRICS.md)
    - [Explanation of R@K, mR@K, zR@K, ng-R@K, ng-mR@K, ng-zR@K, A@K, S2G](METRICS.md#explanation-of-our-metrics)
    - [Output Format](METRICS.md#output-format-of-our-code)
    - [Reported Results](METRICS.md#reported-results) -->


## Overview

Learning from image-text data has demonstrated recent success for many recognition tasks, yet is currently limited to visual features or individual visual concepts such as objects. In this paper, we propose one of the first methods that learn from image-sentence pairs to extract a graphical representation of localized objects and their relationships within an image, known as scene graph. To bridge the gap between images and texts, we leverage an off-the-shelf object detector to identify and localize object instances, match labels of detected regions to concepts parsed from captions, and thus create "pseudo" labels for learning scene graph. Further, we design a Transformer-based model to predict these "pseudo" labels via a masked token prediction task. Learning from only image-sentence pairs, our model achieves 30\% relative gain over a latest method trained with human-annotated unlocalized scene graphs. Our model also shows strong results for weakly and fully supervised scene graph generation. In addition, we explore an open-vocabulary setting for detecting scene graphs, and present the first result for open-set scene graph generation.

## Qualitative Results
### Our generated scene graphs learned from image descriptions
<p align="center">
  <img src="figures/visualization-github.png" alt="overview figure" width="95%" height="95%">
</p>
Partial visualization of Figure 3 in our paper: Our model trained by image-sentence pairs produces scene graphs with a high quality (e.g. "man-on-motorcycle" and "man-wearing-helmet" in first example). More comparison with other models trained by stronger supervision (e.g. unlocalized scene graph labels, localized scene graph labels) can be viewed in the Figure 3 of paper.

### Our generated scene graphs in open-set and closed-set settings
<p align="center">
  <img src="figures/visualization-openset.png" alt="overview figure" width="90%" height="90%">
</p>
Figure 4 in our paper: We explored open-set setting where the categories of target concepts (objects and predicates) are unknown during training. Compared to our closed-set model, our open-set model detects more concepts outside the evaluation dataset, Visual Genome (e.g. "swinge", "mouse", "keyboard"). Our results suggest an exciting avenue of large-scale training of open-set scene graph generation using image captioning dataset such as Conceptual Caption.

## Reference
If you are using our code, please consider citing our paper.
```
@inproceedings{zhong2021SGGfromNLS,
  title={Learning to Generate Scene Graph from Natural Language Supervision},
  author={Zhong, Yiwu and Shi, Jing and Yang, Jianwei and Xu, Chenliang and Li, Yin},
  booktitle={ICCV},
  year={2021}
}
```
