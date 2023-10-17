# ClusVPR for Visual Place Recognition

![image](https://github.com/XuYifan98/ClusVPR/blob/main/figs/VPR_Task.png)

## Introduction
This is a PyTorch implementation for our paper "ClusVPR: Efficient Visual Place Recognition ... ". It is an open-source codebase for visual place recognition.

This pipeline is for the network of VGG-16 + our ClusVPR, trained on Pitts30k dataset. The dimension of the global descriptor is 4096.

![image](https://github.com/XuYifan98/ClusVPR/blob/main/figs/clusvpr_quantitative_results.png)

From the above table, we can see that our model outperforms other baseline models on most of the benchmarks with lower complexity. You can get some ideas of why our trained networks perform better from the following figure:

![image](https://github.com/XuYifan98/ClusVPR/blob/main/figs/clusvpr_qualitative_results.png)

As can be seen, our model focuses on discriminative regions (e.g. buildings, signs), while the other two models falsely focus on dynamic objects or obstacles (e.g.
pedestrians, cars, trees and light).


## Installation
We test this repo with Python 3.8, PyTorch 1.9.0, and CUDA 11.1. However, it should be runnable with recent PyTorch versions (Pytorch >= 1.1.0).
```shell
python setup.py develop
```
In addition, need to install KNN_CUDA from wheel.
```shell
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

## Preparation
### Datasets

We test our models on four geo-localization benchmarks, [MSLS](https://www.mapillary.com/dataset/places), [Pittsburgh](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Torii_Visual_Place_Recognition_2013_CVPR_paper.pdf), [Tokyo 24/7](https://www.di.ens.fr/~josef/publications/Torii15.pdf) and [Tokyo Time Machine](https://arxiv.org/abs/1511.07247) datasets. The last three datasets can be downloaded at [here](https://www.di.ens.fr/willow/research/netvlad/).

The directory of datasets used is like
```shell
datasets/data
├── pitts
│   ├── raw
│   │   ├── pitts250k_test.mat
│   │   ├── pitts250k_train.mat
│   │   ├── pitts250k_val.mat
│   │   ├── pitts30k_test.mat
│   │   ├── pitts30k_train.mat
│   │   ├── pitts30k_val.mat
│   └── └── Pittsburgh
│           ├──images/
│           └──queries/
└── tokyo
    ├── raw
    │   ├── tokyo247
    │   │   ├──images/
    │   │   └──query/
    │   ├── tokyo247.mat
    │   ├── tokyoTM/images/
    │   ├── tokyoTM_train.mat
    └── └── tokyoTM_val.mat
```

### Pre-trained Weights

The file tree we used for storing the pre-trained weights is like
```shell
logs
├── vgg16_pretrained.pth.tar # refer to (1)
└── vgg16_pitts_64_desc_cen.hdf5 # refer to (2)
```

**(1) ImageNet-pretrained weights for CNNs backbone**

The ImageNet-pretrained weights for CNNs backbone or the pretrained weights for the model.

**(2) initial cluster centers for VLAD layer**

Note that the VLAD layer cannot work with random initialization.
The original cluster centers provided by NetVLAD or self-computed cluster centers by running the scripts/cluster.sh.

```shell
bash scripts/cluster.sh vgg16
```

## Training
Train by running script in the terminal. Script location: scripts/train_clusvpr.sh

Format:
```shell
bash scripts/train_clusvpr.sh arch
```
where, **arch** is the backbone name, such as vgg16.

For example:
```shell
bash scripts/train_clusvpr.sh vgg16
```

In the train_clusvpr.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --tuple-size for more tuples on one GPU.
In case your GPU does not have enough memory, reduce --pos-num or --neg-num for fewer positives or negatives in one tuple.

## Testing
Test by running script in the terminal. Script location: scripts/test.sh

Format:
```shell
bash scripts/test.sh resume arch dataset scale
```
where, **resume** is the trained model path.
       **arch** is the backbone name, such as vgg16.
       **dataset scale**, such as pitts 30k, pitts 250k, tokyo.

For example:
1. Test vgg16 on pitts 250k:
```shell
bash scripts/test.sh logs/pitts30k-vgg16/model_best.pth.tar vgg16 pitts 250k
```
In the test.sh.
In case you want to fasten testing, enlarge GPUS for more GPUs, or enlarge the --test-batch-size on one GPU.
In case your GPU does not have enough memory, reduce --test-batch-size on one GPU.
