# Cross-Iteration Batch Normalization

By [Zhuliang Yao](https://scholar.google.com/citations?user=J3kgC1QAAAAJ&hl=en), [Yue Cao](http://yue-cao.me), [Shuxin Zheng](https://scholar.google.co.jp/citations?user=rPhGUw0AAAAJ&hl=en), [Gao Huang](http://www.gaohuang.net/), [Stephen Lin](https://scholar.google.com/citations?user=c3PYmxUAAAAJ&hl=en).

This repo is an official implementation of ["Cross-Iteration Batch Normalization"](https://arxiv.org/abs/2002.05712) on COCO object detection based on open-mmlab's mmdetection. This repository contains a PyTorch implementation of the CBN layer, as well as some training scripts to reproduce the COCO object detection and instance segmentation results reported in our paper.

## Introduction

**CBN** is initially described in [arxiv](https://arxiv.org/abs/2002.05712). A well-known issue of Batch Normalization is its significantly reduced effectiveness in the case of small mini-batch sizes. Here we present Cross-Iteration Batch Normalization (CBN), in which examples from multiple recent iterations are jointly utilized to enhance estimation quality. A challenge is that the network activations from different iterations are not comparable to each other due to changes in network weights. We thus compensate for the network weight changes via a proposed technique based on Taylor polynomials, so that the statistics can be accurately estimated. On object detection and image classification with small mini-batch sizes, CBN is found to outperform the original batch normalization and a direct calculation of statistics over previous iterations without the proposed compensation technique.

## Citing CBN

```
@article{zhu2020CBN,
  title={Cross-Iteration Batch Normalization},
  author={Yao, Zhuliang and Cao, Yue and Zheng, Shuxin and Huang, Gao and Lin, Stephen},
  journal={arXiv preprint arXiv:2002.05712},
  year={2020}
}
```

## Main Results

| Backbone      | Method       | Norm | AP<sup>b</sup> | AP<sup>b</sup><sub>0.50</sub> | AP<sup>b</sup><sub>0.75</sub> | AP<sup>m</sup> | AP<sup>m</sup><sub>0.50</sub> | AP<sup>m</sup><sub>0.75</sub> | Download |
|:-------------:|:------------:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| R-50-FPN | Faster R-CNN | -      | 36.8 | 57.9 | 39.8 | - | - | - | [model](https://drive.google.com/file/d/1BVAFDjJXLDdDX6F0WscvFbnCXY37uZUp/view?usp=sharing) |
| R-50-FPN | Faster R-CNN | SyncBN | 37.5 | 58.4 | 40.6 | - | - | - | [model](https://drive.google.com/file/d/1I0EdPYUUUJfCNb_HMc_EJqD4WsZAeXK1/view?usp=sharing) |
| R-50-FPN | Faster R-CNN | GN     | 37.7 | 59.2 | 41.2 | - | - | - | [model](https://drive.google.com/file/d/1SnGdTSFN0cY6zYiYxdKCZXCronFLJZhp/view?usp=sharing) |
| R-50-FPN | Faster R-CNN | CBN    | 37.6 | 58.5 | 40.9 | - | - | - | [model](https://drive.google.com/file/d/17tIX0hZVPisJpMpsHRlT86ik8DrIV4XG/view?usp=sharing) |
| R-50-FPN | Mask R-CNN | -      | 37.6 | 58.5 | 41.0 | 34.0 | 55.2 | 36.2 | [model](https://drive.google.com/file/d/1YyjL4nLnRvc0VnEN6741av6pjoCYz3va/view?usp=sharing) |
| R-50-FPN | Mask R-CNN | SyncBN | 38.5 | 58.9 | 42.0 | 34.3 | 55.7 | 36.7 | [model](https://drive.google.com/file/d/1w5fzfpItoXGgE8CkuY_hniPhlDpa4I17/view?usp=sharing) |
| R-50-FPN | Mask R-CNN | GN     | 38.5 | 59.4 | 41.8 | 35.0 | 56.4 | 37.3 | [model](https://drive.google.com/file/d/1BTrHh-4Xohhs3JuZaTYbJzegJ4SV01qx/view?usp=sharing) |
| R-50-FPN | Mask R-CNN | CBN    | 38.4 | 58.9 | 42.2 | 34.7 | 55.9 | 37.0 | [model](https://drive.google.com/file/d/1qMxyW8RDJt-LxuNhMj_waK482WDvpddo/view?usp=sharing) |

*All results are trained with 1x schedule. Normalization layers of backbone are fixed by default.


## Installation
Please refer to [INSTALL.md](INSTALL.md) for installation and dataset preparation.


## Demo

### Test
Download the pretrained model
```bash
# Faster R-CNN
python tools/test.py {configs_file} {downloaded model} --gpus 4 --out {tmp.pkl} --eval bbox
# Mask R-CNN
python tools/test.py {configs_file} {downloaded model} --gpus 4 --out {tmp.pkl} --eval bbox segm
```


### Train Mask R-CNN
One node with 4GPUs:
```bash
# SyncBN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_syncbn_1x.py 4
# GN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_gn_1x.py 4
# CBN
./tools/dist_train.sh ./configs/cbn/mask_rcnn_r50_fpn_cbn_buffer3_burnin8_1x.py 4
```


## TODO
- [x] Clean up mmdetection code base
- [x] Add CBN layer support
- [x] Add default configs for training
- [x] Upload pretrained models for quick test demo
- [ ] Provide a conv_module of Conv & CBN
- [ ] Speedup CBN layer with CUDA/CUDNN


## Thanks
This implementation is based on mmdetection. Ref to this link for more details about [mmdetection](https://github.com/open-mmlab/mmdetection).

