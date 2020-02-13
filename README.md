# Cross-Iteration Batch Normalization
This repository contains a PyTorch implementation of the CBN layer, as well as some training scripts to reproduce the COCO object detection and instance segmentation results reported in our paper.


## Results with this code

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


### Train Mask R-CNNN
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

