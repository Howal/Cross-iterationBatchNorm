from __future__ import division
import pprint
import argparse

from mmcv import Config
from mmcv.runner import get_dist_info

from mmdet import __version__
from mmdet.datasets import get_dataset
from mmdet.apis import (train_detector, init_dist, get_root_logger,
                        set_random_seed)
from mmdet.models import build_detector
from mmdet.utils import get_git_hash, summary
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        # save mmdet version in checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__, config=cfg.text)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # init logger before other steps
    logger = get_root_logger(cfg.log_level)
    logger.info('Distributed training: {}'.format(distributed))

    # log cfg
    logger.info('training config:{}\n'.format(pprint.pformat(cfg._cfg_dict)))

    # log git hash
    logger.info('git hash: {}'.format(get_git_hash()))

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    train_dataset = get_dataset(cfg.data.train)

    # update burnin of CBN
    if 'norm_cfg' in cfg and 'CBN' in cfg.norm_cfg['type']:
        if distributed:
            _, world_size = get_dist_info()
            cfg.norm_cfg['burnin'] = int(cfg.norm_cfg['burnin'] * train_dataset.__len__() / cfg.data['imgs_per_gpu'] / world_size)
        else:
            cfg.norm_cfg['burnin'] = int(cfg.norm_cfg['burnin'] * train_dataset.__len__() / cfg.data['imgs_per_gpu'] / cfg.gpus)
        for key in ['bbox_head', 'mask_head']:
            if key in cfg.model and 'norm_cfg' in cfg.model[key]:
                cfg.model[key]['norm_cfg'] = cfg.norm_cfg
        logger.info('dataset length: {}, burnin iter of CBN: {}'.format(train_dataset.__len__(), cfg.norm_cfg['burnin']))

    model = build_detector(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    try:
        logger.info(summary(model, cfg))
    except RuntimeError:
        logger.info('RuntimeError during summary')
        logger.info(str(model))

    train_detector(
        model,
        train_dataset,
        cfg,
        distributed=distributed,
        # validate=args.validate,
        validate=True,
        logger=logger)


if __name__ == '__main__':
    main()
