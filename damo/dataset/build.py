# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import math

import torch.utils.data

from damo.utils import get_world_size

from damo.dataset import datasets as D
from .collate_batch import BatchCollator
from .datasets import MosaicWrapper
# ❌ bỏ DistributedSampler đi nếu không dùng distributed
# from .samplers import DistributedSampler, IterationBasedBatchSampler
from .samplers import IterationBasedBatchSampler
from .transforms import build_transforms

# 👇 Thêm vào
from torch.utils.data import RandomSampler, SequentialSampler


def build_dataset(cfg, ann_files, is_train=True, mosaic_mixup=None):
    if not isinstance(ann_files, (list, tuple)):
        raise RuntimeError(
            'datasets should be a list of strings, got {}'.format(ann_files))
    datasets = []
    for dataset_name in ann_files:
        data = cfg.get_data(dataset_name)
        factory = getattr(D, data['factory'])
        args = data['args']
        args['transforms'] = None
        args['class_names'] = cfg.dataset.class_names
        dataset = factory(**args)

        if is_train and mosaic_mixup is not None:
            dataset = MosaicWrapper(
                dataset=dataset,
                img_size=mosaic_mixup.mosaic_size,
                mosaic_prob=mosaic_mixup.mosaic_prob,
                mixup_prob=mosaic_mixup.mixup_prob,
                transforms=None,
                degrees=mosaic_mixup.degrees,
                translate=mosaic_mixup.translate,
                shear=mosaic_mixup.shear,
                mosaic_scale=mosaic_mixup.mosaic_scale,
                mixup_scale=mosaic_mixup.mixup_scale,
                keep_ratio=mosaic_mixup.keep_ratio
            )

        datasets.append(dataset)

    return datasets


# ✅ Sửa tại đây:
def make_data_sampler(dataset, shuffle):
    return RandomSampler(dataset) if shuffle else SequentialSampler(dataset)


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info['height']) / float(img_info['width'])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_sampler(dataset,
                       sampler,
                       images_per_batch,
                       num_iters=None,
                       start_iter=0,
                       mosaic_warpper=False):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          images_per_batch,
                                                          drop_last=False)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter, enable_mosaic=mosaic_warpper)
    return batch_sampler


def build_dataloader(datasets,
                     augment,
                     batch_size=128,
                     start_epoch=None,
                     total_epochs=None,
                     no_aug_epochs=0,
                     is_train=True,
                     num_workers=0,
                     size_div=32):

    num_gpus = get_world_size()
    assert (
            batch_size % num_gpus == 0
        ), 'training_imgs_per_batch ({}) must be divisible by the number ' \
        'of GPUs ({}) used.'.format(batch_size, num_gpus)
    images_per_gpu = batch_size // num_gpus

    if is_train:
        iters_per_epoch = math.ceil(len(datasets[0]) / batch_size)
        shuffle = True
        num_iters = total_epochs * iters_per_epoch
        start_iter = start_epoch * iters_per_epoch
    else:
        iters_per_epoch = math.ceil(len(datasets[0]) / batch_size)
        shuffle = False
        num_iters = None
        start_iter = 0

    transforms = augment.transform
    enable_mosaic_mixup = 'mosaic_mixup' in augment

    transforms = build_transforms(start_epoch, total_epochs, no_aug_epochs,
                                  iters_per_epoch, num_workers, batch_size,
                                  num_gpus, **transforms)

    for dataset in datasets:
        dataset._transforms = transforms
        if hasattr(dataset, '_dataset'):
            dataset._dataset._transforms = transforms

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle)
        batch_sampler = make_batch_sampler(dataset, sampler, images_per_gpu,
                                           num_iters, start_iter,
                                           enable_mosaic_mixup)
        collator = BatchCollator(size_div)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    if is_train:
        assert len(
            data_loaders) == 1, 'multi-training set is not supported yet!'
        return data_loaders[0]
    return data_loaders
