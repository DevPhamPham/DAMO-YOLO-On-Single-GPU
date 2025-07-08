#!/usr/bin/env python3
import os
from damo.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        # 1) Tên experiment, intervals,...
        self.miscs.exp_name = os.path.splitext(os.path.basename(__file__))[0]
        self.miscs.eval_interval_epochs = 5
        self.miscs.ckpt_interval_epochs = 5
        self.miscs.total_epochs = 150  # Tổng số epoch giảm xuống 150

        # 2) Lịch train
        self.train.batch_size = 32
        self.train.base_lr_per_img = 0.01 / 64
        self.train.min_lr_ratio = 0.05
        self.train.weight_decay = 5e-4
        self.train.momentum = 0.9
        self.train.no_aug_epochs = 16
        self.train.warmup_epochs = 5
        # 2.1) Optimizer config
        self.train.optimizer = {
            'name': 'SGD',
            'lr': 0.001,
            'momentum': 0.9,
            'weight_decay': 0.0005,
            'nesterov': True
        }

        # 3) Augment
        self.train.augment.transform.image_max_range = (640, 640)
        self.train.augment.mosaic_mixup.mixup_prob = 0.15
        self.train.augment.mosaic_mixup.degrees = 10.0
        self.train.augment.mosaic_mixup.translate = 0.2
        self.train.augment.mosaic_mixup.shear = 2.0
        self.train.augment.mosaic_mixup.mosaic_scale = (0.1, 2.0)

        # 4) Dữ liệu COCO
        # self.dataset.data_dir  = 'datasets/coco'
        self.dataset.train_ann = ('train2019_coco',)
        self.dataset.val_ann   = ('val2019_coco',)

        # 5) Backbone / Neck / Head: giữ nguyên kiến trúc gốc
        structure = self.read_structure(
            './damo/base_models/backbones/nas_backbones/tinynas_L35_kxkx.txt')
        self.model.backbone = {
            'name': 'TinyNAS_csp',
            'net_structure_str': structure,
            'out_indices': (2, 3, 4),
            'with_spp': True,
            'use_focus': True,
            'act': 'silu',
            'reparam': True,
        }
        self.model.neck = {
            'name': 'GiraffeNeckV2',
            'depth': 1.5,
            'hidden_ratio': 1.0,
            'in_channels': [128, 256, 512],
            'out_channels': [128, 256, 512],
            'act': 'silu',
            'spp': False,
            'block_name': 'BasicBlock_3x3_Reverse',
        }
        self.model.head = {
            'name': 'ZeroHead',
            'in_channels': [128, 256, 512],
            'stacked_convs': 0,
            'reg_max': 16,
            'act': 'silu',
            'nms_conf_thre': 0.5,
            'nms_iou_thre': 0.45,
            # num_classes giữ nguyên 3
            'num_classes': 3,
            'legacy': False,
        }

        # 6) Danh sách tên lớp
        self.dataset.class_names = [
            'person',
            'car',
            'long vehicle'
        ]
