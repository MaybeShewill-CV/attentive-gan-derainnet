#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-1-31 上午11:21
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/attentive-gan-derainnet
# @File    : global_config.py
# @IDE: PyCharm Community Edition
"""
设置全局变量
"""
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# Train options
__C.TRAIN = edict()

# Set the shadownet training epochs
__C.TRAIN.EPOCHS = 100010
# Set the initial learning rate
__C.TRAIN.LEARNING_RATE = 0.0002
# Set the GPU resource used during training process
__C.TRAIN.GPU_MEMORY_FRACTION = 0.95
# Set the GPU allow growth parameter during tensorflow training process
__C.TRAIN.TF_ALLOW_GROWTH = True
# Set the shadownet training batch size
__C.TRAIN.BATCH_SIZE = 1
# Set train image height
__C.TRAIN.IMG_HEIGHT = 256
# Set train image width
__C.TRAIN.IMG_WIDTH = 376
# Set train image height
__C.TRAIN.CROP_IMG_HEIGHT = 240
# Set train image width
__C.TRAIN.CROP_IMG_WIDTH = 360
# Set cpu multi process thread nums
__C.TRAIN.CPU_MULTI_PROCESS_NUMS = 6
# Set the GPU nums
__C.TRAIN.GPU_NUM = 2

# Test options
__C.TEST = edict()

# Set the GPU resource used during testing process
__C.TEST.GPU_MEMORY_FRACTION = 0.8
# Set the GPU allow growth parameter during tensorflow testing process
__C.TEST.TF_ALLOW_GROWTH = False
# Set the test batch size
__C.TEST.BATCH_SIZE = 1
# Set test image height
__C.TEST.IMG_HEIGHT = 240
# Set test image width
__C.TEST.IMG_WIDTH = 360
