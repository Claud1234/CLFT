#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Configurations of model training and validation

Created on May 12nd, 2021
'''

DATAROOT = '/home/claude/Data/mauro_waymo'
LOG_DIR = '/home/claude/Data/logs/5th_gpu_test/'  # Path to save checkpoints
SPLITS = 'training_0'  # training split file name (.txt file)
EVAL_SPLITS = 'validation_0'  # validaztion split file name (.txt file)

AUGMENT = 'square_crop'  # 'random_crop' 'random_rotate' 'random_colour_jiter'

# rot_range=20,
# factor=4
# crop_size=128

LIDAR_MEAN = [-0.17263354, 0.85321806, 24.5527253]
LIDAR_STD = [7.34546552, 1.17227659, 15.83745082]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


# IMAGE_BRIGHTNESS = 0.4
# IMAGE_CONTRAST = 0.4
# IMAGE_SATURATION = 0.4
# IMAGE_HUE = 0.1

DEVICE = 'cuda:0'  # 'cpu' for CPU training. Default ID for GPU is :0
WORKERS = 16  # number of data loading workers (CPU threads)
BATCH_SIZE = 8  # batch size
EPOCHS = 100  # number of total epochs to run
SAVE_EPOCH = 10  # save the checkpoint after this mount of epochs

EPOCHS_COTRAIN = 300  # number of total epochs to run
CLASS_TOTAL = 4  # number of classes
LR = 0.0001  # initial learning rate)
LR_SEMI = 0.00005


TEST_IMAGE = './test_images/test_img.png'
TEST_LIDAR = './test_images/test_lidar.pkl'
TEST_ANNO = './test_images/test_anno.png'
