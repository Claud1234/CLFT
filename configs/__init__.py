#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Configurations of model training and validation

Created on May 12nd, 2021
'''
LOG_DIR = '/media/storage/data/fusion_logs/asdasd/'  # Path to save checkpoints

# Waymo dataset
WAY_ROOTPATH = '/home/claude/Data/mauro_waymo'
WAY_TRAIN_SPLITS = 'train_all'  # training split file name (.txt file)
WAY_VALID_SPLITS = 'early_stop_valid'  # ES valid split file name
WAY_EVAL_SPLITS = 'eval_night_rain'  # Evaluation split file name (.txt file)

# iseAuto dataset
ISE_ROOTPATH = '/home/claude/Data/claude_iseauto'
ISE_TRAIN_SPLITS = 'train_all'
ISE_SEMI_TRAIN_SPLITS = ''
ISE_VALID_SPLITS = 'early_stop_valid'
ISE_EVAL_SPLITS = 'night_rain_eval'

# Data augment configurations
AUGMENT_SHUFFLE = True  # False
RANDOM_CROP_SIZE = 128
ROTATE_RANGE = 20  # rotate range (-20, 20)
JITTER_PARAM = [0.4, 0.4, 0.4, 0.1]  # [brightness, contrast, saturation, hue]

# Waymo Data normalization
LIDAR_MEAN = [-0.17263354, 0.85321806, 24.5527253]
LIDAR_STD = [7.34546552, 1.17227659, 15.83745082]

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# iseAuto Data normalization
ISE_LIDAR_MEAN = [-0.079, 0.033, 15.90]
ISE_LIDAR_STD = [7.79,  2.156, 7.60]

ISE_IMAGE_MEAN = [0.485, 0.456, 0.406]
ISE_IMAGE_STD = [0.229, 0.224, 0.225]

DEVICE = 'cuda:0'  # 'cpu' for CPU training. Default ID for GPU is :0
CLASS_TOTAL = 4  # number of classes
WORKERS = 16  # number of data loading workers (CPU threads)
BATCH_SIZE = 16  # batch size

EPOCHS = 1000  # number of total epochs for supervised training
SAVE_EPOCH = 10  # save the checkpoint after these epochs if no early-stopping

LR_RGB = 0.00009  # initial learning rate
LR_LIDAR = 0.00008
LR_FUSION = 0.00009


EPOCHS_SEMI = 1500  # number of total epochs for semi-supervised training
LR_SEMI_RGB = 0.00003
LR_SEMI_LIDAR = 0.00003
LR_SEMI_FUSION = 0.00003

# Early stopping
EARLY_STOPPING = True  # or False. When set True, SAVE_EPOCH no longer working
PATIENCE = 200

TEST_IMAGE = './test_images/img_human.png'
TEST_LIDAR = './test_images/lidar_human.pkl'
TEST_ANNO = './test_images/anno_human.png'
