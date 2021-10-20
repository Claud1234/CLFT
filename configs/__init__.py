#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Configurations of model training and validation

Created on May 12nd, 2021
'''
# Waymo dataset
DATAROOT = '/home/claude/Data/mauro_waymo'
LOG_DIR = '/media/storage/data/fusion_logs/phase_0_train_by_all/'  # Path to save checkpoints
TRAIN_SPLITS = 'train_all'  # training split file name (.txt file)
VALID_SPLITS = 'early_stop_valid'  # Validation(while training) split file name
EVAL_SPLITS = 'eval_night_rain'  # Evaluation split file name (.txt file)

# iseAuto dataset
ISE_DATAROOT = '/home/claude/Data/claude_iseauto'
ISE_EVAL_SPLITS = 'day_fair_eval'

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
ISE_LIDAR_MEAN = [0.5, 0.5, 0.5]
ISE_LIDAR_STD = [0.5, 0.5, 0.5]

ISE_IMAGE_MEAN = [0.5, 0.5, 0.5]
ISE_IMAGE_STD = [0.5, 0.5, 0.5]

DEVICE = 'cuda:0'  # 'cpu' for CPU training. Default ID for GPU is :0
CLASS_TOTAL = 4  # number of classes
WORKERS = 16  # number of data loading workers (CPU threads)
BATCH_SIZE = 8  # batch size
EPOCHS = 200  # number of total epochs to run
SAVE_EPOCH = 10  # save the checkpoint after this mount of epochs

EPOCHS_COTRAIN = 200  # number of total epochs to run
LR = 0.0001   # initial learning rate
LR_SEMI = 0.00003

# Early stopping
EARLY_STOPPING = True  # or False. When set True, SAVE_EPOCH no longer working
PATIENCE = 40

TEST_IMAGE = './test_images/img_human.png'
TEST_LIDAR = './test_images/lidar_human.pkl'
TEST_ANNO = './test_images/anno_human.png'
