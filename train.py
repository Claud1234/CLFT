#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import argparse
from tqdm import tqdm

from iseauto.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

import configs
from iseauto.trainer import Trainer
from iseauto.dataset import Dataset


with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-bb', '--backbone', required=True,
                    choices=['fcn', 'dpt'],
                    help='Use the backbone of training, dpt or fcn')
# parser.add_argument('-reset-lr', dest='reset_lr', action='store_true',
#                     help='Reset LR to initial value defined in configs')
# parser.add_argument('-p', '--model_path', dest='model_path',
#                     help='path of checkpoint for training resuming')
# parser.add_argument('-i', '--dataset', dest='dataset', type=str, required=True,
#                     help='select to evaluate waymo or iseauto dataset')
# parser.add_argument('-m', '--model', dest='model', required=True,
#                     choices=['rgb', 'lidar', 'fusion'],
#                     help='Define training modes. (rgb, lidar or fusion)')
args = parser.parse_args()

trainer = Trainer(config, args)

list_datasets = config['Dataset']['paths']['list_datasets']

datasets_train = []
for data_category in list_datasets:
    datasets_train.append(Dataset(config, data_category, 'train'))
train_data = ConcatDataset(datasets_train)
train_dataloader = DataLoader(train_data,
                              batch_size=config['General']['batch_size'],
                              num_workers=configs.WORKERS,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

datasets_valid = []
for data_category in list_datasets:
    datasets_valid.append(Dataset(config, data_category, 'val'))
valid_data = ConcatDataset(datasets_valid)
valid_dataloader = DataLoader(valid_data,
                              batch_size=config['General']['batch_size'],
                              num_workers=configs.WORKERS,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

trainer.train_dpt(train_dataloader, valid_dataloader)


