#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import sys
import argparse
import numpy as np

from torch.utils.data import DataLoader

from tools.trainer import Trainer
from tools.dataset import Dataset


with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description='CLFT and CLFCN Training')
parser.add_argument('-bb', '--backbone', required=True,
                    choices=['clfcn', 'clft'],
                    help='Use the backbone of training, clft or clfcn')
parser.add_argument('-m', '--mode', type=str, required=True,
                    choices=['rgb', 'lidar', 'cross_fusion'],
                    help='Output mode (lidar, rgb or cross_fusion)')
args = parser.parse_args()
np.random.seed(config['General']['seed'])
trainer = Trainer(config, args)

train_data = Dataset(config, 'train', './waymo_dataset/splits_clft/train_all.txt')
train_dataloader = DataLoader(train_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

valid_data = Dataset(config, 'val', './waymo_dataset/splits_clft/early_stop_valid.txt')
valid_dataloader = DataLoader(valid_data,
                              batch_size=config['General']['batch_size'],
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

if args.backbone == 'clft':
    trainer.train_clft(train_dataloader, valid_dataloader, modal=args.mode)

elif args.backbone == 'clfcn':
    trainer.train_clfcn(train_dataloader, valid_dataloader, modal=args.mode)

else:
    sys.exit("A backbone must be specified! (clft or clfcn)")



