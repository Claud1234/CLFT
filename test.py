#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import json
import argparse
import numpy as np
from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset


with open('config.json', 'r') as f:
    config = json.load(f)

parser = argparse.ArgumentParser(description='CLFT and CLFCN Tresting')
parser.add_argument('-bb', '--backbone', required=True,
                    choices=['clfcn', 'clft'],
                    help='Use the backbone of training, clft or clfcn')
parser.add_argument('-m', '--mode', type=str, required=True,
                    choices=['rgb', 'lidar', 'cross_fusion'],
                    help='Output mode (lidar, rgb or cross_fusion)')
parser.add_argument('-p', '--path', type=str, required=True,
                    help='The path of the text file to test the model')
args = parser.parse_args()
np.random.seed(config['General']['seed'])
tester = Tester(config, args)

test_data = Dataset(config, 'test', args.path)
print(f'Testing with the path {args.path}')
test_dataloader = DataLoader(test_data,
                             batch_size=config['General']['batch_size'],
                             shuffle=False,
                             pin_memory=True,
                             drop_last=True)

if args.backbone == 'clft':
    tester.test_clft(test_dataloader, args.mode)

elif args.backbone == 'clfcn':
    tester.test_clfcn(test_dataloader, args.mode)

else:
    sys.exit("A backbone must be specified! (clft or clfcn)")