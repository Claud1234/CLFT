#!/usr/bin/env python3
"""Train CLFT or CLFCN on CUDA, Apple Metal (MPS), or CPU."""
import argparse
import json
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.trainer import Trainer
from tools.dataset import Dataset


def main():
    parser = argparse.ArgumentParser(description='CLFT and CLFCN Training')
    parser.add_argument('-bb', '--backbone', required=True, choices=['clfcn', 'clft'])
    parser.add_argument('-m', '--mode', required=True, choices=['rgb', 'lidar', 'cross_fusion'])
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--device', help='auto, mps, cpu, or cuda:N')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--resume', help='Checkpoint to resume (model and optimizer)')
    parser.add_argument('--reset-lr', action='store_true', help='Restart the epoch/LR schedule on resume')
    parser.add_argument('--train-split', default='./waymo_dataset/splits_clft/train_all.txt')
    parser.add_argument('--val-split', default='./waymo_dataset/splits_clft/early_stop_valid.txt')
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    general = config['General']
    for key in ('device', 'batch_size', 'num_workers', 'epochs'):
        value = getattr(args, key)
        if value is not None:
            general[key] = value
    if args.resume:
        general['resume_training'] = True
        general['resume_training_model_path'] = args.resume
    if args.reset_lr:
        general['reset_lr'] = True
    if general['batch_size'] < 1 or general['epochs'] < 1 or general.get('num_workers', 0) < 0:
        parser.error('batch size and epochs must be positive; num workers must be nonnegative')

    random.seed(general['seed'])
    np.random.seed(general['seed'])
    torch.manual_seed(general['seed'])
    trainer = Trainer(config, args)
    try:
        train_data = Dataset(config, 'train', args.train_split)
        valid_data = Dataset(config, 'val', args.val_split)
        loader_options = dict(batch_size=general['batch_size'],
                              num_workers=general.get('num_workers', 0),
                              pin_memory=trainer.device.type == 'cuda')
        train_loader = DataLoader(train_data, shuffle=True, drop_last=True, **loader_options)
        valid_loader = DataLoader(valid_data, shuffle=False, drop_last=False, **loader_options)
        if len(train_loader) == 0 or len(valid_loader) == 0:
            raise ValueError('Empty loader: check split files and lower the training batch size.')
        train = trainer.train_clft if args.backbone == 'clft' else trainer.train_clfcn
        train(train_loader, valid_loader, modal=args.mode)
    finally:
        trainer.writer.close()


if __name__ == '__main__':
    main()
