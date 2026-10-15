#!/usr/bin/env python3
"""Evaluate a CLFT or CLFCN checkpoint on a selected device."""
import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.tester import Tester
from tools.dataset import Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-bb', '--backbone', required=True, choices=['clfcn', 'clft'])
    parser.add_argument('-m', '--mode', required=True, choices=['rgb', 'lidar', 'cross_fusion'])
    parser.add_argument('-p', '--path', required=True, help='Test split file')
    parser.add_argument('--config', default='config.json')
    parser.add_argument('--device', help='auto, mps, cpu, or cuda:N')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--num-workers', type=int)
    parser.add_argument('--checkpoint', help='Override General.model_path')
    parser.add_argument('--metrics-output', help='Save CLFT metrics as JSON')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    general = config['General']
    for key in ('device', 'batch_size', 'num_workers'):
        value = getattr(args, key)
        if value is not None:
            general[key] = value
    if general['batch_size'] < 1 or general.get('num_workers', 0) < 0:
        parser.error('batch size must be positive; num workers must be nonnegative')
    if args.checkpoint:
        general['model_path'] = args.checkpoint
    random.seed(general['seed'])
    np.random.seed(general['seed'])
    torch.manual_seed(general['seed'])
    tester = Tester(config, args)
    test_data = Dataset(config, 'test', args.path)
    print(f'Testing {len(test_data)} frames from {args.path}')
    loader = DataLoader(test_data, batch_size=general['batch_size'], shuffle=False,
                        num_workers=general.get('num_workers', 0),
                        pin_memory=tester.device.type == 'cuda', drop_last=False)
    test = tester.test_clft if args.backbone == 'clft' else tester.test_clfcn
    metrics = test(loader, args.mode)
    if args.metrics_output and metrics is not None:
        Path(args.metrics_output).write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    main()
