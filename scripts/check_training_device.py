#!/usr/bin/env python3
"""Check real model forward/backward, metrics and Adam on a selected device."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tools.trainer import Trainer
from utils.metrics import find_overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config.macos.json')
    parser.add_argument('-bb', '--backbone', choices=['clft', 'clfcn'], default='clft')
    parser.add_argument('-m', '--mode', choices=['rgb', 'lidar', 'cross_fusion'], default='cross_fusion')
    parser.add_argument('--device', help='Override configured device')
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch size must be positive')
    with open(args.config) as f:
        config = json.load(f)
    config['General']['resume_training'] = False
    config['General']['pretrained'] = False
    if args.device:
        config['General']['device'] = args.device
    with tempfile.TemporaryDirectory(prefix='clft-device-check-') as logdir:
        config['General']['tensorboard_logdir'] = logdir
        trainer = Trainer(config, SimpleNamespace(backbone=args.backbone))
        try:
            size = config['Dataset']['transforms']['resize']
            device = trainer.device
            rgb = torch.randn(args.batch_size, 3, size, size, device=device)
            lidar = torch.randn_like(rgb)
            labels = torch.randint(0, trainer.nclasses - 1, (args.batch_size, size, size), device=device)
            trainer.model.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            outputs = trainer.model(rgb, lidar, args.mode)
            output = outputs[1] if args.backbone == 'clft' else outputs[args.mode]
            if args.backbone == 'clfcn' and args.mode == 'cross_fusion':
                loss = sum(trainer.criterion(outputs[name], labels)
                           for name in ('rgb', 'lidar', 'cross_fusion'))
            else:
                loss = trainer.criterion(output, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite synthetic loss')
            loss.backward()
            gradients = [p.grad for p in trainer.model.parameters() if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all().item() for g in gradients):
                raise RuntimeError('Missing or nonfinite gradients')
            trainer.optimizer.step()
            overlap, _, _, union = find_overlap(trainer.nclasses, output.detach(), labels)
            print(f'PASS: {args.backbone}/{args.mode} on {device}; '
                  f'loss={loss.item():.4f}, IoU={(overlap / union).cpu().tolist()}')
            print('Synthetic check only; this does not measure dataset accuracy or training speed.')
        finally:
            trainer.writer.close()


if __name__ == '__main__':
    main()
