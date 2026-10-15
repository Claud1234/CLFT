#!/usr/bin/env python3
"""Run a bounded CLFT/Waymo GPU training, validation and checkpoint test."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def train_stage(run):
    import torch
    from torch.utils.data import DataLoader
    from tools.dataset import Dataset
    from tools.trainer import Trainer
    from utils.helpers import save_model_dict
    import numpy as np
    import random

    config = json.loads((run / 'config.json').read_text())
    torch.manual_seed(config['General']['seed'])
    np.random.seed(config['General']['seed'])
    random.seed(config['General']['seed'])
    trainer = Trainer(config, SimpleNamespace(backbone='clft'))
    if trainer.device.type != 'mps':
        raise RuntimeError('This check requires the Apple GPU')
    history = []
    add_scalars = trainer.writer.add_scalars

    def record(name, values, epoch):
        history.append({'epoch': epoch, 'metric': name,
                        'values': {key: float(value) for key, value in values.items()}})
        add_scalars(name, values, epoch)

    trainer.writer.add_scalars = record
    head = trainer.model.head_segmentation.head[-1].weight
    initial_head = head.detach().cpu().clone()
    steps = []
    params = dict(trainer.model.named_parameters())
    watched = [head, params['reassembles_RGB.0.read.project.0.weight'],
               params['reassembles_XYZ.0.read.project.0.weight'],
               params['transformer_encoders.blocks.0.attn.qkv.weight']]

    def check_step(optimizer, args, kwargs):
        if not all(p.device.type == 'mps' and p.grad is not None and
                   torch.isfinite(p.grad).all().item() and p.grad.abs().sum().item() > 0
                   for p in watched):
            raise RuntimeError('Missing, zero or nonfinite GPU gradients in encoder/sensor decoders/head')
        torch.mps.synchronize()
        steps.append({'allocated_gib': torch.mps.current_allocated_memory() / 2**30,
                      'driver_gib': torch.mps.driver_allocated_memory() / 2**30})

    handle = trainer.optimizer.register_step_post_hook(check_step)
    batch_size = config['General']['batch_size']
    train = DataLoader(Dataset(config, 'train', run / 'train.txt'), batch_size=batch_size,
                       shuffle=True, num_workers=0, drop_last=True)
    valid = DataLoader(Dataset(config, 'val', run / 'validation.txt'), batch_size=batch_size,
                       shuffle=False, num_workers=0, drop_last=False)
    started = time.perf_counter()
    try:
        trainer.train_clft(train, valid, 'cross_fusion')
        torch.mps.synchronize()
        elapsed = time.perf_counter() - started
        delta = (head.detach().cpu() - initial_head).abs().max().item()
        if delta <= 0 or not all(math.isfinite(v) for row in history for v in row['values'].values()):
            raise RuntimeError('Parameters did not update or logged metrics are nonfinite')
        # Prove real camera and XYZ inputs both influence fusion predictions.
        trainer.model.eval()
        trainer.optimizer.zero_grad(set_to_none=True)
        batch = next(iter(valid))
        rgb = batch['rgb'][:1].to(trainer.device).requires_grad_()
        lidar = batch['lidar'][:1].to(trainer.device).requires_grad_()
        input_stats = {}
        for name, tensor in (('camera', rgb), ('lidar_xyz', lidar)):
            values = tensor.detach().cpu()
            input_stats[name] = {'shape': list(tensor.shape), 'device': str(tensor.device),
                                 'dtype': str(tensor.dtype), 'finite': bool(torch.isfinite(values).all()),
                                 'channel_min': values.flatten(2).min(2).values[0].tolist(),
                                 'channel_max': values.flatten(2).max(2).values[0].tolist()}
        _, logits = trainer.model(rgb, lidar, 'cross_fusion')
        grads = torch.autograd.grad(logits.square().mean(), (rgb, lidar))
        fusion = {'input_gradient_l1': {name: grad.abs().sum().item()
                                       for name, grad in zip(('camera', 'lidar_xyz'), grads)}}
        if not all(torch.isfinite(grad).all().item() and grad.abs().sum().item() > 0 for grad in grads):
            raise RuntimeError('Fusion does not propagate finite nonzero gradients to both sensor inputs')
        baseline = logits.detach()
        with torch.no_grad():
            _, no_camera = trainer.model(torch.zeros_like(rgb), lidar.detach(), 'cross_fusion')
            _, no_lidar = trainer.model(rgb.detach(), torch.zeros_like(lidar), 'cross_fusion')
        fusion['max_logit_change_when_zeroed'] = {
            'camera': (baseline - no_camera).abs().max().item(),
            'lidar_xyz': (baseline - no_lidar).abs().max().item()}
        if not all(math.isfinite(value) and value > 0
                   for value in fusion['max_logit_change_when_zeroed'].values()):
            raise RuntimeError('Zeroing a sensor failed to change fusion predictions')
        epoch = config['General']['epochs'] - 1
        save_model_dict(config, epoch, trainer.model, 'cross_fusion', trainer.optimizer)
        write_json(run / 'training.json', {'status': 'passed', 'device': str(trainer.device),
                   'parameters': sum(p.numel() for p in trainer.model.parameters()),
                   'dtype': str(head.dtype), 'epochs': config['General']['epochs'],
                   'optimizer_steps': len(steps), 'train_frames': len(train.dataset),
                   'validation_frames': len(valid.dataset), 'batch_size': batch_size,
                   'training_and_validation_seconds': elapsed,
                   'max_parameter_change': delta, 'gpu_memory_observations': steps,
                   'input_statistics': input_stats, 'fusion_input_checks': fusion, 'history': history})
    finally:
        handle.remove()
        trainer.writer.close()


def choose_frames(lines, count):
    """Round-robin the four weather/lighting categories without overlap."""
    groups = [[line for line in lines if f'/{category}/camera/' in line]
              for category in ('day/not_rain', 'day/rain', 'night/not_rain', 'night/rain')]
    selected = []
    for index in range(max(map(len, groups))):
        for group in groups:
            if index < len(group):
                selected.append(group[index])
                if len(selected) == count:
                    return selected
    raise ValueError(f'Not enough frames to select {count}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'config.macos.json'))
    parser.add_argument('--output', help='A new result directory; defaults to logs/macos_smoke/TIMESTAMP')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--lidar-format', choices=['pkl', 'png'], help='Override Dataset.lidar_format')
    parser.add_argument('--dataset-root', help='Directory containing labeled/ and splits_clft/')
    parser.add_argument('--train-frames', type=int, default=8)
    parser.add_argument('--validation-frames', type=int, default=5)
    parser.add_argument('--test-frames', type=int, default=8)
    parser.add_argument('--stage', choices=['train'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage:
        train_stage(Path(args.output).resolve())
        return
    if min(args.batch_size, args.train_frames, args.validation_frames, args.test_frames) < 1:
        parser.error('batch size and frame counts must be positive')
    if args.train_frames % args.batch_size:
        parser.error('training frame count must be divisible by batch size')
    import torch
    import timm
    import numpy as np
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable. Run outside the app sandbox using native arm64 Python.')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Disable PYTORCH_ENABLE_MPS_FALLBACK to verify GPU operator support')
    run = Path(args.output).resolve() if args.output else ROOT / 'logs/macos_smoke' / datetime.now().strftime('%Y%m%d-%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(args.config).read_text())
    config['General'].update(device='mps', epochs=2, batch_size=args.batch_size, num_workers=0,
                             pretrained=False, resume_training=False,
                             tensorboard_logdir=str(run / 'tensorboard'), save_epoch=20)
    dataset_root = Path(args.dataset_root or config['Dataset'].get('root', ROOT / 'waymo_dataset')).resolve()
    lidar_format = args.lidar_format or config['Dataset'].get('lidar_format', 'pkl')
    config['Dataset'].update(root=str(dataset_root), lidar_format=lidar_format)
    for key in config['Log']:
        config['Log'][key] = str(run / key) + '/'
    checkpoint = Path(config['Log']['logdir_fusion']) / 'checkpoint_1.pth'
    config['General']['model_path'] = str(checkpoint)
    split_root = dataset_root / 'splits_clft'
    original = {'train': (split_root / 'train_all.txt').read_text().splitlines(),
                'validation': (split_root / 'early_stop_valid.txt').read_text().splitlines(),
                'test': sum([(split_root / f'test_{part}.txt').read_text().splitlines()
                             for part in ('day_fair', 'day_rain', 'night_fair', 'night_rain')], [])}
    audit = {}
    for name, lines in original.items():
        missing = []
        for line in lines:
            camera = dataset_root / line
            lidar = (Path(str(camera).replace('/camera/', '/lidar/')).with_suffix('.pkl')
                     if lidar_format == 'pkl'
                     else Path(str(camera).replace('/camera/', '/lidar_png/')))
            paths = [camera, Path(str(camera).replace('/camera/', '/annotation/')),
                     lidar]
            missing.extend(str(path) for path in paths if not path.is_file())
        if missing:
            raise FileNotFoundError(f'{name}: {len(missing)} missing files, e.g. {missing[:3]}')
        audit[name] = {'frames': len(lines), 'lidar_format': lidar_format,
                       'all_camera_annotation_lidar_present': True}
    if any(set(original[a]) & set(original[b])
           for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Source train/validation/test split frames overlap')
    for name, count in [('train', args.train_frames), ('validation', args.validation_frames), ('test', args.test_frames)]:
        (run / f'{name}.txt').write_text('\n'.join(choose_frames(original[name], count)) + '\n')
    write_json(run / 'config.json', config)
    report = {'status': 'running', 'python': sys.version, 'machine': platform.machine(),
              'macos': platform.mac_ver()[0], 'torch': torch.__version__, 'timm': timm.__version__,
              'numpy': np.__version__, 'device': 'mps', 'cpu_fallback': False,
              'model': config['CLFT']['model_timm'], 'input_size': config['Dataset']['transforms']['resize'],
              'modality': 'cross_fusion', 'initialization': 'random; no pretrained download',
              'lidar_format': lidar_format, 'dataset_root': str(dataset_root),
              'preprocessing': ('front-camera XYZ projection, configured mean/std normalization, '
                                'bottom-half crop, paired spatial augmentation, dilation and resize'
                                if lidar_format == 'pkl' else 'rendered RGB LiDAR PNG'),
              'dataset_audit': audit,
              'checkpoint': str(checkpoint), 'stages': {}}
    write_json(run / 'report.json', report)
    commands = {'training_validation': [sys.executable, '-u', str(Path(__file__).resolve()),
                                       '--stage', 'train', '--output', str(run)],
                'testing': [sys.executable, '-u', str(ROOT / 'test.py'), '--config', str(run / 'config.json'),
                            '-bb', 'clft', '-m', 'cross_fusion', '-p', str(run / 'test.txt'),
                            '--metrics-output', str(run / 'testing.json')]}
    print(f'Results: {run}', flush=True)
    for name, command in commands.items():
        print(f'Running {name} on MPS...', flush=True)
        started = time.perf_counter()
        with (run / f'{name}.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        report['stages'][name] = {'exit_code': result.returncode, 'wall_seconds': time.perf_counter() - started}
        report['status'] = 'failed' if result.returncode else 'running'
        write_json(run / 'report.json', report)
        print((run / f'{name}.log').read_text(), flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {run / (name + ".log")}')
    report.update(status='passed', training=json.loads((run / 'training.json').read_text()),
                  testing=json.loads((run / 'testing.json').read_text()))
    write_json(run / 'report.json', report)
    print(f'PASS: CLFT training, validation and checkpoint testing on MPS. Report: {run / "report.json"}')


if __name__ == '__main__':
    main()
