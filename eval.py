#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation python script

Created on June 5th, 2021
'''
import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

import configs
from fcn.fusion_net import FusionNet
from fcn.dataloader import Dataset


parser = argparse.ArgumentParser(description='Model Evaluation')
parser.add_argument('-p', '--model_path', dest='model_path',
                    help='path of checkpoint for evaluation')
args = parser.parse_args()

eval_dataset = Dataset(dataroot=configs.DATAROOT,
                       split=configs.EVAL_SPLITS,
                       augment=None)
eval_loader = DataLoader(eval_dataset,
                         batch_size=configs.BATCH_SIZE,
                         num_workers=configs.WORKERS,
                         shuffle=True,
                         pin_memory=True,
                         drop_last=True)

model = FusionNet()
device = torch.device(configs.DEVICE)
model.to(device)
print("Use Device: {} for Evaluation".format(configs.DEVICE))

checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print('Validating...')

overlap_cum, union_cum = 0, 0
with torch.no_grad():
    batches_amount = int(len(eval_dataset)/configs.BATCH_SIZE)
    progress_bar = tqdm(eval_loader, total=batches_amount)
    for i, batch in enumerate(progress_bar):
        batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
        batch['lidar'] = batch['lidar'].to(device, non_blocking=True)
        batch['annotation'] = \
            batch['annotation'].to(device, non_blocking=True).squeeze(1)

        outputs = model(batch['rgb'], batch['lidar'], 'all')
        outputs = outputs['fusion']

        annotation = batch['annotation']
        # Select only 1(vehicles) and 2(ped+cyclist) indices in annotation
        labeled = (annotation > 0) * (annotation <= 2)
        _, pred_indices = torch.max(outputs, 1)
        pred_indices = pred_indices * labeled.long()
        overlap = pred_indices * (pred_indices == annotation).long()

        area_pred = torch.histc(pred_indices.float(), bins=2, max=2, min=1)
        area_overlap = torch.histc(overlap.float(), bins=2, max=2, min=1)
        area_label = torch.histc(annotation.float(), bins=2, max=2, min=1)
        area_union = area_pred + area_label - area_overlap

        assert (area_overlap[1:] <= area_union[1:]).all(),\
            "Intersection area should be smaller than Union area"

        overlap_cum += area_overlap
        union_cum += area_union

        batch_IoU = area_overlap.cpu().numpy() / area_union.cpu().numpy()
        progress_bar.set_description(f'IoU: {batch_IoU[0]:.4f}')

IoU = overlap_cum / union_cum
print('Final IoU:', IoU)
print('validation Complete')
