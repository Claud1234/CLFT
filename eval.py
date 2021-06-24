#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation python script

Created on June 5th, 2021
'''
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

import configs
from fcn.fusion_net import FusionNet
from fcn.dataloader import Dataset
from utils.metrics import find_overlap


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
                         shuffle=False,
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

overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
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

        batch_overlap, batch_pred, batch_label, batch_union = \
            find_overlap(outputs, annotation)

        overlap_cum += batch_overlap
        pred_cum += batch_pred
        label_cum += batch_label
        union_cum += batch_union

        batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
        batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
        batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

        progress_bar.set_description(f'BACKGROUND:IoU->{batch_IoU[0]:.4f} '
                                     f'Precision->{batch_precision[0]:.4f} '
                                     f'Recall->{batch_recall[0]:.4f}, '
                                     f'VEHICLE:IoU->{batch_IoU[1]:.4f} '
                                     f'Precision->{batch_precision[1]:.4f} '
                                     f'Recall->{batch_recall[1]:.4f}, '
                                     f'HUMAN:IoU->{batch_IoU[2]:.4f} '
                                     f'Precision->{batch_precision[2]:.4f} '
                                     f'Recall->{batch_recall[2]:.4f} ')

    print('Overall Performance Computing...')
    IoU = overlap_cum / union_cum
    precision = overlap_cum / pred_cum
    recall = overlap_cum / label_cum
    print(f'BACKGROUND:IoU->{IoU[0]:.4f} '
          f'Precision->{precision[0]:.4f} '
          f'Recall->{recall[0]:.4f}, '
          f'VEHICLE:IoU->{IoU[1]:.4f} '
          f'Precision->{precision[1]:.4f} '
          f'Recall->{recall[1]:.4f}, '
          f'HUMAN:IoU->{IoU[2]:.4f} '
          f'Precision->{precision[2]:.4f} '
          f'Recall->{recall[2]:.4f} ')
print('validation Complete')
