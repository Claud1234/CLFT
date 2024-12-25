#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from clfcn.fusion_net import FusionNet
from utils.metrics import find_overlap
from utils.metrics import auc_ap
from clft.clft import CLFT


class Tester(object):
    def __init__(self, config, args):
        super().__init__()
        self.config = config
        self.args = args

        self.device = torch.device(self.config['General']['device'] if torch.cuda.is_available() else "cpu")
        print("device: %s" % self.device)

        if args.backbone == 'clfcn':
            self.model = FusionNet()
            print(f'Using backbone {args.backbone}')
            model_path = config['General']['model_path']
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        elif args.backbone == 'clft':
            resize = config['Dataset']['transforms']['resize']
            self.model = CLFT(RGB_tensor_size=(3, resize, resize),
                              XYZ_tensor_size=(3, resize, resize),
                              patch_size=config['CLFT']['patch_size'],
                              emb_dim=config['CLFT']['emb_dim'],
                              resample_dim=config['CLFT']['resample_dim'],
                              read=config['CLFT']['read'],
                              hooks=config['CLFT']['hooks'],
                              reassemble_s=config['CLFT']['reassembles'],
                              nclasses=len(config['Dataset']['classes']),
                              type=config['CLFT']['type'],
                              model_timm=config['CLFT']['model_timm'], )
            print(f'Using backbone {args.backbone}')

            model_path = config['General']['model_path']
            self.model.load_state_dict(torch.load(model_path, map_location=self.device)['model_state_dict'])

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")

        self.model.to(self.device)
        self.nclasses = len(config['Dataset']['classes'])
        self.model.eval()

    def test_clft(self, test_dataloader, modal):
        print('CLFT Model Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            background_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            background_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            vehicle_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            vehicle_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            human_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            human_rec = torch.zeros((len(progress_bar)), dtype=torch.float)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True)

                _, output_seg = self.model(batch['rgb'], batch['lidar'], modality)

                # 1xHxW -> HxW
                output_seg = output_seg.squeeze(1)
                anno = batch['anno']

                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output_seg, anno)
                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                vehicle_pre[i] = batch_precision[0]
                vehicle_rec[i] = batch_recall[0]
                human_pre[i] = batch_precision[1]
                human_rec[i] = batch_recall[1]

                progress_bar.set_description(f'VEHICLE:IoU->{batch_IoU[0]:.4f} Precision->{batch_precision[0]:.4f}'
                                             f'Recall->{batch_recall[0]:.4f}'
                                             f'HUMAN:IoU->{batch_IoU[1]:.4f} Precision->{batch_precision[1]:.4f}'
                                             f'Recall->{batch_recall[1]:.4f} ')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            vehicle_AP = auc_ap(vehicle_pre, vehicle_rec)
            human_AP = auc_ap(human_pre, human_rec)
            print('-----------------------------------------')
            print(f'VEHICLE:CUM_IoU->{cum_IoU[0]:.4f} CUM_Precision->{cum_precision[0]:.4f}'
                  f'CUM_Recall->{cum_recall[0]:.4f} Average Precision->{vehicle_AP:.4f} \n')
            print(f'HUMAN:CUM_IoU->{cum_IoU[1]:.4f} CUM_Precision->{cum_precision[1]:.4f}'
                  f'CUM_Recall->{cum_recall[1]:.4f} Average Precision->{human_AP:.4f}')
            print('-----------------------------------------')
            print('Testing of the subset completed')

    def test_clfcn(self, test_dataloader, modal):
        print('CLFCN Model Testing...')
        overlap_cum, pred_cum, label_cum, union_cum = 0, 0, 0, 0
        modality = modal
        with torch.no_grad():
            progress_bar = tqdm(test_dataloader)

            background_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            background_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            vehicle_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            vehicle_rec = torch.zeros((len(progress_bar)), dtype=torch.float)
            human_pre = torch.zeros((len(progress_bar)), dtype=torch.float)
            human_rec = torch.zeros((len(progress_bar)), dtype=torch.float)

            for i, batch in enumerate(progress_bar):
                batch['rgb'] = batch['rgb'].to(self.device, non_blocking=True)
                batch['lidar'] = batch['lidar'].to(self.device, non_blocking=True)
                batch['anno'] = batch['anno'].to(self.device, non_blocking=True).squeeze(1)

                outputs = self.model(batch['rgb'], batch['lidar'], modality)

                output = outputs[modality]
                annotation = batch['anno']
                batch_overlap, batch_pred, batch_label, batch_union = find_overlap(self.nclasses, output, annotation)

                overlap_cum += batch_overlap
                pred_cum += batch_pred
                label_cum += batch_label
                union_cum += batch_union

                batch_IoU = 1.0 * batch_overlap / (np.spacing(1) + batch_union)
                batch_precision = 1.0 * batch_overlap / (np.spacing(1) + batch_pred)
                batch_recall = 1.0 * batch_overlap / (np.spacing(1) + batch_label)

                vehicle_pre[i] = batch_precision[0]
                vehicle_rec[i] = batch_recall[0]
                human_pre[i] = batch_precision[1]
                human_rec[i] = batch_recall[1]

                progress_bar.set_description(f'VEHICLE:IoU->{batch_IoU[0]:.4f} Precision->{batch_precision[0]:.4f}'
                                             f'Recall->{batch_recall[0]:.4f}'
                                             f'HUMAN:IoU->{batch_IoU[1]:.4f} Precision->{batch_precision[1]:.4f}'
                                             f'Recall->{batch_recall[1]:.4f} ')

            print('Overall Performance Computing...')
            cum_IoU = overlap_cum / union_cum
            cum_precision = overlap_cum / pred_cum
            cum_recall = overlap_cum / label_cum

            vehicle_AP = auc_ap(vehicle_pre, vehicle_rec)
            human_AP = auc_ap(human_pre, human_rec)
            print('-----------------------------------------')
            print(f'VEHICLE:CUM_IoU->{cum_IoU[0]:.4f} CUM_Precision->{cum_precision[0]:.4f}'
                  f'CUM_Recall->{cum_recall[0]:.4f} Average Precision->{vehicle_AP:.4f} \n')
            print(f'HUMAN:CUM_IoU->{cum_IoU[1]:.4f} CUM_Precision->{cum_precision[1]:.4f}'
                  f'CUM_Recall->{cum_recall[1]:.4f} Average Precision->{human_AP:.4f}')
            print('-----------------------------------------')
            print('Testing of the subset completed')
