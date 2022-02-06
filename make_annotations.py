#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to make annotations for unlabeled iseAuto data

Created on Feb. 6th, 2022
'''
import os
import cv2
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from fcn.fusion_net import FusionNet
from utils.lidar_process import open_lidar
from utils.lidar_process import get_unresized_lid_img_val
from utils.helpers import draw_test_segmentation_map

parser = argparse.ArgumentParser(description='Making annotations')
parser.add_argument('-p', '--model_path', dest='model_path',
                    help='path of checkpoint for evaluation')
parser.add_argument('-m', '--model', dest='model', required=True,
                    choices=['rgb', 'lidar', 'fusion'],
                    help='Define training modes. (rgb, lidar or fusion)')
args = parser.parse_args()


class UnlabeledDataset():
    '''
    No waymo dataset process, no process to annotations, save cam path.
    '''
    def __init__(self, rootpath, split):
        self.rootpath = rootpath
        self.split = split

        self.rgb_normalize = transforms.Normalize(mean=configs.IMAGE_MEAN,
                                                  std=configs.IMAGE_STD)

        list_examples_file = open(os.path.join(
                                rootpath, 'splits', split + '.txt'), 'r')
        self.list_examples_cam = np.array(
                                        list_examples_file.read().splitlines())
        list_examples_file.close()

    def __len__(self):
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        cam_path = os.path.join(self.rootpath, self.list_examples_cam[idx])

        lidar_path = cam_path.replace('/rgb', '/pkl').\
            replace('.png', '.pkl')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)

        rgb = Image.open(cam_path).resize((480, 320), Image.BICUBIC)
        points_set, camera_coord = open_lidar(
                                            lidar_path,
                                            w_ratio=8.84,
                                            h_ratio=8.825,
                                            lidar_mean=configs.ISE_LIDAR_MEAN,
                                            lidar_std=configs.ISE_LIDAR_STD)

        w, h = rgb.size
        X, Y, Z = get_unresized_lid_img_val(h, w, points_set, camera_coord)

        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))
        lid_images = torch.cat((X, Y, Z), 0)
        rgb_copy = TF.to_tensor(np.array(rgb.copy()))[0:3]
        rgb = self.rgb_normalize(TF.to_tensor(np.array(rgb))[0:3])

        return {'cam_path': cam_path, 'rgb': rgb, 'rgb_orig': rgb_copy,
                'lidar': lid_images}


dataset = UnlabeledDataset(configs.ISE_ROOTPATH, configs.ISE_SEMI_TRAIN_SPLITS)

dataloader = DataLoader(dataset,
                        batch_size=configs.BATCH_SIZE,
                        num_workers=configs.WORKERS,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=True)

model = FusionNet()
device = torch.device(configs.DEVICE)
model.to(device)
print("Use Device: {} for making annotations".format(configs.DEVICE))

checkpoint = torch.load(args.model_path)
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print('Making annotations for unlabeled dataset...')
with torch.no_grad():
    batches_amount = int(len(dataset)/configs.BATCH_SIZE)
    progress_bar = tqdm(dataloader, total=batches_amount)

    for i, batch in enumerate(progress_bar):
        batch['rgb'] = batch['rgb'].to(device, non_blocking=True)
        batch['lidar'] = batch['lidar'].to(device, non_blocking=True)

        cam_path_list = batch['cam_path']

        outputs = model(batch['rgb'], batch['lidar'], 'all')
        outputs = outputs[args.model]

        for j in range(0, len(cam_path_list)):
            segmented_image = draw_test_segmentation_map(outputs[j])
            anno_save_dir = os.path.join(
                cam_path_list[j].split('/rgb/')[0], ('machine_annotation_rgb_'
                                                     + args.model))

            if not os.path.exists(anno_save_dir):
                os.makedirs(anno_save_dir)

            anno_path = os.path.join(anno_save_dir,
                                     cam_path_list[j].split('/rgb/')[1])
            cv2.imwrite(anno_path, segmented_image)

        progress_bar.set_description(f'Making annotation of {args.model} mode')
