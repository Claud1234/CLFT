#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Dataloader python script

Created on May 13rd, 2021
'''
import os
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.lidar_process import get_lid_images_val
from utils.data_augment import ImageProcess
import configs


class Dataset():
    def __init__(self, dataroot, split, augment):
        np.random.seed(789)
        self.dataroot = dataroot
        self.split = split
        self.augment = augment

        self.rgb_normalize = transforms.Normalize(mean=configs.IMAGE_MEAN,
                                                  std=configs.IMAGE_STD)

        list_examples_file = open(os.path.join(
                                dataroot, 'splits', split + '.txt'), 'r')
        self.list_examples_cam = np.array(
                                        list_examples_file.read().splitlines())
        list_examples_file.close()

    def __len__(self):
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        cam_path = os.path.join(self.dataroot, self.list_examples_cam[idx])
        annotation_path = cam_path.replace('/camera', '/annotation')
        lidar_path = cam_path.replace('/camera', '/lidar').\
            replace('.png', '.pkl')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        ann_name = annotation_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)
        assert (ann_name == lidar_name)

        if self.augment == 'square_crop':
            rgb, anno, X, Y, Z = ImageProcess(cam_path, annotation_path,
                                              lidar_path).square_crop()

        elif self.augment == 'random_crop':
            return

        elif self.augment == 'random_rotate':
            return

        elif self.augment == 'random_colour_jiter':
            return

        else:   # Only apply the top crop
            rgb, anno, points_set, camera_coord = ImageProcess(
                                                        cam_path,
                                                        annotation_path,
                                                        lidar_path).top_crop()

            w, h = rgb.size
            X, Y, Z = get_lid_images_val(h, w, points_set, camera_coord)

        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))
        lid_images = torch.cat((X, Y, Z), 0)
        rgb_copy = TF.to_tensor(np.array(rgb.copy()))[0:3]
        rgb = self.rgb_normalize(TF.to_tensor(np.array(rgb))[0:3])
        annotation = \
            TF.to_tensor(np.array(anno)).type(torch.LongTensor).squeeze(0)

        return {'rgb': rgb, 'rgb_orig': rgb_copy,
                'lidar': lid_images, 'annotation': annotation}
