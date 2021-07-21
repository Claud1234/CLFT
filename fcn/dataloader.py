#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Dataloader python script

Created on May 13rd, 2021
'''
import os
import random
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from utils.lidar_process import get_resized_lid_img_val
from utils.lidar_process import get_unresized_lid_img_val
from utils.data_augment import ImageProcess


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

        augment_class = ImageProcess(cam_path, annotation_path, lidar_path)

        # Apply top crop for raw data to crop the 1/3 top of the images
        top_crop_rgb, top_crop_anno,\
            top_crop_points_set,\
            top_crop_camera_coord = augment_class.top_crop()

        if self.augment is not None:
            _, _, square_crop_rgb, square_crop_anno, \
                square_crop_points_set, square_crop_camera_coord = \
                augment_class.square_crop(top_crop_rgb,
                                          top_crop_anno,
                                          top_crop_points_set,
                                          top_crop_camera_coord)
            w_rand, h_rand, random_crop_rgb, random_crop_anno, \
                random_crop_points_set, random_crop_camera_coord = \
                augment_class.random_crop(square_crop_rgb,
                                          square_crop_anno,
                                          square_crop_points_set,
                                          square_crop_camera_coord)
            rgb = random_crop_rgb
            anno = random_crop_anno
            X, Y, Z = get_resized_lid_img_val(h_rand, w_rand,
                                              random_crop_points_set,
                                              random_crop_camera_coord)

            if random.random() > 0.5:
                    rgb, anno, X, Y, Z = augment_class.random_rotate(rgb, anno,
                                                                     X, Y, Z)

            rgb = augment_class.colour_jitter(rgb)

        else:  # slef.augment is None
            rgb = top_crop_rgb
            anno = top_crop_anno
            points_set = top_crop_points_set
            camera_coord = top_crop_camera_coord

            w, h = rgb.size
            X, Y, Z = get_unresized_lid_img_val(h, w, points_set, camera_coord)

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
