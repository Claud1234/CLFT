#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Dataloader python script

Created on May 13rd, 2021
'''
import os
import sys
import random
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from utils.lidar_process import get_unresized_lid_img_val
from utils.data_augment import TopCrop
from utils.data_augment import AugmentShuffle


class Dataset():
    def __init__(self, dataset, rootpath, split, augment):
        np.random.seed(789)
        self.dataset = dataset
        self.rootpath = rootpath
        self.split = split
        self.augment = augment
        self.augment_shuffle = configs.AUGMENT_SHUFFLE

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
        if self.dataset == 'waymo':
            cam_path = os.path.join(self.rootpath, self.list_examples_cam[idx])
            annotation_path = cam_path.replace('/camera', '/annotation')
            lidar_path = cam_path.replace('/camera', '/lidar').\
                replace('.png', '.pkl')

        elif self.dataset == 'iseauto':
            cam_path = os.path.join(self.rootpath, self.list_examples_cam[idx])
            annotation_path = cam_path.replace('/rgb', '/annotation_gray')
            lidar_path = cam_path.replace('/rgb', '/pkl').\
                replace('.png', '.pkl')

        else:
            print("Error! Check the dataset arg, either 'waymo' or 'iseauto")
            sys.exit()

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        ann_name = annotation_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)
        assert (ann_name == lidar_name)

        top_crop_class = TopCrop(self.dataset, cam_path,
                                 annotation_path, lidar_path)
        # Apply top crop for raw data to crop the 1/2 top of the images
        top_crop_rgb, top_crop_anno,\
            top_crop_points_set,\
            top_crop_camera_coord = top_crop_class.top_crop()

        augment_class = AugmentShuffle(top_crop_rgb, top_crop_anno,
                                       top_crop_points_set,
                                       top_crop_camera_coord)
        if self.augment is not None:
            if self.augment_shuffle is True:
                aug_list = ['random_crop', 'random_rotate', 'colour_jitter',
                            'random_horizontal_flip', 'random_vertical_flip']
                random.shuffle(aug_list)

                for i in range(len(aug_list)):
                    augment_proc = getattr(augment_class, aug_list[i])
                    rgb, anno, X, Y, Z = augment_proc()

            else:  # self.augment_shuffle is False
                rgb, anno, X, Y, Z = augment_class.random_crop()
                rgb, anno, X, Y, Z = augment_class.random_rotate()
                rgb, anno, X, Y, Z = augment_class.colour_jitter()
                rgb, anno, X, Y, Z = augment_class.random_horizontal_flip()
                rgb, anno, X, Y, Z = augment_class.random_vertical_flip()

        else:  # slef.augment is None, all input data are only top cropped
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
