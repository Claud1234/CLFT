#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Dataloader python script for iseAuto dataset

Created on Oct 11st, 2021
'''
import os
import random
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from utils.iseauto_data_augment import IseAutoTopCrop
from utils.iseauto_data_augment import IseautoAugmentShuffle


class IseautoDataset():
    def __init__(self, dataroot, split, augment):
        np.random.seed(789)
        self.dataroot = dataroot
        self.split = split
        self.augment = augment
        self.augment_shuffle = configs.AUGMENT_SHUFFLE
        self.rgb_normalize = transforms.Normalize(mean=configs.ISE_IMAGE_MEAN,
                                                  std=configs.ISE_IMAGE_STD)
        self.lidar_normalize = transforms.Normalize(
                                                mean=configs.ISE_LIDAR_MEAN,
                                                std=configs.ISE_LIDAR_STD)

        list_examples_file = open(os.path.join(
                                dataroot, 'splits', split + '.txt'), 'r')
        self.list_examples_lidar = np.array(
                                        list_examples_file.read().splitlines())
        list_examples_file.close()

    def __len__(self):
        return len(self.list_examples_lidar)

    def __getitem__(self, idx):
        lidar_path = os.path.join(self.dataroot, self.list_examples_lidar[idx])
        rgb_path = lidar_path.replace('lidar_blank', 'rgb')
        anno_path = lidar_path.replace('lidar_blank', 'annotation_gray')

        rgb_name = rgb_path.split('/')[-1].split('.')[0]
        ann_name = anno_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)
        assert (ann_name == lidar_name)

        top_crop_class = IseAutoTopCrop(rgb_path, anno_path, lidar_path)
        # Apply top crop for raw data to crop the 1/2 top of the images
        top_crop_rgb, top_crop_anno, top_crop_lidar = top_crop_class.top_crop()

        augment_class = IseautoAugmentShuffle(top_crop_rgb, top_crop_anno,
                                              top_crop_lidar)
        if self.augment is not None:
            if self.augment_shuffle is True:
                aug_list = ['random_crop', 'random_rotate', 'colour_jitter',
                            'random_horizontal_flip', 'random_vertical_flip']
                random.shuffle(aug_list)

                for i in range(len(aug_list)):
                    augment_proc = getattr(augment_class, aug_list[i])
                    rgb, anno, lidar = augment_proc()

            else:  # self.augment_shuffle is False
                rgb, anno, lidar = augment_class.random_crop()
                rgb, anno, lidar = augment_class.random_rotate()
                rgb, anno, lidar = augment_class.colour_jitter()
                rgb, anno, lidar = augment_class.random_horizontal_flip()
                rgb, anno, lidar = augment_class.random_vertical_flip()

        else:  # slef.augment is None, all input data are only top cropped
            rgb = top_crop_rgb
            anno = top_crop_anno
            lidar = top_crop_lidar

        lidar = self.lidar_normalize(TF.to_tensor(np.array(lidar))[0:3])
        rgb_copy = TF.to_tensor(np.array(rgb.copy()))[0:3]
        rgb = self.rgb_normalize(TF.to_tensor(np.array(rgb))[0:3])
        annotation = \
            TF.to_tensor(np.array(anno)).type(torch.LongTensor).squeeze(0)

        return {'rgb': rgb, 'rgb_orig': rgb_copy,
                'lidar': lidar, 'annotation': annotation}
