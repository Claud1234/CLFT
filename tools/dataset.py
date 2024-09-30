#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataloader python script

Created on May 13rd, 2021

TODO: get the iseauto dataset dataloder part done.
"""
import os
import sys
import cv2
import random
import numpy as np
from glob import glob
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.helpers import waymo_anno_class_relabel
from utils.helpers import waymo_anno_class_relabel_1
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_unresized_lid_img_val
from utils.data_augment import DataAugment


def lidar_dilation(X, Y, Z):
    kernel = np.ones((3, 3), np.uint8)
    X_dilation = cv2.dilate(np.array(X).astype(np.float32), kernel,
                            iterations=1)
    Y_dilation = cv2.dilate(np.array(Y).astype(np.float32), kernel,
                            iterations=1)
    Z_dilation = cv2.dilate(np.array(Z).astype(np.float32), kernel,
                            iterations=1)

    X_dilation = TF.to_pil_image(X_dilation.astype(np.float32))
    Y_dilation = TF.to_pil_image(Y_dilation.astype(np.float32))
    Z_dilation = TF.to_pil_image(Z_dilation.astype(np.float32))
    return X_dilation, Y_dilation, Z_dilation


class Dataset(object):
    def __init__(self, config, split=None, path=None):
        np.random.seed(789)
        self.config = config

        list_examples_file = open(path, 'r')
        self.list_examples_cam = np.array(list_examples_file.read().splitlines())
        list_examples_file.close()

        if split == 'train':  # only augment for training.
            self.p_flip = config['Dataset']['transforms']['p_flip']
            self.p_crop = config['Dataset']['transforms']['p_crop']
            self.p_rot = config['Dataset']['transforms']['p_rot']
        else:
            self.p_flip = 0
            self.p_crop = 0
            self.p_rot = 0

        self.img_size = config['Dataset']['transforms']['resize']
        self.rgb_normalize = transforms.Compose([transforms.Resize((self.img_size, self.img_size),
                                                interpolation=transforms.InterpolationMode.BILINEAR),
                                                transforms.ToTensor(),
                                                transforms.Normalize(
                                                mean=config['Dataset']['transforms']['image_mean'],
                                                std=config['Dataset']['transforms']['image_mean'])])

        self.anno_resize = transforms.Resize((self.img_size, self.img_size),
                                             interpolation=transforms.InterpolationMode.NEAREST)

    def __len__(self):
        return len(self.list_examples_cam)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        dataroot = './waymo_dataset/'

        if self.config['Dataset']['name'] == 'waymo':
            cam_path = os.path.join(dataroot, self.list_examples_cam[idx])
            anno_path = cam_path.replace('/camera', '/annotation')
            lidar_path = cam_path.replace('/camera', '/lidar').replace('.png', '.pkl')

            # waymo rgb and anno is in 480x320, lidar is in 1920x1280
            rgb = Image.open(cam_path).convert('RGB')

            # Here there are two class relabel functions, go to /utils/helper.py for details.
            anno = waymo_anno_class_relabel(Image.open(anno_path))  # Tensor [1, H, W]
            points_set, camera_coord = open_lidar(lidar_path, w_ratio=4, h_ratio=4,
                                                  lidar_mean=self.config['Dataset']['transforms']['lidar_mean_waymo'],
                                                  lidar_std=self.config['Dataset']['transforms']['lidar_mean_waymo'])

        elif self.config['Dataset']['name'] == 'iseauto':
            cam_path = os.path.join(dataroot, self.list_examples_cam[idx])
            anno_path = cam_path.replace('/rgb', '/annotation_gray')
            lidar_path = cam_path.replace('/rgb', '/pkl').replace('.png', '.pkl')

            rgb = Image.open(cam_path).resize((480, 320), Image.BILINEAR)
            anno = Image.open(anno_path).resize((480, 320), Image.BILINEAR)
            anno = torch.from_numpy(np.array(anno)).unsqueeze(0).long()
            points_set, camera_coord = open_lidar(lidar_path, w_ratio=8.84, h_ratio=8.825,
                                                  lidar_mean=self.config['Dataset']['transforms']['lidar_mean_iseauto'],
                                                  lidar_std=self.config['Dataset']['transforms']['lidar_mean_iseauto'])

        else:
            sys.exit("['Dataset']['name'] must be specified waymo or iseauto")

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == anno_name), "rgb and anno input not matching"
        assert (rgb_name == lidar_name), "rgb and lidar input not matching"

        # Crop the top part 1/2 of the input data
        rgb_orig = rgb.copy()
        w_orig, h_orig = rgb.size  # PIL tuple. (w, h)
        delta = int(h_orig/2)
        top_crop_rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)  # w,h
        top_crop_anno = TF.crop(anno, delta, 0, h_orig-delta, w_orig)

        top_crop_points_set, top_crop_camera_coord, _ = crop_pointcloud(
            points_set, camera_coord, delta, 0, h_orig-delta, w_orig)

        data_augment = DataAugment(self.config, self.p_flip, self.p_crop,
                                   self.p_rot, top_crop_rgb, top_crop_anno,
                                   top_crop_points_set, top_crop_camera_coord)

        aug_list = ['random_horizontal_flip', 'random_crop', 'random_rotate']
        for i in range(len(aug_list)):
            augment_proc = getattr(data_augment, aug_list[i])
            rgb, anno, X, Y, Z = augment_proc()

        if X is None or Y is None or Z is None:
            # either not in train mode or no augment was executed.
            rgb = top_crop_rgb
            anno = top_crop_anno
            points_set = top_crop_points_set
            camera_coord = top_crop_camera_coord

            w, h = rgb.size
            X, Y, Z = get_unresized_lid_img_val(h, w, points_set, camera_coord)

        X, Y, Z = lidar_dilation(X, Y, Z)
        rgb = self.rgb_normalize(rgb)  # Tensor [3, 384, 384]
        anno = self.anno_resize(anno).squeeze(0)  # Tensor

        rgb_orig = transforms.ToTensor()(rgb_orig)
        X = transforms.Resize((self.img_size, self.img_size))(X)
        Y = transforms.Resize((self.img_size, self.img_size))(Y)
        Z = transforms.Resize((self.img_size, self.img_size))(Z)

        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))

        lid_images = torch.cat((X, Y, Z), 0)

        return {'rgb': rgb, 'rgb_orig': rgb_orig,
                'lidar': lid_images, 'anno': anno}


