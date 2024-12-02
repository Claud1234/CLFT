#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RGB, and lidar augmentation operations.
This is only for unlabeled iseAuto splits that have no man-made annotations.

Created on Feb. 5th, 2022
'''
import torch
import random
from PIL import Image

import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2.functional import InterpolationMode

from utils.lidar_process import get_resized_lid_img_val
from utils.lidar_process import get_unresized_lid_img_val
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import open_lidar


class TopCropSemi(object):
    def __init__(self, image_path, lidar_path):
        self.rgb = Image.open(image_path).resize((480, 320), Image.BICUBIC)
        self.points_set, self.camera_coord = open_lidar(
                                        lidar_path,
                                        w_ratio=8.84,
                                        h_ratio=8.825,
                                        lidar_mean=configs.ISE_LIDAR_MEAN,
                                        lidar_std=configs.ISE_LIDAR_STD)
    '''
    Cut the 1/2 top part of the image and lidar, applied before to all of
    augmentation operations
    '''
    def top_crop(self):
        w_orig, h_orig = self.rgb.size
        delta = int(h_orig / 2)
        rgb = TF.crop(self.rgb, delta, 0, h_orig-delta, w_orig)
        points_set, camera_coord, _ = crop_pointcloud(self.points_set,
                                                      self.camera_coord,
                                                      delta, 0,
                                                      h_orig-delta, w_orig)
        return rgb, points_set, camera_coord


class AugmentShuffleSemi(object):
    def __init__(self, rgb, points_set, camera_coord):
        self.rgb = rgb
        self.points_set = points_set
        self.camera_coord = camera_coord
        self.h_resize, self.w_resize = None, None
        self.X, self.Y, self.Z = None, None, None

    def random_crop(self):
        crop_size = configs.RANDOM_CROP_SIZE
        i, j, self.h_resize, self.w_resize = \
            transforms.RandomResizedCrop.get_params(self.rgb, scale=(0.2, 1.),
                                                    ratio=(3./4., 4./3.))
        self.rgb = TF.resized_crop(self.rgb, i, j,
                                   self.h_resize, self.w_resize,
                                   (crop_size, crop_size),
                                   InterpolationMode.BILINEAR)

        if self.X is None and self.Y is None and self.Z is None:
            self.points_set, self.camera_coord, _ = crop_pointcloud(
                                                            self.points_set,
                                                            self.camera_coord,
                                                            i, j,
                                                            self.h_resize,
                                                            self.w_resize)
            self.X, self.Y, self.Z = get_resized_lid_img_val(self.h_resize,
                                                             self.w_resize,
                                                             self.points_set,
                                                             self.camera_coord)
        else:
            self.X = TF.resized_crop(self.X, i, j,
                                     self.h_resize, self.w_resize,
                                     (crop_size, crop_size),
                                     InterpolationMode.BILINEAR)
            self.Y = TF.resized_crop(self.Y, i, j,
                                     self.h_resize, self.w_resize,
                                     (crop_size, crop_size),
                                     InterpolationMode.BILINEAR)
            self.Z = TF.resized_crop(self.Z, i, j,
                                     self.h_resize, self.w_resize,
                                     (crop_size, crop_size),
                                     InterpolationMode.BILINEAR)

        return self.rgb, self.X, self.Y, self.Z

    def random_rotate(self):
        rotate_range = configs.ROTATE_RANGE
        w, h = self.rgb.size
        angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
        self.rgb = TF.affine(self.rgb, angle, (0, 0), 1, 0,
                             InterpolationMode.BILINEAR, fill=0)

        if self.X is None and self.Y is None and self.Z is None:
            self.X, self.Y, self.Z = get_unresized_lid_img_val(
                                                            h, w,
                                                            self.points_set,
                                                            self.camera_coord)

        self.X = TF.affine(self.X, angle, (0, 0), 1, 0,
                           InterpolationMode.NEAREST, fill=0)
        self.Y = TF.affine(self.Y, angle, (0, 0), 1, 0,
                           InterpolationMode.NEAREST, fill=0)
        self.Z = TF.affine(self.Z, angle, (0, 0), 1, 0,
                           InterpolationMode.NEAREST, fill=0)
        return self.rgb, self.X, self.Y, self.Z

    def colour_jitter(self):
        jitter_param = configs.JITTER_PARAM
        rgb_colour_jitter = transforms.ColorJitter(jitter_param[0],
                                                   jitter_param[1],
                                                   jitter_param[2],
                                                   jitter_param[3])
        self.rgb = rgb_colour_jitter(self.rgb)
        return self.rgb, self.X, self.Y, self.Z

    def random_horizontal_flip(self):
        if random.random() > 0.5:
            w, h = self.rgb.size
            self.rgb = TF.hflip(self.rgb)
            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(
                                                            h, w,
                                                            self.points_set,
                                                            self.camera_coord)
            self.X = TF.hflip(self.X)
            self.Y = TF.hflip(self.Y)
            self.Z = TF.hflip(self.Z)
        return self.rgb, self.X, self.Y, self.Z

    def random_vertical_flip(self):
        if random.random() > 0.5:
            w, h = self.rgb.size
            self.rgb = TF.vflip(self.rgb)

            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(
                                                            h, w,
                                                            self.points_set,
                                                            self.camera_coord)
            self.X = TF.vflip(self.X)
            self.Y = TF.vflip(self.Y)
            self.Z = TF.vflip(self.Z)
        return self.rgb, self.X, self.Y, self.Z
