#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RGB, annoation and lidar augmentation operations

Created on May 13rd, 2021
"""
import torch
import random

import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF
from torchvision.transforms.v2.functional import InterpolationMode

from utils.lidar_process import get_resized_lid_img_val
from utils.lidar_process import get_unresized_lid_img_val
from utils.lidar_process import crop_pointcloud


class DataAugment(object):
    def __init__(self, config, p_flip, p_crop, p_rot, rgb, anno, points_set,
                camera_coord):
        self.config = config
        self.p_flip = p_flip
        self.p_crop = p_crop
        self.p_rot = p_rot
        self.rgb = rgb
        self.anno = anno
        self.points_set = points_set
        self.camera_coord = camera_coord
        self.img_size = config['Dataset']['transforms']['resize']
        self.X, self.Y, self.Z = None, None, None

    def random_horizontal_flip(self):
        if random.random() < self.p_flip:
            w, h = self.rgb.size
            self.rgb = TF.hflip(self.rgb)
            self.anno = TF.hflip(self.anno)
            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(
                    h, w, self.points_set, self.camera_coord)
            self.X = TF.hflip(self.X)
            self.Y = TF.hflip(self.Y)
            self.Z = TF.hflip(self.Z)

        return self.rgb, self.anno, self.X, self.Y, self.Z

    def random_crop(self):
        if random.random() < self.p_crop:
            random_size = random.randint(128, self.img_size - 1)
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                self.rgb, scale=[0.2, 1.], ratio=[3. / 4., 4. / 3.])
            self.rgb = TF.resized_crop(self.rgb, i, j, h, w, [random_size, random_size], InterpolationMode.BILINEAR)
            self.anno = TF.resized_crop(self.anno, i, j, h, w, [random_size, random_size], InterpolationMode.NEAREST)

            if self.X is None and self.Y is None and self.Z is None:
                self.points_set, self.camera_coord, _ = crop_pointcloud(
                    self.points_set, self.camera_coord, i, j, h,  w)
                self.X, self.Y, self.Z = get_resized_lid_img_val(
                    random_size, h, w, self.points_set, self.camera_coord)
            else:
                self.X = TF.resized_crop(self.X, i, j, h, w, [random_size, random_size], InterpolationMode.BILINEAR)
                self.Y = TF.resized_crop(self.Y,  i, j, h, w, [random_size, random_size], InterpolationMode.BILINEAR)
                self.Z = TF.resized_crop(self.Z, i, j, h, w, [random_size, random_size], InterpolationMode.BILINEAR)

        return self.rgb, self.anno, self.X, self.Y, self.Z

    def random_rotate(self):
        if random.random() < self.p_rot:
            rotate_range = self.config['Dataset']['transforms']['random_rotate_range']
            w, h = self.rgb.size
            angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
            self.rgb = TF.affine(self.rgb, angle, [0, 0], 1, 0, InterpolationMode.BILINEAR)
            self.anno = TF.affine(self.anno, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)

            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(h, w, self.points_set, self.camera_coord)

            self.X = TF.affine(self.X, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)
            self.Y = TF.affine(self.Y, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)
            self.Z = TF.affine(self.Z, angle, [0, 0], 1, 0, InterpolationMode.NEAREST)

        return self.rgb, self.anno, self.X, self.Y, self.Z

    # def colour_jitter(self):
    #     jitter_param = self.config['Dataset']['transforms']['color_jitter']
    #     rgb_colour_jitter = transforms.ColorJitter(jitter_param[0],
    #                                                jitter_param[1],
    #                                                jitter_param[2],
    #                                                jitter_param[3])
    #     self.rgb = rgb_colour_jitter(self.rgb)
    #     return self.rgb, self.anno, self.X, self.Y, self.Z

    # def random_vertical_flip(self):
    #     if random.random() > 0.5:
    #         w, h = self.rgb.size()[1:]
    #         self.rgb = TF.vflip(self.rgb)
    #         self.anno = TF.vflip(self.anno)
    #
    #         if self.X is None and self.Y is None and self.Z is None:
    #             self.X, self.Y, self.Z = get_unresized_lid_img_val(
    #                                                         h, w,
    #                                                         self.points_set,
    #                                                         self.camera_coord)
    #         self.X = TF.vflip(self.X)
    #         self.Y = TF.vflip(self.Y)
    #         self.Z = TF.vflip(self.Z)
    #     return self.rgb, self.anno, self.X, self.Y, self.Z
