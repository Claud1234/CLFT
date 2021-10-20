#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RGB, annoation and lidar augmentation operations for iseauto dataset

Created on Oct 11st, 2021
'''
import torch
import random
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

import configs


class IseAutoTopCrop(object):
    def __init__(self, image_path, annotation_path, lidar_path):
        self.rgb = Image.open(image_path).resize((480, 320), Image.BICUBIC)
        self.annotation = Image.open(annotation_path).\
            resize((480, 320), Image.BICUBIC).convert('F')
        self.lidar = Image.open(lidar_path).resize((480, 320), Image.BICUBIC)

    '''
    Cut the 1/2 top part of the image and lidar, applied before to all of
    augmentation operations
    '''
    def top_crop(self):
        w_orig, h_orig = self.rgb.size
        delta = int(h_orig / 2)
        rgb = TF.crop(self.rgb, delta, 0, h_orig-delta, w_orig)
        annotation = TF.crop(self.annotation, delta, 0, h_orig-delta, w_orig)
        lidar = TF.crop(self.lidar, delta, 0, h_orig-delta, w_orig)
        return rgb, annotation, lidar


class IseautoAugmentShuffle(object):
    def __init__(self, rgb, anno, lidar):
        self.rgb = rgb
        self.anno = anno
        self.lidar = lidar

    def random_crop(self):
        crop_size = configs.RANDOM_CROP_SIZE
        i, j, self.h_resize, self.w_resize = \
            transforms.RandomResizedCrop.get_params(self.rgb, scale=(0.2, 1.),
                                                    ratio=(3./4., 4./3.))
        self.rgb = TF.resized_crop(self.rgb, i, j,
                                   self.h_resize, self.w_resize,
                                   (crop_size, crop_size),
                                   InterpolationMode.BILINEAR)
        self.anno = TF.resized_crop(self.anno, i, j,
                                    self.h_resize, self.w_resize,
                                    (crop_size, crop_size),
                                    InterpolationMode.NEAREST)
        self.lidar = TF.resized_crop(self.lidar, i, j,
                                     self.h_resize, self.w_resize,
                                     (crop_size, crop_size),
                                     InterpolationMode.NEAREST)
        return self.rgb, self.anno, self.lidar

    def random_rotate(self):
        rotate_range = configs.ROTATE_RANGE
        angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
        self.rgb = TF.affine(self.rgb, angle, (0, 0), 1, 0,
                             InterpolationMode.BILINEAR, fill=0)
        self.anno = TF.affine(self.anno, angle, (0, 0), 1, 0,
                              InterpolationMode.NEAREST, fill=0)

        self.lidar = TF.affine(self.lidar, angle, (0, 0), 1, 0,
                               InterpolationMode.NEAREST, fill=0)
        return self.rgb, self.anno, self.lidar

    def colour_jitter(self):
        jitter_param = configs.JITTER_PARAM
        rgb_colour_jitter = transforms.ColorJitter(jitter_param[0],
                                                   jitter_param[1],
                                                   jitter_param[2],
                                                   jitter_param[3])
        lidar_colour_jitter = transforms.ColorJitter(jitter_param[0],
                                                     jitter_param[1],
                                                     jitter_param[2],
                                                     jitter_param[3])
        self.rgb = rgb_colour_jitter(self.rgb)
        self.ldiar = lidar_colour_jitter(self.lidar)
        return self.rgb, self.anno, self.lidar

    def random_horizontal_flip(self):
        if random.random() > 0.5:
            self.rgb = TF.hflip(self.rgb)
            self.anno = TF.hflip(self.anno)
            self.lidar = TF.hflip(self.lidar)
        return self.rgb, self.anno, self.lidar

    def random_vertical_flip(self):
        if random.random() > 0.5:
            self.rgb = TF.vflip(self.rgb)
            self.anno = TF.vflip(self.anno)
            self.lidar = TF.vflip(self.lidar)
        return self.rgb, self.anno, self.lidar
