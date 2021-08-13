#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RGB, annoation and lidar augmentation operations

Created on May 13rd, 2021
'''
import torch
import random
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

import configs
from utils.lidar_process import get_resized_lid_img_val
from utils.lidar_process import get_unresized_lid_img_val
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import open_lidar


class TopCrop(object):
    def __init__(self, image_path, annotation_path, lidar_path):
        # self.image_path = image_path
        # self.annotation_path = annotation_path
        # self.lidar_path = lidar_path
        self.rgb = Image.open(image_path)
        self.annotation = Image.open(annotation_path).convert('F')
        self.annotation = self.prepare_annotation(self.annotation)
        self.points_set, self.camera_coord = open_lidar(lidar_path)

    '''
    Cut the 1/2 top part of the image and lidar, applied before to all of
    augmentation operations
    '''
    def top_crop(self):
        w_orig, h_orig = self.rgb.size
        delta = int(h_orig / 2)
        rgb = TF.crop(self.rgb, delta, 0, h_orig-delta, w_orig)
        annotation = TF.crop(self.annotation, delta, 0, h_orig-delta, w_orig)
        points_set, camera_coord, _ = crop_pointcloud(self.points_set,
                                                      self.camera_coord,
                                                      delta, 0,
                                                      h_orig-delta, w_orig)
        return rgb, annotation, points_set, camera_coord

    def prepare_annotation(self, annotation):
        '''
        Reassign the indices of the objects in annotation(PointCloud);
        :parameter annotation: 0->ignore 1->vehicle, 2->pedestrian, 3->sign,
                                4->cyclist, 5->background
        :return annotation: 0 -> background+sign, 1->vehicle
                                2->pedestrian+cyclist, 3->ignore
        '''
        annotation = np.array(annotation)

        mask_ignore = annotation == 0
        mask_sign = annotation == 3
        mask_cyclist = annotation == 4
        mask_background = annotation == 5

        annotation[mask_sign] = 0
        annotation[mask_background] = 0
        annotation[mask_cyclist] = 2
        annotation[mask_ignore] = 3

        return TF.to_pil_image(annotation)


class AugmentShuffle(object):
    def __init__(self, rgb, anno, points_set, camera_coord):
        self.rgb = rgb
        self.anno = anno
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
        self.anno = TF.resized_crop(self.anno, i, j,
                                    self.h_resize, self.w_resize,
                                    (crop_size, crop_size),
                                    InterpolationMode.NEAREST)

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

        return self.rgb, self.anno, self.X, self.Y, self.Z

    def random_rotate(self):
        rotate_range = configs.ROTATE_RANGE
        w, h = self.rgb.size
        angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
        self.rgb = TF.affine(self.rgb, angle, (0, 0), 1, 0,
                             InterpolationMode.BILINEAR, fill=0)
        self.anno = TF.affine(self.anno, angle, (0, 0), 1, 0,
                              InterpolationMode.NEAREST, fill=0)

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
        return self.rgb, self.anno, self.X, self.Y, self.Z

    def colour_jitter(self):
        jitter_param = configs.JITTER_PARAM
        rgb_colour_jitter = transforms.ColorJitter(jitter_param[0],
                                                   jitter_param[1],
                                                   jitter_param[2],
                                                   jitter_param[3])
        self.rgb = rgb_colour_jitter(self.rgb)
        return self.rgb, self.anno, self.X, self.Y, self.Z

    def random_horizontal_flip(self):
        if random.random() > 0.5:
            w, h = self.rgb.size
            self.rgb = TF.hflip(self.rgb)
            self.anno = TF.hflip(self.anno)
            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(
                                                            h, w,
                                                            self.points_set,
                                                            self.camera_coord)
            self.X = TF.hflip(self.X)
            self.Y = TF.hflip(self.Y)
            self.Z = TF.hflip(self.Z)
        return self.rgb, self.anno, self.X, self.Y, self.Z

    def random_vertical_flip(self):
        if random.random() > 0.5:
            w, h = self.rgb.size
            self.rgb = TF.vflip(self.rgb)
            self.anno = TF.vflip(self.anno)

            if self.X is None and self.Y is None and self.Z is None:
                self.X, self.Y, self.Z = get_unresized_lid_img_val(
                                                            h, w,
                                                            self.points_set,
                                                            self.camera_coord)
            self.X = TF.vflip(self.X)
            self.Y = TF.vflip(self.Y)
            self.Z = TF.vflip(self.Z)
        return self.rgb, self.anno, self.X, self.Y, self.Z
