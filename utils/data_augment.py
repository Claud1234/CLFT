#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RGB, annoation and lidar augmentation operations

Created on May 13rd, 2021
'''
import torch
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import configs
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import open_lidar


class ImageProcess(object):
    def __init__(self, image_path, annotation_path, lidar_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.lidar_path = lidar_path

    '''
    Cut the 1/3 top part of the image and lidar, applied before to all of
    augmentation operations
    '''
    def top_crop(self):
        rgb = Image.open(self.image_path)
        annotation = Image.open(self.annotation_path).convert('F')
        annotation = self.prepare_annotation(annotation)
        points_set, camera_coord = open_lidar(self.lidar_path)

        w_orig, h_orig = rgb.size
        delta = int(h_orig / 3)
        rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)
        annotation = TF.crop(annotation, delta, 0, h_orig-delta, w_orig)
        points_set, camera_coord, _ = crop_pointcloud(points_set,
                                                      camera_coord,
                                                      delta, 0,
                                                      h_orig-delta, w_orig)
        return rgb, annotation, points_set, camera_coord

    def square_crop(self, rgb, anno, points_set, camera_coord):
        _, h = rgb.size
        i0, j0, h_sq_crop, w_sq_crop = transforms.RandomCrop.get_params(rgb,
                                                                        (h, h))
        square_crop_rgb = TF.crop(rgb, i0, j0, h_sq_crop, w_sq_crop)
        square_crop_anno = TF.crop(anno, i0, j0, h_sq_crop, w_sq_crop)
        square_crop_points_set,\
            square_crop_camera_coord, _ = crop_pointcloud(points_set,
                                                          camera_coord,
                                                          i0, j0,
                                                          h_sq_crop, w_sq_crop)
        return w_sq_crop, h_sq_crop, square_crop_rgb, square_crop_anno, \
            square_crop_points_set, square_crop_camera_coord

    def random_crop(self, rgb, anno, points_set, camera_coord):
        crop_size = configs.RANDOM_CROP_SIZE
        i1, j1, h_rand_crop, w_rand_crop = \
            transforms.RandomResizedCrop.get_params(rgb,
                                                    scale=(0.2, 1.),
                                                    ratio=(3./4., 4./3.))
        random_crop_rgb = TF.resized_crop(rgb, i1, j1,
                                          h_rand_crop, w_rand_crop,
                                          (crop_size, crop_size),
                                          InterpolationMode.BILINEAR)
        random_crop_anno = TF.resized_crop(anno, i1, j1,
                                           h_rand_crop, w_rand_crop,
                                           (crop_size, crop_size),
                                           InterpolationMode.NEAREST)
        random_crop_points_set, \
            random_crop_camera_coord, _ = crop_pointcloud(points_set,
                                                          camera_coord,
                                                          i1, j1,
                                                          h_rand_crop,
                                                          w_rand_crop)
        return w_rand_crop, h_rand_crop, random_crop_rgb, random_crop_anno, \
            random_crop_points_set, random_crop_camera_coord

    def random_rotate(self, rgb, anno, X, Y, Z):
        rotate_range = configs.ROTATE_RANGE
        angle = (-rotate_range + 2 * rotate_range * torch.rand(1)[0]).item()
        rotate_rgb = TF.affine(rgb, angle, (0, 0), 1, 0,
                               InterpolationMode.BILINEAR, fill=0)
        rotate_anno = TF.affine(anno, angle, (0, 0), 1, 0,
                                interpolation=Image.NEAREST, fill=0)
        rotate_X = TF.affine(X, angle, (0, 0), 1, 0,
                             InterpolationMode.NEAREST, fill=0)
        rotate_Y = TF.affine(Y, angle, (0, 0), 1, 0,
                             InterpolationMode.NEAREST, fill=0)
        rotate_Z = TF.affine(Z, angle, (0, 0), 1, 0,
                             InterpolationMode.NEAREST, fill=0)
        return rotate_rgb, rotate_anno, rotate_X, rotate_Y, rotate_Z

    def colour_jitter(self, rgb):
        jitter_param = configs.JITTER_PARAM
        rgb_colour_jitter = transforms.ColorJitter(jitter_param[0],
                                                   jitter_param[1],
                                                   jitter_param[2],
                                                   jitter_param[3])
        jittered_rgb = rgb_colour_jitter(rgb)
        return jittered_rgb

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
