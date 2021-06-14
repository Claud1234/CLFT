#! /usr/bin/env python3
# -*- coding: utf-8 -*-
'''
RGB, annoation and lidar augmentation operations

Created on May 13rd, 2021
'''
import numpy as np
from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils.lidar_process import crop_pointcloud
from utils.lidar_process import open_lidar
from utils.lidar_process import get_lid_images_val


class ImageProcess(object):
    def __init__(self, image_path, annotation_path, lidar_path):
        self.image_path = image_path
        self.annotation_path = annotation_path
        self.lidar_path = lidar_path

    '''
    Cut the top part of the image and lidar, applied before to all of
    augmentation operations
    '''
    def top_crop(self):
        rgb = Image.open(self.image_path)
        annotation = Image.open(self.annotation_path).convert('F')
        annotation = self.prepare_annotation(annotation)
        points_set, camera_coord = open_lidar(self.lidar_path)

        w_orig, h_orig = rgb.size
        delta = int(h_orig / 2)
        rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)
        annotation = TF.crop(annotation, delta, 0, h_orig-delta, w_orig)
        points_set, camera_coord, _ = crop_pointcloud(points_set,
                                                      camera_coord,
                                                      delta, 0,
                                                      h_orig-delta, w_orig)
        return rgb, annotation, points_set, camera_coord

    def square_crop(self):
        top_crop_rgb, top_crop_anno,\
            top_crop_points_set, top_crop_camera_coord = self.top_crop()

        _, h_top_crop = top_crop_rgb.size
        i0, j0, h_sq_crop, w_sq_crop = transforms.RandomCrop.get_params(
                                                                top_crop_rgb,
                                                                (h_top_crop,
                                                                 h_top_crop))
        square_crop_rgb = TF.crop(top_crop_rgb, i0, j0, h_sq_crop, w_sq_crop)
        square_crop_anno = TF.crop(top_crop_anno, i0, j0, h_sq_crop, w_sq_crop)
        square_crop_points_set,\
            square_crop_camera_coord, _ = crop_pointcloud(
                                                        top_crop_points_set,
                                                        top_crop_camera_coord,
                                                        i0, j0,
                                                        h_sq_crop, w_sq_crop)
        X, Y, Z = get_lid_images_val(h_sq_crop, w_sq_crop,
                                     square_crop_points_set,
                                     square_crop_camera_coord)

        return square_crop_rgb, square_crop_anno, X, Y, Z


    # def random_crop(self):
        # i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(
                                # rgb, scale=(0.2, 1.), ratio=(3. / 4., 4. / 3.))
        # rgb = TF.resized_crop(
            # rgb, i1, j1, h1, w1, (self.crop_size, self.crop_size), 
            # Image.BILINEAR)
        # annotation = TF.resized_crop(
            # annotation, i1, j1, h1, w1, (self.crop_size,self.crop_size), 
            # Image.NEAREST)
        # points_set, camera_coord, _ = self.crop_pointcloud(
            # points_set, camera_coord, i1, j1, h1, w1)
        # X,Y,Z = self.get_lid_images(h, w, points_set, camera_coord) 
        #
        # rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
        # rgb = self.normalize(to_tensor(np.array(rgb))[0:3]) 

    # def random_rotate(self):
        # if random.random() > 0.5 and self.rot_augment:
            # angle = -self.rot_range + 2*self.rot_range*torch.rand(1)[0]
            # rgb = TF.affine(rgb, angle, (0,0), 1, 0, 
                            # interpolation=Image.BILINEAR, fill=0)                        
            # annotation = TF.affine(annotation, angle, (0,0), 1, 0, 
                # interpolation=Image.NEAREST, fill=0)                        
            # X = TF.affine(
                # X, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)                        
            # Y = TF.affine(
                # Y, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)                        
            # Z = TF.affine(
                # Z, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)

    # def random_colour_jiter(self):
        # rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
        # rgb = self.normalize(to_tensor(self.colorjitter(rgb))[0:3])#only rgb

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

#     def get_lid_images(self, h, w, points_set, camera_coord):
#         X = np.zeros((self.crop_size, self.crop_size))
#         Y = np.zeros((self.crop_size, self.crop_size))
#         Z = np.zeros((self.crop_size, self.crop_size))
#
#         rows = np.floor(camera_coord[:, 1]*self.crop_size/h)
#         cols = np.floor(camera_coord[:, 0]*self.crop_size/w)
#
#         X[(rows.astype(int), cols.astype(int))] = points_set[:, 0]
#         Y[(rows.astype(int), cols.astype(int))] = points_set[:, 1]
#         Z[(rows.astype(int), cols.astype(int))] = points_set[:, 2]

#         X = TF.to_pil_image(X.astype(np.float32))
#         Y = TF.to_pil_image(Y.astype(np.float32))
#         Z = TF.to_pil_image(Z.astype(np.float32))

#         return X, Y, Z


#####################################################################
# ##functions for data augmentation and normalization
# self.colorjitter = transforms.ColorJitter(brightness=0.4, contrast=0.4,
#                                                   saturation=0.4, hue=0.1)
# #####################################################################
