#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
sys.path.append('.')

import configs
from utils.data_augment import TopCrop
from utils.data_augment import AugmentShuffle

image_path = configs.TEST_IMAGE
anno_path = configs.TEST_ANNO
lidar_path = configs.TEST_LIDAR

top_crop_class = TopCrop(image_path, anno_path, lidar_path)
top_crop_rgb, top_crop_anno, \
    top_crop_points_set, top_crop_camera_coord = top_crop_class.top_crop()

augment_class = AugmentShuffle(top_crop_rgb, top_crop_anno,
                               top_crop_points_set, top_crop_camera_coord)

aug_1 = 'random_crop'
aug_2 = 'random_rotate'
aug_3 = 'random_horizontal_flip'
aug_4 = 'random_vertical_flip'
aug_5 = 'colour_jitter'

aug_list = [aug_4, aug_3, aug_2]

for i in range(len(aug_list)):
    augment_proc = getattr(augment_class, aug_list[i])
    rgb, anno, X, Y, Z = augment_proc()

X = np.array(X)
Y = np.array(Y)
Z = np.array(Z)

if not os.path.exists('test/augment_shuffle_images'):
    os.mkdir('test/augment_shuffle_images')
os.chdir('test/augment_shuffle_images')

plt.subplot(2, 2, 1), plt.imshow(rgb), plt.title('rgb')
plt.subplot(2, 2, 2), plt.imshow(X), plt.title('X')
plt.subplot(2, 2, 3), plt.imshow(Y), plt.title('Y')
plt.subplot(2, 2, 4), plt.imshow(Z), plt.title('Z')
plt.savefig('test.png')
