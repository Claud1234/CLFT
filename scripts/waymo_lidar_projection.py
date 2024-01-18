#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
This script will create an example of Waymo LiDAR points projection.

Read single camera and LiDAR input. Based on the 'camera_coord' of to find the
corresponding pixel in image. Change the pixel color based on the Z of point.

Created on Jan 25th, 2022
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(''))
from utils.lidar_process import open_lidar

# For Waymo example
points_set, camera_coord = open_lidar(
     '../test_images/test_1_lidar.pkl',
     4, 4, lidar_mean=[0, 0, 0], lidar_std=[1, 1, 1])

# points_set, camera_coord = open_lidar(
#    '/home/claude/Data/claude_iseauto/labeled/day_fair/pkl/sq11_000458.pkl',
#    1, 1, lidar_mean=[0, 0, 0], lidar_std=[1, 1, 1])

# For Waymo example
rgb = cv2.imread('../test_images/test_1_img.png')

# For iseAuto example
# rgb = cv2.imread('/home/claude/Data/claude_iseauto/labeled/day_fair/rgb/sq11_000458.png')

cmap = plt.cm.get_cmap('hsv', 256)
cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

for i in range(camera_coord.shape[0]):
    depth = points_set[i, 2]
    color = cmap[int(1280.0 / depth), :]
# For Waymo example
    rgb[camera_coord[i, 1], camera_coord[i, 0]] = color

# For iseAuto example, this is simple upsampling, otherwise points too sparse
# in 4k image.
#    cv2.circle(rgb, (int(np.round(camera_coord[i, 0])),
#                     int(np.round(camera_coord[i, 1]))),
#               2, color=tuple(color), thickness=5)

cv2.imwrite('./result.png', rgb)

while True:
    cv2.imshow("result.jpg", rgb)
    cv2.waitKey(30)
    if cv2.getWindowProperty("result.jpg", cv2.WND_PROP_VISIBLE) <= 0:
        break
