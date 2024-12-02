#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
lidar open and crop and value reading operations

Created on May 13rd, 2021
'''
import pickle
import numpy as np
import torchvision.transforms.v2.functional as TF


def open_lidar(lidar_path, w_ratio, h_ratio, lidar_mean, lidar_std):
    mean_lidar = np.array(lidar_mean)
    std_lidar = np.array(lidar_std)

    file = open(lidar_path, 'rb')
    lidar_data = pickle.load(file)
    file.close()

    points3d = lidar_data['3d_points']
    camera_coord = lidar_data['camera_coordinates']

    # select camera front
    mask = camera_coord[:, 0] == 1
    points3d = points3d[mask, :]
    camera_coord = camera_coord[mask, 1:3]

    x_lid = (points3d[:, 1] - mean_lidar[0])/std_lidar[0]
    y_lid = (points3d[:, 2] - mean_lidar[1])/std_lidar[1]
    z_lid = (points3d[:, 0] - mean_lidar[2])/std_lidar[2]

    camera_coord[:, 1] = (camera_coord[:, 1]/h_ratio).astype(int)
    camera_coord[:, 0] = (camera_coord[:, 0]/w_ratio).astype(int)

    points_set = np.stack((x_lid, y_lid, z_lid), axis=1).astype(np.float32)

    return points_set, camera_coord


def crop_pointcloud(points_set_or, camera_coord_or, i, j, h, w):
    points_set = np.copy(points_set_or)
    camera_coord = np.copy(camera_coord_or)

    camera_coord[:, 1] -= i
    camera_coord[:, 0] -= j
    selected_i = np.logical_and(
                        camera_coord[:, 1] >= 0, camera_coord[:, 1] < h)
    selected_j = np.logical_and(
                        camera_coord[:, 0] >= 0, camera_coord[:, 0] < w)
    selected = np.logical_and(selected_i, selected_j)
    points_set = points_set[selected, :]
    camera_coord = camera_coord[selected, :]

    return points_set, camera_coord, selected


def get_unresized_lid_img_val(h, w, points_set, camera_coord):
    X = np.zeros((h, w))
    Y = np.zeros((h, w))
    Z = np.zeros((h, w))

    rows = np.floor(camera_coord[:, 1])
    cols = np.floor(camera_coord[:, 0])

    X[(rows.astype(int), cols.astype(int))] = points_set[:, 0]
    Y[(rows.astype(int), cols.astype(int))] = points_set[:, 1]
    Z[(rows.astype(int), cols.astype(int))] = points_set[:, 2]

    X = TF.to_pil_image(X.astype(np.float32))
    Y = TF.to_pil_image(Y.astype(np.float32))
    Z = TF.to_pil_image(Z.astype(np.float32))

    return X, Y, Z


def get_resized_lid_img_val(crop_size, h, w, points_set, camera_coord):
    X = np.zeros((crop_size, crop_size))
    Y = np.zeros((crop_size, crop_size))
    Z = np.zeros((crop_size, crop_size))

    rows = np.floor(camera_coord[:, 1]*crop_size/h)
    cols = np.floor(camera_coord[:, 0]*crop_size/w)

    X[(rows.astype(int), cols.astype(int))] = points_set[:, 0]
    Y[(rows.astype(int), cols.astype(int))] = points_set[:, 1]
    Z[(rows.astype(int), cols.astype(int))] = points_set[:, 2]

    X = TF.to_pil_image(X.astype(np.float32))
    Y = TF.to_pil_image(Y.astype(np.float32))
    Z = TF.to_pil_image(Z.astype(np.float32))

    return X, Y, Z
