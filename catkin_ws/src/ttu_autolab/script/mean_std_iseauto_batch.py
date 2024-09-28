#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to compute mean and std of one tools lidar .bin data.
Need bash commands to process the batch data.

Created on Oct 21st, 2021
'''
import sys
import os
import pickle
import numpy as np

__mean_dir = '/home/claude/Data/claude_iseauto/norm_data/mean_dayfair_2k_front_view'
__std_dir = '/home/claude/Data/claude_iseauto/norm_data/std_dayfair_2k_front_view'

if __name__ == '__main__':
    if not os.path.exists(__mean_dir):
        os.mkdir(__mean_dir)
    if not os.path.exists(__std_dir):
        os.mkdir(__std_dir)

    print(sys.argv[1])

    fname = os.path.basename(sys.argv[1])
#     fname_new = fname.replace('.bin', '.txt')
 
#     with open(sys.argv[1], 'rb') as _fd:
#         lidar = np.fromfile(_fd, np.float32).reshape(-1, 4)
#     l_mean = np.mean(lidar[:, :3], axis=0)
#     l_std = np.std(lidar[:, :3], axis=0)
 
#     with open(os.path.join(__mean_dir, fname_new), 'w') as _fd:
#         _fd.write('%f;%f;%f\n' % tuple(l_mean))

#     with open(os.path.join(__std_dir, fname_new), 'w') as _fd:
#         _fd.write('%f;%f;%f\n' % tuple(l_std))

    fname_new = fname.replace('.pkl', '.txt')

    with open(os.path.join('/media/claude/256G', sys.argv[1]), 'rb') as _fd:
        lidar_data = pickle.load(_fd)

    points3d = lidar_data['3d_points']
    camera_coord = lidar_data['camera_coordinates']

    # select camera front
    mask = camera_coord[:, 0] == 1
    points3d = points3d[mask, :]
    camera_coord = camera_coord[mask, 1:3]

    l_mean = np.mean(points3d[:, :3], axis=0)
    l_std = np.std(points3d[:, :3], axis=0)

    with open(os.path.join(__mean_dir, fname_new), 'w') as _fd:
        _fd.write('%f;%f;%f\n' % tuple(l_mean))

    with open(os.path.join(__std_dir, fname_new), 'w') as _fd:
        _fd.write('%f;%f;%f\n' % tuple(l_std))