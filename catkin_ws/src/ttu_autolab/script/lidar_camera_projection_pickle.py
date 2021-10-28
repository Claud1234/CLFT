#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to project lidar points to images.
Extract 3d_points and camera_inds, save as .pkl file

Created on Oct 21st, 2021
'''
import os
import sys
import re
import pickle
import numpy as np

rgx = re.compile('^(.*)_([0-9]+)')


def lidar_projection(png_in_dir, bin_in_dir, pkl_out_path, calib_file_path):
    dir_out_pkl = pkl_out_path
    if not os.path.exists(dir_out_pkl):
        os.mkdir(dir_out_pkl)

    png_file_list = np.array(os.listdir(png_in_dir))
#    bin_list = np.array(os.listdir(bin_in_dir))
    calib = read_calib_file(calib_file_path)

    W, H, C = 4240, 2824, 3

    for _, j in enumerate(png_file_list):
        # This is for selecting previous lidar frame to match current camera
        # frame for better point-image alignment.
        print(j)
        fname = rgx.match(j).groups()[0]
        fidx = rgx.match(j).groups()[1]
        fname_new = '%s_%06d' % (fname, int(fidx)-1)  # -1 frame
        pts_lidar = load_lidar_bin(os.path.join(bin_in_dir,
                                                fname_new + '.bin'))[:, :3]

        pickle_fname = os.path.join(dir_out_pkl, ('%s_%s.pkl' % (fname, fidx)))

        render_lidar_on_image(pts_lidar, (H, W, C), calib, pickle_fname)


def render_lidar_on_image(pts_lidar, img_shape, calib, pickle_fname):
    img_h, img_w, _ = img_shape

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_lidar.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_w) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_h) & (pts_2d[1, :] >= 0) &
                    (pts_lidar[:, 0] > 0)
                    )[0]
    front_view_labels = np.zeros(len(pts_2d.T))
    front_view_labels[inds] = 1

    waymo_struct = {}
    waymo_struct['camera_coordinates'] = \
        np.vstack([front_view_labels, pts_2d]).T.astype(np.uint16)
    waymo_struct['3d_points'] = pts_lidar[:, :3]

    with open(pickle_fname, 'wb') as _fd:
        pickle.dump(waymo_struct, _fd)


def project_velo_to_cam2(calib):
    P_velo2cam_ref = np.vstack((calib['Tr_velo_to_cam'].reshape(3, 4),
                                np.array([0., 0., 0., 1.])))  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = calib['R0_rect'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = calib['P2'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_velo2cam_ref
    return proj_mat


def load_lidar_bin(bin_filename):
    pts = np.fromfile(bin_filename, dtype=np.float32)
    pts = pts.reshape((-1, 4))
    return pts


def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0:
                continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def project_to_image(points, proj_mat):
    """
    Apply the perspective projection
    Args:
        pts_3d:     3D points in camera coordinate [3, npoints]
        proj_mat:   Projection matrix [3, 4]
    """
    num_pts = points.shape[1]

    # Change to homogenous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Input .png .bin .pkl directories and calib file path required!')
        sys.exit(1)
    lidar_projection(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
