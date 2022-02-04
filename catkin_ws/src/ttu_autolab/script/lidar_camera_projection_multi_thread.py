#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to project lidar points to rgb images, create lidar_image.
create lidar_image that background is rgb image
multi-threads process, fast.
'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from multiprocessing import cpu_count
from imutils import paths


def render_lidar_on_image(pts_lidar, rgb, calib):
    img_h, img_w, img_c = rgb.shape
    # blackblankimage = np.zeros(shape=[img_h, img_w, img_c], dtype=np.uint8)

    # projection matrix (project from velo2cam2)
    proj_velo2cam2 = project_velo_to_cam2(calib)

    # apply projection
    pts_2d = project_to_image(pts_lidar.transpose(), proj_velo2cam2)

    # Filter lidar points to be within image FOV
    inds = np.where((pts_2d[0, :] < img_w) & (pts_2d[0, :] >= 0) &
                    (pts_2d[1, :] < img_h) & (pts_2d[1, :] >= 0) &
                    (pts_lidar[:, 0] > 0)
                    )[0]

    # Filter out pixels points
    imgfov_pc_pixel = pts_2d[:, inds]

    # Retrieve depth from lidar
    imgfov_pc_lidar = pts_lidar[inds, :]
    imgfov_pc_lidar = np.hstack((imgfov_pc_lidar,
                                 np.ones((imgfov_pc_lidar.shape[0], 1))))
    imgfov_pc_cam2 = proj_velo2cam2 @ imgfov_pc_lidar.transpose()

    cmap = plt.cm.get_cmap('hsv', 2048)
    cmap = np.array([cmap(i) for i in range(2048)])[:, :3] * 2047

    for i in range(imgfov_pc_pixel.shape[1]):
        depth = imgfov_pc_cam2[2, i]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(rgb, (int(np.round(imgfov_pc_pixel[0, i])),
                         int(np.round(imgfov_pc_pixel[1, i]))),
                   2, color=tuple(color), thickness=5)
        # cv2.circle(blackblankimage, (int(np.round(imgfov_pc_pixel[0, i])),
        #                             int(np.round(imgfov_pc_pixel[1, i]))),
        #           2, color=tuple(color), thickness=5)
    return rgb


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

    # Change to homogeneous coordinate
    points = np.vstack((points, np.ones((1, num_pts))))
    points = proj_mat @ points
    points[:2, :] /= points[2, :]
    return points[:2, :]


def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        # yield the current n-sized chunk to the calling function
        yield l[i: i + n]


def process(payload):
    print("[INFO] starting process {}".format(payload["id"]))

    for single_img_path in payload['input_path']:
        rgb = cv2.cvtColor(cv2.imread(single_img_path), cv2.COLOR_BGR2RGB)

        rgb_sq_num = int(single_img_path.split('_')[1].split('.')[0])
        rgb_sq_num_minus_one = rgb_sq_num-1
        zero_fill_front = str(rgb_sq_num_minus_one).zfill(6)
        pts_lidar = load_lidar_bin(
            os.path.join(sys.argv[2],
                         ('sq06_' + zero_fill_front + '.bin')))[:, :3]
        calib = payload['calib']

        rgb_proj = render_lidar_on_image(pts_lidar, rgb, calib)
        plt.imsave(single_img_path.replace('/png', '/lidar_rgb_-1_frame'), rgb)

    print("[INFO] process {} serializing hashes".format(payload["id"]))


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Input .png and .bin directories and calib file path required!')
        sys.exit(1)

    dir_out_rgb = '/media/claude/256G/sq06/lidar_rgb_-1_frame'
    if not os.path.exists(dir_out_rgb):
        os.mkdir(dir_out_rgb)

    # determine the number of concurrent processes to launch when
    # distributing the load across the system, then create the list
    # of process IDs
    procs = cpu_count()
    procIDs = list(range(0, procs))
    # grab the paths to the input images, then determine the number
    # of images each process will handle
    print("[INFO] grabbing image paths...")
    allImagePaths = sorted(list(paths.list_images(sys.argv[1])))
    numImagesPerProc = len(allImagePaths) / float(procs)
    numImagesPerProc = int(np.ceil(numImagesPerProc))

    # chunk the image paths into N (approximately) equal sets, one
    # set of image paths for each individual process
    chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

#    png_file_list = np.array(os.listdir(png_in_dir))
#    bin_list = np.array(os.listdir(bin_in_dir))
    calib = read_calib_file(sys.argv[3])
    payloads = []

    for i, j in enumerate(chunkedPaths):

            data = {'id': i,
                    'calib': calib,
                    'input_path': j
                    }
            payloads.append(data)

    # construct and launch the processing pool
    print("[INFO] launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process, payloads)
    # close the pool and wait for all processes to finish
    print("[INFO] waiting for processes to finish...")
    pool.close()
    pool.join()
    print("[INFO] multiprocessing complete")
