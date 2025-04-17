#!/usr/bin/env python3
"""
merge the prediction results from small and large specialized model as the new annotation for training.
In small_scale, change cyclist and sign's index from 1, 2 to 4, 3.
In large-scale, vehicle and ped will be same as 1, 2.
When combining, to increase the representation of small_scale objects, 3 and 4 are prior than 1 and 2.

ONLY WORKS FOR CLFT!!
Created on 10th April 2025
"""

import os
import cv2
import argparse

import numpy as np


def run(args):
    data_list = open(args.path, 'r')
    data_cam = np.array(data_list.read().splitlines())
    data_list.close()

    i = 1
    dataroot = '../output/'
    small_scale_pred_path = os.path.join(dataroot, 'clft_model_small_specialization')
    large_scale_pred_path = os.path.join(dataroot, 'clft_model_large_specialization')
    for path in data_cam:
        cam_path = os.path.join(dataroot, path)
        small_scale_pred = cam_path.replace('labeled', small_scale_pred_path)
        large_scale_pred = cam_path.replace('labeled', large_scale_pred_path)
        small_large_merge_path = cam_path.replace('labeled', '../output/clft_small_large_merge')

        small = cv2.imread(small_scale_pred, cv2.IMREAD_UNCHANGED)
        large = cv2.imread(large_scale_pred, cv2.IMREAD_UNCHANGED)

        small[small == 1] = 4
        small[small == 2] = 3

        img = large.copy()
        small_none_0 = small != 0
        img[small_none_0] = small[small_none_0]

        if not os.path.exists(os.path.dirname(small_large_merge_path)):
            os.makedirs(os.path.dirname(small_large_merge_path))
        print(f'merging frame {i}', end='\r')
        cv2.imwrite(small_large_merge_path, img)
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Merge prediction results from small and large')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path of the text file for file list')
    args = parser.parse_args()

    run(args)
