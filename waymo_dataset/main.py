#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Main file to define the path raw and processed data.
  
'''

from waymo_dataset_common import CommonTools
import os

if __name__=="__main__":
    ROOT_DIR = '/home/claude/Data/waymo/training_0'
    FILE_NAME = 'segment-10023947602400723454_1120_000_1140_000_with_camera_labels.tfrecord'
    SAVE_DIR = '/home/claude/Data/waymo/splits/split_0'
    
    common_tools = CommonTools(file_name = os.path.join(ROOT_DIR, FILE_NAME), 
                               save_dir = SAVE_DIR)
    
    common_tools.image_extraction()
    
