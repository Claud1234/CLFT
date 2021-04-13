#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from waymo_dataset_common import CommonTools

if __name__=="__main__":
    FILENAME = '/home/claude/Data/training_segment-10017090168044687777_6380_000_6400_000_with_camera_labels.tfrecord'
    IMAGES_DIR = '/home/claude/Data/1'
    
    common_tools = CommonTools(file_name = FILENAME, images_dir = IMAGES_DIR)
    common_tools.image_extraction()
    
