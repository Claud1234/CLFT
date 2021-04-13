#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Utility Tools to process the Waymo raw dataset.

* Extract and save Images
'''
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import cv2

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


class CommonTools:
    
    def __init__(self, file_name=None, save_dir=None):
        self.raw_file = file_name
        self.save_dir = save_dir #/home/claude/Data/waymo/splits/split_0
        
        self.images_dir = self.save_dir + "/images"
        self.images_labels_dir = self.save_dir + "/image_labels"       
        if not os.path.exists(self.images_dir):
            os.makedirs(self.images_dir)
        if not os.path.exists(self.images_labels_dir):
            os.makedirs(self.images_labels_dir)

    
    def image_extraction(self):
        '''
        Extract images as png and label coordinates as txt
        '''
        dataset = tf.data.TFRecordDataset(self.raw_file, compression_type='')
        dataset_list = list(dataset.as_numpy_iterator())
        amount_of_frames = len(dataset_list)
    
        for i in range(amount_of_frames):
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(dataset_list[i]))
            for index, images in enumerate(frame.images):
                decoded_img = tf.image.decode_jpeg(images.image)
                decoded_img = cv2.cvtColor(decoded_img.numpy(), 
                                           cv2.COLOR_RGB2BGR)
                cv2.imwrite("{}/{}_{}.png".format(
                    self.images_dir, images.name, i), decoded_img)
                
                    
                    
                    
                    
    def image_label_extraction(self):
        break
    
    def lidar_extraction(self): 
        break
            
    def lidar_label_extraction(self):
        break
            
