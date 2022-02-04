#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Script to change the annotation pixel to same as Waymo.
Original RGB image: green->car, red->human
New greyscale image: 1->car, 2->human, 0->all the rest
'''
import cv2
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Annotation color change')
parser.add_argument('anno', type=str, help='Path to original annotation file')
args = parser.parse_args()

im = Image.open(args.anno)
im = im.convert('RGB')

data = np.array(im)
red, green, blue = data.T

car_areas = (red == 0) & (blue == 0) & (green == 255)
human_areas = (red == 255) & (blue == 0) & (green == 0)
data[..., :][car_areas.T] = (1, 1, 1)
data[..., :][human_areas.T] = (2, 2, 2)

data[~(car_areas.T | human_areas.T)] = (0, 0, 0)

im_gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
anno_path = args.anno.replace('annotation_rgb', 'annotation_gray')
cv2.imwrite(anno_path, im_gray)

# im1 = Image.fromarray(data)
# plt.imshow(im_gray)
# plt.show()
