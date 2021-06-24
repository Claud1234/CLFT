#!/usr/bin/env python3
import cv2
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_lid_images_val
from fcn.fusion_net import FusionNet
from utils.helpers import draw_test_segmentation_map, image_overlay

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

# model loading
model = FusionNet()
# checkpoint loading
checkpoint = torch.load(
    '/home/claude/Data/logs/6th_gpu_test/checkpoints_99.pth')
epoch = checkpoint['epoch']
print('Finished Epochs:', epoch)
# trained weights loading
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer loading
# optimizer.load_state_dict(checkpoint['optimizer'])
# load the model to CPU
model.eval().to('cuda')

# Image operations
image = Image.open(configs.TEST_IMAGE).convert('RGB')
w_orig, h_orig = image.size  # original image's w and h
delta = int(h_orig/2)
image = TF.crop(image, delta, 0, h_orig-delta, w_orig)
w_top_crop, h_top_crop = image.size
orig_image = np.array(image.copy())
image = np.array(image)
image = transform(image).to('cuda')
image = image.unsqueeze(0)  # add a batch dimension


# Lidar operations
points_set, camera_coord = open_lidar(configs.TEST_LIDAR)
points_set, camera_coord, _ = crop_pointcloud(points_set, camera_coord,
                                              delta, 0, h_orig-delta, w_orig)
X, Y, Z = get_lid_images_val(h_top_crop, w_top_crop, points_set, camera_coord)
X = TF.to_tensor(np.array(X))
Y = TF.to_tensor(np.array(Y))
Z = TF.to_tensor(np.array(Z))
lidar_image = torch.cat((X, Y, Z), 0)
lidar_image = lidar_image.to('cuda')
lidar_image = lidar_image.unsqueeze(0)  # add a batch dimension
# forward pass through the model
# outputs = model(image, None, 'rgb')
outputs = model(image, lidar_image, 'all')
outputs = outputs['fusion']

# get the segmentation map
segmented_image = draw_test_segmentation_map(outputs)
# image overlay
result = image_overlay(orig_image, segmented_image)

# visualize result
cv2.imshow("result.jpg", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
