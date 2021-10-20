#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import configs
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_unresized_lid_img_val
from fcn.fusion_net import FusionNet
from utils.helpers import draw_test_segmentation_map, image_overlay
import lidar_project_test as lp
from pandas.core.dtypes.cast import _int16_max

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0, 0, 0],
                         std=[1, 1, 1])])

# model loading
model = FusionNet()
# checkpoint loading
checkpoint = torch.load(
    '/media/storage/data/fusion_logs/phase_0_train_by_all/checkpoint_107.pth')
epoch = checkpoint['epoch']
print('Finished Epochs:', epoch)
# trained weights loading
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer loading
# optimizer.load_state_dict(checkpoint['optimizer'])
# load the model to CPU
model.eval().to('cuda')

# Image operations
#image = Image.open(configs.TEST_IMAGE).convert('RGB')
image = Image.open('/home/claude/Data/sq20_000201_rgb.png').resize((480, 320)).convert('RGB')
w_orig, h_orig = image.size  # original image's w and h
#delta = int(h_orig/3)
delta = 0
image = TF.crop(image, delta, 0, h_orig-delta, w_orig)
w_top_crop, h_top_crop = image.size
orig_image = np.array(image.copy())
image = np.array(image)
img_cpu = image.copy()
image = transform(image).to('cuda')
image = image.unsqueeze(0)  # add a batch dimension

# Lidar Image
with open('/home/claude/Data/bin/sq20_000200.bin', 'rb') as _fd:
    _data = _fd.read()
    _lidar = np.frombuffer(_data, np.float32)

_xyzi = _lidar.reshape(-1, 4)
calib = lp.read_calib_file(
    '/home/claude/Dev/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/config/calib.txt')
inds, _pts_2d = lp.render_lidar_on_image(_xyzi[:, :-1], 2824, 4240, calib)
print(inds)
print(img_cpu.shape[:2])
# Xl = np.zeros(img_cpu.shape[:2], np.float32)
# Yl = np.zeros(img_cpu.shape[:2], np.float32)
# Zl = np.zeros(img_cpu.shape[:2], np.float32)
Xl = np.zeros((320, 480), np.float32)
Yl = np.zeros((320, 480), np.float32)
Zl = np.zeros((320, 480), np.float32)


xs, ys = _pts_2d[:, inds].astype(np.int16)
print(xs, ys)
xs = (xs/8.84).astype(int)
ys = (ys/8.875).astype(int)

mean_lidar = configs.LIDAR_MEAN
std_lidar = configs.LIDAR_STD
x_lid = (_xyzi[:, 1] - mean_lidar[0])/std_lidar[0]
y_lid = (_xyzi[:, 2] - mean_lidar[1])/std_lidar[1]
z_lid = (_xyzi[:, 0] - mean_lidar[2])/std_lidar[2]

Xl[ys, xs] = x_lid[inds]
Yl[ys, xs] = y_lid[inds]
Zl[ys, xs] = z_lid[inds]


# with open('/home/claude/xyz.npz', 'wb') as _fd:
#     np.savez_compressed(_fd, X=Xl, Y=Yl, Z=Zl)

# Lidar operations
#points_set, camera_coord = open_lidar(configs.TEST_LIDAR)
#points_set, camera_coord, _ = crop_pointcloud(points_set, camera_coord,
#                                              delta, 0, h_orig-delta, w_orig)
#X, Y, Z = get_unresized_lid_img_val(h_top_crop, w_top_crop,
#                                    points_set, camera_coord)
X = TF.to_tensor(np.array(Xl))
Y = TF.to_tensor(np.array(Yl))
Z = TF.to_tensor(np.array(Zl))
lidar_image = torch.cat((X, Y, Z), 0)
lidar_image = lidar_image.to('cuda')

# a = lidar_image.detach().cpu()
# a = a.permute(1,2,0)
# a = np.array(a)
#
# cv2.imwrite('/home/claude/xyz_0.png', a*255)


lidar_image = lidar_image.unsqueeze(0)  # add a batch dimension
# forward pass through the model
# outputs = model(image, None, 'rgb')
outputs = model(image, lidar_image, 'all')
outputs = outputs['lidar']
print(outputs.shape)

# get the segmentation map
segmented_image = draw_test_segmentation_map(outputs)
# image overlay
result = image_overlay(orig_image, segmented_image)

# visualize result
#labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
#print(labels.any() == 1 )
cv2.imshow("result.jpg", result)
#cv2.imwrite('/home/claude/2.png', np.array(outputs.detach().cpu()))
cv2.waitKey(0)
cv2.destroyAllWindows()
