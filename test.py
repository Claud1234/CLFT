#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
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
from utils.metrics import find_overlap
from utils.metrics import auc_ap
import lidar_project_test as lp
from pandas.core.dtypes.cast import _int16_max


parser = argparse.ArgumentParser(description='test script')
parser.add_argument('-m', '--mode', type=str, required=True,
                    help='Output mode (lidar, rgb ot fusion)', default=None)
args = parser.parse_args()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

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
# image = Image.open(configs.TEST_IMAGE).convert('RGB')
image = Image.open('/home/claude/Data/claude_iseauto/labeled/day_fair/rgb/sq20_001855.png').resize((480, 320)).convert('RGB')
w_orig, h_orig = image.size  # original image's w and h
# delta = int(h_orig/3)
delta = 0
image = TF.crop(image, delta, 0, h_orig-delta, w_orig)
w_top_crop, h_top_crop = image.size
orig_image = np.array(image.copy())
image = np.array(image)
img_cpu = image.copy()
image = transform(image).to('cuda')
image = image.unsqueeze(0)  # add a batch dimension


# Anno operation
annotation =  Image.open('/home/claude/Data/claude_iseauto/labeled/day_fair/annotation_gray/sq20_001855.png').\
        resize((480, 320), Image.BICUBIC).convert('F')
annotation = TF.to_pil_image(np.array(annotation))
annotation = TF.to_tensor(np.array(annotation)).type(torch.LongTensor).unsqueeze(0)  # add a batch dimension
annotation = annotation.to('cuda')

# Lidar Image
with open('/home/claude/Data/bin/sq20_001854.bin', 'rb') as _fd:
    _data = _fd.read()
    _lidar = np.frombuffer(_data, np.float32)

_xyzi = _lidar.reshape(-1, 4)
calib = lp.read_calib_file(
    '/home/claude/Dev/TalTech_DriverlessProject/catkin_ws/src/ttu_autolab/config/calib.txt')
inds, _pts_2d = lp.render_lidar_on_image(_xyzi[:, :-1], 2824, 4240, calib)
# print(inds)
# print(img_cpu.shape[:2])
# Xl = np.zeros(img_cpu.shape[:2], np.float32)
# Yl = np.zeros(img_cpu.shape[:2], np.float32)
# Zl = np.zeros(img_cpu.shape[:2], np.float32)
X = np.zeros((320, 480), np.float32)
Y = np.zeros((320, 480), np.float32)
Z = np.zeros((320, 480), np.float32)


xs, ys = _pts_2d[:, inds].astype(np.int16)
# print(xs, ys)
xs = (xs/8.84).astype(int)
ys = (ys/8.825).astype(int)

mean_lidar = configs.ISE_LIDAR_MEAN
std_lidar = configs.ISE_LIDAR_STD
x_lid = (_xyzi[:, 1] - mean_lidar[0])/std_lidar[0]
y_lid = (_xyzi[:, 2] - mean_lidar[1])/std_lidar[1]
z_lid = (_xyzi[:, 0] - mean_lidar[2])/std_lidar[2]

X[ys, xs] = x_lid[inds]
Y[ys, xs] = y_lid[inds]
Z[ys, xs] = z_lid[inds]


# Lidar operations
# points_set, camera_coord = open_lidar(
#     '/home/claude/Data/claude_iseauto/labeled/day_fair/pkl/sq20_001855.pkl',
#     8.84, 8.825, configs.ISE_LIDAR_MEAN, configs.ISE_LIDAR_STD)
# points_set, camera_coord, _ = crop_pointcloud(points_set, camera_coord,
#                                               delta, 0, h_orig-delta, w_orig)
# X, Y, Z = get_unresized_lid_img_val(h_top_crop, w_top_crop,
#                                     points_set, camera_coord)
X = TF.to_tensor(np.array(X))
Y = TF.to_tensor(np.array(Y))
Z = TF.to_tensor(np.array(Z))
lidar_image = torch.cat((X, Y, Z), 0)
lidar_image = lidar_image.to('cuda')
lidar_image = lidar_image.unsqueeze(0)  # add a batch dimension

# a = lidar_image.detach().cpu()
# a = a.permute(1,2,0)
# a = np.array(a)
# cv2.imwrite('/home/claude/xyz_0.png', a*255)

# forward pass through the model
outputs = model(image, lidar_image, 'all')
outputs = outputs[args.mode]

overlap, pred, label, union = find_overlap(outputs, annotation)
IoU = 1.0 * overlap / (np.spacing(1) + union)
precision = 1.0 * overlap / (np.spacing(1) + pred)
recall = 1.0 * overlap / (np.spacing(1) + label)
print('IoU:', IoU)
print('precision:', precision)
print('recall:', recall)


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
