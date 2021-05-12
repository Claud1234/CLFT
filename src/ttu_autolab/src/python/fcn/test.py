#!/usr/bin/env python
import cv2
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from PIL import Image
from torchvision.transforms.functional import to_tensor
#from model_test import model
from fusion_net import FusionNet
from helpers import draw_test_segmentation_map, image_overlay

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

# model loading
model = FusionNet()
#model = model
# checkpoint loading
checkpoint = torch.load(
    '/home/claude/Data/logs/3rd_test/checkpoint_0009.pth')
# trained weights loading
model.load_state_dict(checkpoint['model_state_dict'])
# optimizer loading
# optimizer.load_state_dict(checkpoint['optimizer'])
# load the model to CPU
model.eval().to('cpu')

# Image operations
image = Image.open('/home/claude/1.png').convert('RGB')
w, h = image.size  # original image's w and h
delta = int(h/2)    
image = TF.crop(image, delta, 0, h-delta, w)
orig_image = np.array(image.copy())
image = np.array(image)
image = transform(image).to('cpu')
image = image.unsqueeze(0) # add a batch dimension


#Lidar operations
def open_lidar():
    mean_lidar = np.array([-0.17263354, 0.85321806, 24.5527253])
    std_lidar = np.array([7.34546552, 1.17227659, 15.83745082])
    
    file = open('/home/claude/2.pkl', 'rb')
    lidar_data = pickle.load(file)
    file.close()

    points3d = lidar_data['3d_points']
    camera_coord = lidar_data['camera_coordinates']

    #select camera front
    mask = camera_coord[:,0] == 1
    points3d = points3d[mask,:]
    camera_coord = camera_coord[mask,1:3]

    x_lid = (points3d[:,1] - mean_lidar[0])/std_lidar[0] 
    y_lid = (points3d[:,2] - mean_lidar[1])/std_lidar[1]
    z_lid = (points3d[:,0] - mean_lidar[2])/std_lidar[2]

    camera_coord[:,1] = (camera_coord[:,1]/4).astype(int)
    camera_coord[:,0] = (camera_coord[:,0]/4).astype(int)

    points_set = np.stack((x_lid,y_lid,z_lid),axis=1).astype(np.float32)

    return points_set, camera_coord

def crop_pointcloud(points_set_or, camera_coord_or, i, j, h, w):
    points_set = np.copy(points_set_or)
    camera_coord = np.copy(camera_coord_or)
    
    camera_coord[:,1] -= i
    camera_coord[:,0] -= j
    selected_i = np.logical_and(camera_coord[:,1] >=0, camera_coord[:,1] < h) 
    selected_j = np.logical_and(camera_coord[:,0] >=0, camera_coord[:,0] < w) 
    selected = np.logical_and(selected_i, selected_j)
    #print (selected)
    points_set = points_set[selected,:]
    camera_coord = camera_coord[selected,:]
    #print ('points_set', points_set.shape)
    #print ('camera_coord', camera_coord.shape)
    return points_set, camera_coord, selected

def get_lid_images_val(h, w, points_set, camera_coord):
    X = np.zeros((h,w))
    Y = np.zeros((h,w))
    Z = np.zeros((h,w))

    rows = np.floor(camera_coord[:,1])
    cols = np.floor(camera_coord[:,0])

    X[(rows.astype(int),cols.astype(int))] = points_set[:,0]
    Y[(rows.astype(int),cols.astype(int))] = points_set[:,1]
    Z[(rows.astype(int),cols.astype(int))] = points_set[:,2]

    X = TF.to_pil_image(X.astype(np.float32))
    Y = TF.to_pil_image(Y.astype(np.float32))
    Z = TF.to_pil_image(Z.astype(np.float32))

    return X, Y, Z

def lidar_process():
    points_set, camera_coord = open_lidar()
    points_set, camera_coord, _ = crop_pointcloud(points_set,camera_coord,
                                                  delta, 0, h-delta, w)
    X,Y,Z = get_lid_images_val(h, w, points_set, camera_coord)
    X = to_tensor(np.array(X))
    Y = to_tensor(np.array(Y))
    Z = to_tensor(np.array(Z))
    lid_images = torch.cat((X,Y,Z),0)
    lid_images = lid_images.unsqueeze(0) # add a batch dimension
    
    return lid_images

# forward pass through the model
#outputs = model(image, None, 'rgb')
outputs = model(image, lidar_process(), 'ind')
outputs = outputs['lidar']
# get the segmentation map
segmented_image = draw_test_segmentation_map(outputs)
# image overlay
result = image_overlay(orig_image, segmented_image)

# visualize result
cv2.imwrite("2.jpg", result)


