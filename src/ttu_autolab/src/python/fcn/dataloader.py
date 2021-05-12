import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
import glob
import pickle
import random

class Dataset():
    def __init__(self, split, dataroot, rot_augment, 
                 rot_range, factor, crop_size):
        np.random.seed(789)
        self.rot_augment = rot_augment
        self.rot_range = rot_range
        self.factor = factor 
        self.crop_size = crop_size             
        self.split = split
        self.dataroot = dataroot

        list_examples_file = open(os.path.join(
                                dataroot, 'splits', split + '.txt'), 'r')
        self.list_examples_cam = np.array(
                                        list_examples_file.read().splitlines()) 
        list_examples_file.close()

        #####################################################################
        ##functions for data augmentation and normalization
        self.mean_lidar = np.array([-0.17263354, 0.85321806, 24.5527253])
        self.std_lidar = np.array([7.34546552, 1.17227659, 15.83745082])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                              std=[0.229, 0.224, 0.225])
        self.colorjitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, 
                                                  saturation=0.4, hue=0.1)
        #####################################################################

    def __len__(self):
        return len(self.list_examples_cam)

    def open_lidar(self, lidar_path, mean_lidar, std_lidar):
        file = open(lidar_path,'rb')
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

        camera_coord[:,1] = (camera_coord[:,1]/self.factor).astype(int)
        camera_coord[:,0] = (camera_coord[:,0]/self.factor).astype(int)

        points_set = np.stack((x_lid,y_lid,z_lid),axis=1).astype(np.float32)

        return points_set, camera_coord

    def crop_pointcloud(self, points_set_or, camera_coord_or, i, j, h, w):
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

    def get_lid_images(self, h, w, points_set, camera_coord):
        X = np.zeros((self.crop_size, self.crop_size))
        Y = np.zeros((self.crop_size, self.crop_size))
        Z = np.zeros((self.crop_size, self.crop_size))

        rows = np.floor(camera_coord[:,1]*self.crop_size/h)
        cols = np.floor(camera_coord[:,0]*self.crop_size/w)

        X[(rows.astype(int),cols.astype(int))] = points_set[:,0]
        Y[(rows.astype(int),cols.astype(int))] = points_set[:,1]
        Z[(rows.astype(int),cols.astype(int))] = points_set[:,2]

        X = TF.to_pil_image(X.astype(np.float32))
        Y = TF.to_pil_image(Y.astype(np.float32))
        Z = TF.to_pil_image(Z.astype(np.float32))

        return X, Y, Z

    def get_lid_images_val(self, h, w, points_set, camera_coord):
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

    def prepare_annotation(self, annotation):
        annotation = np.array(annotation)

        mask_ignore = annotation == 0
        mask_sign = annotation == 3
        mask_cyclist = annotation == 4
        mask_background = annotation == 5

        annotation[mask_sign] = 0
        annotation[mask_background] = 0
        annotation[mask_cyclist] = 2
        annotation[mask_ignore] = 3

        return TF.to_pil_image(annotation)
        
    def __getitem__(self, idx):
        # cam_path = os.path.join(self.dataroot, 'scale_' + str(self.factor), 
                                                # self.list_examples_cam[idx])
        cam_path = os.path.join(self.dataroot, self.list_examples_cam[idx])                                       
        
        lidar_path = cam_path.replace('/camera','/lidar').replace('.png','.pkl')
        annotation_path = cam_path.replace('/camera','/annotation')

        rgb = Image.open(cam_path)          
        annotation = Image.open(annotation_path).convert('F')    
        annotation = self.prepare_annotation(annotation)

        points_set, camera_coord = self.open_lidar(
            lidar_path, self.mean_lidar, self.std_lidar)

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        ann_name = annotation_path.split('/')[-1].split('.')[0]

        assert(rgb_name==lidar_name)
        assert(ann_name==lidar_name)

        # Crop top part
        w, h = rgb.size  # original image's w and h
        delta = int(h/2)    
        rgb = TF.crop(rgb, delta, 0, h-delta, w)
        annotation = TF.crop(annotation, delta, 0, h-delta, w)
        points_set, camera_coord, _ = self.crop_pointcloud(points_set,
                                                           camera_coord,
                                                           delta, 0, h-delta, w)
        # print ('pt_set_0:', points_set.shape)
        # print ('cam_coord_0:', camera_coord.shape)
        
        if self.split == '1' or self.split == '2':   
            '''
            #### Square crop
            '''
            w, h = rgb.size # Top cropped image's w and h
            #print ('sqaure crop:',(w,h))
            i0, j0, h0, w0 = transforms.RandomCrop.get_params(rgb, (h,h))        
            rgb = TF.crop(rgb, i0, j0, h0, w0)
            annotation = TF.crop(annotation, i0, j0, h0, w0)
            points_set, camera_coord, _ = self.crop_pointcloud(points_set, 
                                                               camera_coord,
                                                               i0, j0, h0, w0)
            # print ('pt_set:', points_set.shape)
            # print ('cam_coor:', camera_coord.shape)
            X,Y,Z = self.get_lid_images_val(h0, w0, points_set, camera_coord)
            
            rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
            rgb = self.normalize(to_tensor(np.array(rgb))[0:3])
            
            '''
            #### Random cropping
            '''
            # i1, j1, h1, w1 = transforms.RandomResizedCrop.get_params(
                                # rgb, scale=(0.2, 1.), ratio=(3. / 4., 4. / 3.))
            # rgb = TF.resized_crop(
                # rgb, i1, j1, h1, w1, (self.crop_size, self.crop_size), 
                # Image.BILINEAR)
            # annotation = TF.resized_crop(
                # annotation, i1, j1, h1, w1, (self.crop_size,self.crop_size), 
                # Image.NEAREST)
            # points_set, camera_coord, _ = self.crop_pointcloud(
                # points_set, camera_coord, i1, j1, h1, w1)
            # X,Y,Z = self.get_lid_images(h, w, points_set, camera_coord) 
            #
            # rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
            # rgb = self.normalize(to_tensor(np.array(rgb))[0:3])       

            '''
            #### Random rotation
            '''
            # if random.random() > 0.5 and self.rot_augment:
                # angle = -self.rot_range + 2*self.rot_range*torch.rand(1)[0]
                # rgb = TF.affine(rgb, angle, (0,0), 1, 0, 
                                # interpolation=Image.BILINEAR, fill=0)                        
                # annotation = TF.affine(annotation, angle, (0,0), 1, 0, 
                    # interpolation=Image.NEAREST, fill=0)                        
                # X = TF.affine(
                    # X, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)                        
                # Y = TF.affine(
                    # Y, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)                        
                # Z = TF.affine(
                    # Z, angle, (0,0), 1, 0, interpolation=Image.NEAREST, fill=0)

            '''
            #### Random color jittering
            '''
            # rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
            # rgb = self.normalize(to_tensor(self.colorjitter(rgb))[0:3])#only rgb
                        
        else:
            w, h = rgb.size
            #print ('no aug:',(w,h))
            X,Y,Z = self.get_lid_images_val(h, w, points_set, camera_coord)        
            rgb_copy = to_tensor(np.array(rgb.copy()))[0:3]
            rgb = self.normalize(to_tensor(np.array(rgb))[0:3])  #only rgb

        X = to_tensor(np.array(X))
        Y = to_tensor(np.array(Y))
        Z = to_tensor(np.array(Z))
        lid_images = torch.cat((X,Y,Z),0)
        annotation = to_tensor(np.array(annotation)).type(torch.LongTensor).squeeze(0)
    
        return {'rgb':rgb, 'rgb_orig':rgb_copy, 'lidar':lid_images, 
                'annotation':annotation}   