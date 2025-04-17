#!/usr/bin/env python3
"""
This is the script for model specialization prediction. It saves the prediction as 384x384x1 PNG image, pixel values
are 0, 1, 2. 0 is background, 1 and 2 are objects, in small_scale, it is cyclist and sign, in large scale, it is
vehicle and pedestrian.

ONLY WORKS FOR CLFT!!
Created on 10th April 2025
"""
import os
import cv2
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF

import json
from clft.clft import CLFT
from clfcn.fusion_net import FusionNet
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_unresized_lid_img_val
from tools.dataset import lidar_dilation


class OpenInput(object):
    def __init__(self, backbone, cam_mean, cam_std, lidar_mean, lidar_std, w_ratio, h_ratio):
        self.backbone = backbone
        self.cam_mean = cam_mean
        self.cam_std = cam_std
        self.lidar_mean = lidar_mean
        self.lidar_std = lidar_std
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

    def open_rgb(self, image_path):
        clft_rgb_normalize = transforms.Compose(
            [transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cam_mean,
                    std=self.cam_std)])

        clfcn_rgb_normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=self.cam_mean, std=self.cam_std)])

        rgb = Image.open(image_path).convert('RGB')
        # image = Image.open(
        #       '/home/claude/Data/claude_iseauto/labeled/night_fair/rgb/sq14_000061.png').\
        #         resize((480, 320)).convert('RGB')
        w_orig, h_orig = rgb.size  # original image's w and h
        delta = int(h_orig/2)
        top_crop_rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)
        if self.backbone == 'clft':
            top_rgb_norm = clft_rgb_normalize(top_crop_rgb)
        elif self.backbone == 'clfcn':
            top_rgb_norm = clfcn_rgb_normalize(top_crop_rgb)
        return top_rgb_norm

    def open_lidar(self, lidar_path):
        points_set, camera_coord = open_lidar(lidar_path,
                                              w_ratio=self.w_ratio, h_ratio=self.h_ratio,
                                              lidar_mean=self.lidar_mean, lidar_std=self.lidar_std)

        top_crop_points_set, top_crop_camera_coord, _ = crop_pointcloud(
            points_set, camera_coord, 160, 0, 160, 480)
        X, Y, Z = get_unresized_lid_img_val(160, 480,
                                            top_crop_points_set,
                                            top_crop_camera_coord)
        X, Y, Z = lidar_dilation(X, Y, Z)

        if self.backbone == 'clft':
            X = transforms.Resize((384, 384))(X)
            Y = transforms.Resize((384, 384))(Y)
            Z = transforms.Resize((384, 384))(Z)

        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))

        lid_images = torch.cat((X, Y, Z), 0)
        return lid_images


def run(modality, backbone, config):
    device = torch.device(config['General']['device'] if torch.cuda.is_available() else "cpu")
    open_input = OpenInput(backbone,
                           cam_mean=config['Dataset']['transforms']['image_mean'],
                           cam_std=config['Dataset']['transforms']['image_mean'],
                           lidar_mean=config['Dataset']['transforms']['lidar_mean_waymo'],
                           lidar_std=config['Dataset']['transforms']['lidar_mean_waymo'],
                           w_ratio=4, h_ratio=4)

    if args.specialization == 'large':
        n_classes = len(config['Dataset']['class_large_scale'])
    elif args.specialization == 'small':
        n_classes = len(config['Dataset']['class_small_scale'])
    elif args.specialization == 'all':
        n_classes = len(config['Dataset']['class_all_scale'])
    elif args.specialization == 'cross':
        n_classes = len(config['Dataset']['class_cross_scale'])
    else:
        sys.exit("A specialization must be specified! (large or small or all or cross)")

    if backbone == 'clfcn':
        model = FusionNet()
        print(f'Using backbone {args.backbone}')
        checkpoint = torch.load(args.model_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    elif backbone == 'clft':
        resize = config['Dataset']['transforms']['resize']
        model = CLFT(RGB_tensor_size=(3, resize, resize),
                     XYZ_tensor_size=(3, resize, resize),
                     patch_size=config['CLFT']['patch_size'],
                     emb_dim=config['CLFT']['emb_dim'],
                     resample_dim=config['CLFT']['resample_dim'],
                     hooks=config['CLFT']['hooks'],
                     reassemble_s=config['CLFT']['reassembles'],
                     nclasses=n_classes,
                     model_timm=config['CLFT']['model_timm'], )
        print(f'Using backbone {args.backbone}')

        model_path = args.model_path
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()

    else:
        sys.exit("A backbone must be specified! (clft or clfcn)")

    data_list = open(args.path, 'r')
    data_cam = np.array(data_list.read().splitlines())
    data_list.close()

    i = 1
    dataroot = './waymo_dataset/'
    for path in data_cam:
        cam_path = os.path.join(dataroot, path)
        lidar_path = cam_path.replace('/camera', '/lidar').replace('.png', '.pkl')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)

        rgb = open_input.open_rgb(cam_path).to(device, non_blocking=True)
        rgb = rgb.unsqueeze(0)  # add a batch dimension
        lidar = open_input.open_lidar(lidar_path).to(device, non_blocking=True)
        lidar = lidar.unsqueeze(0)

        if backbone == 'clft':
            with torch.no_grad():
                output_seg = model(rgb, lidar, modality)
                pred_index = torch.argmax(output_seg.squeeze(), dim=0).detach().cpu().numpy()

                specialization_path = f'output/clft_model_{args.specialization}_specialization'
                pred_path = cam_path.replace('waymo_dataset/labeled', specialization_path)

                if not os.path.exists(os.path.dirname(pred_path)):
                    os.makedirs(os.path.dirname(pred_path))
                print(f'saving prediction result {i}...', end='\r')
                cv2.imwrite(pred_path, pred_index)

        elif backbone == 'clfcn':
            # TODO: Do it if there is a need in future.
            pass

        else:
            sys.exit("A backbone must be specified! (clft or clfcn)")
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visual run script')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Output mode (lidar, rgb or cross_fusion)')
    parser.add_argument('-s', '--specialization', type=str, required=True,
                        choices=['small', 'large', 'all', 'cross'],
                        help='Model specialization. (large or small or all or cross)')
    parser.add_argument('-bb', '--backbone', required=True,
                        choices=['clfcn', 'clft'],
                        help='Use the backbone of training, clft or clfcn')
    parser.add_argument('-p', '--path', type=str, required=True,
                        help='The path of the text file to visualize')
    parser.add_argument('-mp', '--model_path', type=str, required=True,
                        help='The model path checkpoint.')
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args.mode, args.backbone, configs)
