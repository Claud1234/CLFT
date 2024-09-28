#!/usr/bin/env python3
"""
This is the script to load the all input frames to feed to model path file to compute the qualitative overlay results.
Currently it only works for CLFT model paths and Waymo dataset. It loads the config.json file for important
inforamtion, so you have to set the things like CLFT variants, model path, and other things in json file.
If you want to see how it works for single input frame, you can refer the ipython notebook in
ipython/make_qualitative_images.ipynb
"""
import os
import cv2
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import json
from clft.clft import CLFT
from clfcn.fusion_net import FusionNet
from utils.helpers import waymo_anno_class_relabel
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_unresized_lid_img_val
from tools.dataset import lidar_dilation

from utils.helpers import draw_test_segmentation_map, image_overlay


class OpenInput(object):
    def __init__(self, cam_mean, cam_std, lidar_mean, lidar_std, w_ratio, h_ratio):
        self.cam_mean = cam_mean
        self.cam_std = cam_std
        self.lidar_mean = lidar_mean
        self.lidar_std = lidar_std
        self.w_ratio = w_ratio
        self.h_ratio = h_ratio

    def open_rgb(self, image_path):
        rgb_normalize = transforms.Compose(
            [transforms.Resize((384, 384),
                    interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.cam_mean,
                    std=self.cam_std)])

        rgb = Image.open(image_path).convert('RGB')
        # image = Image.open(
        #       '/home/claude/Data/claude_iseauto/labeled/night_fair/rgb/sq14_000061.png').\
        #         resize((480, 320)).convert('RGB')
        w_orig, h_orig = rgb.size  # original image's w and h
        delta = int(h_orig/2)
        top_crop_rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)
        top_rgb_norm = rgb_normalize(top_crop_rgb)
        return top_rgb_norm

    def open_anno(self, anno_path):
        anno_resize = transforms.Resize((384, 384),
                        interpolation=transforms.InterpolationMode.NEAREST)
        anno = Image.open(anno_path)
        anno = waymo_anno_class_relabel(anno)
        # annotation = Image.open(
        #       '/home/claude/Data/claude_iseauto/labeled/night_fair/annotation_rgb/sq14_000061.png').\
        #          resize((480, 320), Image.BICUBIC).convert('F')
        w_orig, h_orig = anno.size  # PIL tuple. (w, h)
        delta = int(h_orig/2)
        top_crop_anno = TF.crop(anno, delta, 0, h_orig - delta, w_orig)
        anno_resize = anno_resize(top_crop_anno).squeeze(0)
        return anno_resize

    def open_lidar(self, lidar_path):
        points_set, camera_coord = open_lidar(
           lidar_path,
            w_ratio=self.w_ratio,
            h_ratio=self.h_ratio,
            lidar_mean=self.lidar_mean,
            lidar_std=self.lidar_std)

        top_crop_points_set, top_crop_camera_coord, _ = crop_pointcloud(
            points_set, camera_coord, 160, 0, 160, 480)
        X, Y, Z = get_unresized_lid_img_val(160, 480,
                                            top_crop_points_set,
                                            top_crop_camera_coord)
        X, Y, Z = lidar_dilation(X, Y, Z)
        X = transforms.Resize((384, 384))(X)
        Y = transforms.Resize((384, 384))(Y)
        Z = transforms.Resize((384, 384))(Z)

        X = TF.to_tensor(np.array(X))
        Y = TF.to_tensor(np.array(Y))
        Z = TF.to_tensor(np.array(Z))

        lid_images = torch.cat((X, Y, Z), 0)
        return lid_images


def run(modality, backbone, config):
    device = torch.device(config['General']['device']
                          if torch.cuda.is_available() else "cpu")
    open_input = OpenInput(cam_mean=config['Dataset']['transforms']['image_mean'],
                           cam_std=config['Dataset']['transforms']['image_mean'],
                           lidar_mean=config['Dataset']['transforms']['lidar_mean_waymo'],
                           lidar_std=config['Dataset']['transforms']['lidar_mean_waymo'],
                           w_ratio=4,
                           h_ratio=4)

    # TODO: need to fisish the CLFCN part.
    if backbone == 'clfcn':
        model = FusionNet()
        print(f'Using backbone {args.backbone}')
        checkpoint = torch.load(
            './checkpoint_289_fusion.pth', map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

    elif backbone == 'clft':
        resize = config['Dataset']['transforms']['resize']
        model = CLFT(
            RGB_tensor_size=(3, resize, resize),
            XYZ_tensor_size=(3, resize, resize),
            emb_dim=config['General']['emb_dim'],
            resample_dim=config['General']['resample_dim'],
            read=config['General']['read'],
            nclasses=len(config['Dataset']['classes']),
            hooks=config['General']['hooks'],
            model_timm=config['General']['model_timm'],
            type=config['General']['type'],
            patch_size=config['General']['patch_size'], )
        print(f'Using backbone {args.backbone}')

        model_path = config['General']['model_path']
        model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        model.to(device)
        model.eval()

    else:
        sys.exit("A backbone must be specified! (clft or clfcn)")

    waymo_all_list = open('/home/autolab/Data/waymo/splits_clft/all.txt', 'r')
    waymo_all_cam = np.array(waymo_all_list.read().splitlines())
    waymo_all_list.close()

    i = 1
    dataroot = '/home/autolab/Data/waymo/'
    for path in waymo_all_cam:
        cam_path = os.path.join(dataroot, path)
        anno_path = cam_path.replace('/camera', '/annotation')
        lidar_path = cam_path.replace('/camera', '/lidar').replace('.png', '.pkl')

        rgb_name = cam_path.split('/')[-1].split('.')[0]
        anno_name = anno_path.split('/')[-1].split('.')[0]
        lidar_name = lidar_path.split('/')[-1].split('.')[0]
        assert (rgb_name == lidar_name)
        assert (anno_name == lidar_name)

        rgb = open_input.open_rgb(cam_path).to(device, non_blocking=True)
        rgb = rgb.unsqueeze(0)  # add a batch dimension
        lidar = open_input.open_lidar(lidar_path).to(device, non_blocking=True)
        lidar = lidar.unsqueeze(0)

        with torch.no_grad():
            output_seg = model(rgb, lidar, modality)

            segmented_image = draw_test_segmentation_map(output_seg)
            seg_resize = cv2.resize(segmented_image, (480, 160))

        seg_path = cam_path.replace('waymo/labeled', 'clft_seg_results/base_fusion/segment')
        overlay_path = cam_path.replace('waymo/labeled', 'clft_seg_results/base_fusion/overlay')

        print(f'saving segment result {i}...')
        cv2.imwrite(seg_path, seg_resize)

        rgb_cv2 = cv2.imread(cam_path)
        rgb_cv2_top = rgb_cv2[160:320, 0:480]
        overlay = image_overlay(rgb_cv2_top, seg_resize)
        print(f'saving overlay result {i}...')
        cv2.imwrite(overlay_path, overlay)
        i+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visual run script')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Output mode (lidar, rgb or cross_fusion)')
    parser.add_argument('-bb', '--backbone', required=True,
                        choices=['clfcn', 'clft'],
                        help='Use the backbone of training, clft or clfcn')
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args.mode, args.backbone, configs)
