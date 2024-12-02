#!/usr/bin/env python3
"""
This is the script to compute the inference time of CLFT and CLFCN. It will load only one frame as input, execute
the GPU warm up, then repeat the output computation loop 2000 (number you can decide) times, then only capture
CUDA event time and computer average. This part was explained in paper.
"""
import sys
import torch
import json
import argparse
import numpy as np
from PIL import Image
import torchvision.transforms.v2 as transforms
import torchvision.transforms.v2.functional as TF

from clft.clft import CLFT
from clfcn.fusion_net import FusionNet
from utils.helpers import waymo_anno_class_relabel
from utils.lidar_process import open_lidar
from utils.lidar_process import crop_pointcloud
from utils.lidar_process import get_unresized_lid_img_val
from tools.dataset import lidar_dilation


class OpenInput(object):
    def __init__(self, config):
        self.config = config

    def open_rgb(self):
        rgb_normalize = transforms.Compose(
            [transforms.Resize((384, 384),
                    interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config['Dataset']['transforms']['image_mean'],
                    std=self.config['Dataset']['transforms']['image_mean'])])

        rgb = Image.open('./test_images/test_1_img.png').convert('RGB')
        # image = Image.open(
        #       '/home/claude/Data/claude_iseauto/labeled/night_fair/rgb/sq14_000061.png').\
        #         resize((480, 320)).convert('RGB')
        w_orig, h_orig = rgb.size  # original image's w and h
        delta = int(h_orig/2)
        top_crop_rgb = TF.crop(rgb, delta, 0, h_orig-delta, w_orig)
        top_rgb_norm = rgb_normalize(top_crop_rgb)
        return top_rgb_norm

    def open_anno(self):
        anno_resize = transforms.Resize((384, 384),
                        interpolation=transforms.InterpolationMode.NEAREST)
        anno = Image.open('./test_images/test_1_anno.png')
        anno = waymo_anno_class_relabel(anno)
        # annotation = Image.open(
        #       '/home/claude/Data/claude_iseauto/labeled/night_fair/annotation_rgb/sq14_000061.png').\
        #          resize((480, 320), Image.BICUBIC).convert('F')
        w_orig, h_orig = anno.size  # PIL tuple. (w, h)
        delta = int(h_orig/2)
        top_crop_anno = TF.crop(anno, delta, 0, h_orig - delta, w_orig)
        anno_resize = anno_resize(top_crop_anno).squeeze(0)
        return anno_resize

    def open_lidar(self):
        points_set, camera_coord = open_lidar(
            './test_images/test_1_lidar.pkl',
            w_ratio=4,
            h_ratio=4,
            lidar_mean=self.config['Dataset']['transforms']['lidar_mean_waymo'],
            lidar_std=self.config['Dataset']['transforms']['lidar_mean_waymo'])

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
    open_input = OpenInput(config)
    rgb = open_input.open_rgb().to(device, non_blocking=True)
    rgb = rgb.unsqueeze(0)  # add a batch dimension
    lidar = open_input.open_lidar().to(device, non_blocking=True)
    lidar = lidar.unsqueeze(0)

    if backbone == 'clfcn':
        model = FusionNet()
        print(f'Using backbone {args.backbone}')
        checkpoint = torch.load('./model_path/clfcn/checkpoint_289_fusion.pth', map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        # init time logger
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 2000
        timings=np.zeros((repetitions,1))

        # GPU-WARM-UP
        for _ in range(2000):
            _ = model(rgb, lidar, 'cross_fusion')
        print('GPU warm up is done with 2000 iterations')

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                output_seg = model(rgb, lidar, 'cross_fusion')
                ender.record()
                # wait for GPU sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f'Mean execute time of 2000 iterations is {mean_syn} milliseconds')

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
        model.load_state_dict(torch.load(model_path, map_location=device)[
                                  'model_state_dict'])

        model.to(device)
        model.eval()

        # init time logger
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 2000
        timings=np.zeros((repetitions,1))

        # GPU-WARM-UP
        for _ in range(2000):
            _,_ = model(rgb, lidar, modality)
        print('GPU warm up is done with 2000 iterations')

        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _, output_seg = model(rgb, lidar, modality)
                ender.record()
                # wait for GPU sync
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time

        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        print(f'Mean execute time of 2000 iterations is {mean_syn} milliseconds')

    else:
        sys.exit("A backbone must be specified! (dpt or fcn)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visual run script')
    parser.add_argument('-m', '--mode', type=str, required=True,
                        choices=['rgb', 'lidar', 'cross_fusion'],
                        help='Output mode (lidar, rgb or cross_fusion)')
    parser.add_argument('-bb', '--backbone', required=True,
                        choices=['fcn', 'dpt'],
                        help='Use the backbone of training, dpt or fcn')
    args = parser.parse_args()

    with open('config.json', 'r') as f:
        configs = json.load(f)

    run(args.mode, args.backbone, configs)
