#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import torchvision


class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet, self).__init__()

        self.backbone_rgb, self.intermediate_single_rgb, self.classifier_rgb =self.get_split_model()

        self.backbone_lidar, self.intermediate_single_lidar, self.classifier_lidar = self.get_split_model()

        _, self.intermediate_single_fusion, self.classifier_fusion = self.get_split_model(isFusion=True)

    def get_split_model(self, isFusion=False):
        full_model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=4)
        backbone = nn.Sequential(*list(full_model.backbone.children())[:-1])
        intermediate_single = nn.Sequential(*list(full_model.backbone.children())[-1:])
        classifier = nn.Sequential(*list(full_model.classifier.children()))

        if isFusion:
            intermediate_single[0][0].conv1 = nn.Conv2d(
                2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            intermediate_single[0][0].downsample[0] = nn.Conv2d(
                2048, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)

        return backbone, intermediate_single, classifier

    def forward(self, rgb, lidar, modal):

        _, _, h, w = rgb.shape

        if modal == 'cross_fusion':
            features_rgb = self.backbone_rgb(rgb)
            features_lidar = self.backbone_lidar(lidar)
            features_fusion = torch.cat((features_rgb, features_lidar), dim=1)

            out_rgb = self.classifier_rgb(
                self.intermediate_single_rgb(features_rgb))
            out_lidar = self.classifier_lidar(
                self.intermediate_single_lidar(features_lidar))
            out_fusion = self.classifier_fusion(
                self.intermediate_single_fusion(features_fusion))

            out_rgb = F.interpolate(out_rgb, size=(h, w), mode='bilinear', align_corners=False)
            out_lidar = F.interpolate(out_lidar, size=(h, w), mode='bilinear', align_corners=False)
            out_fusion = F.interpolate(out_fusion, size=(h, w),  mode='bilinear', align_corners=False)

            out = {}
            out['rgb'] = out_rgb
            out['lidar'] = out_lidar
            out['cross_fusion'] = out_fusion

        elif modal == 'rgb':
            features_rgb = self.backbone_rgb(rgb)
            out_rgb = self.classifier_rgb(
                self.intermediate_single_rgb(features_rgb))
            out_rgb = F.interpolate(out_rgb, size=(h, w), mode='bilinear', align_corners=False)

            out = {}
            out['rgb'] = out_rgb

        elif modal == 'lidar':
            features_lidar = self.backbone_lidar(lidar)
            out_lidar = self.classifier_lidar(
                self.intermediate_single_lidar(features_lidar))
            out_lidar = F.interpolate(out_lidar, size=(h, w), mode='bilinear', align_corners=False)

            out = {}
            out['lidar'] = out_lidar

        return out
