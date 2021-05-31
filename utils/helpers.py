#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np

import configs

from tensorboardX import SummaryWriter


logdir = configs.LOG_DIR
if not os.path.exists(logdir):
    os.makedirs(logdir)

label_colors_list = [
        (255, 0, 0),  # animal
        (0, 255, 0),  # archway
        (0, 0, 255)]  # bicyclist]

# all the classes that are present in the dataset
ALL_CLASSES = ['sign', 'bicyclist', 'background']

"""
This (`class_values`) assigns a specific class label to each of the classes.
For example, `animal=0`, `archway=1`, and so on.
"""
class_values = [ALL_CLASSES.index(cls.lower()) for cls in ALL_CLASSES]


class TensorboardWriter():
    def __init__(self):
        super(TensorboardWriter, self).__init__()
    # initialize `SummaryWriter()`
        self.writer = SummaryWriter()

    def tensorboard_writer(self, loss, mIoU, pix_acc, iterations, phase=None):
        if phase == 'train':
            self.writer.add_scalar('Train Loss', loss, iterations)
            self.writer.add_scalar('Train mIoU', mIoU, iterations)
            self.writer.add_scalar('Train Pixel Acc', pix_acc, iterations)
        if phase == 'valid':
            self.writer.add_scalar('Valid Loss', loss, iterations)
            self.writer.add_scalar('Valid mIoU', mIoU, iterations)
            self.writer.add_scalar('Valid Pixel Acc', pix_acc, iterations)


def draw_test_segmentation_map(outputs):
    """
    This function will apply color mask as per the output that we
    get when executing `test.py` or `test_vid.py` on a single image
    or a video. NOT TO BE USED WHILE TRAINING OR VALIDATING.
    """
    labels = torch.argmax(outputs.squeeze(), dim=0).detach().cpu().numpy()
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_colors_list)):
        if label_num in class_values:
            idx = labels == label_num
            red_map[idx] = np.array(label_colors_list)[label_num, 0]
            green_map[idx] = np.array(label_colors_list)[label_num, 1]
            blue_map[idx] = np.array(label_colors_list)[label_num, 2]

    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return segmented_image


def image_overlay(image, segmented_image):
    """
    This function will apply an overlay of the output segmentation
    map on top of the orifinal input image. MAINLY TO BE USED WHEN
    EXECUTING `test.py` or `test_vid.py`.
    """
    alpha = 0.6  # how much transparency to apply
    beta = 1 - alpha  # alpha + beta should equal 1
    gamma = 0  # scalar added to each sum
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
    return image


def save_model_dict(epoch, model, optimizer):
    torch.save({'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               logdir+f"checkpoints_{epoch}.pth")


def adjust_learning_rate(optimizer, epoch, epoch_max):
    """Decay the learning rate based on schedule"""
    lr = configs.LR * (1 - epoch/epoch_max)**0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def adjust_learning_rate_semi(optimizer, epoch, epoch_max, args):
    mid_epoch = epoch_max/2
    if epoch <= mid_epoch:
        lr = np.exp(-(1-epoch/mid_epoch)**2)*args.lrsemi
    else:
        lr = args.lrsemi * (1 - epoch/epoch_max)**0.9

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
