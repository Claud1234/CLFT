#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Model evaluation metrics Python scripts

Created on June 18th, 2021
'''
import torch
import configs


'''
                           annotation
           |--------------|---------------|------------|------------
           |              | background(1) | vehicle(2) | human(3)
prediction |background(1) |       a       |     b      |     c
           |vehicle(2)    |       d       |     e      |     f
           |human(3)      |       g       |     h      |     i

IoU(1) = a / (a + b  + c + d + g)
IoU(2) = e / (d + e + f + b + h)
IoU(3) = i / (g + h + i + c + f)

precision(1) = a / (a + b + c)
precision(2) = e / (d + e + f)
precision(3) = i / (g + h + i)

recall(1) = a / (a + d + g)
recall(2) = e / (b + e + h)
recall(3) = i / (c + f + i)
'''


def find_overlap(output, anno):
    '''
    :param output: 'fusion' output batch (8, 4, 160, 480)
    :param anno: annotation batch (8, 160, 480)
    :return: histogram statistic of overlap, prediction and annotation, union
    '''
    # 0 -> background, 1-> vehicle,  2-> human (ped+cyclist), 3 -> ignore
    n_classes = configs.CLASS_TOTAL - 1
    # Return each pixel value as either 0 or 1 or 2 or 3, which
    # represent different classes.
    _, pred_indices = torch.max(output, 1)  # (8, 160, 480)
    # 1 -> background, 2-> vehicle,  3-> human, (ped+cyclist,) 4 -> ignore
    pred_indices = pred_indices + 1  # (8, 160, 480)
    anno = anno + 1  # (8, 160, 480)
    # If pixel value in anno is 4(ignore), then it is False;
    # if pixel value if anno is 1 or 2 or 3, then it is True.
    labeled = (anno > 0) * (anno <= n_classes)  # (8, 160, 480)
    # For 'label.long()', True will be 1, False will be 0.
    # In prediction, if value is 1 or 2 or 3, then no change;
    #                if value is 4, then change to 0
    pred_indices = pred_indices * labeled.long()
    # If pixel value in prediction is same as anno, then keep it same;
    # if pixel value in prediction is not same as anno, then change to 0
    overlap = pred_indices * (pred_indices == anno).long()  # (8, 160, 480)

    # (a, e, i)
    area_overlap = torch.histc(overlap.float(),
                               bins=n_classes, max=n_classes, min=1)
    # ((a + b + c), (d + e + f), (g + h + i))
    area_pred = torch.histc(pred_indices.float(),
                            bins=n_classes, max=n_classes, min=1)
    # ((a + d + j), (b + e + h), (c + f + i))
    area_label = torch.histc(anno.float(),
                             bins=n_classes, max=n_classes, min=1)
    # ((a + b  + c + d + g), (d + e + f + b + h), (g + h + i + c + f))
    area_union = area_pred + area_label - area_overlap

    assert (area_overlap <= area_union).all(),\
        "Intersection area should be smaller than Union area"

    return area_overlap, area_pred, area_label, area_union
