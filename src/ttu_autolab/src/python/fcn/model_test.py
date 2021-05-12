"""
We will use the FCN ResNet50 from the PyTorch model. We will not
use any pretrained weights. Training from scratch.
"""

import torchvision.models as models
import torch.nn as nn

def model(pretrained, requires_grad):
    model = models.segmentation.fcn_resnet50(
        pretrained=pretrained, progress=True)

    return model


model = model(pretrained=False, requires_grad=False)
print(model)

