import numpy as np
import torch
import torch.nn as nn
import timm

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadSeg

torch.manual_seed(0)


class CLFT(nn.Module):
    def __init__(self,
                 RGB_tensor_size=None,
                 XYZ_tensor_size=None,
                 patch_size=None,
                 emb_dim=None,
                 resample_dim=None,
                 hooks=None,
                 reassemble_s=None,
                 nclasses=None,
                 model_timm=None
                 ):
        """
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)

        # Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles_RGB.append(Reassemble(RGB_tensor_size, patch_size, s, emb_dim, resample_dim))
            self.reassembles_XYZ.append(Reassemble(XYZ_tensor_size, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles_RGB = nn.ModuleList(self.reassembles_RGB)
        self.reassembles_XYZ = nn.ModuleList(self.reassembles_XYZ)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, rgb, lidar, modal=None):
        t = self.transformer_encoders(lidar)
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            if modal == 'rgb':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB)
            if modal == 'lidar':
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ)
            if modal == 'cross_fusion':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result)
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result)
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal)
            previous_stage = fusion_result

        out_segmentation = None
        if self.head_segmentation is not None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_segmentation

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            #self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))