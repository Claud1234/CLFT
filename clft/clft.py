import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from clft.reassemble import Reassemble
from clft.fusion import Fusion
from clft.head import HeadDepth, HeadSeg

torch.manual_seed(0)


class CLFT(nn.Module):
    def __init__(self,
                 RGB_tensor_size=(3, 384, 384),
                 XYZ_tensor_size=(3, 384, 384),
                 patch_size=16,
                 emb_dim=1024,
                 resample_dim=256,
                 read='projection',
                 num_layers_encoder = 24,
                 hooks=[5, 11, 17, 23],
                 reassemble_s=[4, 8, 16, 32],
                 transformer_dropout=0,
                 nclasses=2,
                 type="full",
                 model_timm="vit_large_patch16_384"):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        #Splitting img into patches
        # channels, image_height, image_width = image_size
        # assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        # num_patches = (image_height // patch_size) * (image_width // patch_size)
        # patch_dim = channels * patch_size * patch_size
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        #     nn.Linear(patch_dim, emb_dim),
        # )
        # #Embedding
        # self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        #Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        # self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        self.transformer_encoders = timm.create_model(model_timm,
                                                      pretrained=True)
        self.type_ = type

        #Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        #Reassembles Fusion
        self.reassembles_RGB = []
        self.reassembles_XYZ = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles_RGB.append(Reassemble(RGB_tensor_size, read, patch_size, s, emb_dim, resample_dim))
            self.reassembles_XYZ.append(Reassemble(XYZ_tensor_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles_RGB = nn.ModuleList(self.reassembles_RGB)
        self.reassembles_XYZ = nn.ModuleList(self.reassembles_XYZ)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            self.head_segmentation = None
        else:
            self.head_depth = None
            self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, rgb, lidar, modal = 'rgb'):
        # x = self.to_patch_embedding(img)
        # b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # x += self.pos_embedding[:, :(n + 1)]
        # t = self.transformer_encoders(x)

        t = self.transformer_encoders(lidar)
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])
            activation_result = self.activation[hook_to_take]
            if modal == 'rgb':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result) #claude check here
                reassemble_result_XYZ = torch.zeros_like(reassemble_result_RGB) # this is just to keep the space allocated but it will not be used later in fusion
            if modal == 'lidar':
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result) #claude check here
                reassemble_result_RGB = torch.zeros_like(reassemble_result_XYZ) # this is just to keep the space allocated but it will not be used later in fusion
            if modal == 'cross_fusion':
                reassemble_result_RGB = self.reassembles_RGB[i](activation_result) #claude check here
                reassemble_result_XYZ = self.reassembles_XYZ[i](activation_result) #claude check here
            
            fusion_result = self.fusions[i](reassemble_result_RGB, reassemble_result_XYZ, previous_stage, modal) #claude check here
            previous_stage = fusion_result
        out_depth = None
        out_segmentation = None
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        if self.head_segmentation != None:
            out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, out_segmentation
        
        
        


    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            #self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))