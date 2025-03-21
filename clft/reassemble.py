import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class Read_projection(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(Read_projection, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())

    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)


class Resample(nn.Module):
    def __init__(self, p, s, h, emb_dim, resample_dim):
        super(Resample, self).__init__()
        assert (s in [4, 8, 16, 32]), "s must be in [0.5, 4, 8, 16, 32]"
        self.conv1 = nn.Conv2d(emb_dim, resample_dim, kernel_size=1, stride=1, padding=0)
        if s == 4:
            self.conv2 = nn.ConvTranspose2d(resample_dim, resample_dim,
                                            kernel_size=4, stride=4, padding=0,
                                            bias=True, dilation=1, groups=1)
        elif s == 8:
            self.conv2 = nn.ConvTranspose2d(resample_dim, resample_dim,
                                            kernel_size=2, stride=2, padding=0,
                                            bias=True, dilation=1, groups=1)
        elif s == 16:
            self.conv2 = nn.Identity()
        else:
            self.conv2 = nn.Conv2d(resample_dim, resample_dim, kernel_size=2,stride=2, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Reassemble(nn.Module):
    def __init__(self, image_size, p, s, emb_dim, resample_dim):
        """
        p = patch size
        s = coefficient resample
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        """
        super(Reassemble, self).__init__()
        channels, image_height, image_width = image_size

        # Read
        self.read = Read_projection(emb_dim)

        # Concat after read
        self.concat = Rearrange('b (h w) c -> b c h w',
                                c=emb_dim,
                                h=(image_height // p),
                                w=(image_width // p))

        # Projection + Resample
        self.resample = Resample(p, s, image_height, emb_dim, resample_dim)

    def forward(self, x):
        x = self.read(x)
        x = self.concat(x)
        x = self.resample(x)
        return x
