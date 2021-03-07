import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
    def forward(self, x):
        x_shortcut = x
        x = self.conv1(F.relu(self.norm1(x)))
        x = self.conv2(F.relu(self.norm2(x)))
        return x + x_shortcut

class WideResnet(nn.Module):
    """WRN without the final mean-pool. I handle the downsampling slightly
    differently, for simplicity."""
    def __init__(self, N=2, k=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16*k, 3, padding=1)
        self.conv2 = nn.Sequential(*[ResBlock(16*k) for _ in range(N)])
        self.pre_conv3 = nn.Conv2d(16*k, 32*k, 1, stride=2, bias=False)
        self.conv3 = nn.Sequential(*[ResBlock(32*k) for _ in range(N)])
        self.pre_conv4 = nn.Conv2d(32*k, 64*k, 1, stride=2, bias=False)
        self.conv4 = nn.Sequential(*[ResBlock(64*k) for _ in range(N)])
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pre_conv3(x)
        x = self.conv3(x)
        x = self.pre_conv4(x)
        x = self.conv4(x)
        return x

class WideResnetDecoder(nn.Module):
    """Decoder network that's exactly a mirror-image of WRN."""
    def __init__(self, N=2, k=1):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16*k, 3, padding=1)
        self.conv2 = nn.Sequential(*[ResBlock(16*k) for _ in range(N)])
        self.pre_conv3 = nn.Conv2dTranspose(32*k, 16*k, 1, stride=2,
            output_padding=1, bias=False)
        self.conv3 = nn.Sequential(*[ResBlock(32*k) for _ in range(N)])
        self.pre_conv4 = nn.Conv2dTranspose(64*k, 32*k, 1, stride=2,
            output_padding=1, bias=False)
        self.conv4 = nn.Sequential(*[ResBlock(64*k) for _ in range(N)])
    def forward(self, x):
        x = self.conv4(x)
        x = self.pre_conv4(x)
        x = self.conv3(x)
        x = self.pre_conv3(x)
        x = self.conv2(x)
        x = self.conv1(x)
        return x