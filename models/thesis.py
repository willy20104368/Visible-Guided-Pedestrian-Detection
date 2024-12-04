import math
from copy import copy
from pathlib import Path
from re import X

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import torchvision.ops
from PIL import Image
from torch.cuda import amp

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh
from utils.plots import color_list, plot_one_box
from utils.torch_utils import time_synchronized
from models.common import *



class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()
 
        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )
 
    def forward(self, x):
        x = self.upsample(x)
 
        return x
 
 
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()
 
        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )
 
    def forward(self, x):
        x = self.downsample(x)
 
        return x
 

class ASFF_2(nn.Module):
    #https://arxiv.org/pdf/2306.15988v1.pdf
    def __init__(self, inter_dim=512, level=0, channel=[64, 128]):
        super(ASFF_2, self).__init__()
 
        self.inter_dim = inter_dim
        compress_c = 8
 
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
 
        self.weight_levels = nn.Conv2d(compress_c * 2, 2, kernel_size=1, stride=1, padding=0)
 
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        
        self.level = level

        if self.level == 0:
            self.upsample = Upsample(channel[1], channel[0])
        elif self.level == 1:
            self.downsample = Downsample(channel[0], channel[1])
 
    def forward(self, x):
        input1, input2 = x
        if self.level == 0:
            input2 = self.upsample(input2)
        elif self.level == 1:
            input1 = self.downsample(input1)
 
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
 
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
 
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :]
 
        out = self.conv(fused_out_reduced)
 
        return out
 
 
class ASFF_3(nn.Module):
    #https://arxiv.org/pdf/2306.15988v1.pdf
    def __init__(self, inter_dim=512, level=0, channel=[64, 128, 256]):
        super(ASFF_3, self).__init__()
 
        self.inter_dim = inter_dim
        compress_c = 8
 
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)
 
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
 
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
 
        self.level = level
        if self.level == 0:
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample2x1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x1 = Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
 
    def forward(self, x):
        input1, input2, input3 = x
        if self.level == 0:
            input2 = self.upsample2x(input2)
            input3 = self.upsample4x(input3)
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            input1 = self.downsample4x(input1)
            input2 = self.downsample2x(input2)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)
 
        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)
 
        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
 
        out = self.conv(fused_out_reduced)
 
        return out
    

class ASFF_4(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[32, 64, 128, 256]):
        super(ASFF_4, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 4, 4, kernel_size=1, stride=1, padding=0)

        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

        self.level = level

        if self.level == 0:
            self.upsample8x = Upsample(channel[3], channel[0], scale_factor=8)
            self.upsample4x = Upsample(channel[2], channel[0], scale_factor=4)
            self.upsample2x = Upsample(channel[1], channel[0], scale_factor=2)
        elif self.level == 1:
            self.upsample4x_1 = Upsample(channel[3], channel[1], scale_factor=4)
            self.upsample2x_1 = Upsample(channel[2], channel[1], scale_factor=2)
            self.downsample2x_1 =  Downsample(channel[0], channel[1], scale_factor=2)
        elif self.level == 2:
            self.upsample2x_2 = Upsample(channel[3], channel[2], scale_factor=2)
            self.downsample2x_2 = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x_2 = Downsample(channel[0], channel[2], scale_factor=4)
        elif self.level == 3:
            self.downsample2x_3 = Downsample(channel[2], channel[3], scale_factor=2)
            self.downsample4x_3 = Downsample(channel[1], channel[3], scale_factor=4)
            self.downsample8x_3 = Downsample(channel[0], channel[3], scale_factor=8)

    def forward(self, x):
        input0, input1, input2, input3 = x

        if self.level == 0:
            input1 = self.upsample2x(input1)
            input2 = self.upsample4x(input2)
            input3 = self.upsample8x(input3)
        elif self.level == 1:
            input0 = self.downsample2x_1(input0)
            input2 = self.upsample2x_1(input2)
            input3 = self.upsample4x_1(input3)
        elif self.level == 2:
            input0 = self.downsample4x_2(input0)
            input1 = self.downsample2x_2(input1)
            input3 = self.upsample2x_2(input3)
        elif self.level == 3:
            input0 = self.downsample8x_3(input0)
            input1 = self.downsample4x_3(input1)
            input2 = self.downsample2x_3(input2)

        level_0_weight_v = self.weight_level_0(input0)
        level_1_weight_v = self.weight_level_1(input1)
        level_2_weight_v = self.weight_level_2(input2)
        level_3_weight_v = self.weight_level_3(input3)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = input0 * levels_weight[:, 0:1, :, :] + \
                            input1 * levels_weight[:, 1:2, :, :] + \
                            input2 * levels_weight[:, 2:3, :, :] + \
                            input3 * levels_weight[:, 3:, :, :]

        out = self.conv(fused_out_reduced)

        return out
    



class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out