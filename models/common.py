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
from models.ConvMLP.convmlp import *
from models.ops_dcnv3.modules import DCNv3, DCNv3_pytorch
##### basic ####

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class MP(nn.Module):
    def __init__(self, k=2):
        super(MP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return self.m(x)


class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)
    
    
class ReOrg(nn.Module):
    def __init__(self):
        super(ReOrg, self).__init__()

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Chuncat(nn.Module):
    def __init__(self, dimension=1):
        super(Chuncat, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1 = []
        x2 = []
        for xi in x:
            xi1, xi2 = xi.chunk(2, self.d)
            x1.append(xi1)
            x2.append(xi2)
        return torch.cat(x1+x2, self.d)


class Shortcut(nn.Module):
    def __init__(self, dimension=0):
        super(Shortcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        return x[0]+x[1]


class Foldcut(nn.Module):
    def __init__(self, dimension=0):
        super(Foldcut, self).__init__()
        self.d = dimension

    def forward(self, x):
        x1, x2 = x.chunk(2, self.d)
        return x1+x2


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))
    
class DConv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(DConv, self).__init__()
        self.k = k if d == 1 else k + (k-1)*(d-1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(self.k, p), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class Conv_LN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv_LN, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.norm = nn.LayerNorm(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0,2,3,1).contiguous()
        x = self.norm(x)
        return self.act(x).permute(0,3,1,2).contiguous()


class RobustConv(nn.Module):
    # Robust convolution (use high kernel size 7-11 for: downsampling and other layers). Train for 300 - 450 epochs.
    def __init__(self, c1, c2, k=7, s=1, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv, self).__init__()
        self.conv_dw = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv1x1 = nn.Conv2d(c1, c2, 1, 1, 0, groups=1, bias=True)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv1x1(self.conv_dw(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x


class RobustConv2(nn.Module):
    # Robust convolution 2 (use [32, 5, 2] or [32, 7, 4] or [32, 11, 8] for one of the paths in CSP).
    def __init__(self, c1, c2, k=7, s=4, p=None, g=1, act=True, layer_scale_init_value=1e-6):  # ch_in, ch_out, kernel, stride, padding, groups
        super(RobustConv2, self).__init__()
        self.conv_strided = Conv(c1, c1, k=k, s=s, p=p, g=c1, act=act)
        self.conv_deconv = nn.ConvTranspose2d(in_channels=c1, out_channels=c2, kernel_size=s, stride=s, 
                                              padding=0, bias=True, dilation=1, groups=1
        )
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(c2)) if layer_scale_init_value > 0 else None

    def forward(self, x):
        x = self.conv_deconv(self.conv_strided(x))
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1)) 
        return x
    

def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super(GhostConv, self).__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class Stem(nn.Module):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Stem, self).__init__()
        c_ = int(c2/2)  # hidden channels
        self.cv1 = Conv(c1, c_, 3, 2)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 2)
        self.pool = torch.nn.MaxPool2d(2, stride=2)
        self.cv4 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x = self.cv1(x)
        return self.cv4(torch.cat((self.cv3(self.cv2(x)), self.pool(x)), dim=1))


class DownC(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, n=1, k=2):
        super(DownC, self).__init__()
        c_ = int(c1)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2//2, 3, k)
        self.cv3 = Conv(c1, c2//2, 1, 1)
        self.mp = nn.MaxPool2d(kernel_size=k, stride=k)

    def forward(self, x):
        return torch.cat((self.cv2(self.cv1(x)), self.cv3(self.mp(x))), dim=1)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))
    

class Bottleneck(nn.Module):
    # Darknet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class Res(nn.Module):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Res, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 3, 1, g=g)
        self.cv3 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))


class ResX(Res):
    # ResNet bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels


class Ghost(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super(Ghost, self).__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)

##### end of basic #####


##### cspnet #####

class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class SPPCSPC_LN(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC_LN, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv_LN(c1, c_, 1, 1)
        self.cv2 = Conv_LN(c1, c_, 1, 1)
        self.cv3 = Conv_LN(c_, c_, 3, 1)
        self.cv4 = Conv_LN(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv_LN(4 * c_, c_, 1, 1)
        self.cv6 = Conv_LN(c_, c_, 3, 1)
        self.cv7 = Conv_LN(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class GhostSPPCSPC(SPPCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 1, 1)
        self.cv2 = GhostConv(c1, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 1)
        self.cv4 = GhostConv(c_, c_, 1, 1)
        self.cv5 = GhostConv(4 * c_, c_, 1, 1)
        self.cv6 = GhostConv(c_, c_, 3, 1)
        self.cv7 = GhostConv(2 * c_, c2, 1, 1)


class GhostStem(Stem):
    # Stem
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, p, g, act)
        c_ = int(c2/2)  # hidden channels
        self.cv1 = GhostConv(c1, c_, 3, 2)
        self.cv2 = GhostConv(c_, c_, 1, 1)
        self.cv3 = GhostConv(c_, c_, 3, 2)
        self.cv4 = GhostConv(2 * c_, c2, 1, 1)
        

class BottleneckCSPA(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPB(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class BottleneckCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))


class ResCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class ResXCSPA(ResCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPB(ResCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class ResXCSPC(ResCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Res(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class GhostCSPA(BottleneckCSPA):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPB(BottleneckCSPB):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])


class GhostCSPC(BottleneckCSPC):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[Ghost(c_, c_) for _ in range(n)])

##### end of cspnet #####


##### yolor #####

class ImplicitA(nn.Module):
    def __init__(self, channel, mean=0., std=.02):
        super(ImplicitA, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.zeros(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit + x
    

class ImplicitM(nn.Module):
    def __init__(self, channel, mean=1., std=.02):
        super(ImplicitM, self).__init__()
        self.channel = channel
        self.mean = mean
        self.std = std
        self.implicit = nn.Parameter(torch.ones(1, channel, 1, 1))
        nn.init.normal_(self.implicit, mean=self.mean, std=self.std)

    def forward(self, x):
        return self.implicit * x
    
##### end of yolor #####


##### repvgg #####

class RepConv(nn.Module):
    # Represented convolution
    # https://arxiv.org/abs/2101.03697

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, act=True, deploy=False):
        super(RepConv, self).__init__()

        self.deploy = deploy
        self.groups = g
        self.in_channels = c1
        self.out_channels = c2

        assert k == 3
        assert autopad(k, p) == 1

        padding_11 = autopad(k, p) - k // 2

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

        if deploy:
            self.rbr_reparam = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=True)

        else:
            self.rbr_identity = (nn.BatchNorm2d(num_features=c1) if c2 == c1 and s == 1 else None)

            self.rbr_dense = nn.Sequential(
                nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

            self.rbr_1x1 = nn.Sequential(
                nn.Conv2d( c1, c2, 1, s, padding_11, groups=g, bias=False),
                nn.BatchNorm2d(num_features=c2),
            )

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            return self.act(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.act(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1].running_mean
            running_var = branch[1].running_var
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1].eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return (
            kernel.detach().cpu().numpy(),
            bias.detach().cpu().numpy(),
        )

    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv

    def fuse_repvgg_block(self):    
        if self.deploy:
            return
        print(f"RepConv.fuse_repvgg_block")
                
        self.rbr_dense = self.fuse_conv_bn(self.rbr_dense[0], self.rbr_dense[1])
        
        self.rbr_1x1 = self.fuse_conv_bn(self.rbr_1x1[0], self.rbr_1x1[1])
        rbr_1x1_bias = self.rbr_1x1.bias
        weight_1x1_expanded = torch.nn.functional.pad(self.rbr_1x1.weight, [1, 1, 1, 1])
        
        # Fuse self.rbr_identity
        if (isinstance(self.rbr_identity, nn.BatchNorm2d) or isinstance(self.rbr_identity, nn.modules.batchnorm.SyncBatchNorm)):
            # print(f"fuse: rbr_identity == BatchNorm2d or SyncBatchNorm")
            identity_conv_1x1 = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=self.groups, 
                    bias=False)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.to(self.rbr_1x1.weight.data.device)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.squeeze().squeeze()
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")
            identity_conv_1x1.weight.data.fill_(0.0)
            identity_conv_1x1.weight.data.fill_diagonal_(1.0)
            identity_conv_1x1.weight.data = identity_conv_1x1.weight.data.unsqueeze(2).unsqueeze(3)
            # print(f" identity_conv_1x1.weight = {identity_conv_1x1.weight.shape}")

            identity_conv_1x1 = self.fuse_conv_bn(identity_conv_1x1, self.rbr_identity)
            bias_identity_expanded = identity_conv_1x1.bias
            weight_identity_expanded = torch.nn.functional.pad(identity_conv_1x1.weight, [1, 1, 1, 1])            
        else:
            # print(f"fuse: rbr_identity != BatchNorm2d, rbr_identity = {self.rbr_identity}")
            bias_identity_expanded = torch.nn.Parameter( torch.zeros_like(rbr_1x1_bias) )
            weight_identity_expanded = torch.nn.Parameter( torch.zeros_like(weight_1x1_expanded) )            
        

        #print(f"self.rbr_1x1.weight = {self.rbr_1x1.weight.shape}, ")
        #print(f"weight_1x1_expanded = {weight_1x1_expanded.shape}, ")
        #print(f"self.rbr_dense.weight = {self.rbr_dense.weight.shape}, ")

        self.rbr_dense.weight = torch.nn.Parameter(self.rbr_dense.weight + weight_1x1_expanded + weight_identity_expanded)
        self.rbr_dense.bias = torch.nn.Parameter(self.rbr_dense.bias + rbr_1x1_bias + bias_identity_expanded)
                
        self.rbr_reparam = self.rbr_dense
        self.deploy = True

        if self.rbr_identity is not None:
            del self.rbr_identity
            self.rbr_identity = None

        if self.rbr_1x1 is not None:
            del self.rbr_1x1
            self.rbr_1x1 = None

        if self.rbr_dense is not None:
            del self.rbr_dense
            self.rbr_dense = None


class RepBottleneck(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut=True, g=1, e=0.5)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c2, 3, 1, g=g)


class RepBottleneckCSPA(BottleneckCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPB(BottleneckCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepBottleneckCSPC(BottleneckCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])


class RepRes(Res):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResCSPA(ResCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPB(ResCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResCSPC(ResCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepRes(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResX(ResX):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv2 = RepConv(c_, c_, 3, 1, g=g)


class RepResXCSPA(ResXCSPA):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPB(ResXCSPB):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])


class RepResXCSPC(ResXCSPC):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=32, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*[RepResX(c_, c_, shortcut, g, e=0.5) for _ in range(n)])

##### end of repvgg #####


##### transformer #####

class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x

##### end of transformer #####


##### yolov5 #####

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))
        

class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))
    
    
class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        print('autoShape already enabled, skipping... ')  # model already converted to model.autoshape()
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/samples/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[:, :, :3] if im.ndim == 3 else np.tile(im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im  # update
        shape1 = [make_divisible(x, int(self.stride.max())) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, render=False, save_dir=''):
        colors = color_list()
        for i, (img, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {img.shape[0]}x{img.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(box, img, label=label, color=colors[int(cls) % 10])
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                img.show(self.files[i])  # show
            if save:
                f = self.files[i]
                img.save(Path(save_dir) / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i < self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(img)

    def print(self):
        self.display(pprint=True)  # print results
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/hub/exp')  # increment save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        self.display(save=True, save_dir=save_dir)  # save results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

##### end of yolov5 ######


##### orepa #####

def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn.running_var + bn.eps).sqrt()
    return kernel * ((gamma / std).reshape(-1, 1, 1, 1)), bn.bias - bn.running_mean * gamma / std
    
    
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                             stride=1, padding=0, dilation=1, groups=1, deploy=False, nonlinear=None):
        super().__init__()
        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear
        if deploy:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                      stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if hasattr(self, 'bn'):
            return self.nonlinear(self.bn(self.conv(x)))
        else:
            return self.nonlinear(self.conv(x))

    def switch_to_deploy(self):
        kernel, bias = transI_fusebn(self.conv.weight, self.bn)
        conv = nn.Conv2d(in_channels=self.conv.in_channels, out_channels=self.conv.out_channels, kernel_size=self.conv.kernel_size,
                                      stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups, bias=True)
        conv.weight.data = kernel
        conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv')
        self.__delattr__('bn')
        self.conv = conv    

class OREPA_3x3_RepConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 internal_channels_1x1_3x3=None,
                 deploy=False, nonlinear=None, single_init=False):
        super(OREPA_3x3_RepConv, self).__init__()
        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nonlinear

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.branch_counter = 0

        self.weight_rbr_origin = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_origin, a=math.sqrt(1.0))
        self.branch_counter += 1


        if groups < out_channels:
            self.weight_rbr_avg_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            self.weight_rbr_pfir_conv = nn.Parameter(torch.Tensor(out_channels, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_avg_conv, a=1.0)
            nn.init.kaiming_uniform_(self.weight_rbr_pfir_conv, a=1.0)
            self.weight_rbr_avg_conv.data
            self.weight_rbr_pfir_conv.data
            self.register_buffer('weight_rbr_avg_avg', torch.ones(kernel_size, kernel_size).mul(1.0/kernel_size/kernel_size))
            self.branch_counter += 1

        else:
            raise NotImplementedError
        self.branch_counter += 1

        if internal_channels_1x1_3x3 is None:
            internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels   # For mobilenet, it is better to have 2X internal channels

        if internal_channels_1x1_3x3 == in_channels:
            self.weight_rbr_1x1_kxk_idconv1 = nn.Parameter(torch.zeros(in_channels, int(in_channels/self.groups), 1, 1))
            id_value = np.zeros((in_channels, int(in_channels/self.groups), 1, 1))
            for i in range(in_channels):
                id_value[i, i % int(in_channels/self.groups), 0, 0] = 1
            id_tensor = torch.from_numpy(id_value).type_as(self.weight_rbr_1x1_kxk_idconv1)
            self.register_buffer('id_tensor', id_tensor)

        else:
            self.weight_rbr_1x1_kxk_conv1 = nn.Parameter(torch.Tensor(internal_channels_1x1_3x3, int(in_channels/self.groups), 1, 1))
            nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv1, a=math.sqrt(1.0))
        self.weight_rbr_1x1_kxk_conv2 = nn.Parameter(torch.Tensor(out_channels, int(internal_channels_1x1_3x3/self.groups), kernel_size, kernel_size))
        nn.init.kaiming_uniform_(self.weight_rbr_1x1_kxk_conv2, a=math.sqrt(1.0))
        self.branch_counter += 1

        expand_ratio = 8
        self.weight_rbr_gconv_dw = nn.Parameter(torch.Tensor(in_channels*expand_ratio, 1, kernel_size, kernel_size))
        self.weight_rbr_gconv_pw = nn.Parameter(torch.Tensor(out_channels, in_channels*expand_ratio, 1, 1))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_dw, a=math.sqrt(1.0))
        nn.init.kaiming_uniform_(self.weight_rbr_gconv_pw, a=math.sqrt(1.0))
        self.branch_counter += 1

        if out_channels == in_channels and stride == 1:
            self.branch_counter += 1

        self.vector = nn.Parameter(torch.Tensor(self.branch_counter, self.out_channels))
        self.bn = nn.BatchNorm2d(out_channels)

        self.fre_init()

        nn.init.constant_(self.vector[0, :], 0.25)    #origin
        nn.init.constant_(self.vector[1, :], 0.25)      #avg
        nn.init.constant_(self.vector[2, :], 0.0)      #prior
        nn.init.constant_(self.vector[3, :], 0.5)    #1x1_kxk
        nn.init.constant_(self.vector[4, :], 0.5)     #dws_conv


    def fre_init(self):
        prior_tensor = torch.Tensor(self.out_channels, self.kernel_size, self.kernel_size)
        half_fg = self.out_channels/2
        for i in range(self.out_channels):
            for h in range(3):
                for w in range(3):
                    if i < half_fg:
                        prior_tensor[i, h, w] = math.cos(math.pi*(h+0.5)*(i+1)/3)
                    else:
                        prior_tensor[i, h, w] = math.cos(math.pi*(w+0.5)*(i+1-half_fg)/3)

        self.register_buffer('weight_rbr_prior', prior_tensor)

    def weight_gen(self):

        weight_rbr_origin = torch.einsum('oihw,o->oihw', self.weight_rbr_origin, self.vector[0, :])

        weight_rbr_avg = torch.einsum('oihw,o->oihw', torch.einsum('oihw,hw->oihw', self.weight_rbr_avg_conv, self.weight_rbr_avg_avg), self.vector[1, :])
        
        weight_rbr_pfir = torch.einsum('oihw,o->oihw', torch.einsum('oihw,ohw->oihw', self.weight_rbr_pfir_conv, self.weight_rbr_prior), self.vector[2, :])

        weight_rbr_1x1_kxk_conv1 = None
        if hasattr(self, 'weight_rbr_1x1_kxk_idconv1'):
            weight_rbr_1x1_kxk_conv1 = (self.weight_rbr_1x1_kxk_idconv1 + self.id_tensor).squeeze()
        elif hasattr(self, 'weight_rbr_1x1_kxk_conv1'):
            weight_rbr_1x1_kxk_conv1 = self.weight_rbr_1x1_kxk_conv1.squeeze()
        else:
            raise NotImplementedError
        weight_rbr_1x1_kxk_conv2 = self.weight_rbr_1x1_kxk_conv2

        if self.groups > 1:
            g = self.groups
            t, ig = weight_rbr_1x1_kxk_conv1.size()
            o, tg, h, w = weight_rbr_1x1_kxk_conv2.size()
            weight_rbr_1x1_kxk_conv1 = weight_rbr_1x1_kxk_conv1.view(g, int(t/g), ig)
            weight_rbr_1x1_kxk_conv2 = weight_rbr_1x1_kxk_conv2.view(g, int(o/g), tg, h, w)
            weight_rbr_1x1_kxk = torch.einsum('gti,gothw->goihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2).view(o, ig, h, w)
        else:
            weight_rbr_1x1_kxk = torch.einsum('ti,othw->oihw', weight_rbr_1x1_kxk_conv1, weight_rbr_1x1_kxk_conv2)

        weight_rbr_1x1_kxk = torch.einsum('oihw,o->oihw', weight_rbr_1x1_kxk, self.vector[3, :])

        weight_rbr_gconv = self.dwsc2full(self.weight_rbr_gconv_dw, self.weight_rbr_gconv_pw, self.in_channels)
        weight_rbr_gconv = torch.einsum('oihw,o->oihw', weight_rbr_gconv, self.vector[4, :])    

        weight = weight_rbr_origin + weight_rbr_avg + weight_rbr_1x1_kxk + weight_rbr_pfir + weight_rbr_gconv

        return weight

    def dwsc2full(self, weight_dw, weight_pw, groups):
        
        t, ig, h, w = weight_dw.size()
        o, _, _, _ = weight_pw.size()
        tg = int(t/groups)
        i = int(ig*groups)
        weight_dw = weight_dw.view(groups, tg, ig, h, w)
        weight_pw = weight_pw.squeeze().view(o, groups, tg)
        
        weight_dsc = torch.einsum('gtihw,ogt->ogihw', weight_dw, weight_pw)
        return weight_dsc.view(o, i, h, w)

    def forward(self, inputs):
        weight = self.weight_gen()
        out = F.conv2d(inputs, weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        return self.nonlinear(self.bn(out))

class RepConv_OREPA(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False, nonlinear=nn.SiLU()):
        super(RepConv_OREPA, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = c1
        self.out_channels = c2

        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        assert k == 3
        assert padding == 1

        padding_11 = padding - k // 2

        if nonlinear is None:
            self.nonlinearity = nn.Identity()
        else:
            self.nonlinearity = nonlinear

        if use_se:
            self.se = SEBlock(self.out_channels, internal_neurons=self.out_channels // 16)
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=self.in_channels) if self.out_channels == self.in_channels and s == 1 else None
            self.rbr_dense = OREPA_3x3_RepConv(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=k, stride=s, padding=padding, groups=groups, dilation=1)
            self.rbr_1x1 = ConvBN(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=s, padding=padding_11, groups=groups, dilation=1)
            print('RepVGG Block, identity = ', self.rbr_identity)


    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        out1 = self.rbr_dense(inputs)
        out2 = self.rbr_1x1(inputs)
        out3 = id_out
        out = out1 + out2 + out3

        return self.nonlinearity(self.se(out))


    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    # Not used for OREPA
    def get_custom_L2(self):
        K3 = self.rbr_dense.weight_gen()
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight / ((self.rbr_dense.bn.running_var + self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var + self.rbr_1x1.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()

        l2_loss_circle = (K3 ** 2).sum() - (K3[:, :, 1:2, 1:2] ** 2).sum()      # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:2] * t3 + K1 * t1                           # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel ** 2 / (t3 ** 2 + t1 ** 2)).sum()        # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1,1,1,1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if not isinstance(branch, nn.BatchNorm2d):
            if isinstance(branch, OREPA_3x3_RepConv):
                kernel = branch.weight_gen()
            elif isinstance(branch, ConvBN):
                kernel = branch.conv.weight
            else:
                raise NotImplementedError
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        print(f"RepConv_OREPA.switch_to_deploy")
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.in_channels, out_channels=self.rbr_dense.out_channels,
                                     kernel_size=self.rbr_dense.kernel_size, stride=self.rbr_dense.stride,
                                     padding=self.rbr_dense.padding, dilation=self.rbr_dense.dilation, groups=self.rbr_dense.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity') 

##### end of orepa #####


##### swin transformer #####    
    
class WindowAttention(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # print(attn.dtype, v.dtype)
        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            #print(attn.dtype, v.dtype)
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):

    B, H, W, C = x.shape
    assert H % window_size == 0, 'feature map h and w can not divide by window size'
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer(nn.Module):

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        # if min(self.input_resolution) <= self.window_size:
        #     # if window size is larger than input resolution, we don't partition windows
        #     self.shift_size = 0
        #     self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w

        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=8):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class STCSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class STCSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(STCSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformerBlock(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

##### end of swin transformer #####   


##### swin transformer v2 ##### 
  
class WindowAttention_v2(nn.Module):

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        try:
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        except:
            x = (attn.half() @ v).transpose(1, 2).reshape(B_, N, C)
            
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    
class Mlp_v2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition_v2(x, window_size):
    
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse_v2(windows, window_size, H, W):
    
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerLayer_v2(nn.Module):

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.SiLU, norm_layer=nn.LayerNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        #self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        #if min(self.input_resolution) <= self.window_size:
        #    # if window size is larger than input resolution, we don't partition windows
        #    self.shift_size = 0
        #    self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention_v2(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=(pretrained_window_size, pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_v2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def create_mask(self, H, W):
        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x):
        # reshape x[b c h w] to x[b l c]
        _, _, H_, W_ = x.shape

        Padding = False
        if min(H_, W_) < self.window_size or H_ % self.window_size!=0 or W_ % self.window_size!=0:
            Padding = True
            # print(f'img_size {min(H_, W_)} is less than (or not divided by) window_size {self.window_size}, Padding.')
            pad_r = (self.window_size - W_ % self.window_size) % self.window_size
            pad_b = (self.window_size - H_ % self.window_size) % self.window_size
            x = F.pad(x, (0, pad_r, 0, pad_b))

        # print('2', x.shape)
        B, C, H, W = x.shape
        L = H * W
        x = x.permute(0, 2, 3, 1).contiguous().view(B, L, C)  # b, L, c

        # create mask from init to forward
        if self.shift_size > 0:
            attn_mask = self.create_mask(H, W).to(x.device)
        else:
            attn_mask = None

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition_v2(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse_v2(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)  # b c h w
        
        if Padding:
            x = x[:, :, :H_, :W_]  # reverse padding

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class SwinTransformer2Block(nn.Module):
    def __init__(self, c1, c2, num_heads, num_layers, window_size=7):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)

        # remove input_resolution
        self.blocks = nn.Sequential(*[SwinTransformerLayer_v2(dim=c2, num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2) for i in range(num_layers)])

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        x = self.blocks(x)
        return x


class ST2CSPA(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPA, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.m(self.cv1(x))
        y2 = self.cv2(x)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPB(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPB, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(torch.cat((y1, y2), dim=1))


class ST2CSPC(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(ST2CSPC, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 1, 1)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        num_heads = c_ // 32
        self.m = SwinTransformer2Block(c_, c_, num_heads, n)
        #self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(torch.cat((y1, y2), dim=1))

##### end of swin transformer v2 #####   
class DFL(nn.Module):
    # DFL module
    def __init__(self, c1=17):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)) # / 120.0
        self.c1 = c1
        # self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)



##### start of custom function module #####

# Strip_Conv_k3
class SC_k3(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function
    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, act=True): 
        super(SC_k3, self).__init__()
        self.identity_conv1 = nn.Conv2d(cin, cin, 1, 1, groups=g, bias=False)
        self.conv1 = nn.Conv2d(cin, cout, k, s, autopad(k,p), groups=g, bias=False)
        self.identity_strip = nn.Conv2d(cin, cin, 1, 1, groups=g, bias=False)
        self.strip_conv = nn.Conv2d(cin,cout, (k,1), s, (autopad(k,p),0), groups=g, bias=False)
        self.conv2 = nn.Conv2d(cin, cout, k, s, autopad(k,p), groups=g, bias=False)
        self.identity = nn.Conv2d(cin, cout, 1, 1, groups=g, bias=False)
        # bn for conv1
        self.bn_conv1 = nn.BatchNorm2d(cout)
        # bn for strip conv
        self.bn_strip = nn.BatchNorm2d(cout)
        # bn for conv2
        self.bn_conv2 = nn.BatchNorm2d(cout)
        # bn for indentiy conv
        self.bn_identity = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # training
    def forward(self,x):
        return self.act( self.bn_conv1(self.conv1(self.identity_conv1(x)))
                        +self.bn_strip(self.strip_conv(self.identity_strip(x)))
                        +self.bn_identity(self.identity(x))
                        +self.bn_conv2(self.conv2(x))
                        )

    # inferencing
    def fuseforward(self, x):
        return self.act(self.conv1(x))
    
    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv
    
    def fuseSCk3(self):
        # fuse conv and bn first
        print("fuse SCk3")
        self.conv1 = self.fuse_conv_bn(self.conv1, self.bn_conv1)
        self.strip_conv = self.fuse_conv_bn(self.strip_conv, self.bn_strip)
        self.identity = self.fuse_conv_bn(self.identity,self.bn_identity)
        self.conv2 = self.fuse_conv_bn(self.conv2, self.bn_conv2)
        # fuse four conv in one
        self.conv1.weight = nn.Parameter(nn.functional.conv2d(self.conv1.weight.data, self.identity_conv1.weight.data.permute(1,0,2,3)))
        strip_weight = nn.functional.conv2d(nn.functional.pad(self.strip_conv.weight.data,(1,1,0,0),'constant',0), self.identity_strip.weight.data.permute(1,0,2,3))
        all_weight = self.conv1.weight.data + self.conv2.weight.data + strip_weight
        all_weight[:,:,1,1] += self.identity.weight.data[:,:,0,0]
        self.conv1.weight = nn.Parameter(all_weight)
        self.conv1.bias = nn.Parameter(self.conv1.bias.data + self.conv2.bias.data + self.strip_conv.bias.data + self.identity.bias.data)

        # delete non-used layers
        del self.bn_conv1
        self.bn_conv1 = None
        del self.identity_conv1
        self.identity_conv1 = None
        del self.bn_strip
        self.bn_strip = None
        del self.strip_conv
        self.strip_conv = None
        del self.identity_strip
        self.identity_strip = None
        del self.conv2
        self.conv2 = None
        del self.identity
        self.identity = None
        del self.bn_conv2
        self.bn_conv2 = None
        del self.bn_identity
        self.bn_identity = None

        # set new forward path
        self.forward = self.fuseforward

# SC_k3_tiny
class SC_k3_tiny(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function
    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, act=True): 
        super(SC_k3_tiny, self).__init__()
        self.conv = nn.Conv2d(cin, cout, 3, s, autopad(3,p), groups=g, bias=False)
        self.strip_conv = nn.Conv2d(cin, cout, (3,1), s, (autopad(3,p),0), groups=g, bias=False)
        # 可實驗是否需要此分支
        self.identity = nn.Conv2d(cin, cout, 1, 1, groups=g, bias=False)
        # bn for conv
        self.bn1 = nn.BatchNorm2d(cout)
        # bn for strip conv
        self.bn2 = nn.BatchNorm2d(cout)
        # bn for indentiy conv
        self.bn3 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # training    
    def forward(self,x):
        return self.act(self.bn1(self.conv(x))+self.bn2(self.strip_conv(x))+self.bn3(self.identity(x)))
    # inferencing
    def fuseforward(self,x):
        return self.act(self.conv(x))
    
    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv
    
    # fusing all layers
    def fuseSCk3_tiny(self):
        # fuse conv and bn first
        print("fuse fuseSCk3_tiny")
        self.conv = self.fuse_conv_bn(self.conv,self.bn1)
        self.strip_conv = self.fuse_conv_bn(self.strip_conv,self.bn2)
        self.identity = self.fuse_conv_bn(self.identity,self.bn3)

        strip_weight = nn.functional.pad(self.strip_conv.weight.data,(1,1,0,0),'constant',0)
        all_weight = self.conv.weight.data + strip_weight
        all_weight[:,:,1,1] += self.identity.weight.data[:,:,0,0]
        self.conv.weight = nn.Parameter(all_weight)
        self.conv.bias = nn.Parameter(self.conv.bias.data + self.strip_conv.bias.data + self.identity.bias.data)

        # delete non-used layers
        del self.bn1
        self.bn1 = None
        del self.bn2
        self.bn2 = None
        del self.strip_conv
        self.strip_conv = None
        del self.bn3
        self.bn3 = None
        del self.identity
        self.identity = None

        # set new forward path
        self.forward = self.fuseforward


# Dilated_Strip_Conv_k5 
class DSC_k5(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function
    def __init__(self, cin, cout, k=5, s=1, p=None, g=1, act=True): 
        super(DSC_k5, self).__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k,p), groups=g, bias=False)
        self.strip_conv = nn.Conv2d(cin, cout, (k,1), s, (autopad(k,p),0), groups=g, bias=False)
        self.dilated = nn.Conv2d(cin, cout, k//2+1, s, autopad(k,p), groups=g, dilation=2, bias=False)
        # 可實驗是否需要此分支
        self.identity = nn.Conv2d(cin, cout, 1, 1, groups=g, bias=False)
        # bn for conv
        self.bn1 = nn.BatchNorm2d(cout)
        # bn for strip conv
        self.bn2 = nn.BatchNorm2d(cout)
        # bn for dilated conv
        self.bn3 = nn.BatchNorm2d(cout)
        # bn for indentiy conv
        self.bn4 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # training    
    def forward(self,x):
        return self.act(self.bn1(self.conv(x))+self.bn2(self.strip_conv(x))+self.bn3(self.dilated(x))+self.bn4(self.identity(x)))
    # inferencing
    def fuseforward(self,x):
        return self.act(self.conv(x))
    
    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv
    
    # testing dilated conv and conv results
    def dilated3x3toConv5x5(self, dilated,x):
        conv = nn.Conv2d(in_channels = dilated.in_channels,
                              out_channels = dilated.out_channels,
                              kernel_size = dilated.kernel_size[0]*2-1,
                              stride=dilated.stride,
                              padding = autopad(dilated.kernel_size[0]*2-1),
                              dilation = 1,
                              groups = dilated.groups,
                              bias = True,
                              padding_mode = dilated.padding_mode)
        dilated_weight = torch.zeros_like(self.conv.weight.data)
        for i in range(3):
            for j in range(3):
                dilated_weight[:,:,i*2,j*2] = dilated.weight.data[:,:,i,j]
        conv.weight = nn.Parameter(dilated_weight)
        conv.bias = dilated.bias
        return conv(x)
    
    # fusing all layers
    def fuseDSCk5(self):
        # fuse conv and bn first
        print("fuse DSC_k5")
        self.conv = self.fuse_conv_bn(self.conv,self.bn1)
        self.strip_conv = self.fuse_conv_bn(self.strip_conv,self.bn2)
        self.dilated = self.fuse_conv_bn(self.dilated,self.bn3)
        self.identity = self.fuse_conv_bn(self.identity,self.bn4)
        # fuse four conv in one
        dilated_identity_weight = torch.zeros_like(self.conv.weight.data)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    dilated_identity_weight[:,:,i*2,j*2] = self.dilated.weight.data[:,:,i,j] + self.identity.weight.data[:,:,0,0]
                else:
                    dilated_identity_weight[:,:,i*2,j*2] = self.dilated.weight.data[:,:,i,j]
        strip_weight = nn.functional.pad(self.strip_conv.weight.data,(2,2,0,0),'constant',0)
        self.conv.weight = nn.Parameter(self.conv.weight.data + dilated_identity_weight + strip_weight)
        self.conv.bias = nn.Parameter(self.conv.bias.data + self.dilated.bias.data + self.strip_conv.bias.data + self.identity.bias.data)

        # delete non-used layers
        del self.bn1
        self.bn1 = None
        del self.dilated
        self.dilated = None
        del self.bn2
        self.bn2 = None
        del self.strip_conv
        self.strip_conv = None
        del self.bn3
        self.bn3 = None
        del self.identity
        self.identity = None
        del self.bn4
        self.bn4 = None

        # set new forward path
        self.forward = self.fuseforward

# DSC_k5 with 2 strip conv 變差
class DSC_k5_v1(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function
    def __init__(self, cin, cout, k=5, s=1, p=None, g=1, act=True): 
        super(DSC_k5_v1, self).__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, autopad(k,p), groups=g, bias=False)
        self.strip_conv5 = nn.Conv2d(cin, cout, (k,1), s, (autopad(k,p),0), groups=g, bias=False)
        self.dilated = nn.Conv2d(cin, cout, k//2+1, s, autopad(k,p), groups=g, dilation=2, bias=False)
        # 可實驗是否需要此分支
        self.identity = nn.Conv2d(cin, cout, 1, 1, groups=g, bias=False)
        self.strip_conv3 = nn.Conv2d(cin, cout, (3,1), s, (autopad(3,p),0), groups=g, bias=False)
        # bn for conv
        self.bn1 = nn.BatchNorm2d(cout)
        # bn for strip conv5
        self.bn2 = nn.BatchNorm2d(cout)
        # bn for dilated conv
        self.bn3 = nn.BatchNorm2d(cout)
        # bn for indentiy conv
        self.bn4 = nn.BatchNorm2d(cout)
        # bn for strip conv5
        self.bn5 = nn.BatchNorm2d(cout)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # training    
    def forward(self,x):
        return self.act(self.bn1(self.conv(x))+self.bn2(self.strip_conv5(x))+self.bn3(self.dilated(x))+self.bn4(self.identity(x))+self.bn5(self.strip_conv3(x)))
    # inferencing
    def fuseforward(self,x):
        return self.act(self.conv(x))
    
    def fuse_conv_bn(self, conv, bn):

        std = (bn.running_var + bn.eps).sqrt()
        bias = bn.bias - bn.running_mean * bn.weight / std

        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        weights = conv.weight * t

        bn = nn.Identity()
        conv = nn.Conv2d(in_channels = conv.in_channels,
                              out_channels = conv.out_channels,
                              kernel_size = conv.kernel_size,
                              stride=conv.stride,
                              padding = conv.padding,
                              dilation = conv.dilation,
                              groups = conv.groups,
                              bias = True,
                              padding_mode = conv.padding_mode)

        conv.weight = torch.nn.Parameter(weights)
        conv.bias = torch.nn.Parameter(bias)
        return conv
    
    # testing dilated conv and conv results
    def dilated3x3toConv5x5(self, dilated,x):
        conv = nn.Conv2d(in_channels = dilated.in_channels,
                              out_channels = dilated.out_channels,
                              kernel_size = dilated.kernel_size[0]*2-1,
                              stride=dilated.stride,
                              padding = autopad(dilated.kernel_size[0]*2-1),
                              dilation = 1,
                              groups = dilated.groups,
                              bias = True,
                              padding_mode = dilated.padding_mode)
        dilated_weight = torch.zeros_like(self.conv.weight.data)
        for i in range(3):
            for j in range(3):
                dilated_weight[:,:,i*2,j*2] = dilated.weight.data[:,:,i,j]
        conv.weight = nn.Parameter(dilated_weight)
        conv.bias = dilated.bias
        return conv(x)
    
    # fusing all layers
    def fuseDSCk5(self):
        # fuse conv and bn first
        print("fuse DSC_k5_v1")
        self.conv = self.fuse_conv_bn(self.conv,self.bn1)
        self.strip_conv5 = self.fuse_conv_bn(self.strip_conv5,self.bn2)
        self.dilated = self.fuse_conv_bn(self.dilated,self.bn3)
        self.identity = self.fuse_conv_bn(self.identity,self.bn4)
        self.strip_conv3 = self.fuse_conv_bn(self.strip_conv3,self.bn5)
        # fuse four conv in one
        dilated_identity_weight = torch.zeros_like(self.conv.weight.data)
        for i in range(3):
            for j in range(3):
                if i == 1 and j == 1:
                    dilated_identity_weight[:,:,i*2,j*2] = self.dilated.weight.data[:,:,i,j] + self.identity.weight.data[:,:,0,0]
                else:
                    dilated_identity_weight[:,:,i*2,j*2] = self.dilated.weight.data[:,:,i,j]
        strip_weight5 = nn.functional.pad(self.strip_conv5.weight.data,(2,2,0,0),'constant',0)
        strip_weight3 = nn.functional.pad(self.strip_conv3.weight.data,(2,2,1,1),'constant',0)
        self.conv.weight = nn.Parameter(self.conv.weight.data + dilated_identity_weight + strip_weight5 + strip_weight3)
        self.conv.bias = nn.Parameter(self.conv.bias.data + self.dilated.bias.data + self.strip_conv5.bias.data + self.identity.bias.data + self.strip_conv3.bias.data)

        # delete non-used layers
        del self.bn1
        self.bn1 = None
        del self.dilated
        self.dilated = None
        del self.bn2
        self.bn2 = None
        del self.strip_conv5
        self.strip_conv5 = None
        del self.bn3
        self.bn3 = None
        del self.identity
        self.identity = None
        del self.bn4
        self.bn4 = None
        del self.strip_conv3
        self.strip_conv3 = None
        del self.bn5
        self.bn5 = None
        # set new forward path
        self.forward = self.fuseforward
# C4
class C4(nn.Module):
    # channel_in, channel_out, split_ratio=1 or 2
    def __init__(self, cin, cout,r=1): 
        super(C4, self).__init__()
        # conv1 output = DSC_k5 input tensor
        self.conv1 = Conv(cin, cin//r, k=1, s=1)
        self.identity = Conv(cin, cin//r, k=1, s=1)
        self.DSC_k5 = DSC_k5(cin//r, cin, k=5, s=1)
        self.conv2 = Conv(cin+2*(cin//r), cout, k=1, s=1)
    def forward(self,x):
        x1 = self.conv1(x)
        x2 = self.DSC_k5(x1)
        #shortcut
        sc = self.identity(x)
        return self.conv2(torch.cat((x1,x2,sc),1))


class ConvMLP(nn.Module):
    def __init__(self,pretrained=True):
        super(ConvMLP, self).__init__()
        self.backbone = convmlp_s(classifier_head=None, pretrained=pretrained)
    def forward(self, x):
        return self.backbone(x)

class ConvMLP_FPN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, fpn, k=1, s=1, p=None, g=1, act=True):  # fpn_num, ch_in, ch_out, kernel, stride, padding, groups
        super(ConvMLP_FPN, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.fpn = fpn

    def forward(self, x):
        return self.act(self.bn(self.conv(x[self.fpn])))

    def fuseforward(self, x):
        return self.act(self.conv(x[self.fpn]))
    
# --------------------------DCNv2 start--------------------------
class DCNv2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        # mask = torch.Softmax(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.norm(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

class DCNv2_1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, groups=1, act=True, dilation=1, deformable_groups=1):
        super(DCNv2_1, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size)
        self.stride = (stride, stride)
        self.padding = (autopad(kernel_size, padding), autopad(kernel_size, padding))
        self.dilation = (dilation, dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(out_channels))

        out_channels_offset_mask = (self.deformable_groups * 3 *
                                    self.kernel_size[0] * self.kernel_size[1])
        self.conv_offset_mask = nn.Conv2d(
            self.in_channels,
            out_channels_offset_mask,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True,
        )
        # self.norm = nn.BatchNorm2d(out_channels)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.reset_parameters()

    def forward(self, x):
        offset_mask = self.conv_offset_mask(x)
        o1, o2, mask = torch.chunk(offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        # mask = torch.sigmoid(mask)
        mask = torch.softmax(mask)
        x = torch.ops.torchvision.deform_conv2d(
            x,
            self.weight,
            offset,
            mask,
            self.bias,
            self.stride[0], self.stride[1],
            self.padding[0], self.padding[1],
            self.dilation[0], self.dilation[1],
            self.groups,
            self.deformable_groups,
            True
        )
        x = self.norm(x)
        x = self.act(x)
        return x

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        std = 1. / math.sqrt(n)
        self.weight.data.uniform_(-std, std)
        self.bias.data.zero_()
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()
# ---------------------------DCNv2 end---------------------------

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class MLPLayer(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act=True,
                 drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1 = x.permute(0,2,3,1).contiguous()
        x1 = self.norm1(x1)
        x1 = self.fc1(x1)
        x1 = self.act(x1)
        x1 = self.fc2(x1)
        x1 = self.norm2(x1)
        x1 = self.drop(x1)
        return x + x1.permute(0,3,1,2).contiguous()

#old SC_k3_m, SC_k3_m_att_win_400
# class MLPLayer(nn.Module):
#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act=True,
#                  drop=0.1):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.norm = nn.LayerNorm(out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x1 = x.permute(0,2,3,1).contiguous()
#         x1 = self.fc1(x1)
#         x1 = self.act(x1)
#         x1 = self.fc2(x1)
#         x1 = self.norm(x1)
#         x1 = self.drop(x1)
#         return x + x1.permute(0,3,1,2).contiguous()

class DCNv2_block(nn.Module):
    def __init__(self, c1, c2, num=3, k=3, s=1, p=1, g=1, act=True):
        super(DCNv2_block, self).__init__()
        self.DCN1 = DCNv2(c1, c2, k, s, p, g)
        DCNs = [DCNv2(c2, c2, k, s, p, g) for i in range(num-1)]
        self.DCNs = nn.ModuleList(DCNs)
        self.MLP = MLPLayer(c2, c2*2, c2)

    def forward(self, x):
        short_cut = x
        x = self.DCN1(x)
        for blk in self.DCNs:
            x = blk(x)
        x = self.MLP(x + short_cut)
        return x

class DCNv3_block_MLP(nn.Module):
    def __init__(self, c2, num=3, k=3, s=1, p=1, g=1, act=True, drop=0.1):
        super(DCNv3_block_MLP, self).__init__()
        DCNs = [DCNv3_block(c2, k, s, p, g=g) for i in range(num)]
        self.DCNs = nn.ModuleList(DCNs)
        self.MLP = MLPLayer(c2, c2*2, c2)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        short_cut = x
        for blk in self.DCNs:
            x = blk(x)
        dcn = x + short_cut       
        return self.MLP(dcn) + dcn

class DCNv3_block_cnn(nn.Module):
    def __init__(self, c2, num=3, k=3, s=1, p=1, g=1, act=True, drop=0.1):
        super(DCNv3_block_cnn, self).__init__()
        DCNs = [DCNv3_block_BN(c2, k, s, p, g=g, act=act) for i in range(num)]
        self.DCNs = nn.ModuleList(DCNs)
        self.mlp = Conv_LN(c2, c2, k=1, s=1, p=None, g=g, act=act)
    def forward(self, x):
        short_cut = x
        for blk in self.DCNs:
            x = blk(x)
        x = self.mlp(x)
        return x + short_cut


class DCNv3_block(nn.Module):
    def __init__(self, c2, k=3, s=1, p=1, g=1, act=True):
        super(DCNv3_block, self).__init__()
        self.dcnv3 = DCNv3_pytorch(c2, kernel_size=k, stride=s, group=g, dilation=1)
        self.norm = nn.LayerNorm(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        x = self.dcnv3(x)
        x = self.norm(x)
        x = self.act(x)
        return x.permute(0,3,1,2).contiguous()

class DCNv3_block_BN(nn.Module):
    def __init__(self, c2, k=3, s=1, p=1, g=1, act=True):
        super(DCNv3_block_BN, self).__init__()
        self.dcnv3 = DCNv3_pytorch(c2, kernel_size=k, stride=s, group=g, dilation=1)
        self.norm = nn.BatchNorm2d(c2)
        self.act = nn.GELU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        x = self.dcnv3(x)
        x = x.permute(0,3,1,2).contiguous()
        x = self.norm(x)
        x = self.act(x)
        return x
    
class SelfAttention(nn.Module):
    """ Self attention Layer"""
    def __init__(self,c1, c2):
        super(SelfAttention,self).__init__()
        self.chanel_in = c1
        
        self.query_conv = nn.Conv2d(in_channels = c1 , out_channels = c1//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = c1 , out_channels = c1//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = c1 , out_channels = c1 , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma * out + x
        # out = self.MLP(out) + out
        out = self.MLP(out)
        return out
    
class atten(nn.Module):
    def __init__(self, c1, c2):
        super(atten, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)
    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize,-1,width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = self.gamma * out + x
        # out = self.MLP(out) + out
        out = self.MLP(out)
        return out

class atten_v2(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v2, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma = nn.Parameter(torch.zeros(1))


        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2), c2)
        self.SE = SELayer(c1)

    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize,-1,width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,width,height)

        out = (self.gamma * out) + self.SE(x) + x
        out = self.MLP(out) + out
        return out

class atten_v3(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v3, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)
        self.SE = ChannelAttention(c1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize,-1,width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,height, width)

        out = (self.gamma1 * out) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        # out = self.MLP(out)
        return out


class atten_v3_origin(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v3_origin, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None))
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None))
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1)

        self.value = nn.Conv2d(c1, c1, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None))
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(in_features=c1, hidden_features=(c1+c2)//2, out_features=c2, act=nn.SiLU())
        self.SE = ChannelAttention(c1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize, -1, width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,height, width)

        out = (self.gamma1 * out) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out)
        return out


class atten_v3_c(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v3_c, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        # self.MLP = MLPLayer(c1, (c1+c2)//2, c2)
        self.SE = ChannelAttention(c1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize,-1,width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,height, width)

        out = (self.gamma1 * out) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        # out = self.MLP(out)
        return out
    
class atten_v3_head(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v3_head, self).__init__()
        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None))
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None))
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.SE = ChannelAttention(c1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        q = torch.cat((self.dw1(x), self.dw2(x)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x).view(m_batchsize,-1,width*height)
        att = torch.bmm(q,k)
        att = self.softmax(att)
        v = self.value(x).view(m_batchsize,-1,width*height)

        out = torch.bmm(v, att.permute(0,2,1))
        out = out.view(m_batchsize,C,height, width)

        out = (self.gamma1 * out) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        return out
    
class atten_win(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for i in range(self.split):
                for j in range(self.split):
                    win_q.append(q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_k.append(k[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_v.append(v[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])

        #############
        ## check correctness
        # test_q = k.clone()
        # for i in range(self.split):
        #         for j in range(self.split):
        #             test_q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = win_q[self.split*i+j]
        # print(torch.sum(test_q-q))
        ##############

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_w,win_h))
        x_c = x.clone()
        for i in range(self.split):
            for j in range(self.split):
                x_c[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = out[2*i+j]

        out = self.gamma * x_c + x 
        out = self.MLP(out) + out
        return out

class atten_win_origin(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_origin, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for j in range(self.split):
                for i in range(self.split):
                    win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])


        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))
        x_c = x.clone()
        for j in range(self.split):
            for i in range(self.split):
                x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

        out = self.gamma * x_c + x 
        out = self.MLP(out)
        return out

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        reduction (int): Channel reduction factor. Default: 16.
    """

    def __init__(self, num_feat, reduction=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // reduction, 1, padding=0),
            # nn.ReLU(inplace=True),
            nn.SiLU(),
            nn.Conv2d(num_feat // reduction, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.attention(x)
        return x * y

# class SELayer(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SELayer,self).__init__()
#         self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace = True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#             )
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)

class atten_win_v2(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v2, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma = nn.Parameter(torch.zeros(1))

        # self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2), c2)

        self.SE = SELayer(c1)
    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for i in range(self.split):
                for j in range(self.split):
                    win_q.append(q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_k.append(k[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_v.append(v[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])

        #############
        ## check correctness
        # test_q = k.clone()
        # for i in range(self.split):
        #         for j in range(self.split):
        #             test_q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = win_q[self.split*i+j]
        # print(torch.sum(test_q-q))
        ##############

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_w,win_h))
        x_c = x.clone()
        for i in range(self.split):
            for j in range(self.split):
                x_c[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = out[2*i+j]

        out = (self.gamma * x_c) + self.SE(x) + x
        out = self.MLP(out) + out
        return out
# sc-k3-att-v2
class atten_win_v3(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v3, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for i in range(self.split):
                for j in range(self.split):
                    win_q.append(q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_k.append(k[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_v.append(v[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])

        #############
        ## check correctness
        # test_q = k.clone()
        # for i in range(self.split):
        #         for j in range(self.split):
        #             test_q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = win_q[self.split*i+j]
        # print(torch.sum(test_q-q))
        ##############

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_w,win_h))
        x_c = x.clone()
        for i in range(self.split):
            for j in range(self.split):
                x_c[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = out[2*i+j]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        return out


class atten_win_v3_origin(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v3_origin, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for i in range(self.split):
                for j in range(self.split):
                    win_q.append(q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_k.append(k[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_v.append(v[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])

        #############
        ## check correctness
        # test_q = k.clone()
        # for i in range(self.split):
        #         for j in range(self.split):
        #             test_q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = win_q[self.split*i+j]
        # print(torch.sum(test_q-q))
        ##############

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_w,win_h))
        x_c = x.clone()
        for i in range(self.split):
            for j in range(self.split):
                x_c[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = out[self.split*i+j]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        return out

class atten_win_v3_2(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v3_2, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, width ,height = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for i in range(self.split):
                for j in range(self.split):
                    win_q.append(q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_k.append(k[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])
                    win_v.append(v[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)])

        #############
        ## check correctness
        # test_q = k.clone()
        # for i in range(self.split):
        #         for j in range(self.split):
        #             test_q[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = win_q[self.split*i+j]
        # print(torch.sum(test_q-q))
        ##############

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_w,win_h))
        x_c = x.clone()
        for i in range(self.split):
            for j in range(self.split):
                x_c[:, :, win_w*(j) :win_w*(j+1), win_h*(i):win_h*(i+1)] = out[3*i+j]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        return out

class atten_win_v3_c(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v3_c, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        # self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, height ,width = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for j in range(self.split):
                for i in range(self.split):
                    win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])

        att = list()
        for i in range(self.split ** 2):
            att.append(self.softmax(torch.bmm(win_q[i].contiguous().view(m_batchsize, -1, win_w*win_h).permute(0,2,1), win_k[i].contiguous().view(m_batchsize, -1, win_w*win_h))))

        
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))
        x_c = x.clone()
        for j in range(self.split):
            for i in range(self.split):
                x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        # out = self.MLP(out)
        return out

class atten_win_v4(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v4, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, height ,width = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for j in range(self.split):
                for i in range(self.split):
                    win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
        att = list()

        for i in range(self.split ** 2):

            win_q_i = win_q[i].contiguous().view(m_batchsize, -1, win_w * win_h).permute(0, 2, 1)
            tmp = torch.bmm(win_q_i, win_k[i].contiguous().view(m_batchsize, -1, win_w * win_h))
            neighbors = []
            
            # build neighbor list
            row_index = i // self.split
            col_index = i % self.split
            for row in range(-1, 2):
                for col in range(-1, 2):
                    new_row = row_index + row
                    new_col = col_index + col
                    if 0 <= new_row < self.split and 0 <= new_col < self.split:
                        neighbors.append(new_row * self.split + new_col)
            neighbors.remove(i)
            for neighbor in neighbors:
                tmp += torch.bmm(win_q_i, win_k[neighbor].contiguous().view(m_batchsize, -1, win_w * win_h))
            # tmp = tmp / len(neighbors)
            att.append(self.softmax(tmp))

        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))

        x_c = x.clone()
        for j in range(self.split):
            for i in range(self.split):
                x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        return out

class atten_win_v4_1(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        '''
        super(atten_win_v4_1, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)
    def forward(self, x):
        m_batchsize, C, height ,width = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        if self.split >= 2:
            for j in range(self.split):
                for i in range(self.split):
                    win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
        att = list()

        for i in range(self.split ** 2):

            win_q_i = win_q[i].contiguous().view(m_batchsize, -1, win_w * win_h).permute(0, 2, 1)
            tmp = torch.bmm(win_q_i, win_k[i].contiguous().view(m_batchsize, -1, win_w * win_h))
            neighbors = []
            
            # build neighbor list
            row_index = i // self.split
            col_index = i % self.split
            for row in range(-1, 2):
                for col in range(-1, 2):
                    new_row = row_index + row
                    new_col = col_index + col
                    if 0 <= new_row < self.split and 0 <= new_col < self.split:
                        neighbors.append(new_row * self.split + new_col)
            neighbors.remove(i)
            for neighbor in neighbors:
                tmp += torch.bmm(win_q_i, win_k[neighbor].contiguous().view(m_batchsize, -1, win_w * win_h))
            tmp = tmp / (len(neighbors)+1)
            att.append(self.softmax(tmp))

        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))

        x_c = x.clone()
        for j in range(self.split):
            for i in range(self.split):
                x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        return out

# sc-k3-att-win-v5-correct
class atten_win_v5(nn.Module):
    def __init__(self, c1, c2, split=2):
        '''
        split: 一邊切成幾塊
        v5: 3*3 grid information, window position weights
        '''
        super(atten_win_v5, self).__init__()
        self.split = split

        self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
        self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
        self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

        self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        
        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

        self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

        self.SE = ChannelAttention(c1)

        self.w_position_weigths = nn.Parameter(torch.zeros((self.split**2) * 9)) # 3*3 grid

    def forward(self, x):
        m_batchsize, C, height ,width = x.size()
        win_w = width // (self.split)
        win_h = height // (self.split)

        q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
        k = self.key_conv(x)
        v = self.value(x)

        win_q = list()
        win_k = list()
        win_v = list()

        # split window
        if self.split >= 2:
            for j in range(self.split):
                for i in range(self.split):
                    win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
                    win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])

        # calculate position_weight * (q*k)
        att = list()
        for i in range(self.split ** 2):

            win_q_i = win_q[i].contiguous().view(m_batchsize, -1, win_w * win_h).permute(0, 2, 1)
            tmp = self.w_position_weigths[i*9 + 4] * torch.bmm(win_q_i, win_k[i].contiguous().view(m_batchsize, -1, win_w * win_h))
            neighbors = []
            position_idx = []

            # build neighbor list & position list
            row_index = i // self.split
            col_index = i % self.split
            for row in range(-1, 2):
                for col in range(-1, 2):
                    new_row = row_index + row
                    new_col = col_index + col
                    if 0 <= new_row < self.split and 0 <= new_col < self.split:
                        neighbors.append(new_row * self.split + new_col)
                        position_idx.append((row+1) * 3 + col+1)
            neighbors.remove(i)
            position_idx.remove(4)

            neighbors = zip(neighbors, position_idx)

            for neighbor, position in neighbors:
                tmp += self.w_position_weigths[i*9 + position] * torch.bmm(win_q_i, win_k[neighbor].contiguous().view(m_batchsize, -1, win_w * win_h))
            att.append(self.softmax(tmp))

        # calculate (q*k)*v
        out = list()
        for i in range(self.split ** 2):
            out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))
        # restore window to whole feature map
        x_c = x.clone()
        for j in range(self.split):
            for i in range(self.split):
                x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

        out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
        out = self.MLP(out) + out
        # out = self.MLP(out)
        return out

#sc-k3-att-win-v53
# class atten_win_v5(nn.Module):
#     def __init__(self, c1, c2, split=2):
#         '''
#         split: 一邊切成幾塊
#         '''
#         super(atten_win_v5, self).__init__()
#         self.split = split

#         self.dw1 = nn.Conv2d(c1, c1//16, kernel_size=7, padding=autopad(7,None), groups=c1//16)
#         self.dw2 = nn.Conv2d(c1, c1//16, kernel_size=11, padding=autopad(11,None), groups=c1//16)
        
#         self.key_conv = nn.Conv2d(c1, c1//8, kernel_size=1, groups=c1//8)

#         self.value = nn.Conv2d(c1, c1, kernel_size=1, groups=c1)

#         self.gamma1 = nn.Parameter(torch.zeros(1))
        
#         self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
#         self.act = nn.SiLU()
#         self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
#         self.gamma2 = nn.Parameter(torch.zeros(1))

#         self.softmax  = nn.Softmax(dim=-1)

#         self.MLP = MLPLayer(c1, (c1+c2)//2, c2)

#         self.SE = ChannelAttention(c1)

#         self.w_position_weigths = nn.Parameter(torch.zeros(9)) # 3*3 grid

#     def forward(self, x):
#         m_batchsize, C, height ,width = x.size()
#         win_w = width // (self.split)
#         win_h = height // (self.split)

#         q = torch.cat((self.dw1(x), self.dw2(x)), 1)        
#         k = self.key_conv(x)
#         v = self.value(x)

#         win_q = list()
#         win_k = list()
#         win_v = list()

#         # split window
#         if self.split >= 2:
#             for j in range(self.split):
#                 for i in range(self.split):
#                     win_q.append(q[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
#                     win_k.append(k[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])
#                     win_v.append(v[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)])

#         # calculate position_weight * (q*k)
#         att = list()
#         for i in range(self.split ** 2):

#             win_q_i = win_q[i].contiguous().view(m_batchsize, -1, win_w * win_h).permute(0, 2, 1)
#             tmp = self.w_position_weigths[4] * torch.bmm(win_q_i, win_k[i].contiguous().view(m_batchsize, -1, win_w * win_h))
#             neighbors = []
#             position_idx = []

#             # build neighbor list & position list
#             row_index = i // self.split
#             col_index = i % self.split
#             for row in range(-1, 2):
#                 for col in range(-1, 2):
#                     new_row = row_index + row
#                     new_col = col_index + col
#                     if 0 <= new_row < self.split and 0 <= new_col < self.split:
#                         neighbors.append(new_row * self.split + new_col)
#                         position_idx.append((row+1) * 3 + col+1)
#             neighbors.remove(i)
#             position_idx.remove(4)

#             neighbors = zip(neighbors, position_idx)

#             for neighbor, position in neighbors:
#                 tmp += self.w_position_weigths[position] * torch.bmm(win_q_i, win_k[neighbor].contiguous().view(m_batchsize, -1, win_w * win_h))
#             att.append(self.softmax(tmp))

#         # calculate (q*k)*v
#         out = list()
#         for i in range(self.split ** 2):
#             out.append(torch.bmm(win_v[i].contiguous().view(m_batchsize, -1, win_w*win_h), att[i].permute(0,2,1)).contiguous().view(m_batchsize,C,win_h,win_w))
#         # restore window to whole feature map
#         x_c = x.clone()
#         for j in range(self.split):
#             for i in range(self.split):
#                 x_c[:, :, win_h*(j) :win_h*(j+1), win_w*(i):win_w*(i+1)] = out[self.split*j+i]

#         out = (self.gamma1 * x_c) + (self.gamma2 * self.SE(self.conv2(self.act(self.conv1(x))))) + x
#         out = self.MLP(out) + out
#         return out
# class CHSPP(nn.Module):
#     def __init__(self, c1, c2):
#         super(CHSPP, self).__init__()

#         self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
#         self.act = nn.SiLU()
#         self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
#         self.ch = ChannelAttention(c1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#         self.m1 = Conv(c1, c2, k=3, s=1)
#         self.m2 = DConv(c1, c2, k=3, s=1, d=2)
#         self.m3 = DConv(c1, c2, k=3, s=1, d=3)

#         self.conv3 = Conv(3*c2, c2//2, k=1, s=1)
#         self.conv4 = Conv(3*c2, c2//2, k=1, s=1)
#         self.sck3 = SC_k3(c2//2, c2//2)
#         self.conv5 = Conv(c2, c2, k=1, s=1)
#     def forward(self, x):
#         x1 = self.conv2(self.act(self.conv1(x)))
#         x1 = self.ch(x1)
#         x1 = x + self.gamma * x1

#         m1 = self.m1(x1)
#         m2 = self.m2(x1)
#         m3 = self.m3(x1)
#         m = torch.cat((m1, m2, m3), dim=1)
#         x2 = self.sck3(self.conv3(m))
#         x3 = self.conv4(m)
#         return self.conv5(torch.cat((x2, x3), dim=1))
    
#     def fuse(self):
#         self.sck3.fuseSCk3()

##SC_k3_m_att_win_v3_23
class CHSPP(nn.Module):
    def __init__(self, c1, c2):
        super(CHSPP, self).__init__()

        self.conv1 = nn.Conv2d(c1, c1//16, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(c1//16, c1, kernel_size=3, padding=autopad(3,None), groups=c1//16)
        self.ch = ChannelAttention(c1)
        self.gamma = nn.Parameter(torch.zeros(1))


        self.conv3 = Conv(c1, c2//2, k=1, s=1)
        self.conv4 = Conv(c1, c2//2, k=1, s=1)
        self.sck3 = SC_k3(c2//2, c2//2)
        self.conv5 = Conv(c2, c2, k=1, s=1)
    def forward(self, x):
        x1 = self.conv2(self.act(self.conv1(x)))
        x1 = self.ch(x1)
        x1 = x + self.gamma * x1

        x2 = self.sck3(self.conv3(x1))
        x3 = self.conv4(x1)
        return self.conv5(torch.cat((x2, x3), dim=1))
    
    def fuse(self):
        self.sck3.fuseSCk3()

#SC_k3_m_att_win_v3_24
# class CHSPP(nn.Module):
#     def __init__(self, c1, c2):
#         super(CHSPP, self).__init__()

#         self.conv1 = Conv(c1, c2, k=1, s=1)
        

#         self.m1 = Conv(c2, c2//2, k=1, s=1)
#         self.m2 = DConv(c2, c2//2, k=3, s=1, d=2)
#         self.m3 = DConv(c2, c2//2, k=3, s=1, d=3)
#         self.sck3 = SC_k3(c2, c2//2)

#         self.ch = ChannelAttention(2*c2)

#         self.conv2 = Conv(2*c2, c2, k=1, s=1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         x1 = self.conv1(x)

#         m1 = self.m1(x1)
#         m2 = self.m2(x1)
#         m3 = self.m3(x1)
#         m4 = self.sck3(x1)

#         chx = self.ch(torch.cat((m1, m2, m3, m4), dim=1))
#         x2 = self.conv2(chx)
        
#         return x1 + self.gamma * x2
    
#     def fuse(self):
#         self.sck3.fuseSCk3()

##### end of custom function module #####

##### start of custom detect head module  #####

# Decouple cls & reg(bbox
##### v1 
class DecoupledHead(nn.Module):
    def __init__(self, ch=256, nc=1,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 , 1, 1)
        # self.cls_convs1 = Conv(256 , 256 , 3, 1, 1)
        # self.cls_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        # x1 = self.cls_convs1(x)
        # x1 = self.cls_convs2(x1)
        
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        x1 = self.cls_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out

class DecoupledHead_v2(nn.Module):
    def __init__(self, ch=256, nc=1,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 , 1, 1)
        self.cls_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.cls_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs1 = Conv(256 , 256 , 3, 1, 1)
        self.reg_convs2 = Conv(256 , 256 , 3, 1, 1)
        self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
       
        out = torch.cat([x21, x22, x1], 1)

        return out
### v2 seems to be failed
# class DecoupledHead(nn.Module):
#     def __init__(self, ch=256, nc=1,  anchors=()):
#         super().__init__()
#         self.nc = nc  # number of classes
#         self.nl = len(anchors)  # number of detection layers
#         self.na = len(anchors[0]) // 2  # number of anchors
#         self.merge = Conv(ch, 256 , 1, 1)
#         self.cls_conv = SC_k3(256 , 256 , k=3, s=1)
#         self.reg_conv = SC_k3(256 , 256 , k=3, s=1)
#         self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
#         self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
#         self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

#     def forward(self, x):
#         x = self.merge(x)
#         x1 = self.cls_conv(x)
#         x1 = self.cls_preds(x1)
#         x2 = self.reg_conv(x)
#         x21 = self.reg_preds(x2)
#         x22 = self.obj_preds(x2)
#         out = torch.cat([x21, x22, x1], 1)
#         return out
    
#     def fuse(self):
#         self.cls_conv.fuseSCk3()
#         self.reg_conv.fuseSCk3()

# v3
class DecoupledHead_tiny(nn.Module):
    def __init__(self, ch=256, nc=1,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors

        self.merge = Conv(ch, 256 , 1, 1)
        self.cls_convs1 = SC_k3_tiny(256 , 128 , k=3, s=1)
        self.cls_convs2 = SC_k3_tiny(128 , 64 , k=3, s=1)
        self.reg_convs1 = SC_k3_tiny(256 , 128 , k=3, s=1)
        self.reg_convs2 = SC_k3_tiny(128 , 128 , k=3, s=1)
        self.cls_preds = nn.Conv2d(64 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(128 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(128 , 1 * self.na, 1)

    def forward(self, x):
        x = self.merge(x)
        x1 = self.cls_convs1(x)
        x1 = self.cls_convs2(x1)
        x1 = self.cls_preds(x1)
        x2 = self.reg_convs1(x)
        x2 = self.reg_convs2(x2)
        x21 = self.reg_preds(x2)
        x22 = self.obj_preds(x2)
        out = torch.cat([x21, x22, x1], 1)
        return out
    
    def fuse(self):
        self.cls_convs1.fuseSCk3_tiny()
        self.cls_convs2.fuseSCk3_tiny()
        self.reg_convs1.fuseSCk3_tiny()
        self.reg_convs2.fuseSCk3_tiny()
##### end of custom detect head module  #####

class Classify(nn.Module):
    # YOLOv5 classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 dropout_p=0.0):  # ch_in, ch_out, kernel, stride, padding, groups, dropout probability
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, autopad(k, p), g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=dropout_p, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.cat(x, 1)
        return self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))