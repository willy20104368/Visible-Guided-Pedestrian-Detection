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
from models.common import *

class PConv(nn.Module):
    "https://arxiv.org/pdf/2303.03667.pdf 2023CVPR"

    """ Partial convolution (PConv).
    """
    def __init__(self,
                 dim: int,
                #  n_div: int = 4, # default from paper
                 n_div: int = 2,
                 kernel_size: int = 3) -> None:
        """ Construct a PConv layer.

        :param dim: Number of input/output channels
        :param n_div: Reciprocal of the partial ratio.
        :param kernel_size: Kernel size.
        """
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv

        self.conv = nn.Conv2d(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False
        )


    def forward_slicing(self, x):
        """ Apply forward pass for inference. """
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])

        return x

    def forward(self, x):
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.conv(x1)
        x = torch.cat((x1, x2), 1)

        return x


class PAtten(nn.Module):

    def __init__(self, dim, n_div=4):
        super().__init__()

        self.dim_att = dim // n_div
        self.dim_untouched = dim - self.dim_att

        self.dw1 = nn.Conv2d(self.dim_att, self.dim_att//2, kernel_size=7, padding=autopad(7,None))
        self.dw2 = nn.Conv2d(self.dim_att, self.dim_att//2, kernel_size=11, padding=autopad(11,None))
        
        self.key_conv = nn.Conv2d(self.dim_att, self.dim_att, kernel_size=1)

        self.value = nn.Conv2d(self.dim_att, self.dim_att, kernel_size=1)

        self.softmax  = nn.Softmax(dim=-1)

        self.conv1 = Conv(dim, dim, 1, 1)

        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, padding=autopad(1,None))

    def forward_slicing(self, x):

        x1 = x[:,:self.dim_att, :, :]
        m_batchsize, C ,height, width = x1.size() 

        q = torch.cat((self.dw1(x1), self.dw2(x1)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x1).view(m_batchsize,-1,width*height)
        v = self.value(x1).view(m_batchsize,-1,height*width)
        
        att = self.softmax(torch.bmm(q,k))
        out1 = torch.bmm(v, att.permute(0,2,1))
        out1= out1.view(m_batchsize, C, height, width)

        x[:,:self.dim_att, :, :] = out1

        return self.conv2(self.conv1(x))

    def forward(self, x):
        # x1: [b, dim_att, h, w], x2: [b, dim_untouched, h, w] 
        x1, x2 = torch.split(x, [self.dim_att, self.dim_untouched], dim=1)
        m_batchsize, C ,height, width = x1.size()

        q = torch.cat((self.dw1(x1), self.dw2(x1)), 1).view(m_batchsize,-1,width*height).permute(0,2,1)
        k = self.key_conv(x1).view(m_batchsize,-1,width*height)
        v = self.value(x1).view(m_batchsize,-1,height*width)

        att = self.softmax(torch.bmm(q,k))
        out1 = torch.bmm(v, att.permute(0,2,1))
        out1= out1.view(m_batchsize, C, height, width)

        out1 = torch.cat((out1, x2), dim=1)

        return self.conv2(self.conv1(out1))


class PDHead(nn.Module):
    def __init__(self, ch=256, nc=1,  anchors=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.merge = Conv(ch, 256 , 1, 1)
        self.cls_convs1 = PConv(256, 2, 3)
        self.cls_convs11 = Conv(256, 256, 1 ,1)
        self.cls_convs12 = nn.Conv2d(256, 256, 1)

        self.cls_convs2 = PConv(256, 2, 3)
        self.cls_convs21 = Conv(256, 256, 1 ,1)
        self.cls_convs22 = nn.Conv2d(256, 256, 1)

        self.reg_convs1 = PConv(256, 2, 3)
        self.reg_convs11 = Conv(256, 256, 1 ,1)
        self.reg_convs12 = nn.Conv2d(256, 256, 1)

        self.reg_convs2 = PConv(256, 2, 3)
        self.reg_convs21 = Conv(256, 256, 1 ,1)
        self.reg_convs22 = nn.Conv2d(256, 256, 1)

        self.cls_preds = nn.Conv2d(256 , self.nc * self.na, 1)
        self.reg_preds = nn.Conv2d(256 , 4 * self.na, 1)
        self.obj_preds = nn.Conv2d(256 , 1 * self.na, 1)

    def forward(self, x):
        merge = self.merge(x)
        x1 = self.cls_convs1(merge)
        x11 = self.cls_convs11(x1)
        # x11 = self.cls_convs12(x11) + x1
        x11 = self.cls_convs12(x11) 

        x2 = self.cls_convs2(x11)
        x21 = self.cls_convs21(x2)
        # x21 = self.cls_convs22(x21) + x2
        x21 = self.cls_convs22(x21)

        class_pred = self.cls_preds(x21)

        x3 = self.reg_convs1(merge)
        x31 = self.reg_convs11(x3)
        # x31 = self.reg_convs12(x31) + x3
        x31 = self.reg_convs12(x31)

        x4 = self.reg_convs2(x31)
        x41 = self.reg_convs21(x4)
        # x41 = self.reg_convs22(x41) + x4
        x41 = self.reg_convs22(x41)

        bbx_pred = self.reg_preds(x41)
        ob_pred = self.obj_preds(x41)
       
        out = torch.cat([bbx_pred, ob_pred, class_pred], 1)

        return out


class atten_v3_head(nn.Module):
    def __init__(self, c1, c2):
        super(atten_v3_head, self).__init__()
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
    
class RepC4(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function
    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, act=True): 
        super(RepC4, self).__init__()
        self.identity_conv1 = nn.Conv2d(cin, cin, 1, 1, groups=g, bias=False)
        self.conv1 = nn.Conv2d(cin, cout, k, s, autopad(k,p), groups=g, bias=False)

        self.identity_strip_v = nn.Conv2d(cin, cin, 1, 1, groups=g, bias=False)
        self.strip_conv_v = nn.Conv2d(cin, cout, (k,1), s, (autopad(k,p),0), groups=g, bias=False)

        self.identity_strip_h = nn.Conv2d(cin, cin, 1, 1, groups=g, bias=False)
        self.strip_conv_h = nn.Conv2d(cin, cout, (1,k), s, (0,autopad(k,p)), groups=g, bias=False)

        self.identity = nn.Conv2d(cin, cout, 1, 1, groups=g, bias=False)

        # bn for conv1
        self.bn_conv1 = nn.BatchNorm2d(cout)
        # bn for strip conv Vert
        self.bn_strip_v = nn.BatchNorm2d(cout)
        # bn for strip conv Hor
        self.bn_strip_h = nn.BatchNorm2d(cout)
        # bn for indentiy conv
        self.bn_identity = nn.BatchNorm2d(cout)

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    # training
    def forward(self,x):
        return self.act( self.bn_conv1(self.conv1(self.identity_conv1(x)))
                        +self.bn_strip_v(self.strip_conv_v(self.identity_strip_v(x)))
                        +self.bn_strip_h(self.strip_conv_h(self.identity_strip_h(x)))
                        +self.bn_identity(self.identity(x))
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
    
    def fuseRepC4(self):
        # fuse conv and bn first
        print("fuse RepC4")
        self.conv1 = self.fuse_conv_bn(self.conv1, self.bn_conv1)
        self.strip_conv_v = self.fuse_conv_bn(self.strip_conv_v, self.bn_strip_v)
        self.strip_conv_h = self.fuse_conv_bn(self.strip_conv_h, self.bn_strip_h)
        self.identity = self.fuse_conv_bn(self.identity,self.bn_identity)
        # fuse four conv in one
        self.conv1.weight = nn.Parameter(nn.functional.conv2d(self.conv1.weight.data, self.identity_conv1.weight.data.permute(1,0,2,3)))
        strip_weight_v = nn.functional.conv2d(nn.functional.pad(self.strip_conv_v.weight.data,(1,1,0,0),'constant',0), self.identity_strip_v.weight.data.permute(1,0,2,3))
        strip_weight_h = nn.functional.conv2d(nn.functional.pad(self.strip_conv_h.weight.data,(0,0,1,1),'constant',0), self.identity_strip_h.weight.data.permute(1,0,2,3))
        all_weight = self.conv1.weight.data + strip_weight_v + strip_weight_h
        all_weight[:,:,1,1] += self.identity.weight.data[:,:,0,0]
        self.conv1.weight = nn.Parameter(all_weight)
        self.conv1.bias = nn.Parameter(self.conv1.bias.data + self.strip_conv_v.bias.data + self.strip_conv_h.bias.data + self.identity.bias.data)

        # delete non-used layers
        del self.bn_conv1
        self.bn_conv1 = None
        del self.identity_conv1
        self.identity_conv1 = None
        del self.bn_strip_v
        self.bn_strip_v = None
        del self.strip_conv_v
        self.strip_conv_v = None
        del self.identity_strip_v
        self.identity_strip_v = None
        del self.bn_strip_h
        self.bn_strip_h = None
        del self.strip_conv_h
        self.strip_conv_h = None
        del self.identity_strip_h
        self.identity_strip_h = None
        del self.identity
        self.identity = None
        del self.bn_identity
        self.bn_identity = None

        # set new forward path
        self.forward = self.fuseforward

class PSC_k3(nn.Module):
    # channel_in, channel_out, kernel_size, strides, padding, group, activation function, preserve ratio
    def __init__(self, cin, cout, k=3, s=1, p=None, g=1, act=True, n_div = 2): 
        super(PSC_k3, self).__init__()
        self.dim_conv = cin // n_div
        self.dim_untouched = cin - self.dim_conv

        
        self.repconv = SC_k3(self.dim_conv, self.dim_conv, 3, 1)
        self.conv1 = Conv(cin, cout, k=1, s=1)
    
    def forward_slicing(self, x):
        """ Apply forward pass for inference. """
        x[:, :self.dim_conv, :, :] = self.repconv(x[:, :self.dim_conv, :, :])
        return self.conv1(x)

    def forward(self, x):
        """ Apply forward pass for training. """
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.repconv(x1)
        x = self.conv1(torch.cat((x1, x2), 1))

        return x
    def fusePSC_k3(self):
        print("fuse PSC_k3")
        self.forward = self.forward_slicing


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