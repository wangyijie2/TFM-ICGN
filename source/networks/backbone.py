# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import torch.nn as nn
from .trident_conv import MultiScaleTridentConv

class ResidualBlock(nn.Module):

    def __init__(self, in_planes, planes, norm_layer=nn.InstanceNorm2d, stride=1, dilation=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.norm1 = norm_layer(planes)
        self.norm2 = norm_layer(planes)
        if stride != 1 or in_planes != planes:
            self.norm3 = norm_layer(planes)
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False), self.norm3)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)

class ResNetEncoder(nn.Module):

    def __init__(self, output_dim=128, norm_layer=nn.InstanceNorm2d, num_output_scales=1, **kwargs):
        super(ResNetEncoder, self).__init__()
        self.num_branch = num_output_scales
        feature_dims = [64, 96, 128]
        self.conv1 = nn.Conv2d(1, feature_dims[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = norm_layer(feature_dims[0])
        self.relu1 = nn.ReLU(inplace=True)
        self.in_planes = feature_dims[0]
        self.layer1 = self._make_layer(feature_dims[0], stride=1, norm_layer=norm_layer)
        self.layer2 = self._make_layer(feature_dims[1], stride=1, norm_layer=norm_layer)
        self.layer3 = self._make_layer(feature_dims[2], stride=1, norm_layer=norm_layer)
        self.conv2 = nn.Conv2d(feature_dims[2], output_dim, kernel_size=1, stride=1, padding=0)
        if self.num_branch > 1:
            if self.num_branch == 4:
                strides = (1, 2, 4, 8)
            elif self.num_branch == 3:
                strides = (1, 2, 4)
            elif self.num_branch == 2:
                strides = (1, 2)
            else:
                raise ValueError
            self.trident_conv = MultiScaleTridentConv(output_dim, output_dim, kernel_size=3, strides=strides, paddings=1, num_branch=self.num_branch)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        block1 = ResidualBlock(self.in_planes, dim, norm_layer=norm_layer, stride=stride, dilation=dilation)
        block2 = ResidualBlock(dim, dim, norm_layer=norm_layer, stride=1, dilation=dilation)
        self.in_planes = dim
        return nn.Sequential(block1, block2)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)
        if self.num_branch > 1:
            out = self.trident_conv([x] * self.num_branch)
        else:
            out = [x]
        return out
