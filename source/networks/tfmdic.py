# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .backbone import ResNetEncoder
from .transformer import TransformerEncoder, FeatureFlowAttention
from .matching import global_correlation_softmax, local_correlation_softmax
from .geometry import flow_warp
from .utils import normalize_img, feature_add_position

class tfmdic(nn.Module):

    def __init__(self, num_scales=1, upsample_factor=8, feature_channels=128, attention_type='swin', num_transformer_blocks=12, ffn_dim_expansion=4, num_head=1, **kwargs):
        super(tfmdic, self).__init__()
        self.num_scales = num_scales
        self.feature_channels = feature_channels
        self.upsample_factor = upsample_factor
        self.attention_type = attention_type
        self.num_transformer_blocks = num_transformer_blocks
        self.backbone = ResNetEncoder(output_dim=feature_channels, num_output_scales=num_scales)
        self.transformer = TransformerEncoder(num_blocks=num_transformer_blocks, d_model=feature_channels, nhead=num_head, attention_type=attention_type, ffn_dim_expansion=ffn_dim_expansion)
        self.upsampler = nn.Sequential(nn.Conv2d(2 + feature_channels, 256, 3, 1, 1), nn.ReLU(inplace=True), nn.Conv2d(256, upsample_factor ** 2 * 9, 1, 1, 0))

    def extract_feature(self, img0, img1):
        concat = torch.cat((img0, img1), dim=0)
        features = self.backbone(concat)
        features = features[::-1]
        (feature0, feature1) = ([], [])
        for i in range(len(features)):
            feature = features[i]
            chunks = torch.chunk(feature, 2, 0)
            feature0.append(chunks[0])
            feature1.append(chunks[1])
        return (feature0, feature1)

    def upsample_flow(self, flow, feature, bilinear=False, upsample_factor=8):
        if bilinear:
            up_flow = F.interpolate(flow, scale_factor=upsample_factor, mode='bilinear', align_corners=True) * upsample_factor
        else:
            concat = torch.cat((flow, feature), dim=1)
            mask = self.upsampler(concat)
            (b, flow_channel, h, w) = flow.shape
            mask = mask.view(b, 1, 9, self.upsample_factor, self.upsample_factor, h, w)
            mask = torch.softmax(mask, dim=2)
            up_flow = F.unfold(self.upsample_factor * flow, [3, 3], padding=1)
            up_flow = up_flow.view(b, flow_channel, 9, 1, 1, h, w)
            up_flow = torch.sum(mask * up_flow, dim=2)
            up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
            up_flow = up_flow.reshape(b, flow_channel, self.upsample_factor * h, self.upsample_factor * w)
        return up_flow

    def forward(self, img0, img1, attn_splits_list=None, corr_radius_list=None, **kwargs):
        results_dict = {}
        flow_preds = []
        (img0, img1) = normalize_img(img0, img1)
        t0 = time.time()
        (feature0_list, feature1_list) = self.extract_feature(img0, img1)
        t1 = time.time()
        feature_extract_time = t1 - t0
        flow = None
        assert len(attn_splits_list) == len(corr_radius_list) == self.num_scales
        transformer_time_total = 0
        matching_time_total = 0
        for scale_idx in range(self.num_scales):
            (feature0, feature1) = (feature0_list[scale_idx], feature1_list[scale_idx])
            upsample_factor = self.upsample_factor * 2 ** (self.num_scales - 1 - scale_idx)
            if scale_idx > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            if flow is not None:
                flow = flow.detach()
                feature1 = flow_warp(feature1, flow)
            attn_splits = attn_splits_list[scale_idx]
            corr_radius = corr_radius_list[scale_idx]
            t2 = time.time()
            (feature0, feature1) = feature_add_position(feature0, feature1, attn_splits, self.feature_channels)
            (feature0, feature1) = self.transformer(feature0, feature1, attn_num_splits=attn_splits)
            t3 = time.time()
            transformer_time_total += t3 - t2
            t4 = time.time()
            if corr_radius == -1:
                flow_pred = global_correlation_softmax(feature0, feature1)[0]
            else:
                flow_pred = local_correlation_softmax(feature0, feature1, corr_radius)[0]
            t5 = time.time()
            matching_time_total += t5 - t4
            flow = flow + flow_pred if flow is not None else flow_pred
            if self.training:
                flow_bilinear = self.upsample_flow(flow, None, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_bilinear)
            if self.training and scale_idx < self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0, bilinear=True, upsample_factor=upsample_factor)
                flow_preds.append(flow_up)
            if scale_idx == self.num_scales - 1:
                flow_up = self.upsample_flow(flow, feature0)
                flow_preds.append(flow_up)
        results_dict.update({'flow_preds': flow_preds, 'time_feature_extract': feature_extract_time, 'time_transformer': transformer_time_total, 'time_matching': matching_time_total})
        return results_dict
