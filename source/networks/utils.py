# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

from .position import PositionEmbeddingSine

def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        (b, h, w, c) = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)
    else:
        (b, c, h, w) = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits
        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)
    return feature

def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        (b, h, w, c) = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(new_b, num_splits * h, num_splits * w, c)
    else:
        (b, c, h, w) = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(new_b, c, num_splits * h, num_splits * w)
    return merge

def normalize_img(img0, img1):
    img0 = img0 / 255
    img1 = img1 / 255
    return (img0, img1)

def feature_add_position(feature0, feature1, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        feature0_splits = split_feature(feature0, num_splits=attn_splits)
        feature1_splits = split_feature(feature1, num_splits=attn_splits)
        position = pos_enc(feature0_splits)
        feature0_splits = feature0_splits + position
        feature1_splits = feature1_splits + position
        feature0 = merge_splits(feature0_splits, num_splits=attn_splits)
        feature1 = merge_splits(feature1_splits, num_splits=attn_splits)
    else:
        position = pos_enc(feature0)
        feature0 = feature0 + position
        feature1 = feature1 + position
    return (feature0, feature1)
