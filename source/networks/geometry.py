# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import torch
import torch.nn.functional as F

def coords_grid(b, h, w, homogeneous=False, device=None):
    (y, x) = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    stacks = [x, y]
    if homogeneous:
        ones = torch.ones_like(x)
        stacks.append(ones)
    grid = torch.stack(stacks, dim=0).float()
    grid = grid[None].repeat(b, 1, 1, 1)
    if device is not None:
        grid = grid.to(device)
    return grid

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None
    (x, y) = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device), torch.linspace(h_min, h_max, len_h, device=device)], indexing='ij')
    grid = torch.stack((x, y), -1).transpose(0, 1).float()
    return grid

def normalize_coords(coords, h, w):
    c = torch.Tensor([(w - 1) / 2.0, (h - 1) / 2.0]).float().to(coords.device)
    return (coords - c) / c

def bilinear_sample(img, sample_coords, mode='bilinear', padding_mode='zeros', return_mask=False):
    if sample_coords.size(1) != 2:
        sample_coords = sample_coords.permute(0, 3, 1, 2)
    (b, _, h, w) = sample_coords.shape
    x_grid = 2 * sample_coords[:, 0] / (w - 1) - 1
    y_grid = 2 * sample_coords[:, 1] / (h - 1) - 1
    grid = torch.stack([x_grid, y_grid], dim=-1)
    img = F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)
    if return_mask:
        mask = (x_grid >= -1) & (y_grid >= -1) & (x_grid <= 1) & (y_grid <= 1)
        return (img, mask)
    return img

def flow_warp(feature, flow, mask=False, padding_mode='zeros'):
    (b, c, h, w) = feature.size()
    assert flow.size(1) == 2
    grid = coords_grid(b, h, w).to(flow.device) + flow
    return bilinear_sample(feature, grid, padding_mode=padding_mode, return_mask=mask)
