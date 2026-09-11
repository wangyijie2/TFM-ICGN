# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np

def fftcc_estimate(ref_img, tar_img, poi_queue, subset_radius_x=16, subset_radius_y=16, odd_subset=False):
    (H, W) = ref_img.shape
    subset_w = subset_radius_x * 2 + int(odd_subset)
    subset_h = subset_radius_y * 2 + int(odd_subset)
    subset_size = subset_w * subset_h
    results = [None] * len(poi_queue)
    for (idx, (px, py)) in enumerate(poi_queue):
        (px, py) = (int(px), int(py))
        (initial_u, initial_v) = (0.0, 0.0)
        if px < subset_radius_x or px >= W - subset_radius_x or py < subset_radius_y or (py >= H - subset_radius_y) or (int(px + initial_u) < subset_radius_x) or (int(px + initial_u) >= W - subset_radius_x) or (int(py + initial_v) < subset_radius_y) or (int(py + initial_v) >= H - subset_radius_y):
            results[idx] = None
            continue
        ref_subset = np.zeros((subset_h, subset_w), dtype=np.float32)
        tar_subset = np.zeros((subset_h, subset_w), dtype=np.float32)
        for r in range(subset_h):
            for c in range(subset_w):
                rx = px + c - subset_radius_x
                ry = py + r - subset_radius_y
                ref_subset[r, c] = ref_img[ry, rx]
                tx = rx + int(initial_u)
                ty = ry + int(initial_v)
                tar_subset[r, c] = tar_img[ty, tx]
        ref_mean = ref_subset.mean()
        tar_mean = tar_subset.mean()
        ref_zm = ref_subset - ref_mean
        tar_zm = tar_subset - tar_mean
        ref_norm = np.sum(ref_zm ** 2)
        tar_norm = np.sum(tar_zm ** 2)
        ref_fft = np.fft.rfft2(ref_zm)
        tar_fft = np.fft.rfft2(tar_zm)
        zncc_freq = np.conj(ref_fft) * tar_fft
        zncc_map = np.fft.irfft2(zncc_freq, s=(subset_h, subset_w))
        max_idx = np.argmax(zncc_map)
        local_v = max_idx // subset_w
        local_u = max_idx % subset_w
        if local_u > subset_radius_x:
            local_u -= subset_w
        if local_v > subset_radius_y:
            local_v -= subset_h
        max_zncc_val = zncc_map[local_v % subset_h, local_u % subset_w]
        results[idx] = {'u': float(local_u + initial_u), 'v': float(local_v + initial_v), 'zncc': float(max_zncc_val / (np.sqrt(ref_norm * tar_norm) * subset_size))}
    return results
