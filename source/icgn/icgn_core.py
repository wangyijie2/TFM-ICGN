# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np
from .deformation import Deformation2D1

class ICGN2D1:

    def __init__(self, subset_radius_x: int=15, subset_radius_y: int=15, conv_criterion: float=0.001, stop_condition: int=50):
        self.subset_rx = subset_radius_x
        self.subset_ry = subset_radius_y
        self.conv_criterion = conv_criterion
        self.stop_condition = stop_condition
        self.ref_gradient_x = None
        self.ref_gradient_y = None
        self.tar_interp = None
        self.ref_img = None
        self.tar_img = None

    def prepare(self, ref_img: np.ndarray, tar_img: np.ndarray, ref_gradient_x: np.ndarray, ref_gradient_y: np.ndarray, tar_interp):
        self.ref_img = ref_img.astype(np.float32)
        self.tar_img = tar_img.astype(np.float32)
        self.ref_gradient_x = ref_gradient_x
        self.ref_gradient_y = ref_gradient_y
        self.tar_interp = tar_interp

    def compute_single(self, poi_x: float, poi_y: float, initial_u: float, initial_v: float) -> dict:
        rx = self.subset_rx
        ry = self.subset_ry
        subset_w = 2 * rx + 1
        subset_h = 2 * ry + 1
        if poi_y - ry < 0 or poi_x - rx < 0 or poi_y + ry >= self.ref_img.shape[0] or (poi_x + rx >= self.ref_img.shape[1]) or (abs(initial_u) >= self.ref_img.shape[1]) or (abs(initial_v) >= self.ref_img.shape[0]):
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        if np.isnan(initial_u) or np.isnan(initial_v):
            return {'u': 0.0, 'v': 0.0, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        y_start = int(poi_y) - ry
        x_start = int(poi_x) - rx
        ref_subset = self.ref_img[y_start:y_start + subset_h, x_start:x_start + subset_h].copy()
        ref_mean = ref_subset.mean()
        ref_subset -= ref_mean
        ref_norm = np.sqrt(np.sum(ref_subset ** 2))
        if ref_norm < 1e-10:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        grad_x_sub = self.ref_gradient_x[y_start:y_start + subset_h, x_start:x_start + subset_w]
        grad_y_sub = self.ref_gradient_y[y_start:y_start + subset_h, x_start:x_start + subset_w]
        sd = np.zeros((subset_h, subset_w, 6), dtype=np.float32)
        y_coords = np.arange(subset_h, dtype=np.float32) - ry
        x_coords = np.arange(subset_w, dtype=np.float32) - rx
        (xx, yy) = np.meshgrid(x_coords, y_coords)
        sd[:, :, 0] = grad_x_sub
        sd[:, :, 1] = grad_x_sub * xx
        sd[:, :, 2] = grad_x_sub * yy
        sd[:, :, 3] = grad_y_sub
        sd[:, :, 4] = grad_y_sub * xx
        sd[:, :, 5] = grad_y_sub * yy
        sd_flat = sd.reshape(-1, 6)
        H = sd_flat.T @ sd_flat
        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        p_current = Deformation2D1(u=initial_u, v=initial_v)
        p_initial = Deformation2D1(u=initial_u, v=initial_v)
        iteration = 0
        dp_norm_max = 999.0
        znssd = 0.0
        while iteration < self.stop_condition and dp_norm_max >= self.conv_criterion:
            iteration += 1
            tar_subset = np.zeros((subset_h, subset_w), dtype=np.float32)
            for r in range(subset_h):
                for c in range(subset_w):
                    x_local = x_coords[c]
                    y_local = y_coords[r]
                    (wx, wy) = p_current.warp(x_local, y_local)
                    gx = poi_x + wx
                    gy = poi_y + wy
                    tar_subset[r, c] = self.tar_interp.compute(gx, gy)
            if np.any(tar_subset < -0.5):
                break
            tar_mean = tar_subset.mean()
            tar_subset_zm = tar_subset - tar_mean
            tar_norm = np.sqrt(np.sum(tar_subset_zm ** 2))
            if tar_norm < 1e-10:
                break
            scale = ref_norm / tar_norm
            error = tar_subset_zm * scale - ref_subset
            znssd = np.sum(error ** 2) / ref_norm ** 2
            num = sd_flat.T @ error.ravel()
            dp = inv_H @ num
            p_inc = Deformation2D1()
            p_inc.set_deformation(dp)
            p_current.icgn_update(p_inc)
            rx2 = rx * rx
            ry2 = ry * ry
            dp_norm_max = np.sqrt(p_inc.u ** 2 + p_inc.v ** 2 + p_inc.ux ** 2 * rx2 + p_inc.uy ** 2 * ry2 + p_inc.vx ** 2 * rx2 + p_inc.vy ** 2 * ry2)
        zncc = 0.5 * (2.0 - znssd)
        if dp_norm_max >= self.conv_criterion and iteration >= self.stop_condition:
            zncc = -4.0
        if zncc >= 0 and zncc < 0.7:
            zncc = -4.0
        if np.isnan(zncc) or np.isnan(p_current.u) or np.isnan(p_current.v):
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -5.0, 'iteration': iteration, 'convergence': dp_norm_max}
        if zncc < 0:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': float(zncc), 'iteration': iteration, 'convergence': dp_norm_max, 'u0': float(initial_u), 'v0': float(initial_v)}
        return {'u': float(p_current.u), 'v': float(p_current.v), 'ux': float(p_current.ux), 'uy': float(p_current.uy), 'vx': float(p_current.vx), 'vy': float(p_current.vy), 'zncc': float(zncc), 'iteration': iteration, 'convergence': float(dp_norm_max), 'u0': float(initial_u), 'v0': float(initial_v)}

    def compute_batch(self, pois: list) -> list:
        results = []
        for poi in pois:
            result = self.compute_single(poi_x=poi['x'], poi_y=poi['y'], initial_u=poi['u0'], initial_v=poi['v0'])
            results.append(result)
        return results
