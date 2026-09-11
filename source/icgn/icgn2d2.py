# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np
from .deformation import Deformation2D1

class Deformation2D2:

    def __init__(self):
        self.u = 0.0
        self.ux = 0.0
        self.uy = 0.0
        self.uxx = 0.0
        self.uxy = 0.0
        self.uyy = 0.0
        self.v = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vxx = 0.0
        self.vxy = 0.0
        self.vyy = 0.0
        self.warp_matrix = np.eye(6, dtype=np.float32)

    def set_deformation_from_p(self, p):
        (self.u, self.ux, self.uy, self.uxx, self.uxy, self.uyy, self.v, self.vx, self.vy, self.vxx, self.vxy, self.vyy) = map(float, p)
        self._set_warp()

    def set_deformation_from_1st_order(self, p1: Deformation2D1):
        self.u = p1.u
        self.ux = p1.ux
        self.uy = p1.uy
        self.uxx = 0.0
        self.uxy = 0.0
        self.uyy = 0.0
        self.v = p1.v
        self.vx = p1.vx
        self.vy = p1.vy
        self.vxx = 0.0
        self.vxy = 0.0
        self.vyy = 0.0
        self._set_warp()

    def _set_warp(self):
        W = self.warp_matrix
        (u, ux, uy) = (self.u, self.ux, self.uy)
        (uxx, uxy, uyy) = (self.uxx, self.uxy, self.uyy)
        (v, vx, vy) = (self.v, self.vx, self.vy)
        (vxx, vxy, vyy) = (self.vxx, self.vxy, self.vyy)
        W[0, 0] = 1.0 + 2.0 * ux + ux * ux + u * uxx
        W[0, 1] = 2.0 * u * uxy + 2.0 * (1.0 + ux) * uy
        W[0, 2] = uy * uy + u * uyy
        W[0, 3] = 2.0 * u * (1.0 + ux)
        W[0, 4] = 2.0 * u * uy
        W[0, 5] = u * u
        W[1, 0] = 0.5 * (v * uxx + 2.0 * (1.0 + ux) * vx + u * vxx)
        W[1, 1] = 1.0 + uy * vx + ux * vy + v * uxy + u * vxy + vy + ux
        W[1, 2] = 0.5 * (v * uyy + 2.0 * uy * (1.0 + vy) + u * vyy)
        W[1, 3] = v + v * ux + u * vx
        W[1, 4] = u + v * uy + u * vy
        W[1, 5] = u * v
        W[2, 0] = vx * vx + v * vxx
        W[2, 1] = 2.0 * v * vxy + 2.0 * vx * (1.0 + vy)
        W[2, 2] = 1.0 + 2.0 * vy + vy * vy + v * vyy
        W[2, 3] = 2.0 * v * vx
        W[2, 4] = 2.0 * v * (1.0 + vy)
        W[2, 5] = v * v
        W[3, 0] = 0.5 * uxx
        W[3, 1] = uxy
        W[3, 2] = 0.5 * uyy
        W[3, 3] = 1.0 + ux
        W[3, 4] = uy
        W[3, 5] = u
        W[4, 0] = 0.5 * vxx
        W[4, 1] = vxy
        W[4, 2] = 0.5 * vyy
        W[4, 3] = vx
        W[4, 4] = 1.0 + vy
        W[4, 5] = v
        W[5, 0] = 0.0
        W[5, 1] = 0.0
        W[5, 2] = 0.0
        W[5, 3] = 0.0
        W[5, 4] = 0.0
        W[5, 5] = 1.0

    def _extract_from_warp(self):
        W = self.warp_matrix
        self.u = float(W[3, 5])
        self.ux = float(W[3, 3] - 1.0)
        self.uy = float(W[3, 4])
        self.uxx = float(W[3, 0] * 2.0)
        self.uxy = float(W[3, 1])
        self.uyy = float(W[3, 2] * 2.0)
        self.v = float(W[4, 5])
        self.vx = float(W[4, 3])
        self.vy = float(W[4, 4] - 1.0)
        self.vxx = float(W[4, 0] * 2.0)
        self.vxy = float(W[4, 1])
        self.vyy = float(W[4, 2] * 2.0)

    def warp(self, x_local: float, y_local: float):
        pt = np.array([x_local * x_local, x_local * y_local, y_local * y_local, x_local, y_local, 1.0], dtype=np.float32)
        warped = self.warp_matrix @ pt
        return (float(warped[3]), float(warped[4]))

    def icgn_update(self, p_increment):
        W_inc_inv = np.linalg.inv(p_increment.warp_matrix)
        self.warp_matrix = self.warp_matrix @ W_inc_inv
        self._extract_from_warp()

    def to_array(self):
        return np.array([self.u, self.ux, self.uy, self.uxx, self.uxy, self.uyy, self.v, self.vx, self.vy, self.vxx, self.vxy, self.vyy], dtype=np.float32)

class ICGN2D2:

    def __init__(self, subset_radius_x=16, subset_radius_y=16, conv_criterion=0.001, stop_condition=50):
        self.subset_rx = subset_radius_x
        self.subset_ry = subset_radius_y
        self.conv_criterion = conv_criterion
        self.stop_condition = stop_condition
        self.ref_gradient_x = None
        self.ref_gradient_y = None
        self.tar_interp = None
        self.ref_img = None

    def prepare(self, ref_img, tar_img, ref_gradient_x, ref_gradient_y, tar_interp):
        self.ref_img = ref_img.astype(np.float32)
        self.ref_gradient_x = ref_gradient_x
        self.ref_gradient_y = ref_gradient_y
        self.tar_interp = tar_interp

    def compute_single(self, poi_x, poi_y, initial_u, initial_v, *, initial_gradients=None):
        gradients = np.asarray((0.0, 0.0, 0.0, 0.0) if initial_gradients is None else initial_gradients, dtype=float)
        if gradients.shape != (4,) or not np.isfinite(gradients).all():
            raise ValueError('initial_gradients must be finite (ux, uy, vx, vy)')
        (rx, ry) = (self.subset_rx, self.subset_ry)
        subset_w = 2 * rx + 1
        subset_h = 2 * ry + 1
        if poi_y - ry < 0 or poi_x - rx < 0 or poi_y + ry >= self.ref_img.shape[0] or (poi_x + rx >= self.ref_img.shape[1]) or (abs(initial_u) >= self.ref_img.shape[1]) or (abs(initial_v) >= self.ref_img.shape[0]) or np.isnan(initial_u) or np.isnan(initial_v):
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0, 'uxy': 0.0, 'uyy': 0.0, 'vx': 0.0, 'vy': 0.0, 'vxx': 0.0, 'vxy': 0.0, 'vyy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        y_s = int(poi_y) - ry
        x_s = int(poi_x) - rx
        ref_subset = self.ref_img[y_s:y_s + subset_h, x_s:x_s + subset_w].copy()
        ref_mean = ref_subset.mean()
        ref_subset -= ref_mean
        ref_norm = np.sqrt(np.sum(ref_subset ** 2))
        if ref_norm < 1e-10:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0, 'uxy': 0.0, 'uyy': 0.0, 'vx': 0.0, 'vy': 0.0, 'vxx': 0.0, 'vxy': 0.0, 'vyy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        gx = self.ref_gradient_x[y_s:y_s + subset_h, x_s:x_s + subset_w]
        gy = self.ref_gradient_y[y_s:y_s + subset_h, x_s:x_s + subset_w]
        y_coords = np.arange(subset_h, dtype=np.float32) - ry
        x_coords = np.arange(subset_w, dtype=np.float32) - rx
        (xx, yy) = np.meshgrid(x_coords, y_coords)
        xx2 = xx * xx * 0.5
        xy2 = xx * yy
        yy2 = yy * yy * 0.5
        sd = np.zeros((subset_h, subset_w, 12), dtype=np.float32)
        sd[:, :, 0] = gx
        sd[:, :, 1] = gx * xx
        sd[:, :, 2] = gx * yy
        sd[:, :, 3] = gx * xx2
        sd[:, :, 4] = gx * xy2
        sd[:, :, 5] = gx * yy2
        sd[:, :, 6] = gy
        sd[:, :, 7] = gy * xx
        sd[:, :, 8] = gy * yy
        sd[:, :, 9] = gy * xx2
        sd[:, :, 10] = gy * xy2
        sd[:, :, 11] = gy * yy2
        sd_flat = sd.reshape(-1, 12)
        H = sd_flat.T @ sd_flat
        try:
            inv_H = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0, 'uxy': 0.0, 'uyy': 0.0, 'vx': 0.0, 'vy': 0.0, 'vxx': 0.0, 'vxy': 0.0, 'vyy': 0.0, 'zncc': -3.0, 'iteration': 0, 'convergence': 999.0}
        p1_init = Deformation2D1(u=initial_u, ux=gradients[0], uy=gradients[1], v=initial_v, vx=gradients[2], vy=gradients[3])
        p_current = Deformation2D2()
        p_current.set_deformation_from_1st_order(p1_init)
        iteration = 0
        dp_norm_max = 999.0
        znssd = 0.0
        while iteration < self.stop_condition and dp_norm_max >= self.conv_criterion:
            iteration += 1
            tar_subset = np.zeros((subset_h, subset_w), dtype=np.float32)
            for r in range(subset_h):
                for c in range(subset_w):
                    (wx, wy) = p_current.warp(x_coords[c], y_coords[r])
                    gx_global = poi_x + wx
                    gy_global = poi_y + wy
                    tar_subset[r, c] = self.tar_interp.compute(gx_global, gy_global)
            if np.any(tar_subset < -1):
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
            p_inc = Deformation2D2()
            p_inc.set_deformation_from_p(dp)
            p_current.icgn_update(p_inc)
            (rx2, ry2) = (rx * rx, ry * ry)
            rx4 = rx2 * rx2 * 0.25
            ry4 = ry2 * ry2 * 0.25
            rxy2 = rx2 * ry2
            dp_norm_max = np.sqrt(p_inc.u ** 2 + p_inc.v ** 2 + p_inc.ux ** 2 * rx2 + p_inc.uy ** 2 * ry2 + p_inc.vx ** 2 * rx2 + p_inc.vy ** 2 * ry2 + p_inc.uxx ** 2 * rx4 + p_inc.uyy ** 2 * ry4 + p_inc.vxx ** 2 * rx4 + p_inc.vyy ** 2 * ry4 + p_inc.uxy ** 2 * rxy2 + p_inc.vxy ** 2 * rxy2)
        zncc = 0.5 * (2.0 - znssd)
        if dp_norm_max >= self.conv_criterion and iteration >= self.stop_condition:
            zncc = -4.0
        if zncc >= 0 and zncc < 0.7:
            zncc = -4.0
        if np.isnan(zncc) or np.isnan(p_current.u) or np.isnan(p_current.v):
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0, 'uxy': 0.0, 'uyy': 0.0, 'vx': 0.0, 'vy': 0.0, 'vxx': 0.0, 'vxy': 0.0, 'vyy': 0.0, 'zncc': -5.0, 'iteration': iteration, 'convergence': dp_norm_max}
        if zncc < 0:
            return {'u': initial_u, 'v': initial_v, 'ux': 0.0, 'uy': 0.0, 'uxx': 0.0, 'uxy': 0.0, 'uyy': 0.0, 'vx': 0.0, 'vy': 0.0, 'vxx': 0.0, 'vxy': 0.0, 'vyy': 0.0, 'zncc': float(zncc), 'iteration': iteration, 'convergence': float(dp_norm_max), 'u0': float(initial_u), 'v0': float(initial_v)}
        return {'u': float(p_current.u), 'v': float(p_current.v), 'ux': float(p_current.ux), 'uy': float(p_current.uy), 'uxx': float(p_current.uxx), 'uxy': float(p_current.uxy), 'uyy': float(p_current.uyy), 'vx': float(p_current.vx), 'vy': float(p_current.vy), 'vxx': float(p_current.vxx), 'vxy': float(p_current.vxy), 'vyy': float(p_current.vyy), 'zncc': float(zncc), 'iteration': iteration, 'convergence': float(dp_norm_max), 'u0': float(initial_u), 'v0': float(initial_v)}
