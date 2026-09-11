# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np

class Deformation2D1:

    def __init__(self, u: float=0.0, ux: float=0.0, uy: float=0.0, v: float=0.0, vx: float=0.0, vy: float=0.0):
        self.u = u
        self.ux = ux
        self.uy = uy
        self.v = v
        self.vx = vx
        self.vy = vy
        self.warp_matrix = np.eye(3, dtype=np.float32)
        self._set_warp()

    def _set_warp(self):
        self.warp_matrix[0, 0] = 1.0 + self.ux
        self.warp_matrix[0, 1] = self.uy
        self.warp_matrix[0, 2] = self.u
        self.warp_matrix[1, 0] = self.vx
        self.warp_matrix[1, 1] = 1.0 + self.vy
        self.warp_matrix[1, 2] = self.v
        self.warp_matrix[2, 0] = 0.0
        self.warp_matrix[2, 1] = 0.0
        self.warp_matrix[2, 2] = 1.0

    def _extract_from_warp(self):
        self.u = float(self.warp_matrix[0, 2])
        self.ux = float(self.warp_matrix[0, 0] - 1.0)
        self.uy = float(self.warp_matrix[0, 1])
        self.v = float(self.warp_matrix[1, 2])
        self.vx = float(self.warp_matrix[1, 0])
        self.vy = float(self.warp_matrix[1, 1] - 1.0)

    def set_deformation(self, p=None):
        if p is not None:
            (self.u, self.ux, self.uy, self.v, self.vx, self.vy) = map(float, p)
        self._set_warp()

    def warp(self, x_local: float, y_local: float) -> tuple:
        pt = np.array([x_local, y_local, 1.0], dtype=np.float32)
        warped = self.warp_matrix @ pt
        return (float(warped[0]), float(warped[1]))

    def icgn_update(self, p_increment: 'Deformation2D1'):
        W_inc_inv = np.linalg.inv(p_increment.warp_matrix)
        self.warp_matrix = self.warp_matrix @ W_inc_inv
        self._extract_from_warp()

    def to_array(self) -> np.ndarray:
        return np.array([self.u, self.ux, self.uy, self.v, self.vx, self.vy], dtype=np.float32)
