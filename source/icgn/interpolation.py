# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np

class BicubicBspline:
    BC_MATRIX = np.array([[-144.0 / 336.0, 384.0 / 336.0, -384.0 / 336.0, 144.0 / 336.0], [342.0 / 336.0, -702.0 / 336.0, 450.0 / 336.0, -90.0 / 336.0], [-198.0 / 336.0, -18.0 / 336.0, 270.0 / 336.0, -54.0 / 336.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)

    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        (self.height, self.width) = image.shape
        self.coefficient = None

    def prepare(self):
        (H, W) = (self.height, self.width)
        BC = self.BC_MATRIX
        self.coefficient = np.zeros((H, W, 4, 4), dtype=np.float32)
        for r in range(1, H - 2):
            for c in range(1, W - 2):
                Q = self.image[r - 1:r + 3, c - 1:c + 3]
                mat_p = BC @ Q @ BC.T
                self.coefficient[r, c] = mat_p[::-1, ::-1]

    def compute(self, x: float, y: float) -> float:
        if x < 1.0 or y < 1.0 or x >= self.width - 2 or (y >= self.height - 2):
            return -1.0
        if np.isnan(x) or np.isnan(y):
            return -1.0
        x_int = int(np.floor(x))
        y_int = int(np.floor(y))
        dx = x - x_int
        dy = y - y_int
        dx2 = dx * dx
        dy2 = dy * dy
        dx3 = dx2 * dx
        dy3 = dy2 * dy
        coeff = self.coefficient[y_int, x_int]
        value = coeff[0, 0] + coeff[0, 1] * dx + coeff[0, 2] * dx2 + coeff[0, 3] * dx3 + coeff[1, 0] * dy + coeff[1, 1] * dy * dx + coeff[1, 2] * dy * dx2 + coeff[1, 3] * dy * dx3 + coeff[2, 0] * dy2 + coeff[2, 1] * dy2 * dx + coeff[2, 2] * dy2 * dx2 + coeff[2, 3] * dy2 * dx3 + coeff[3, 0] * dy3 + coeff[3, 1] * dy3 * dx + coeff[3, 2] * dy3 * dx2 + coeff[3, 3] * dy3 * dx3
        return float(value)
