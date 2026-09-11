# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np

class Gradient2D4:

    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        (self.height, self.width) = image.shape
        self.gradient_x = np.zeros((self.height, self.width), dtype=np.float32)
        self.gradient_y = np.zeros((self.height, self.width), dtype=np.float32)

    def compute_gradient_x(self) -> np.ndarray:
        c1 = 1.0 / 12.0
        c2 = 2.0 / 3.0
        grad = np.zeros_like(self.image)
        grad[:, 2:-2] = -c1 * self.image[:, 4:] + c2 * self.image[:, 3:-1] - c2 * self.image[:, 1:-3] + c1 * self.image[:, :-4]
        self.gradient_x = grad
        return grad

    def compute_gradient_y(self) -> np.ndarray:
        c1 = 1.0 / 12.0
        c2 = 2.0 / 3.0
        grad = np.zeros_like(self.image)
        grad[2:-2, :] = -c1 * self.image[4:, :] + c2 * self.image[3:-1, :] - c2 * self.image[1:-3, :] + c1 * self.image[:-4, :]
        self.gradient_y = grad
        return grad
