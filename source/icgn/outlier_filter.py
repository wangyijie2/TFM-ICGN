# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np
from scipy.ndimage import median_filter

def filter_outliers(u, v, threshold=5.0, kernel_size=5):
    u_med = median_filter(u, size=kernel_size)
    v_med = median_filter(v, size=kernel_size)
    valid = ~np.isnan(u) & ~np.isnan(v)
    diff = np.sqrt((u - u_med) ** 2 + (v - v_med) ** 2)
    outlier = (diff > threshold) & valid
    u_f = u.copy()
    v_f = v.copy()
    u_f[outlier] = np.nan
    v_f[outlier] = np.nan
    return (u_f, v_f, int(outlier.sum()))
