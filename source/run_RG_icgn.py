# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2017, Justin Blaber. All rights reserved.
# See LICENSES/BSD-3-Clause.txt for conditions and disclaimer.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import cv2
import numpy as np
import time
import os
import glob
import heapq
import itertools
import matplotlib.pyplot as plt
from icgn.fftcc import fftcc_estimate
from icgn import Gradient2D4, BicubicBspline
from icgn.icgn2d2 import ICGN2D2
from icgn.format_result import print_results
DATA_DIR = 'E:\\codeimagedata\\image\\complex4.2'
SUBSET_R = 10
CONV_CRIT = 0.0001
MAX_ITER = 50
STEP_X = 1
MARGIN = SUBSET_R + 2

def find_file(pattern, dir_path):
    matches = glob.glob(os.path.join(dir_path, pattern))
    if not matches:
        raise FileNotFoundError(f"找不到 '{pattern}': {dir_path}")
    return matches[0]

def save_visualization(u_arr, v_arr, out_dir, prefix='RG_ICGN2'):
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(u_arr, cmap='jet')
    axes[0].set_title(f'{prefix}: U Displacement')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(v_arr, cmap='jet')
    axes[1].set_title(f'{prefix}: V Displacement')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    png_path = os.path.join(out_dir, f'{prefix}_displacement.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → {prefix}_displacement.png')
ref_path = find_file('*REF*.*', DATA_DIR)
tar_path = find_file('*TAR*.*', DATA_DIR)
print(f'REF: {os.path.basename(ref_path)}')
print(f'TAR: {os.path.basename(tar_path)}')
ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
tar = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
(H, W) = ref.shape
print(f'Size: {W}x{H}')
print('\n[Stage 1] Precomputing gradients and B-spline...')
grad2d = Gradient2D4(ref)
gx = grad2d.compute_gradient_x()
gy = grad2d.compute_gradient_y()
tar_interp = BicubicBspline(tar)
tar_interp.prepare()
icgn = ICGN2D2(SUBSET_R, SUBSET_R, CONV_CRIT, MAX_ITER)
icgn.prepare(ref, tar, gx, gy, tar_interp)
print('\n[Stage 2] Seed point selection and initialization...')
if STEP_X < 1 or not isinstance(STEP_X, int):
    raise ValueError('STEP_X 必须为正整数')
grid_x = range(MARGIN, W - MARGIN, STEP_X)
grid_y = range(MARGIN, H - MARGIN, STEP_X)
if not grid_x or not grid_y:
    raise ValueError('图像尺寸不足以容纳当前子区和边缘范围')
seed_x = min(grid_x, key=lambda x: abs(x - W // 2))
seed_y = min(grid_y, key=lambda y: abs(y - H // 2))
total_pois = len(grid_x) * len(grid_y)
t_init = time.perf_counter()
fftcc_out = fftcc_estimate(ref, tar, [(seed_x, seed_y)], SUBSET_R, SUBSET_R)
init_time = time.perf_counter() - t_init
seed_init_ok = bool(fftcc_out) and fftcc_out[0] is not None and np.isfinite([fftcc_out[0]['u'], fftcc_out[0]['v']]).all()
seed_u0 = float(fftcc_out[0]['u']) if seed_init_ok else 0.0
seed_v0 = float(fftcc_out[0]['v']) if seed_init_ok else 0.0

def result_is_success(result):
    keys = ('u', 'v', 'ux', 'uy', 'vx', 'vy', 'uxx', 'uxy', 'uyy', 'vxx', 'vxy', 'vyy', 'zncc', 'convergence', 'iteration')
    return all((np.isfinite(result[k]) for k in keys)) and result['zncc'] >= 0.7 and (0 <= result['convergence'] < CONV_CRIT) and (1 <= result['iteration'] <= MAX_ITER)
print('\n[Stage 3] Reliability-guided propagation...')
u_out = np.full((H, W), np.nan, dtype=np.float32)
v_out = np.full((H, W), np.nan, dtype=np.float32)
status = np.full((H, W), -1, dtype=np.int8)
(seed_deltas, succ_iters) = ([], [])
succ_time_total = 0.0
processed = successes = fails = total_iter = solver_calls = 0
propagation_blocked = 0
queue = []
sequence = itertools.count()
neighbors4 = [(0, -STEP_X), (STEP_X, 0), (0, STEP_X), (-STEP_X, 0)]

def solve_point(px, py, u0, v0, gradients=None):
    global processed, successes, fails, total_iter, solver_calls
    global succ_time_total, propagation_blocked
    processed += 1
    status[py, px] = 1
    if not np.isfinite([u0, v0]).all():
        fails += 1
        propagation_blocked += 1
        return
    t_poi = time.perf_counter()
    result = icgn.compute_single(float(px), float(py), float(u0), float(v0), initial_gradients=gradients)
    dt_poi = time.perf_counter() - t_poi
    solver_calls += 1
    total_iter += result['iteration']
    if result_is_success(result):
        (u_out[py, px], v_out[py, px]) = (result['u'], result['v'])
        status[py, px] = 0
        successes += 1
        seed_deltas.append(np.hypot(result['u'] - u0, result['v'] - v0))
        succ_iters.append(result['iteration'])
        succ_time_total += dt_poi
        heapq.heappush(queue, (-result['zncc'], next(sequence), px, py, result))
    else:
        (u_out[py, px], v_out[py, px]) = (u0, v0)
        fails += 1
        propagation_blocked += 1
t0 = time.perf_counter()
if seed_init_ok:
    print(f'  Seed at ({seed_x},{seed_y}), FFT-CC init: ({seed_u0:.3f}, {seed_v0:.3f}) px')
    solve_point(seed_x, seed_y, seed_u0, seed_v0)
else:
    processed = fails = propagation_blocked = 1
    status[seed_y, seed_x] = 1
    print('  WARNING: seed FFT-CC failed; propagation stopped.')
if not queue and seed_init_ok:
    print('  WARNING: seed rejected; propagation stopped.')
while queue:
    (_, _, parent_x, parent_y, parent) = heapq.heappop(queue)
    for (dx, dy) in neighbors4:
        (px, py) = (parent_x + dx, parent_y + dy)
        if not (MARGIN <= px < W - MARGIN and MARGIN <= py < H - MARGIN):
            continue
        if status[py, px] >= 0:
            continue
        u0 = parent['u'] + parent['ux'] * dx + parent['uy'] * dy
        v0 = parent['v'] + parent['vx'] * dx + parent['vy'] * dy
        gradients = (parent['ux'], parent['uy'], parent['vx'], parent['vy'])
        solve_point(px, py, u0, v0, gradients)
        if processed % max(1, total_pois // 10) == 0:
            avg_iter = total_iter / solver_calls if solver_calls else 0.0
            print(f'  {processed / total_pois * 100:.0f}% processed={processed}/{total_pois}, avg_iter={avg_iter:.1f}, blocked={propagation_blocked}')
icgn_time = time.perf_counter() - t0
unvisited = total_pois - processed
u_crop = u_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
v_crop = v_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
np.savetxt(os.path.join(DATA_DIR, 'RG_ICGN2_U.csv'), np.nan_to_num(u_crop, nan=0.0), delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(DATA_DIR, 'RG_ICGN2_V.csv'), np.nan_to_num(v_crop, nan=0.0), delimiter=',', fmt='%.6f')
save_visualization(u_crop, v_crop, DATA_DIR, 'RG_ICGN2')
avg_iter = total_iter / solver_calls if solver_calls > 0 else 0
sd = np.array(seed_deltas) if seed_deltas else np.array([0.0])
print_results('RG-ICGN + ICGN2D2', init_time, icgn_time, total_pois, avg_iter, successes / total_pois * 100, successes, fails + unvisited, sd.mean(), sd.max(), succ_time=succ_time_total, succ_iters=np.mean(succ_iters) if succ_iters else 0, n_succ=successes)
print(f'  Seed at ({seed_x},{seed_y}), rejected: {fails}, unvisited: {unvisited}')
print('  Avg iterations: solver calls only; precomputation excluded from timing.')
print(f'Output: {DATA_DIR}\\RG_ICGN2_U.csv, RG_ICGN2_V.csv, RG_ICGN2_displacement.png')
