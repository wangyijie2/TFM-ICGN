# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import cv2
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt
from icgn.fftcc import fftcc_estimate
from icgn import Gradient2D4, BicubicBspline
from icgn.icgn2d2 import ICGN2D2
from icgn.paper_metrics import print_results, evaluate_initialization
from icgn.outlier_filter import filter_outliers
DATA_DIR = 'E:\\codeimagedata\\image\\tension009'

def find_file(pattern, dir_path):
    matches = glob.glob(os.path.join(dir_path, pattern))
    if not matches:
        raise FileNotFoundError(f"找不到 '{pattern}': {dir_path}")
    return matches[0]

def save_visualization(u_arr, v_arr, out_dir, prefix='FFT_ICGN2'):
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(u_arr, cmap='jet')
    axes[0].set_title(f'FFT-ICGN: U Displacement')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(v_arr, cmap='jet')
    axes[1].set_title(f'FFT-ICGN: V Displacement')
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
SUBSET_R = 10
CONV_CRIT = 0.0001
MAX_ITER = 50
STEP_X = 1
MARGIN = SUBSET_R + 2
poi_list = [(x, y) for y in range(MARGIN, H - MARGIN) for x in range(MARGIN, W - MARGIN, STEP_X)]
print(f'POIs: {len(poi_list)}')
print('\n[Stage 1] FFT-CC initial estimation...')
t0 = time.time()
fftcc_out = fftcc_estimate(ref, tar, poi_list, SUBSET_R, SUBSET_R, odd_subset=True)
fftcc_time = time.time() - t0
u_ini = np.zeros((H, W), dtype=np.float32)
v_ini = np.zeros((H, W), dtype=np.float32)
fftcc_fail = 0
for (k, (px, py)) in enumerate(poi_list):
    r = fftcc_out[k]
    if r is not None:
        u_ini[py, px] = r['u']
        v_ini[py, px] = r['v']
    else:
        fftcc_fail += 1
print(f'  Time: {fftcc_time:.2f}s, FFT-CC boundary fails: {fftcc_fail}')
print('\n[Stage 2] ICGN2D2 sub-pixel refinement...')
grad2d = Gradient2D4(ref)
gx = grad2d.compute_gradient_x()
gy = grad2d.compute_gradient_y()
tar_interp = BicubicBspline(tar)
tar_interp.prepare()
icgn = ICGN2D2(SUBSET_R, SUBSET_R, CONV_CRIT, MAX_ITER)
icgn.prepare(ref, tar, gx, gy, tar_interp)
metric_converged = np.zeros((H, W), dtype=bool)
metric_seconds = np.full((H, W), np.nan)
u_out = u_ini.copy()
v_out = v_ini.copy()
iterations = []
successes = 0
fails = 0
succ_iters = []
succ_time_total = 0.0
t0 = time.time()
for (k, (px, py)) in enumerate(poi_list):
    t_poi = time.perf_counter()
    result = icgn.compute_single(poi_x=float(px), poi_y=float(py), initial_u=float(u_ini[py, px]), initial_v=float(v_ini[py, px]))
    dt_poi = time.perf_counter() - t_poi
    metric_seconds[py, px] = dt_poi
    metric_converged[py, px] = np.isfinite(result['convergence']) and 0 <= result['convergence'] < CONV_CRIT and (1 <= result['iteration'] <= MAX_ITER) and (result['zncc'] >= 0.7)
    u_out[py, px] = result['u']
    v_out[py, px] = result['v']
    iterations.append(result['iteration'])
    if result['zncc'] >= 0:
        successes += 1
        succ_iters.append(result['iteration'])
        succ_time_total += dt_poi
    else:
        fails += 1
        u_out[py, px] = np.nan
        v_out[py, px] = np.nan
    if (k + 1) % max(1, len(poi_list) // 10) == 0:
        pct = (k + 1) / len(poi_list) * 100
        valid_iter = [i for i in iterations if i > 0]
        print(f'  {pct:.0f}% avg_iter={(np.mean(valid_iter) if valid_iter else 0):.1f}')
icgn_time = time.time() - t0
(u_out, v_out, n_outliers) = filter_outliers(u_out, v_out, threshold=12.0)
fails += n_outliers
if n_outliers > 0:
    print(f'  Outlier filter removed: {n_outliers} points')
u_crop = u_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
v_crop = v_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
np.savetxt(os.path.join(DATA_DIR, 'FFT_ICGN2_U.csv'), u_crop, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(DATA_DIR, 'FFT_ICGN2_V.csv'), v_crop, delimiter=',', fmt='%.6f')
save_visualization(u_crop, v_crop, DATA_DIR, 'FFT_ICGN2')
valid_mask = ~np.isnan(u_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN])
valid_delta = np.sqrt((u_crop[valid_mask] - u_ini[MARGIN:H - MARGIN, MARGIN:W - MARGIN][valid_mask]) ** 2 + (v_crop[valid_mask] - v_ini[MARGIN:H - MARGIN, MARGIN:W - MARGIN][valid_mask]) ** 2)
valid_iter = [i for i in iterations if i > 0]
metric_selected = metric_converged & np.isfinite(u_out) & np.isfinite(v_out)
(metric_y, metric_x) = np.indices((H, W))
paper_metrics = evaluate_initialization(DATA_DIR, (H, W), metric_x, metric_y, u_ini, v_ini, u_out, v_out, metric_selected, metric_seconds)
print_results('FFT-ICGN', fftcc_time, icgn_time, len(poi_list), np.mean(valid_iter) if valid_iter else 0, successes / len(poi_list) * 100, successes, fails, valid_delta.mean() if len(valid_delta) > 0 else 0, valid_delta.max() if len(valid_delta) > 0 else 0, succ_time=succ_time_total, succ_iters=np.mean(succ_iters) if succ_iters else 0, n_succ=successes, paper_metrics=paper_metrics, table_directory=DATA_DIR, table_method='FFT-ICGN')
print(f'Output: {DATA_DIR}\\FFT_ICGN2_U.csv, FFT_ICGN2_V.csv, FFT_ICGN2_displacement.png')
print(f'Output: {DATA_DIR}\\FFT_ICGN2_U.csv, FFT_ICGN2_V.csv, FFT_ICGN2_displacement.png')
