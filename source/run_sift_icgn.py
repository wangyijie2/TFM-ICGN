# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import cv2
import numpy as np
import time
import os
import glob
import matplotlib.pyplot as plt
from icgn.sift_affine import match_sift_features, feature_affine_estimate
from icgn import Gradient2D4, BicubicBspline
from icgn.icgn2d2 import ICGN2D2
from icgn.paper_metrics import print_results, evaluate_initialization
DATA_DIR = 'E:\\codeimagedata\\image\\Uniaxial compression'

def find_file(pattern, dir_path):
    matches = glob.glob(os.path.join(dir_path, pattern))
    if not matches:
        raise FileNotFoundError(f"找不到 '{pattern}': {dir_path}")
    return matches[0]

def save_visualization(u_arr, v_arr, out_dir, prefix='SIFT_ICGN2'):
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(u_arr, cmap='jet')
    axes[0].set_title(f'SIFT-ICGN: U Displacement')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(v_arr, cmap='jet')
    axes[1].set_title(f'SIFT-ICGN: V Displacement')
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
print('\n[Stage 1] SIFT feature extraction and matching...')
t0 = time.time()
(ref_kp, tar_kp, n_match) = match_sift_features(ref, tar, contrast_threshold=0.02)
sift_match_time = time.time() - t0
print(f'  Matched features: {n_match}')
print(f'  Time: {sift_match_time:.2f}s')
if n_match < 3:
    raise RuntimeError('Not enough SIFT features for affine estimation!')
print('\n[Stage 2] FeatureAffine deformation estimation...')
search_r = np.sqrt(SUBSET_R ** 2 + SUBSET_R ** 2)
t0 = time.time()
affine_out = feature_affine_estimate(ref_kp, tar_kp, poi_list, search_radius=search_r, neighbor_min=7, ransac_samples=3, ransac_trials=20, error_threshold=1.5)
affine_time = time.time() - t0
u_ini = np.zeros((H, W), dtype=np.float32)
v_ini = np.zeros((H, W), dtype=np.float32)
sift_ok = np.zeros((H, W), dtype=bool)
affine_fails = 0
for (k, (px, py)) in enumerate(poi_list):
    r = affine_out[k]
    if r is not None and r['zncc'] >= 0:
        u_ini[py, px] = r['u']
        v_ini[py, px] = r['v']
        sift_ok[py, px] = True
    else:
        affine_fails += 1
print(f'  Time: {affine_time:.2f}s, estimation fails: {affine_fails}')
init_total = sift_match_time + affine_time
print('\n[Stage 3] ICGN2D2 sub-pixel refinement...')
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
fails = affine_fails
succ_iters = []
succ_time_total = 0.0
t0 = time.time()
for (k, (px, py)) in enumerate(poi_list):
    if not sift_ok[py, px]:
        u_out[py, px] = np.nan
        v_out[py, px] = np.nan
        iterations.append(0)
        continue
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
        print(f'  {pct:.0f}% avg_iter={np.mean(iterations):.1f}')
icgn_time = time.time() - t0
u_crop = u_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
v_crop = v_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN]
np.savetxt(os.path.join(DATA_DIR, 'SIFT_ICGN2_U.csv'), u_crop, delimiter=',', fmt='%.6f')
np.savetxt(os.path.join(DATA_DIR, 'SIFT_ICGN2_V.csv'), v_crop, delimiter=',', fmt='%.6f')
save_visualization(u_crop, v_crop, DATA_DIR, 'SIFT_ICGN2')
valid_mask = ~np.isnan(u_out[MARGIN:H - MARGIN, MARGIN:W - MARGIN])
valid_delta = np.sqrt((u_crop[valid_mask] - u_ini[MARGIN:H - MARGIN, MARGIN:W - MARGIN][valid_mask]) ** 2 + (v_crop[valid_mask] - v_ini[MARGIN:H - MARGIN, MARGIN:W - MARGIN][valid_mask]) ** 2)
metric_selected = metric_converged & np.isfinite(u_out) & np.isfinite(v_out)
(metric_y, metric_x) = np.indices((H, W))
paper_metrics = evaluate_initialization(DATA_DIR, (H, W), metric_x, metric_y, u_ini, v_ini, u_out, v_out, metric_selected, metric_seconds)
print_results('SIFT-ICGN', init_total, icgn_time, len(poi_list), np.mean([i for i in iterations if i > 0]) if any((i > 0 for i in iterations)) else 0, successes / len(poi_list) * 100, successes, fails, valid_delta.mean() if len(valid_delta) > 0 else 0, valid_delta.max() if len(valid_delta) > 0 else 0, succ_time=succ_time_total, succ_iters=np.mean(succ_iters) if succ_iters else 0, n_succ=successes, paper_metrics=paper_metrics, table_directory=DATA_DIR, table_method='SIFT-ICGN')
print(f'Output: {DATA_DIR}\\SIFT_ICGN2_U.csv, SIFT_ICGN2_V.csv, SIFT_ICGN2_displacement.png')
