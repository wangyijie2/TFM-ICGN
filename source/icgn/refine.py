# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np
import time
from .gradient import Gradient2D4
from .interpolation import BicubicBspline
from .icgn_core import ICGN2D1
from .icgn2d2 import ICGN2D2

def refine_displacement(ref_img: np.ndarray, tar_img: np.ndarray, u_init: np.ndarray, v_init: np.ndarray, subset_radius: int=15, conv_criterion: float=0.001, stop_condition: int=50, step_size: int=1, order: int=2, verbose: bool=True, paper_timing: bool=False) -> dict:
    if ref_img.ndim == 2:
        ref_img = ref_img.astype(np.float32)
    if tar_img.ndim == 2:
        tar_img = tar_img.astype(np.float32)
    (H, W) = ref_img.shape
    margin = subset_radius + 2
    if verbose:
        print('[IC-GN Refine] Computing reference image gradients...')
    grad2d = Gradient2D4(ref_img)
    grad_x = grad2d.compute_gradient_x()
    grad_y = grad2d.compute_gradient_y()
    if verbose:
        print('[IC-GN Refine] Precomputing B-spline interpolation coefficients...')
    t0 = time.time()
    tar_interp = BicubicBspline(tar_img)
    tar_interp.prepare()
    if verbose:
        print(f'  B-spline prepare: {time.time() - t0:.2f}s')
    if order == 2:
        icgn = ICGN2D2(subset_radius_x=subset_radius, subset_radius_y=subset_radius, conv_criterion=conv_criterion, stop_condition=stop_condition)
    else:
        icgn = ICGN2D1(subset_radius_x=subset_radius, subset_radius_y=subset_radius, conv_criterion=conv_criterion, stop_condition=stop_condition)
    icgn.prepare(ref_img, tar_img, grad_x, grad_y, tar_interp)
    y_range = range(margin, H - margin, step_size)
    x_range = range(margin, W - margin, step_size)
    n_pois = len(y_range) * len(x_range)
    u_out = u_init.copy().astype(np.float32)
    v_out = v_init.copy().astype(np.float32)
    ux_out = np.zeros_like(u_init, dtype=np.float32)
    uy_out = np.zeros_like(u_init, dtype=np.float32)
    vx_out = np.zeros_like(u_init, dtype=np.float32)
    vy_out = np.zeros_like(u_init, dtype=np.float32)
    zncc_out = np.zeros_like(u_init, dtype=np.float32)
    convergence_out = np.full(u_out.shape, np.nan, dtype=float)
    point_seconds_out = np.full(u_out.shape, np.nan, dtype=float)
    iter_out = np.zeros_like(u_init, dtype=np.int32)
    if verbose:
        print(f'[IC-GN Refine] Processing {n_pois} POIs (step={step_size}, subset={2 * subset_radius + 1}x{2 * subset_radius + 1})...')
    t0 = time.time()
    success_count = 0
    fail_count = 0
    total_iterations = 0
    succ_iterations = 0
    succ_time_total = 0.0
    for (idx_y, poi_y) in enumerate(y_range):
        for (idx_x, poi_x) in enumerate(x_range):
            t_poi = time.perf_counter()
            result = icgn.compute_single(poi_x=float(poi_x), poi_y=float(poi_y), initial_u=float(u_init[poi_y, poi_x]), initial_v=float(v_init[poi_y, poi_x]))
            dt_poi = time.perf_counter() - t_poi
            convergence_out[poi_y, poi_x] = result['convergence']
            point_seconds_out[poi_y, poi_x] = dt_poi
            u_out[poi_y, poi_x] = result['u']
            v_out[poi_y, poi_x] = result['v']
            ux_out[poi_y, poi_x] = result['ux']
            uy_out[poi_y, poi_x] = result['uy']
            vx_out[poi_y, poi_x] = result['vx']
            vy_out[poi_y, poi_x] = result['vy']
            zncc_out[poi_y, poi_x] = result['zncc']
            iter_out[poi_y, poi_x] = result['iteration']
            if result['zncc'] >= 0:
                success_count += 1
                succ_time_total += dt_poi
                succ_iterations += result['iteration']
            else:
                fail_count += 1
            total_iterations += result['iteration']
            if verbose and (idx_y * len(x_range) + idx_x + 1) % max(1, n_pois // 10) == 0:
                pct = (idx_y * len(x_range) + idx_x + 1) / n_pois * 100
                print(f'  {pct:.0f}% done, avg_iter={total_iterations / max(1, success_count + fail_count):.1f}')
    elapsed = time.time() - t0
    stats = {'n_pois': n_pois, 'success_count': success_count, 'fail_count': fail_count, 'success_rate': success_count / n_pois if n_pois > 0 else 0, 'avg_iterations': total_iterations / n_pois if n_pois > 0 else 0, 'succ_time_total': succ_time_total, 'succ_iterations': succ_iterations, 'avg_zncc': float(np.mean(zncc_out[margin:-margin:step_size, margin:-margin:step_size][zncc_out[margin:-margin:step_size, margin:-margin:step_size] >= 0])), 'time_elapsed': elapsed, 'time_per_poi': elapsed / n_pois * 1000 if n_pois > 0 else 0}
    if verbose:
        if not paper_timing:
            print(f"  Done in {elapsed:.2f}s ({stats['time_per_poi']:.2f}ms/POI)")
        print(f"  Success: {success_count}/{n_pois} ({stats['success_rate'] * 100:.1f}%)")
        print(f"  Avg iterations: {stats['avg_iterations']:.2f}")
        print(f"  Avg ZNCC: {stats['avg_zncc']:.4f}")
    return {'u': u_out, 'v': v_out, 'ux': ux_out, 'uy': uy_out, 'vx': vx_out, 'vy': vy_out, 'zncc': zncc_out, 'iteration': iter_out, 'convergence': convergence_out, 'point_seconds': point_seconds_out, 'stats': stats}
