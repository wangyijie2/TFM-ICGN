# SPDX-License-Identifier: MPL-2.0
# Copyright (C) 2021-2025, Zhenyu Jiang <zhenyujiang@scut.edu.cn>
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. See LICENSES/MPL-2.0.txt or https://mozilla.org/MPL/2.0/.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import numpy as np
import cv2

def match_sift_features(ref_img, tar_img, nfeatures=0, contrast_threshold=0.005, edge_threshold=10, sigma=1.6, match_ratio=0.7):
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold, sigma=sigma)
    (kp1, des1) = sift.detectAndCompute(ref_img.astype(np.uint8), None)
    (kp2, des2) = sift.detectAndCompute(tar_img.astype(np.uint8), None)
    if des1 is None or des2 is None or len(kp1) < 2 or (len(kp2) < 2):
        return ([], [], 0)
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    good = []
    for m_n in matches:
        if len(m_n) == 2:
            (m, n) = m_n
            if m.distance < match_ratio * n.distance:
                good.append(m)
    ref_kp = [(kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]) for m in good]
    tar_kp = [(kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]) for m in good]
    return (ref_kp, tar_kp, len(good))

def _solve_affine(ref_pts, tar_pts):
    N = len(ref_pts)
    R = np.ones((N, 3), dtype=np.float32)
    T = np.ones((N, 3), dtype=np.float32)
    for i in range(N):
        R[i, 0] = ref_pts[i][0]
        R[i, 1] = ref_pts[i][1]
        T[i, 0] = tar_pts[i][0]
        T[i, 1] = tar_pts[i][1]
    (A, _, _, _) = np.linalg.lstsq(R, T, rcond=None)
    return A

def _affine_to_deformation(A):
    return {'u': float(A[2, 0]), 'ux': float(A[0, 0] - 1.0), 'uy': float(A[1, 0]), 'v': float(A[2, 1]), 'vx': float(A[0, 1]), 'vy': float(A[1, 1] - 1.0)}

def feature_affine_estimate(ref_kp, tar_kp, poi_queue, search_radius, neighbor_min=7, ransac_samples=3, ransac_trials=20, error_threshold=1.5):
    n_kp = len(ref_kp)
    n_pois = len(poi_queue)
    results = [None] * n_pois
    for (idx, (px, py)) in enumerate(poi_queue):
        distances = []
        for k in range(n_kp):
            dx = ref_kp[k][0] - px
            dy = ref_kp[k][1] - py
            d = np.sqrt(dx * dx + dy * dy)
            if d < search_radius:
                distances.append((d, k))
        distances.sort(key=lambda x: x[0])
        neighbor_ids = [d[1] for d in distances]
        neighbor_num = len(neighbor_ids)
        if neighbor_num < ransac_samples:
            max_distance = 4.0 * search_radius
            nearby = []
            for k in range(n_kp):
                d = np.hypot(ref_kp[k][0] - px, ref_kp[k][1] - py)
                if d <= max_distance:
                    nearby.append((d, k))
            nearby.sort(key=lambda item: item[0])
            neighbor_ids = [item[1] for item in nearby[:neighbor_min]]
            neighbor_num = len(neighbor_ids)
            if neighbor_num < ransac_samples:
                results[idx] = {'u': 0.0, 'ux': 0.0, 'uy': 0.0, 'v': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -1.0, 'feature_count': neighbor_num}
                continue
        elif neighbor_num < neighbor_min:
            distances_all = []
            for k in range(n_kp):
                dx = ref_kp[k][0] - px
                dy = ref_kp[k][1] - py
                d = np.sqrt(dx * dx + dy * dy)
                distances_all.append((d, k))
            distances_all.sort(key=lambda x: x[0])
            neighbor_ids = [d[1] for d in distances_all[:neighbor_min]]
            neighbor_num = len(neighbor_ids)
        ref_local = []
        tar_local = []
        for nid in neighbor_ids:
            ref_local.append((ref_kp[nid][0] - px, ref_kp[nid][1] - py))
            tar_local.append((tar_kp[nid][0] - px, tar_kp[nid][1] - py))
        rng = np.random.RandomState(42)
        max_set = []
        location_mean_error = 0.0
        for trial in range(ransac_trials):
            sample_ids = list(rng.choice(neighbor_num, size=min(ransac_samples, neighbor_num), replace=False))
            ref_sample = [ref_local[i] for i in sample_ids]
            tar_sample = [tar_local[i] for i in sample_ids]
            A = _solve_affine(ref_sample, tar_sample)
            trial_set = []
            err_sum = 0.0
            for j in range(neighbor_num):
                (rx, ry) = ref_local[j]
                tx_pred = rx * A[0, 0] + ry * A[1, 0] + A[2, 0]
                ty_pred = rx * A[0, 1] + ry * A[1, 1] + A[2, 1]
                err = np.sqrt((tx_pred - tar_local[j][0]) ** 2 + (ty_pred - tar_local[j][1]) ** 2)
                if err < error_threshold:
                    trial_set.append(j)
                    err_sum += err
            if len(trial_set) > len(max_set):
                max_set = trial_set[:]
                location_mean_error = err_sum / max(1, len(trial_set))
            if len(max_set) >= neighbor_min and location_mean_error <= error_threshold / neighbor_min:
                break
        if len(max_set) < 3:
            results[idx] = {'u': 0.0, 'ux': 0.0, 'uy': 0.0, 'v': 0.0, 'vx': 0.0, 'vy': 0.0, 'zncc': -2.0, 'feature_count': len(max_set)}
            continue
        ref_inliers = [ref_local[i] for i in max_set]
        tar_inliers = [tar_local[i] for i in max_set]
        A_final = _solve_affine(ref_inliers, tar_inliers)
        deform = _affine_to_deformation(A_final)
        results[idx] = {**deform, 'zncc': 0.0, 'feature_count': len(max_set)}
    return results
