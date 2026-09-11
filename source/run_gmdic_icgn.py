# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

import cv2
import numpy as np
import time
import os
from pathlib import Path
import glob
import matplotlib.pyplot as plt
import torch
from networks.tfmdic import tfmdic
from utils.utils import InputPadder
from icgn import refine_displacement
from icgn.paper_metrics import print_results, evaluate_initialization
DATA_DIR = 'E:\\codeimagedata\\image\\tension009'
CHECKPOINT = str(Path(__file__).resolve().parents[1] / 'checkpoints' / 'step_250000.pth')

def find_file(pattern, dir_path):
    matches = glob.glob(os.path.join(dir_path, pattern))
    if not matches:
        raise FileNotFoundError(f"找不到匹配 '{pattern}' 的文件: {dir_path}")
    return matches[0]

def estimate_gmdic(ref, tar):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Loading: {CHECKPOINT}')
    model = tfmdic(num_scales=1, num_transformer_blocks=16, upsample_factor=2, feature_channels=128, attention_type='swin', ffn_dim_expansion=4, num_head=1).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt, strict=True)
    model.eval()
    ref_t = torch.as_tensor(ref, dtype=torch.float32, device=device)[None, None]
    tar_t = torch.as_tensor(tar, dtype=torch.float32, device=device)[None, None]
    padder = InputPadder(ref_t.shape, padding_factor=16)
    (ref_p, tar_p) = padder.pad(ref_t, tar_t)
    with torch.no_grad():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        results = model(ref_p, tar_p, attn_splits_list=[8], corr_radius_list=[5], prop_radius_list=False)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        init_time = time.perf_counter() - started
        flow = padder.unpad(results['flow_preds'][-1][0]).detach().cpu().numpy()
    if flow.shape != (2, *ref.shape) or not np.isfinite(flow).all():
        raise RuntimeError('GM-DIC 输出尺寸错误或包含非有限值')
    print(f'GM-DIC forward (single call, no warm-up): {init_time:.6f} s')
    return (flow[0], flow[1], init_time)

def save_gmdic_initial(u, v, out_dir):
    for (name, values) in [('GM_U.csv', u), ('GM_V.csv', v)]:
        np.savetxt(os.path.join(out_dir, name), values, fmt='%.6f', delimiter=',', newline=',\n')
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    for (ax, values, label) in zip(axes, (u, v), ('U', 'V')):
        im = ax.imshow(values, cmap='jet', origin='upper')
        fig.colorbar(im, ax=ax, label=f'Displacement ({label})', fraction=0.046, pad=0.04)
        ax.set_title(f'{label} Direction Displacement')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'GM_displacement.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: GM_U.csv, GM_V.csv, GM_displacement.png')

def save_visualization(u_icgn, v_icgn, out_dir):
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    im0 = axes[0].imshow(u_icgn, cmap='jet')
    axes[0].set_title('TFM-ICGN: U Displacement')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(v_icgn, cmap='jet')
    axes[1].set_title('TFM-ICGN: V Displacement')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'GMGN_displacement.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'  → GMGN_displacement.png')

def main():
    ref_path = find_file('*REF*.*', DATA_DIR)
    tar_path = find_file('*TAR*.*', DATA_DIR)
    print(f'REF: {os.path.basename(ref_path)}')
    print(f'TAR: {os.path.basename(tar_path)}')
    print(f'Dir:  {DATA_DIR}')
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    tar = cv2.imread(tar_path, cv2.IMREAD_GRAYSCALE)
    if ref is None or tar is None or ref.shape != tar.shape:
        raise ValueError('REF/TAR读取失败或尺寸不一致')
    ref = ref.astype(np.float32)
    tar = tar.astype(np.float32)
    print(f'Size: {ref.shape[1]}x{ref.shape[0]}')
    (u_gmdic, v_gmdic, init_time) = estimate_gmdic(ref, tar)
    save_gmdic_initial(u_gmdic, v_gmdic, DATA_DIR)
    print(f'GM-DIC: u [{u_gmdic.min():.3f}, {u_gmdic.max():.3f}], v [{v_gmdic.min():.3f}, {v_gmdic.max():.3f}]')
    t0 = time.time()
    result = refine_displacement(ref, tar, u_init=u_gmdic, v_init=v_gmdic, subset_radius=10, conv_criterion=0.0001, stop_condition=50, step_size=1, order=2, verbose=True, paper_timing=True)
    MARGIN = 12
    u_crop = result['u'][MARGIN:-MARGIN, MARGIN:-MARGIN]
    v_crop = result['v'][MARGIN:-MARGIN, MARGIN:-MARGIN]
    np.savetxt(os.path.join(DATA_DIR, 'GMGN_U.csv'), u_crop, delimiter=',', fmt='%.6f')
    np.savetxt(os.path.join(DATA_DIR, 'GMGN_V.csv'), v_crop, delimiter=',', fmt='%.6f')
    save_visualization(u_crop, v_crop, DATA_DIR)
    delta = np.sqrt((result['u'] - u_gmdic) ** 2 + (result['v'] - v_gmdic) ** 2)
    stats = result['stats']
    succ_iter = stats['succ_iterations'] / stats['success_count'] if stats['success_count'] > 0 else 0
    metric_selected = np.isfinite(result['convergence']) & (result['convergence'] >= 0) & (result['convergence'] < 0.0001) & (result['iteration'] >= 1) & (result['iteration'] <= 50) & (result['zncc'] >= 0.7)
    (metric_y, metric_x) = np.indices(ref.shape)
    paper_metrics = evaluate_initialization(DATA_DIR, ref.shape, metric_x, metric_y, u_gmdic, v_gmdic, result['u'], result['v'], metric_selected, result['point_seconds'])
    print_results('TFM-ICGN', init_time, stats['time_elapsed'], stats['n_pois'], stats['avg_iterations'], stats['success_rate'] * 100, stats['success_count'], stats['fail_count'], delta.mean(), delta.max(), succ_time=stats['succ_time_total'], succ_iters=succ_iter, n_succ=stats['success_count'], paper_metrics=paper_metrics, table_directory=DATA_DIR, table_method='TFM-ICGN')
    print(f'Output: {DATA_DIR}\\GMGN_U.csv, GMGN_V.csv, GMGN_displacement.png')
if __name__ == '__main__':
    main()
