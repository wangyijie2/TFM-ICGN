# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

from pathlib import Path
import numpy as np
import csv
import os
import tempfile

def read_gt(path):
    a = np.genfromtxt(path, delimiter=',')
    if a.ndim != 2:
        raise ValueError(f'GT must be a matrix: {path}')
    if path.read_text().splitlines()[0].rstrip().endswith(','):
        a = a[:, :-1]
    return a

def evaluate_initialization(directory, shape, x, y, u0, v0, u, v, selected, seconds):
    (x, y, u0, v0, u, v, selected, seconds) = [np.asarray(a).ravel() for a in (x, y, u0, v0, u, v, selected, seconds)]
    selected = selected.astype(bool)
    count = int(selected.sum())
    if np.any(selected & (~np.isfinite(u0) | ~np.isfinite(v0) | ~np.isfinite(u) | ~np.isfinite(v) | ~np.isfinite(seconds) | (seconds < 0))):
        raise ValueError('Selected successful points contain missing values/timing; cannot report a complete mean')
    files = sorted(Path(directory).glob('*_GT_U.csv'))
    orphan = sorted(Path(directory).glob('*_GT_V.csv'))
    if len(files) > 1 or (not files and orphan):
        raise ValueError('Ambiguous or incomplete GT files; select one paired *_GT_U/V.csv')
    label = 'IC-GN correction'
    sources = []
    values = np.hypot(u - u0, v - v0)
    if files:
        up = files[0]
        vp = up.with_name(up.name[:-5] + 'V.csv')
        if not vp.exists():
            raise FileNotFoundError(vp)
        (gu, gv) = (read_gt(up), read_gt(vp))
        if gu.shape != gv.shape:
            raise ValueError('GT U/V shapes differ')
        (dy, dx) = (int(shape[0]) - gu.shape[0], int(shape[1]) - gu.shape[1])
        if dy < 0 or dx < 0 or dy % 2 or dx % 2:
            raise ValueError('GT cannot be aligned by symmetric border cropping')
        xx = x[selected].astype(int) - dx // 2
        yy = y[selected].astype(int) - dy // 2
        if np.any(xx < 0) | np.any(yy < 0) | np.any(xx >= gu.shape[1]) | np.any(yy >= gu.shape[0]):
            raise ValueError('GT does not cover every selected point; no silent subset averaging')
        if not (np.isfinite(gu[yy, xx]).all() and np.isfinite(gv[yy, xx]).all()):
            raise ValueError('GT contains missing values at selected points')
        values = np.hypot(u0[selected] - gu[yy, xx], v0[selected] - gv[yy, xx])
        label = 'IDD'
        sources = [str(up), str(vp)]
    else:
        values = values[selected]
    return {'metric_name': label, 'metric_mean_px': float(values.mean()) if count else None, 'metric_max_px': float(values.max()) if count else None, 'metric_point_count': count, 'icgn_refinement_ms_per_success': float(seconds[selected].mean() * 1000) if count else None, 'gt_files': sources}

def print_results(method_name, init_time, icgn_time, n_pois, avg_iter, success_rate, successes, fails, delta_mean=None, delta_max=None, succ_time=None, succ_iters=None, n_succ=None, *, paper_metrics, table_directory=None, table_method=None):

    def fmt(v):
        return f'{v:.6f}' if v is not None and np.isfinite(v) else 'N/A'
    print('\n' + '=' * 55 + '\n  ' + method_name + '\n' + '-' * 55)
    print(f'  Initial guess time:       {init_time:8.2f} s')
    print(f"  IC-GN refinement time:    {fmt(paper_metrics['icgn_refinement_ms_per_success'])} ms/POI")
    print(f'  Avg iterations:          {avg_iter:8.2f}')
    print(f'  Success rate:            {success_rate:7.1f}% ({successes}/{n_pois})')
    print(f"  {paper_metrics['metric_name']} (mean): {fmt(paper_metrics['metric_mean_px'])} px")
    print(f"  Metric POIs (converged): {paper_metrics['metric_point_count']}")
    if succ_iters is not None:
        print(f'  Avg iterations (success): {succ_iters:.2f}')
    if paper_metrics['gt_files']:
        print('  GT: ' + ', '.join(paper_metrics['gt_files']))
    else:
        print('  No GT detected; reporting IC-GN correction.')
    print('  Timing/IDD subset requires actual convergence; existing success-rate rules unchanged.')
    print('=' * 55 + '\n')
    if table_directory is not None:
        save_summary_table(table_directory, table_method, success_rate, succ_iters, init_time, paper_metrics)

def save_summary_table(directory, method, csr, ain, init_seconds, metrics):
    aliases = {'TMF-ICGN': 'TFM-ICGN', 'GMGN': 'TFM-ICGN', 'GM-GN': 'TFM-ICGN', 'FFT': 'FFT-ICGN', 'SIFT': 'SIFT-ICGN', 'Ncorr': 'RG-ICGN'}
    method = aliases.get(method, method)
    methods = ('TFM-ICGN', 'FFT-ICGN', 'SIFT-ICGN', 'RG-ICGN')
    if method not in methods:
        raise ValueError(f'Unknown table method: {method}')
    metric_header = 'IDD' if metrics['metric_name'] == 'IDD' else 'IC-GN 修正量'
    headers = ['方法', 'CSR', metric_header, 'AIN', '初估时间', 'IC-GN 精化时间']
    target = Path(directory) / '实验指标汇总.csv'
    rows = {name: [name, '', '', '', '', ''] for name in methods}
    if target.exists():
        with target.open(encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f)
            previous_header = next(reader, [])
            if previous_header != headers:
                raise ValueError(f'{target}: 表头与本次指标不同，不能混合IDD和修正量；请移走旧表后重新生成。')
            for row in reader:
                if row:
                    row[0] = aliases.get(row[0], row[0])
                if len(row) != 6 or row[0] not in rows:
                    raise ValueError(f'{target}: 非预期的表格行')
                rows[row[0]] = row

    def fmt(value, digits, unit=''):
        return f'{value:.{digits}f}{unit}' if value is not None and np.isfinite(value) else 'N/A'
    rows[method] = [method, fmt(csr, 1, '%'), fmt(metrics['metric_mean_px'], 4, ' px'), fmt(ain, 3), fmt(init_seconds, 4, ' s'), fmt(metrics['icgn_refinement_ms_per_success'], 4, ' ms/POI')]
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8-sig', dir=target.parent, prefix='.summary_', suffix='.tmp', delete=False) as f:
            temporary = Path(f.name)
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows((rows[name] for name in methods))
        os.replace(temporary, target)
    except PermissionError as exc:
        raise PermissionError(f'请关闭Excel中打开的 {target.name} 后再保存。原表未覆盖。') from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    print(f'  实验指标汇总: {target}')
    return target
