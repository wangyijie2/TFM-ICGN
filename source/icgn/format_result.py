# SPDX-License-Identifier: Apache-2.0
# See LICENSE and NOTICE for license terms and retained notices.
# Copyright 2026 Wang Yijie, modifications. Modified for this release, 2026-09-11.

def print_results(method_name, init_time, icgn_time, n_pois, avg_iter, success_rate, successes, fails, delta_mean=None, delta_max=None, succ_time=None, succ_iters=None, n_succ=None):
    print()
    print('=' * 55)
    print(f'  {method_name}')
    print('-' * 55)
    print(f'  Initial guess time:       {init_time:>8.2f} s')
    print(f'  IC-GN refinement time:    {icgn_time:>8.2f} s  ({icgn_time / n_pois * 1000:.2f} ms/POI)')
    print(f'  Total time:               {init_time + icgn_time:>8.2f} s')
    print(f'  Avg iterations:           {avg_iter:>8.2f}')
    print(f'  Success rate:             {success_rate:>7.1f}%  ({successes}/{successes + fails})')
    if delta_mean is not None:
        print(f'  IC-GN correction (mean):  {delta_mean:>8.4f} px')
    if delta_max is not None:
        print(f'  IC-GN correction (max):   {delta_max:>8.4f} px')
    if succ_time is not None and n_succ and (n_succ > 0):
        print(f'  --- successful POIs only ({n_succ}) ---')
        print(f'  IC-GN time (success):     {succ_time:>8.2f} s  ({succ_time / n_succ * 1000:.2f} ms/POI)')
        print(f'  Avg iterations (success): {succ_iters:>8.2f}')
    print('=' * 55)
    print()
