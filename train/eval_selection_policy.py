"""후보 선택 정책 오프라인 평가 — q(inlier×cov−free) vs 보정 p (calib_v1).

입력: uam_results_t0c (모드 B 후보 전체의 특징·위치 기록)
평가: 참조 있는 샘플에서 각 정책이 고른 후보의 dt<1m 안착률 + ACCEPT 시 정/오.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CD = Path(__file__).parents[1] / 'eval' / 'external_test' / 'calib_data'
RES = Path(__file__).parents[1] / 'eval' / 'external_test' / 'uam_results_t0c' / 'uam_results.json'


def main():
    z = np.load(CD / 'calib_v1.npz')
    w, b, th = z['w'], float(z['b']), float(z['theta_acc'])
    R = json.load(open(RES))

    stats = {p: dict(near=0, n=0, acc_true=0, acc_false=0) for p in ('q', 'p_calib')}
    for rec in R:
        m = rec.get('B')
        if not m or not m.get('ok') or not m.get('cands'):
            continue
        ref = rec.get('ref_t')
        cands = m['cands']
        for c in cands:
            c['p_new'] = float(1 / (1 + np.exp(-(w @ np.array(c['x']) + b))))
        for pol, keyf in (('q', lambda c: c['q']), ('p_calib', lambda c: c['p_new'])):
            best = max(cands, key=keyf)
            if ref is None:
                # 무참조: z 휴리스틱 오수락만 집계 (θ=0.7 ACCEPT 기준)
                if best['p_new'] >= 0.7 and (best['t'][2] > 9.5 or best['t'][2] < 2):
                    stats[pol]['acc_false'] += 1
                continue
            dt = float(np.linalg.norm(np.array(best['t']) - np.array(ref)))
            s = stats[pol]
            s['n'] += 1
            if dt < 1.0:
                s['near'] += 1
            if best['p_new'] >= 0.7 if pol == 'p_calib' else best['verdict'] == 'ACCEPT':
                if dt < 1.0:
                    s['acc_true'] += 1
                elif dt > 2.0:
                    s['acc_false'] += 1
    for pol, s in stats.items():
        print(f'{pol:8s}: 안착 {s["near"]}/{s["n"]} · ACCEPT 정 {s["acc_true"]} / 오 {s["acc_false"]}')


if __name__ == '__main__':
    main()
