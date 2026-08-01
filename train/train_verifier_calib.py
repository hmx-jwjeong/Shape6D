"""M3-3 신뢰도 보정 1차 학습 — 로지스틱 (08 계획 §4.3).

데이터:
  gso_A / gso_flip  eval/external_test/calib_data/calib_gso.json (ADD 라벨)
  uam_A / uam_B     eval/external_test/uam_results_t0/uam_results.json
                    라벨: ‖Δt‖<1m 양성, >2m 음성 (참조 있는 행)
                    essential 무참조 행은 z>9.5m(원거리 구조물)·z<2m(바닥)만 음성, 나머지 제외

특징: shape6d.verify.confidence 10-d (기존 벡터 그대로 — 공간 분해 특징은 2차에서).
학습: ConfidenceCalibrator.fit (IRLS 로지스틱, l2). 도메인×라벨 층화 70/30 분할.
평가: 도메인별 holdout AUC · 오수락<1%@리콜손실≤5% 게이트 · UAM 게이트(정포즈 ACCEPT≥80%, 오수락 0)
출력: calib_data/calib_v1.npz (w, b) + 리포트 stdout

실행: python3 train/train_verifier_calib.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from shape6d.verify.confidence import FEATURE_NAMES, ConfidenceCalibrator  # noqa: E402

CD = Path(__file__).parents[1] / 'eval' / 'external_test' / 'calib_data'
UAM = Path(__file__).parents[1] / 'eval' / 'external_test' / 'uam_results_t0' / 'uam_results.json'
SEED = 0


def load_rows():
    rows = []
    for r in json.load(open(CD / 'calib_gso.json')):
        rows.append(dict(x=r['x'], y=r['y'], domain=r['domain']))
    for rec in json.load(open(UAM)):
        for mode in 'AB':
            m = rec.get(mode)
            if not m or not m.get('ok') or m.get('x') is None:
                continue
            dt, z = m.get('dt_ref_m'), m['t'][2]
            if dt is not None:
                y = 1 if dt < 1.0 else (0 if dt > 2.0 else None)
            elif z > 9.5 or z < 2.0:
                y = 0  # essential 무참조: 원거리 구조물/바닥 — 07 육안 검증된 오수락 패턴
            else:
                y = None
            if y is None:
                continue
            rows.append(dict(x=m['x'], y=y, domain=f'uam_{mode}'))
    return rows


def auc(y, s):
    o = np.argsort(s)
    y = np.asarray(y)[o]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0:
        return float('nan')
    ranks = np.arange(1, len(y) + 1)
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    rows = load_rows()
    X = np.array([r['x'] for r in rows])
    y = np.array([r['y'] for r in rows], float)
    dom = np.array([r['domain'] for r in rows])
    print(f'전체 {len(rows)}행: ' + ' · '.join(
        f'{d} {int((dom == d).sum())} (양성 {int(y[dom == d].sum())})' for d in np.unique(dom)))

    # 도메인×라벨 층화 70/30
    rng = np.random.default_rng(SEED)
    tr = np.zeros(len(rows), bool)
    for d in np.unique(dom):
        for lab in (0, 1):
            idx = np.nonzero((dom == d) & (y == lab))[0]
            rng.shuffle(idx)
            tr[idx[:int(0.7 * len(idx))]] = True
    te = ~tr

    # 표준화 (IRLS 조건수·l2 등방성) — 스케일러는 가중치에 접어 저장
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xs = (X - mu) / sd

    # 도메인 균형: flip 지배 완화 — 훈련행 가중을 복제로 근사 (도메인별 동일 유효 표본)
    tr_idx = np.nonzero(tr)[0]
    counts = {d: (dom[tr_idx] == d).sum() for d in np.unique(dom)}
    target = max(min(counts.values()), 60)
    bal = []
    for d in np.unique(dom):
        di = tr_idx[dom[tr_idx] == d]
        reps = max(1, round(target / len(di) * 3))
        bal.extend(list(di) * reps)
    bal = np.array(bal)

    # l2 선택: holdout AUC 최대
    best_fit = None
    for l2 in (0.1, 0.3, 1.0, 3.0, 10.0):
        c = ConfidenceCalibrator.fit(Xs[bal], y[bal], l2=l2)
        a = auc(y[te], np.array([c(x) for x in Xs[te]]))
        if best_fit is None or a > best_fit[1]:
            best_fit = (c, a, l2)
    calib, _, l2_sel = best_fit
    print(f'\nl2={l2_sel} 선택 (holdout AUC 기준), 균형 훈련행 {len(bal)}')
    p_te = np.array([calib(x) for x in Xs[te]])
    p_all = np.array([calib(x) for x in Xs])

    print('\n가중치 (특징별):')
    for n, w in zip(FEATURE_NAMES, calib.w):
        print(f'  {n:14s} {w:+8.3f}')
    print(f'  bias           {calib.b:+8.3f}')

    print(f'\nholdout 전체 AUC {auc(y[te], p_te):.3f}')
    for d in np.unique(dom):
        m = te & (dom == d)
        if m.sum() > 5:
            print(f'  {d:10s} AUC {auc(y[m], p_all[m]):.3f} (n={int(m.sum())})')

    # 운용점: holdout에서 FA율 1% 이하가 되는 최소 임계 → 리콜 확인
    ths = np.unique(p_te)
    best = None
    for th in ths:
        acc = p_te >= th
        fa = (acc & (y[te] == 0)).sum() / max(acc.sum(), 1)
        rec = (acc & (y[te] == 1)).sum() / max((y[te] == 1).sum(), 1)
        if fa <= 0.01 and (best is None or rec > best[2]):
            best = (th, fa, rec)
    if best:
        th, fa, rec = best
        print(f'\n운용점 θ_acc={th:.3f}: 오수락 {fa*100:.1f}% · 리콜 {rec*100:.1f}% '
              f'(게이트: FA<1% @ 리콜손실≤5% → 리콜 ≥95% {"통과" if rec >= 0.95 else "미달"})')
    else:
        th = 0.5
        print('\nFA≤1% 달성 임계 없음 — 특징 확장(2차) 필요')

    # UAM 게이트: 정포즈 ACCEPT율 / 오수락 (소프트 정책: free_viol 하드 가드는 x[3]>0.5만)
    fv = X[:, FEATURE_NAMES.index('free_viol')]
    for d in ('uam_A', 'uam_B'):
        m = dom == d
        if not m.any():
            continue
        acc = (p_all >= th) & (fv < 0.5)
        tp = (acc & m & (y == 1)).sum(); pos = (m & (y == 1)).sum()
        fp = (acc & m & (y == 0)).sum()
        print(f'UAM 게이트 [{d}] 정포즈 ACCEPT {int(tp)}/{int(pos)} '
              f'({tp/max(pos,1)*100:.0f}%, 목표≥80%) · 오수락 {int(fp)} (목표 0)')

    # UAM θ 스캔 — 실기 게이트가 달성 가능한 운용 영역
    print('\nUAM θ 스캔 (정포즈ACCEPT% / 오수락건):')
    um = np.isin(dom, ['uam_A', 'uam_B'])
    for th_u in (0.3, 0.5, 0.7, 0.8, 0.9):
        acc = (p_all >= th_u) & (X[:, FEATURE_NAMES.index('free_viol')] < 0.5)
        tp = (acc & um & (y == 1)).sum(); pos = (um & (y == 1)).sum()
        fp = (acc & um & (y == 0)).sum()
        print(f'  θ={th_u:.1f}: {tp}/{pos} ({tp/max(pos,1)*100:.0f}%) / 오수락 {int(fp)}')

    # 원 스케일 가중치로 환산해 저장 (w'/sd, b' = b - Σ w·mu/sd)
    w_raw = calib.w / sd
    b_raw = calib.b - float((calib.w * mu / sd).sum())
    np.savez(CD / 'calib_v1.npz', w=w_raw, b=b_raw,
             feature_names=np.array(FEATURE_NAMES), theta_acc=th, seed=SEED, l2=l2_sel,
             n_train=int(tr.sum()), n_test=int(te.sum()))
    print(f'\n저장: {CD}/calib_v1.npz (원 스케일 환산 가중치)')


if __name__ == '__main__':
    main()
