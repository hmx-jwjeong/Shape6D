"""uam_results.json 집계 + 보고서 그림 생성 (docs/assets_07)."""
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'uam_results')
ASSETS = os.path.join(HERE, '..', '..', 'docs', 'assets_07')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in font_manager.findSystemFonts(fontpaths=['/usr/share/fonts/opentype/noto']):
    if 'NotoSansCJK-Regular' in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
        break

C_A, C_B = '#e8873a', '#1baf7a'
INK, SUB, LINE = '#1a1d23', '#5a6270', '#dde2ea'


def agg(results):
    rows = {'A': [], 'B': []}
    for r in results:
        for m in 'AB':
            if m in r and isinstance(r[m], dict):
                rows[m].append({**r[m], 'folder': r['folder'], 'sample': r['sample'],
                                'ms': r.get(f'{m}_ms')})
    stats = {}
    for m in 'AB':
        R = rows[m]
        ok = [x for x in R if x.get('ok')]
        verd = {}
        for x in R:
            v = x.get('verdict', x.get('reason', 'error'))
            verd[v] = verd.get(v, 0) + 1
        dts = np.array([x['dt_ref_m'] for x in ok if x.get('dt_ref_m') is not None])
        stats[m] = dict(
            n=len(R), n_ok=len(ok), verdicts=verd,
            dt=dict(n=len(dts),
                    near1=int((dts < 1.0).sum()), near2=int((dts < 2.0).sum()),
                    med=float(np.median(dts)) if len(dts) else None,
                    p90=float(np.percentile(dts, 90)) if len(dts) else None),
            free_med=float(np.median([x['free_viol'] for x in ok])) if ok else None,
            inl_med=float(np.median([x['inlier_ratio'] for x in ok])) if ok else None,
            cov_med=float(np.median([x['coverage'] for x in ok])) if ok else None,
            guard_rej=sum(1 for x in ok if x.get('verdict') == 'REJECT'
                          and x['free_viol'] > 0.15),
            ms_med=float(np.median([x['ms'] for x in ok if x.get('ms')])) if ok else None,
        )
    return rows, stats


def fig_dt_hist(rows, path):
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    bins = np.arange(0, 8.5, 0.5)
    for m, c, h, lbl in [('B', C_B, '//', '모드 B (풀 파이프라인)'),
                         ('A', C_A, None, '모드 A (ROI 보조)')]:
        d = [x['dt_ref_m'] for x in rows[m] if x.get('ok') and x.get('dt_ref_m') is not None]
        ax.hist(d, bins=bins, histtype='step' if m == 'B' else 'bar',
                facecolor=c if m == 'A' else 'none', edgecolor=c, hatch=h,
                linewidth=2, alpha=0.85 if m == 'A' else 1.0, label=f'{lbl} (n={len(d)})')
    ax.set_xlabel('로그된 SAM-6D 위치와의 편차 ‖Δt‖ (m)', fontsize=10.5)
    ax.set_ylabel('샘플 수', fontsize=10.5)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3, linewidth=0.5, axis='y')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def fig_scatter(rows, path):
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    for m, c, mk, lbl in [('B', C_B, 'o', '모드 B'), ('A', C_A, '^', '모드 A')]:
        ok = [x for x in rows[m] if x.get('ok')]
        ax.scatter([x['inlier_ratio'] * x['coverage'] for x in ok],
                   [x['free_viol'] for x in ok], s=34, c=c, marker=mk,
                   alpha=0.75, edgecolors='white', linewidths=0.7, label=lbl)
    ax.axhline(0.15, color='#b4232a', linewidth=1.4, linestyle='--')
    ax.annotate('free-space 하드 가드 (0.15) — 위는 무조건 REJECT', xy=(0.99, 0.157),
                xycoords=('axes fraction', 'data'), ha='right', fontsize=9.5, color='#b4232a')
    ax.set_xlabel('정합 품질 (inlier비 × 커버리지)', fontsize=10.5)
    ax.set_ylabel('free-space 위반율', fontsize=10.5)
    ax.legend(fontsize=10, framealpha=0.9, loc='upper left')
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    results = json.load(open(os.path.join(OUT, 'uam_results.json')))
    rows, stats = agg(results)
    os.makedirs(ASSETS, exist_ok=True)
    fig_dt_hist(rows, os.path.join(ASSETS, 'dt_hist.png'))
    fig_scatter(rows, os.path.join(ASSETS, 'quality_scatter.png'))
    print(json.dumps(stats, indent=1, ensure_ascii=False))

    # 대표 케이스 이미지 → assets 복사 (인자로 샘플명 나열 시 그것만)
    names = sys.argv[1:]
    for n in names:
        src = os.path.join(OUT, n + '.jpg')
        if os.path.exists(src):
            shutil.copy(src, os.path.join(ASSETS, 'case_' + n + '.jpg'))
            print('copied', n)


if __name__ == '__main__':
    main()
