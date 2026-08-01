"""Track0+보정 보고서용 비교 오버레이 — 같은 프레임에서 q 선택 vs 보정 p 선택.

세션분할 가중치(calib_v1_sessionsplit: UAM 0716·17 미학습) 사용 —
시험 세션 케이스는 out-of-domain 렌더링이다.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from run_uam_log import (DATA, K, MESH, S1_CFG, load_frame, verify_candidate,  # noqa: E402
                         planarity_ratio, PLANARITY_MIN)
from harness import onboard_mesh  # noqa: E402

from shape6d.common.types import Candidate, Proposal  # noqa: E402
from shape6d.identify.size_gate import SizeGate  # noqa: E402
from shape6d.proposal.prompt_gen import LidarPromptGenerator  # noqa: E402
from shape6d.verify.symmetry_eval import SymmetryHandler  # noqa: E402

ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'assets_09')
CALIB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'calib_data',
                     'calib_v1_sessionsplit.npz')

SAMPLES = [
    ('robot_log', '20260716_151300_810_sam6d'),
    ('robot_log', '20260717_104956_828_sam6d'),
    ('robot_log', '20260717_154115_026_sam6d'),
    ('robot_log', '20260716_133531_534_sam6d'),
    ('robot_log_essential', '20260710_182410_897_sam6d'),
    ('robot_log_essential', '20260710_192818_093_sam6d'),
]


def overlay_compare(rgb, fb, onb, res_q, res_p, p_q, p_p, ref_t, path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for f in font_manager.findSystemFonts(fontpaths=['/usr/share/fonts/opentype/noto']):
        if 'NotoSansCJK-Regular' in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams['font.family'] = 'Noto Sans CJK JP'
            break
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.imshow(rgb)
    P = fb.lidar_points[::4]
    ax.scatter(K.fx * P[:, 0] / P[:, 2] + K.cx, K.fy * P[:, 1] / P[:, 2] + K.cy,
               s=1.0, c='#4a90d9', alpha=0.3, linewidths=0, label='LiDAR 관측')
    for res, color, name, pv in [(res_q, '#1baf7a', '기존 q 선택', p_q),
                                 (res_p, '#6d28d9', '보정 p 선택', p_p)]:
        if res is None:
            continue
        M = onb['master'][::5] @ res.pose[:3, :3].T + res.pose[:3, 3]
        ax.scatter(K.fx * M[:, 0] / M[:, 2] + K.cx, K.fy * M[:, 1] / M[:, 2] + K.cy,
                   s=0.8, c=color, alpha=0.65, linewidths=0,
                   label=f'{name} (p={pv:.2f})')
    if ref_t is not None:
        ax.scatter([K.fx * ref_t[0] / ref_t[2] + K.cx], [K.fy * ref_t[1] / ref_t[2] + K.cy],
                   marker='x', s=120, c='#d33', linewidths=2.5, label='SAM-6D 로그 위치')
    ax.legend(loc='upper right', fontsize=9, markerscale=8, framealpha=0.85)
    ax.set_xlim(0, K.width)
    ax.set_ylim(K.height, 0)
    ax.axis('off')
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    z = np.load(CALIB)
    w, b = z['w'], float(z['b'])
    mesh = trimesh.load(MESH, force='mesh')
    onb = onboard_mesh(mesh)
    sym_h = SymmetryHandler(onb['sym'].sym_rots, onb['sym'].sym_axes)

    caps = {}
    for folder, name in SAMPLES:
        rgb, fb = load_frame(f'{DATA}/{folder}/{name}')
        est = json.load(open(f'{DATA}/{folder}/{name}/estimate.json'))
        ref_t = None
        if est.get('num_detections', 0) > 0:
            p = est['sam6d_pose_camera']
            ref_t = np.array([p['x'], p['y'], p['z']])

        gen = LidarPromptGenerator(**S1_CFG)
        _, clusters = gen(fb)
        cands = [Candidate(
            proposal=Proposal(mask=None, bbox=np.zeros(4), score=.5, source='lidar_hull',
                              cluster_id=c.id, lidar_idx=c.point_indices,
                              n_lidar=len(c.point_indices)),
            pts=fb.lidar_points[c.point_indices], uv=fb.lidar_pixels[c.point_indices])
            for c in clusters]
        gated = [c for c in SizeGate()(cands, onb['diam'])
                 if planarity_ratio(c.pts) >= PLANARITY_MIN]
        gated = sorted(gated, key=lambda c: -c.pts.shape[0])[:5]
        evals = []
        for cand in gated:
            res, s_depth = verify_candidate(onb, fb, cand, sym_h)
            st = res.diag['stats']
            q = st['inlier_ratio'] * st['coverage'] - st['free_viol']
            pn = float(1 / (1 + np.exp(-(w @ np.array(res.diag['features']) + b))))
            evals.append((res, q, pn, s_depth))
        ok_q = [e for e in evals if e[3] >= 0.25]
        res_q, q_q, p_q, _ = max(ok_q or evals, key=lambda e: e[1])
        res_p, q_p, p_p, _ = max(evals, key=lambda e: e[2])

        def dt(res):
            return (float(np.linalg.norm(res.pose[:3, 3] - ref_t)) if ref_t is not None
                    else None)
        cap = dict(dt_q=dt(res_q), dt_p=dt(res_p), p_q=p_q, p_p=p_p,
                   t_p=[round(float(v), 2) for v in res_p.pose[:3, 3]])
        caps[f'{folder}__{name}'] = cap
        title = (f'q 선택 dt={cap["dt_q"]:.2f}m / 보정 p 선택 dt={cap["dt_p"]:.2f}m'
                 if ref_t is not None else '참조 없음 (essential)')
        overlay_compare(rgb, fb, onb, res_q, res_p, p_q, p_p, ref_t,
                        os.path.join(ASSETS, f'sel_{folder}__{name}.jpg'), title)
        print(name, cap, flush=True)
    json.dump(caps, open(os.path.join(ASSETS, 'sel_captions.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
