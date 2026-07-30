"""UAM 전시장 로봇 로그(robot_log / robot_log_essential)에서 v0-geo 학습 전 성능 평가.

데이터: /mnt/disk/UAM TEST DATA — *_sam6d 스냅샷 (rgb.png 1280×800 + depth.png
uint16 mm, depth는 Ruby128 LiDAR의 카메라 투영·충전율 ~10%). 카메라 K는 본 저장소
스크래치 캘리브레이션(LiDAR↔depth 정합, inlier 96.5%, 잔차 중앙 4.2mm)으로 추정.

모드 B: 풀 파이프라인 (S1 장면 튜닝: ransac_dist 0.03 / max_planes 3 / voxel 0.15)
모드 A: ROI 보조 — 로봇이 기록한 SAM-6D 위치 주변 반경 crop으로 S1 우회 (S2·coarse·S4 절연)

참조: 로그된 SAM-6D 포즈(카메라계)와의 위치 편차. GT 아님 — 일치도 지표.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from harness import onboard_mesh, frame_from_points  # noqa: E402

from shape6d.common.frame_bundle import CameraIntrinsics  # noqa: E402
from shape6d.common.types import Candidate, Proposal  # noqa: E402
from shape6d.identify.depth_match import PointToTemplateMatcher  # noqa: E402
from shape6d.identify.size_gate import SizeGate  # noqa: E402
from shape6d.pose.template_init import coarse_poses_from_match  # noqa: E402
from shape6d.proposal.prompt_gen import LidarPromptGenerator  # noqa: E402
from shape6d.verify.symmetry_eval import SymmetryHandler  # noqa: E402
from shape6d.verify.verifier import Verifier  # noqa: E402

DATA = '/mnt/disk/UAM TEST DATA'
OUT = os.path.join(os.path.dirname(__file__), 'uam_results')
MESH = os.path.expanduser('~/Downloads/SKID_Scalled_Upright_Widnshield.obj')

# LiDAR↔depth.png 정합 캘리브레이션 결과 (2026-07-22)
K = CameraIntrinsics(fx=608.581, fy=608.858, cx=647.712, cy=407.744, width=1280, height=800)

SIGMA = 0.008
ZMIN, ZMAX = 0.3, 12.0
S1_CFG = dict(voxel=0.15, ransac_dist=0.03, max_planes=3, min_cluster_pts=30)

# Track 0 (08 계획 §2): T0-1 평면성 게이트 / T0-2 FOV 커버리지
# 실측 근거: 기체 λ3/λ1 0.24~0.33 vs 평면 클러터(게시판·부스·펜스) 0.0001~0.011
PLANARITY_MIN = 0.02   # λ3/λ1 < 이 값 → 부피 없는 평면으로 배제
FOV_FRAC_MIN = 0.6     # 모델 투영의 화면 내 비율 < 이 값 → ACCEPT 강등


def planarity_ratio(P: np.ndarray) -> float:
    Pc = P.astype(np.float64) - P.mean(0)
    ev = np.linalg.eigvalsh(Pc.T @ Pc / len(P))
    return float(ev[0] / max(ev[2], 1e-12))


def fov_fraction(pose: np.ndarray, master: np.ndarray) -> float:
    M = master @ pose[:3, :3].T + pose[:3, 3]
    ok = M[:, 2] > 1e-6
    if not ok.any():
        return 0.0
    u = K.fx * M[ok, 0] / M[ok, 2] + K.cx
    v = K.fy * M[ok, 1] / M[ok, 2] + K.cy
    inside = (u >= 0) & (u < K.width) & (v >= 0) & (v < K.height)
    return float(inside.sum() / len(M))


def load_frame(sample: str):
    rgb = np.array(Image.open(f'{sample}/rgb.png'))[:, :, :3]
    depth = np.array(Image.open(f'{sample}/depth.png')).astype(np.float64) / 1000.0
    vs, us = np.nonzero((depth > ZMIN) & (depth < ZMAX))
    z = depth[vs, us]
    pts = np.stack([(us - K.cx) * z / K.fx, (vs - K.cy) * z / K.fy, z], 1).astype(np.float32)
    return rgb, frame_from_points(rgb, pts, K)


def frame_obs_of(fb):
    vs, us = np.nonzero(fb.valid_mask)
    step = max(1, len(vs) // 20000)
    return (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
            fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))


def verify_candidate(onb, fb, cand, sym_h, k=5):
    matcher = PointToTemplateMatcher(onb['tpl']['tdf'], onb['tpl']['tpl_center'],
                                     onb['diam'], top_views_pass2=k)
    m = matcher.match(cand.pts, k=k)
    cand.scores['depth'] = m.s_depth
    hyps = coarse_poses_from_match(m, onb['tpl']['tpl_pose'], onb['tpl']['tpl_center'])
    ver = Verifier(K, sym_h, sigma_lidar=SIGMA)
    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb['master'],
              onb['master_n'], onb['X_verify'], onb['diam'],
              s2_scores=cand.scores, frame_obs=frame_obs_of(fb))
    return res, float(m.s_depth)


def run_mode_b(onb, fb, sym_h, max_candidates=5):
    gen = LidarPromptGenerator(**S1_CFG)
    _, clusters = gen(fb)
    if not clusters:
        return None, {'reason': 'no_clusters'}
    cands = [Candidate(
        proposal=Proposal(mask=None, bbox=np.zeros(4), score=.5, source='lidar_hull',
                          cluster_id=c.id, lidar_idx=c.point_indices,
                          n_lidar=len(c.point_indices)),
        pts=fb.lidar_points[c.point_indices], uv=fb.lidar_pixels[c.point_indices])
        for c in clusters]
    gated = SizeGate()(cands, onb['diam'])
    if not gated:
        return None, {'reason': 'gated_out', 'n_clusters': len(clusters)}
    n_before = len(gated)
    gated = [c for c in gated if planarity_ratio(c.pts) >= PLANARITY_MIN]  # T0-1
    if not gated:
        return None, {'reason': 'all_planar', 'n_gated_presize': n_before}
    gated = sorted(gated, key=lambda c: -c.pts.shape[0])[:max_candidates]

    best, best_diag, cand_rows = None, None, []
    for cand in gated:
        res, s_depth = verify_candidate(onb, fb, cand, sym_h)
        st = res.diag['stats']
        q = st['inlier_ratio'] * st['coverage'] - st['free_viol']
        # 후보 전체 기록 — Track 2 보정 학습·선택 정책 오프라인 평가용
        cand_rows.append({'x': [float(v) for v in res.diag['features']],
                          't': [float(v) for v in res.pose[:3, 3]],
                          'q': float(q), 's_depth': float(s_depth),
                          'verdict': res.verdict, 'p_conf': float(res.p_conf),
                          'n_pts': int(cand.pts.shape[0])})
        if s_depth < 0.25:
            continue
        if best is None or q > best_diag['quality']:
            best = res
            best_diag = {'quality': q, 's_depth': s_depth,
                         'n_pts': int(cand.pts.shape[0]), 'n_gated': len(gated)}
    if best is None:
        return None, {'reason': 'no_match', 'n_gated': len(gated), 'cands': cand_rows}
    best_diag['cands'] = cand_rows
    return best, best_diag


def run_mode_a(onb, fb, sym_h, center: np.ndarray, radius: float):
    d = np.linalg.norm(fb.lidar_points - center[None], axis=1)
    idx = np.nonzero(d < radius)[0]
    if len(idx) < 30:
        return None, {'reason': 'roi_empty', 'n_roi': int(len(idx))}
    cand = Candidate(
        proposal=Proposal(mask=None, bbox=np.zeros(4), score=1.0, source='roi',
                          lidar_idx=idx, n_lidar=len(idx)),
        pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx])
    res, s_depth = verify_candidate(onb, fb, cand, sym_h)
    return res, {'s_depth': s_depth, 'n_pts': int(len(idx))}


def sam6d_ref(sample: str):
    try:
        est = json.load(open(f'{sample}/estimate.json'))
    except Exception:
        return None, None
    if est.get('num_detections', 0) > 0:
        p = est['sam6d_pose_camera']
        return np.array([p['x'], p['y'], p['z']]), est
    return None, est


def result_row(res, diag, ref_t, master=None):
    if res is None:
        return {'ok': False, **diag}
    st = res.diag['stats']
    t = res.pose[:3, 3]
    fov = fov_fraction(res.pose, master) if master is not None else None
    # T0-2: 시야 밖 모델은 커버리지가 감지 못함 → ACCEPT 강등 (원판정은 verdict_raw로 보존)
    verdict = res.verdict
    if fov is not None and fov < FOV_FRAC_MIN and verdict == 'ACCEPT':
        verdict = 'REJECT'
    x = res.diag.get('features')
    row = {'ok': True, 'verdict': verdict, 'verdict_raw': res.verdict,
           'fov_frac': fov, 'p_conf': float(res.p_conf),
           'x': None if x is None else [float(v) for v in x],
           'inlier_ratio': float(st['inlier_ratio']), 'coverage': float(st['coverage']),
           'free_viol': float(st['free_viol']),
           's_depth': diag.get('s_depth'), 'n_pts': diag.get('n_pts'),
           't': [float(x) for x in t], 'pose': [[float(x) for x in r] for r in res.pose]}
    if 'cands' in diag:
        row['cands'] = diag['cands']
    if ref_t is not None:
        row['dt_ref_m'] = float(np.linalg.norm(t - ref_t))
    return row


def overlay(rgb, fb, onb, resA, resB, ref_t, path):
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
               s=1.0, c='#4a90d9', alpha=0.35, linewidths=0, label='LiDAR 관측')
    for res, color, name in [(resB, '#1baf7a', '모드 B(풀)'), (resA, '#e8873a', '모드 A(ROI)')]:
        if res is None:
            continue
        M = onb['master'][::5] @ res.pose[:3, :3].T + res.pose[:3, 3]
        ax.scatter(K.fx * M[:, 0] / M[:, 2] + K.cx, K.fy * M[:, 1] / M[:, 2] + K.cy,
                   s=0.8, c=color, alpha=0.6, linewidths=0,
                   label=f'{name} {res.verdict} p={res.p_conf:.2f}')
    if ref_t is not None:
        ax.scatter([K.fx * ref_t[0] / ref_t[2] + K.cx], [K.fy * ref_t[1] / ref_t[2] + K.cy],
                   marker='x', s=120, c='#d33', linewidths=2.5, label='SAM-6D 로그 위치')
    ax.legend(loc='upper right', fontsize=9, markerscale=8, framealpha=0.85)
    ax.set_xlim(0, K.width)
    ax.set_ylim(K.height, 0)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)


def pick_samples(folder: str, n: int):
    dirs = sorted(glob.glob(f'{DATA}/{folder}/*_sam6d'))
    dirs = [d for d in dirs if os.path.exists(f'{d}/rgb.png') and os.path.exists(f'{d}/depth.png')]
    if n >= len(dirs):
        return dirs
    idx = np.linspace(0, len(dirs) - 1, n).round().astype(int)
    return [dirs[i] for i in sorted(set(idx))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30, help='폴더당 샘플 수 (시간순 균등)')
    ap.add_argument('--out', default=OUT, help='결과 디렉토리 (07 정본 보존용)')
    ap.add_argument('--overlay-all', action='store_true')
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    t0 = time.time()
    mesh = trimesh.load(MESH, force='mesh')
    onb = onboard_mesh(mesh)
    sym_h = SymmetryHandler(onb['sym'].sym_rots, onb['sym'].sym_axes)
    print(f'onboard {time.time()-t0:.1f}s diam={onb["diam"]:.3f} '
          f'sym_rots={len(onb["sym"].sym_rots)}', flush=True)

    results = []
    for folder in ['robot_log', 'robot_log_essential']:
        samples = pick_samples(folder, args.n)
        print(f'== {folder}: {len(samples)} samples', flush=True)
        for i, s in enumerate(samples):
            name = os.path.basename(s)
            rec = {'folder': folder, 'sample': name}
            try:
                rgb, fb = load_frame(s)
                rec['n_scene_pts'] = int(len(fb.lidar_points))
                ref_t, est = sam6d_ref(s)
                rec['ref_t'] = None if ref_t is None else [float(x) for x in ref_t]
                rec['sam6d_scores'] = est.get('scores') if est else None

                t1 = time.time()
                resB, diagB = run_mode_b(onb, fb, sym_h)
                rec['B_ms'] = (time.time() - t1) * 1e3
                rec['B'] = result_row(resB, diagB, ref_t, onb['master'])

                resA = None
                if ref_t is not None:
                    t1 = time.time()
                    resA, diagA = run_mode_a(onb, fb, sym_h, ref_t, radius=0.75 * onb['diam'])
                    rec['A_ms'] = (time.time() - t1) * 1e3
                    rec['A'] = result_row(resA, diagA, ref_t, onb['master'])

                overlay(rgb, fb, onb, resA, resB, ref_t,
                        f'{out_dir}/{folder}__{name}.jpg')
            except Exception as e:
                rec['error'] = repr(e)
            results.append(rec)
            bt = rec.get('B', {})
            print(f'[{i+1}/{len(samples)}] {name} '
                  f'B={bt.get("verdict", bt.get("reason", "ERR"))} '
                  f'dtB={bt.get("dt_ref_m", float("nan")):.2f} '
                  f'A={rec.get("A", {}).get("verdict", "-")} '
                  f'dtA={rec.get("A", {}).get("dt_ref_m", float("nan")):.2f} '
                  f'({rec.get("B_ms", 0)/1e3:.0f}s)', flush=True)

    json.dump(results, open(f'{out_dir}/uam_results.json', 'w'), indent=1, ensure_ascii=False)
    print(f'done {len(results)} samples, total {time.time()-t0:.0f}s -> {out_dir}/uam_results.json')


if __name__ == '__main__':
    main()
