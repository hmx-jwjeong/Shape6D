"""M3-3 보정 학습 데이터 생성 — GSO 풀셋에서 (특징 x, 라벨 y, 도메인) 수집.

도메인 (08 계획 §4.2):
  gso_A     GT 마스크 보조 정합 결과 — 정/오는 sym-aware ADD<0.1D로 라벨 (자연 발생 플립 포함)
  gso_flip  의도적 오포즈: GT에 주축 180°/90° 회전 적용 후 ICP 수렴 → 대부분 음성 라벨
            (수렴이 정포즈로 돌아오면 add 기준으로 양성 처리 — 라벨은 항상 ADD가 결정)

특징 = shape6d.verify.confidence.make_features 10-d (VerifyResult.diag['features'] 그대로).
출력: calib_gso.json — [{x, y, add_rel, domain, verdict, p_conf, key, obj_id, ...}]

실행: python3 eval/external_test/gen_calib_set.py --shards 2 --per-shard 60
"""
from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2]))
from harness import frame_from_points, onboard_mesh, sparsify_fixed_grid  # noqa: E402

from shape6d.common.frame_bundle import CameraIntrinsics  # noqa: E402
from shape6d.common.types import Candidate, PoseHypothesis, Proposal  # noqa: E402
from shape6d.identify.depth_match import PointToTemplateMatcher  # noqa: E402
from shape6d.pose.template_init import coarse_poses_from_match  # noqa: E402
from shape6d.verify.symmetry_eval import SymmetryHandler  # noqa: E402
from shape6d.verify.verifier import Verifier  # noqa: E402

DSET = Path('/mnt/samsung2tb/datasets/megapose')
SHARDS = DSET / 'MegaPose-GSO-fixed'
GSO_RAW = DSET / 'gso_raw'
OUT = Path(__file__).parent / 'calib_data'
SIGMA = 0.008
FUEL = 'https://fuel.gazebosim.org/1.0/GoogleResearch/models/{}.zip'


def rle_to_mask(rle) -> np.ndarray:
    arr = np.zeros(int(np.prod(rle['size'])), dtype=bool)
    v, pos = False, 0
    for run in rle['counts']:
        if v:
            arr[pos:pos + run] = True
        pos += run
        v = not v
    return arr.reshape(rle['size'], order='F')


def fetch_mesh(gso_id: str) -> Path | None:
    d = GSO_RAW / gso_id
    obj = d / 'meshes' / 'model.obj'
    if obj.exists():
        return obj
    d.mkdir(parents=True, exist_ok=True)
    try:
        buf = urllib.request.urlopen(FUEL.format(gso_id), timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(buf)) as z:
            for n in z.namelist():
                if n.endswith('meshes/model.obj') or n.endswith('meshes/model.mtl'):
                    (d / 'meshes').mkdir(exist_ok=True)
                    (d / 'meshes' / os.path.basename(n)).write_bytes(z.read(n))
        return obj if obj.exists() else None
    except Exception as e:
        print(f'[mesh fail] {gso_id}: {e}', flush=True)
        return None


def preprocess_mesh(obj_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_path, force='mesh')
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    s = 0.2 / (mesh.bounds[1] - mesh.bounds[0]).max()   # 실증 규약 (06)
    mesh.apply_translation(-c)
    mesh.apply_scale(s)
    return mesh


def flip_hyps(R_gt: np.ndarray, t_gt: np.ndarray) -> list[PoseHypothesis]:
    """의도적 오포즈: 물체 주축 180° 2종 + 90° 1종."""
    def rot(axis, deg):
        a = np.deg2rad(deg)
        c, s = np.cos(a), np.sin(a)
        M = {'x': [[1, 0, 0], [0, c, -s], [0, s, c]],
             'y': [[c, 0, s], [0, 1, 0], [-s, 0, c]],
             'z': [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
        return np.array(M)
    out = []
    for axis, deg in [('x', 180), ('y', 180), ('z', 90)]:
        out.append(PoseHypothesis(R=(R_gt @ rot(axis, deg)).astype(np.float64),
                                  t=t_gt.astype(np.float64), score=0.5, refined=False))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=2)
    ap.add_argument('--per-shard', type=int, default=60)
    ap.add_argument('--visib-min', type=float, default=0.75)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    gso = {e['obj_id']: e['gso_id'] for e in
           json.load(open(DSET / 'MegaPose-GSO' / 'gso_models.json'))}

    rows, t_start = [], time.time()
    for si in range(args.shards):
        sp = SHARDS / f'shard-{si:06d}.tar'
        if not sp.exists():
            continue
        t = tarfile.open(sp)
        keys = sorted({m.name.split('.')[0] for m in t.getmembers()})
        n_done = 0
        onb_cache: dict[int, dict | None] = {}
        for key in keys:
            if n_done >= args.per_shard:
                break
            try:
                gts = json.load(t.extractfile(f'{key}.gt.json'))
                gis = json.load(t.extractfile(f'{key}.gt_info.json'))
                cam = json.load(t.extractfile(f'{key}.camera.json'))
                cands_i = [i for i, gi in enumerate(gis)
                           if gi['px_count_all'] > 0
                           and gi['px_count_visib'] / gi['px_count_all'] >= args.visib_min
                           and gi['px_count_visib'] > 1500]
                if not cands_i:
                    continue
                gi_idx = max(cands_i, key=lambda i: gis[i]['px_count_visib'])
                obj_id = gts[gi_idx]['obj_id']

                if obj_id not in onb_cache:
                    obj = fetch_mesh(gso[obj_id])
                    onb_cache[obj_id] = None if obj is None else onboard_mesh(preprocess_mesh(obj))
                onb = onb_cache[obj_id]
                if onb is None:
                    continue

                Km = np.array(cam['cam_K']).reshape(3, 3)
                depth = np.array(Image.open(io.BytesIO(
                    t.extractfile(f'{key}.depth.png').read()))).astype(np.float32) \
                    * cam['depth_scale'] / 1000.0
                rgb = np.array(Image.open(io.BytesIO(t.extractfile(f'{key}.rgb.jpg').read())))
                H, W = depth.shape
                K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2],
                                     width=W, height=H)
                R_gt = np.array(gts[gi_idx]['cam_R_m2c']).reshape(3, 3)
                t_gt = np.array(gts[gi_idx]['cam_t_m2c']) / 1000.0
                mask = rle_to_mask(json.load(t.extractfile(f'{key}.mask_visib.json'))[str(gi_idx)])

                pts = sparsify_fixed_grid(depth, K)
                fb = frame_from_points(rgb[:, :, :3], pts, K)
                idx = fb.object_points(mask, erosion_px=2)
                if len(idx) < 40:
                    continue
                sym_h = SymmetryHandler(onb['sym'].sym_rots, onb['sym'].sym_axes)
                cand = Candidate(
                    proposal=Proposal(mask=None, bbox=np.zeros(4), score=1.0, source='gt_mask',
                                      lidar_idx=idx, n_lidar=len(idx)),
                    pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx])
                matcher = PointToTemplateMatcher(onb['tpl']['tdf'], onb['tpl']['tpl_center'],
                                                 onb['diam'], top_views_pass2=5)
                m = matcher.match(cand.pts, k=5)
                cand.scores['depth'] = m.s_depth
                ver = Verifier(K, sym_h, sigma_lidar=SIGMA)
                vs, us = np.nonzero(fb.valid_mask)
                step = max(1, len(vs) // 20000)
                fobs = (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
                        fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))

                def run_and_record(hyps, domain):
                    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb['master'],
                              onb['master_n'], onb['X_verify'], onb['diam'],
                              s2_scores=cand.scores, frame_obs=fobs)
                    e_pos, e_rot = sym_h.sym_aware_error(
                        res.pose[:3, :3], res.pose[:3, 3], R_gt, t_gt, onb['X_verify'])
                    rows.append(dict(
                        x=[float(v) for v in res.diag['features']],
                        y=int(e_pos / onb['diam'] < 0.1),
                        add_rel=float(e_pos / onb['diam']), e_rot=float(e_rot),
                        domain=domain, verdict=res.verdict, p_conf=float(res.p_conf),
                        key=key, shard=si, obj_id=int(obj_id), n_pts=int(len(idx))))

                hyps = coarse_poses_from_match(m, onb['tpl']['tpl_pose'], onb['tpl']['tpl_center'])
                run_and_record(hyps, 'gso_A')
                for h in flip_hyps(R_gt, t_gt):
                    run_and_record([h], 'gso_flip')
                n_done += 1
                if n_done % 10 == 0:
                    pos = sum(r['y'] for r in rows)
                    print(f'shard{si} {n_done}/{args.per_shard} rows={len(rows)} '
                          f'pos={pos} ({time.time()-t_start:.0f}s)', flush=True)
            except Exception as e:
                print(f'[skip] {key}: {e}', flush=True)
        t.close()

    json.dump(rows, open(OUT / 'calib_gso.json', 'w'))
    pos = sum(r['y'] for r in rows)
    print(f'done rows={len(rows)} pos={pos} neg={len(rows)-pos} '
          f'({time.time()-t_start:.0f}s) -> {OUT}/calib_gso.json')


if __name__ == '__main__':
    main()
