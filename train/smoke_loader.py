"""M2-A1 스모크: GSO 샤드 → 학습 샘플 (희소 LiDAR 증강 #1 포함) → GPU 텐서.

목적: 본학습 전 데이터 경로 검증 — (i) 샤드 스트리밍 파싱 (ii) dense depth →
ML-X 격자 희소화 (iii) 물체 crop [N,3]+GT 포즈 텐서화 (iv) 처리율 측정.
PEM 모델·loss는 M2-B에서 — 여기는 입력 파이프라인만.

실행: python3 train/smoke_loader.py --shard 0 --n 32
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
sys.path.insert(0, str(Path(__file__).parents[1] / 'eval' / 'external_test'))
from harness import sparsify_fixed_grid  # noqa: E402
from shape6d.common.frame_bundle import CameraIntrinsics  # noqa: E402

SHARDS = Path('/mnt/samsung2tb/datasets/megapose/MegaPose-GSO-fixed')


def iter_samples(shard_idx: int, visib_min: float = 0.5, npt_range=(300, 12000), seed: int = 0):
    """샤드에서 (obj_pts[N,3], R_gt, t_gt, obj_id, meta)를 스트림 생성.

    증강 #1: 격자 희소화 후 물체 위 포인트를 npt_range 로그 균등 목표로 랜덤 서브샘플.
    """
    rng = np.random.default_rng(seed)
    t = tarfile.open(SHARDS / f'shard-{shard_idx:06d}.tar')
    keys = sorted({m.name.split('.')[0] for m in t.getmembers()})
    for key in keys:
        try:
            gts = json.load(t.extractfile(f'{key}.gt.json'))
            gis = json.load(t.extractfile(f'{key}.gt_info.json'))
            cam = json.load(t.extractfile(f'{key}.camera.json'))
            masks = json.load(t.extractfile(f'{key}.mask_visib.json'))
            depth = np.array(Image.open(io.BytesIO(
                t.extractfile(f'{key}.depth.png').read()))).astype(np.float32) \
                * cam['depth_scale'] / 1000.0
            H, W = depth.shape
            Km = np.array(cam['cam_K']).reshape(3, 3)
            K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2],
                                 width=W, height=H)
            pts = sparsify_fixed_grid(depth, K, sigma=0.003, seed=int(rng.integers(1 << 31)))
            u = np.round(K.fx * pts[:, 0] / pts[:, 2] + K.cx).astype(int).clip(0, W - 1)
            v = np.round(K.fy * pts[:, 1] / pts[:, 2] + K.cy).astype(int).clip(0, H - 1)
            for gi_idx, (g, gi) in enumerate(zip(gts, gis)):
                if gi['px_count_all'] <= 0:
                    continue
                if gi['px_count_visib'] / gi['px_count_all'] < visib_min:
                    continue
                rle = masks.get(str(gi_idx))
                if rle is None:
                    continue
                arr = np.zeros(int(np.prod(rle['size'])), dtype=bool)
                val, pos = False, 0
                for run in rle['counts']:
                    if val:
                        arr[pos:pos + run] = True
                    pos += run
                    val = not val
                mask = arr.reshape(rle['size'], order='F')
                sel = mask[v, u]
                obj = pts[sel]
                if len(obj) < 100:
                    continue
                n_tgt = int(np.exp(rng.uniform(*np.log(npt_range))))
                if len(obj) > n_tgt:
                    obj = obj[rng.choice(len(obj), n_tgt, replace=False)]
                yield dict(pts=obj,
                           R=np.array(g['cam_R_m2c']).reshape(3, 3),
                           t=np.array(g['cam_t_m2c']) / 1000.0,
                           obj_id=g['obj_id'], key=key, n_raw=int(sel.sum()))
        except Exception:
            continue
    t.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n', type=int, default=32)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={dev}')

    t0, n, npts = time.time(), 0, []
    for s in iter_samples(args.shard):
        P = torch.from_numpy(s['pts']).to(dev)          # [N,3] float32
        Rt = torch.from_numpy(np.hstack([s['R'], s['t'][:, None]])).to(dev)
        assert P.ndim == 2 and P.shape[1] == 3 and Rt.shape == (3, 4)
        npts.append(len(P))
        n += 1
        if n >= args.n:
            break
    dt = time.time() - t0
    print(f'{n} samples in {dt:.1f}s = {n/dt:.1f} samples/s (단일 프로세스, 압축 해제 포함)')
    print(f'obj pts: min {min(npts)} med {int(np.median(npts))} max {max(npts)}')
    print('GPU tensor OK — M2-A1 경로 검증 완료 (본학습은 다중 워커로 병렬화)')


if __name__ == '__main__':
    main()
