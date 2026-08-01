"""MegaPose 렌더 모델계 복원 — 물체별 similarity 보정 (s_o, R_o, t_o).

발견(2026-07-30): MegaPose-GSO 렌더는 raw GSO 메시가 아니라 **물체별로
리스케일된** 메시를 사용 (실측 s ∈ [0.55, 1.35], 회전 ≤2°). raw 메시로 만든
라벨은 nn/D p50 0.10으로 붕괴 — Phase A 학습 실패의 근본 원인.

방법: 물체별로 GT 포즈로 장면점을 모델계에 병합(→ 렌더에 쓰인 메시 표면이
복원됨) → raw 메시 master에 similarity ICP → X_render = s·X_raw@R.T + t.
포즈 라벨을 쓰지만 이는 추정기 학습이 아니라 **물체 메타데이터(메시 스케일)
복원 전처리**다 — train 물체는 train 샘플로, val 물체는 val 샘플로 추정.

산출: phase_a/frame_correction.npz {oi, s, R, t, res_p50_rel, n_samples}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
import train.train_phase_a as T                                        # noqa: E402

DEV = "cuda"


def merged_cloud(bank, tr, oi_all, o, cap=6000, max_samples=80):
    idx = np.nonzero(oi_all == o)[0][:max_samples]
    if len(idx) == 0:
        return None, 0
    pts = tr["pts"][idx].to(DEV).float()
    npt = tr["npt"][idx].to(DEV)
    R = tr["R"][idx].to(DEV).view(-1, 3, 3)
    t = tr["t"][idx].to(DEV)
    te = t + torch.einsum("bij,j->bi", R, bank.c[o])
    out = []
    for b in range(len(idx)):
        v = torch.arange(pts.shape[1], device=DEV) < npt[b]
        out.append(torch.einsum("ji,kj->ki", R[b], pts[b][v] - te[b]))
    P = torch.cat(out)
    return P[torch.randperm(len(P), device=DEV)[:cap]], len(idx)


def sim_icp(P, M, iters=40, trim=0.8):
    s = torch.tensor(1.0, device=DEV)
    R = torch.eye(3, device=DEV)
    tt = torch.zeros(3, device=DEV)
    for _ in range(iters):
        Q = s * (P @ R.T) + tt
        d = torch.cdist(Q, M)
        nnd, nn = d.min(-1)
        keep = nnd < torch.quantile(nnd, trim)
        Pq, Tq = P[keep], M[nn[keep]]
        muP, muT = Pq.mean(0), Tq.mean(0)
        Pc, Tc = Pq - muP, Tq - muT
        H = Pc.T @ Tc
        U, S, Vh = torch.linalg.svd(H)
        Rn = Vh.T @ torch.diag(torch.tensor(
            [1., 1., float(torch.det(Vh.T @ U.T))], device=DEV)) @ U.T
        sn = S.sum() / (Pc ** 2).sum()
        R, s = Rn, sn
        tt = muT - s * (muP @ R.T)
    Q = s * (P @ R.T) + tt
    res = torch.cdist(Q, M).min(-1).values
    return s, R, tt, res


def main():
    bank = T.ObjBank()
    ld = lambda sp: {k: torch.from_numpy(v) for k, v in
                     np.load(T.DATA / f"phase_a_{sp}.npz").items()}
    tr, va = ld("train"), ld("val")
    obj_ids = np.load(T.DATA / "phase_a_objs.npz")["obj_id"]
    n_obj = len(obj_ids)
    S = np.ones(n_obj, np.float32)
    Rc = np.tile(np.eye(3, dtype=np.float32), (n_obj, 1, 1))
    Tc = np.zeros((n_obj, 3), np.float32)
    RES = np.full(n_obj, np.nan, np.float32)
    NS = np.zeros(n_obj, np.int32)
    for o in range(n_obj):
        src = va if obj_ids[o] % 5 == 0 else tr
        P, ns = merged_cloud(bank, src, src["oi"].numpy(), o)
        NS[o] = ns
        if P is None or len(P) < 300:
            print(f"obj {o}: 샘플 부족({ns}) — 보정 항등 유지", flush=True)
            continue
        s, R, tt, res = sim_icp(P, bank.master[o])
        # 주의: sim_icp는 '장면병합 → raw'가 아니라 raw 비교 기준. 우리가 원하는 건
        # raw → 렌더 프레임: X_render = 역변환. P(렌더 프레임) → M(raw c-센터):
        # M ≈ s·P@R.T + t  ⇒  X_render = (X_rawC − t)@R / s
        D = float(bank.diam[o])
        S[o], Rc[o], Tc[o] = float(s), R.cpu().numpy(), tt.cpu().numpy()
        RES[o] = float(res.median()) / D
        if o % 40 == 0:
            print(f"obj {o:3d}: s={S[o]:.3f} res/D={RES[o]:.4f} (n={ns})", flush=True)
    np.savez(T.DATA / "frame_correction.npz", obj_id=obj_ids, s=S, R=Rc, t=Tc,
             res_rel=RES, n_samples=NS)
    ok = RES[~np.isnan(RES)]
    print(f"\n보정 완료 {len(ok)}/{n_obj}종 · res/D p50={np.median(ok):.4f} "
          f"p90={np.percentile(ok, 90):.4f} · s 범위 [{S.min():.2f}, {S.max():.2f}]",
          flush=True)


if __name__ == "__main__":
    main()
