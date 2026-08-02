"""Phase B 1단계 — 신규 GSO 865종 렌더 프레임 보정 복원 (train/recover_gso_scale.py 일반화).

원본(168종)은 phase_a 팩(phase_a_{train,val}.npz)에 의존하지만, 신규 물체는 팩이
없으므로 MegaPose-GSO-fixed 샤드에서 직접 장면 샘플(crop 포인트 + GT 포즈)을
수집한다. 방법은 원본과 동일: 물체별 GT 포즈로 장면점을 모델계에 병합(→ 렌더에
쓰인 리스케일 메시 표면 복원) → raw 메시 master에 similarity ICP →
X_render = ((X_rawC − t)/s) @ R.

- 입력 메시 루트 / 출력 파일명 파라미터화 (원본 스크립트·frame_correction.npz 불변)
- 물체↔obj_id 매핑: MegaPose-GSO/gso_models.json (gso_id ↔ obj_id)
- 샤드 30개면 물체당 후보 p50 ~200샘플 → max_samples=80 충분 (실측: 샤드 0에서
  943종·물체당 p50 7샘플)

[v2] 원본의 s=1 단일 시작 ICP는 진값 s가 1에서 먼 물체(캔팩 s~1.9, 배낭 s~2.2 실측)
에서 붕괴 국소해(s→0, 전방 잔차 trivially 0)에 빠짐 — 반복 구조(12캔 팩)에서 특히
심함. 대책: 스케일 그리드 다중 시작 + rms비 시작, 후보 선택은 **양방향** 잔차
(전방 P→M + 역방향 M→Q). 붕괴 해는 역방향 잔차가 크게(≥0.25D) 벌어져 배제됨
(건강 물체 역방향 p50 0.006~0.016D 실측). res_bwd_rel 키로 역방향 잔차도 저장.

실행: python3 tools/recover_gso_scale_b.py \
    --mesh-root /mnt/samsung2tb/datasets/gso_meshes_raw/unpacked \
    --out /mnt/samsung2tb/datasets/megapose/phase_a/frame_correction_b.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import trimesh

sys.path.insert(0, str(Path(__file__).parents[1]))
from train.smoke_loader import iter_samples                            # noqa: E402

GSO_JSON = "/mnt/samsung2tb/datasets/megapose/MegaPose-GSO/gso_models.json"
DEV = "cuda"
CLOUD_CAP = 6000       # 병합 장면점 상한 (원본 merged_cloud cap과 동일)
PTS_PER_SAMPLE = 400   # 샘플당 저장 포인트 (80샘플 × 400 ≥ CLOUD_CAP 확보)
MASTER_N = 4096        # raw 메시 표면 서브샘플 (ICP 타깃)


# --- train/recover_gso_scale.py 에서 그대로 복사 (원본 수정 금지 제약) -----------
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
# ------------------------------------------------------------------------------

S0_GRID = (0.10, 0.16, 0.25, 0.40, 0.65, 1.00, 1.60, 2.50)


def multistart_icp(P, M, D):
    """스케일 그리드+rms비 다중 시작 sim_icp → 양방향 잔차 최소 후보.

    반환: (s, R, t, fwd_rel, bwd_rel) — s·R·t 규약은 원본과 동일 (M ≈ s·P@R.T + t).
    모든 후보 발산(NaN) 시 None.
    """
    rms0 = float((M.pow(2).sum(1).mean().sqrt()
                  / (P - P.mean(0)).pow(2).sum(1).mean().sqrt()))
    best = None
    for s0 in (*S0_GRID, rms0):
        si, Ri, ti, ri = sim_icp(P * s0, M, iters=30)
        st = float(si) * s0
        if not np.isfinite(st) or st <= 0:
            continue
        fwd = float(ri.median()) / D
        Q = st * (P @ Ri.T) + ti
        bwd = float(torch.cdist(M, Q).min(-1).values.median()) / D
        if not (np.isfinite(fwd) and np.isfinite(bwd)):
            continue
        if best is None or fwd + bwd < best[0]:
            best = (fwd + bwd, s0)
    if best is None:
        return None
    # 승자 시작점에서 원본과 동일한 40회 정련
    s0 = best[1]
    si, Ri, ti, ri = sim_icp(P * s0, M, iters=40)
    st = float(si) * s0
    fwd = float(ri.median()) / D
    Q = st * (P @ Ri.T) + ti
    bwd = float(torch.cdist(M, Q).min(-1).values.median()) / D
    if not (np.isfinite(st) and np.isfinite(fwd) and np.isfinite(bwd)):
        return None
    return st, Ri, ti, fwd, bwd


def load_mesh(args):
    """(obj_id, path) → dict(master[MASTER_N,3] centered, c, diam) 또는 실패 str."""
    obj_id, path = args
    try:
        m = trimesh.load(path, force="mesh")
        surf, _ = trimesh.sample.sample_surface(m, 40000, seed=obj_id)
        surf = np.asarray(surf, np.float64)
        c = surf.mean(0)
        diam = float(np.linalg.norm(surf.max(0) - surf.min(0)))
        rng = np.random.default_rng(obj_id)
        master = (surf[rng.choice(len(surf), MASTER_N, replace=False)] - c)
        return dict(obj_id=obj_id, master=master.astype(np.float32),
                    c=c.astype(np.float32), diam=diam)
    except Exception as e:
        return f"{obj_id}:{type(e).__name__}:{e}"


def scan_shard(args):
    """샤드에서 대상 물체 샘플 수집 → [(obj_id, pts[≤400]f16, R f32, t f32)]."""
    shard, targets, per_obj_cap = args
    rng = np.random.default_rng(shard + 7777)
    cnt: dict[int, int] = {}
    out = []
    try:
        for s in iter_samples(shard, seed=shard):
            o = int(s["obj_id"])
            if o not in targets or cnt.get(o, 0) >= per_obj_cap:
                continue
            pts = s["pts"]
            if len(pts) > PTS_PER_SAMPLE:
                pts = pts[rng.choice(len(pts), PTS_PER_SAMPLE, replace=False)]
            out.append((o, pts.astype(np.float16),
                        s["R"].astype(np.float32), s["t"].astype(np.float32)))
            cnt[o] = cnt.get(o, 0) + 1
    except Exception as e:  # 샤드 단위 격리
        print(f"shard {shard}: {type(e).__name__} {e}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", default="/mnt/samsung2tb/datasets/gso_meshes_raw/unpacked")
    ap.add_argument("--out", default="/mnt/samsung2tb/datasets/megapose/phase_a/frame_correction_b.npz")
    ap.add_argument("--shards", type=int, default=30)
    ap.add_argument("--max-samples", type=int, default=80)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out_p = Path(a.out)
    assert out_p.name != "frame_correction.npz", "기존 보정 파일 덮어쓰기 금지"

    # 1) 대상 목록: 메시 루트 ∩ gso_models.json
    name_of = {e["gso_id"]: int(e["obj_id"]) for e in json.load(open(GSO_JSON))}
    root = Path(a.mesh_root)
    tasks, unmapped, no_mesh = [], [], []
    for d in sorted(root.iterdir()):
        p = d / "meshes" / "model.obj"
        if not p.exists():
            no_mesh.append(d.name)
            continue
        if d.name not in name_of:
            unmapped.append(d.name)
            continue
        tasks.append((name_of[d.name], str(p)))
    tasks.sort()
    print(f"대상 {len(tasks)}종 · 미매핑 {len(unmapped)} {unmapped} · "
          f"메시 부재 {len(no_mesh)} {no_mesh}", flush=True)

    # 2) 메시 로드 (병렬)
    t0 = time.time()
    with Pool(a.workers) as p:
        mres = p.map(load_mesh, tasks)
    meshes = {q["obj_id"]: q for q in mres if isinstance(q, dict)}
    mesh_fail = [q for q in mres if isinstance(q, str)]
    print(f"[mesh] {len(meshes)}종 로드 {time.time()-t0:.0f}s · 실패 {len(mesh_fail)} "
          f"{mesh_fail[:10]}", flush=True)

    # 3) 샤드 스캔 (병렬) — 물체당 샤드별 상한으로 편중 방지
    t0 = time.time()
    per_obj_cap = max(4, (a.max_samples * 3) // a.shards)
    targets = frozenset(meshes)
    with Pool(a.workers) as p:
        parts = p.map(scan_shard, [(s, targets, per_obj_cap) for s in range(a.shards)])
    by_obj: dict[int, list] = {o: [] for o in meshes}
    for part in parts:
        for o, pts, R, t in part:
            if len(by_obj[o]) < a.max_samples:
                by_obj[o].append((pts, R, t))
    ns_all = np.array([len(v) for v in by_obj.values()])
    print(f"[scan] 샤드 {a.shards}개 {time.time()-t0:.0f}s · 물체당 샘플 "
          f"p10={np.percentile(ns_all,10):.0f} p50={np.median(ns_all):.0f} "
          f"p90={np.percentile(ns_all,90):.0f} · 0샘플 {int((ns_all==0).sum())}종", flush=True)

    # 4) 물체별 병합 → similarity ICP (GPU)
    t0 = time.time()
    obj_ids = np.array(sorted(meshes), np.int32)
    n_obj = len(obj_ids)
    S = np.ones(n_obj, np.float32)
    Rc = np.tile(np.eye(3, dtype=np.float32), (n_obj, 1, 1))
    Tc = np.zeros((n_obj, 3), np.float32)
    RES = np.full(n_obj, np.nan, np.float32)
    BWD = np.full(n_obj, np.nan, np.float32)
    NS = np.zeros(n_obj, np.int32)
    gen = torch.Generator(device=DEV)
    for i, o in enumerate(obj_ids):
        q = meshes[int(o)]
        samples = by_obj[int(o)]
        NS[i] = len(samples)
        if len(samples) == 0:
            print(f"obj {o}: 샘플 0 — 보정 항등 유지", flush=True)
            continue
        c = torch.from_numpy(q["c"]).to(DEV).float()
        merged = []
        for pts, R, t in samples:
            P = torch.from_numpy(pts).to(DEV).float()
            Rg = torch.from_numpy(R).to(DEV)
            tg = torch.from_numpy(t).to(DEV).float()
            te = tg + Rg @ c                       # 원본 merged_cloud와 동일
            merged.append(torch.einsum("ji,kj->ki", Rg, P - te))
        P = torch.cat(merged)
        if len(P) < 300:
            print(f"obj {o}: 병합점 부족({len(P)}) — 보정 항등 유지", flush=True)
            continue
        gen.manual_seed(int(o))
        P = P[torch.randperm(len(P), generator=gen, device=DEV)[:CLOUD_CAP]]
        M = torch.from_numpy(q["master"]).to(DEV)
        r = multistart_icp(P, M, q["diam"])
        if r is None:
            print(f"obj {o}: 전 시작점 발산 — 보정 항등 유지", flush=True)
            continue
        st, R, tt, fwd, bwd = r
        S[i], Rc[i], Tc[i] = st, R.cpu().numpy(), tt.cpu().numpy()
        RES[i], BWD[i] = fwd, bwd
        if i % 25 == 0 or i == n_obj - 1:
            print(f"[{i+1}/{n_obj}] obj {o}: s={S[i]:.3f} res/D={RES[i]:.4f} "
                  f"bwd/D={BWD[i]:.4f} (n={NS[i]}) {time.time()-t0:.0f}s", flush=True)

    # unpacked/{name}/meshes/model.obj — 로드 실패 물체 제외 후 obj_id 순 정렬 유지
    name_by_id = {oid: pth.split("/")[-3] for oid, pth in tasks}
    names = np.array([name_by_id[int(o)] for o in obj_ids])
    np.savez(out_p, obj_id=obj_ids, s=S, R=Rc, t=Tc, res_rel=RES, n_samples=NS,
             res_bwd_rel=BWD, gso_id=names)
    fin = ~np.isnan(RES)
    ok = RES[fin]
    print(f"\n보정 완료 {len(ok)}/{n_obj}종 · res/D p50={np.median(ok):.4f} "
          f"p90={np.percentile(ok, 90):.4f} · bwd/D p50={np.median(BWD[fin]):.4f} "
          f"p90={np.percentile(BWD[fin], 90):.4f} · s 범위 "
          f"[{S[fin].min():.2f}, {S[fin].max():.2f}] · 의심(bwd/D>0.05) "
          f"{int((BWD[fin] > 0.05).sum())}종 → {out_p}", flush=True)


if __name__ == "__main__":
    main()
