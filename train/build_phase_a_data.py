"""Phase A 데이터 구축 — 오브젝트 팩(42뷰) + 장면 샘플(포즈 회귀 라벨).

[클린룸 규약] SAM-6D 소스 미열람. 데이터 규약은 docs/03 §2·§6.4와 BOP 포맷 문서 기준.
[17 §5.5] 희소화는 센서 스펙 합성(shape6d.common.sparsify 경유 smoke_loader) — UAM 미사용.

산출:
  phase_a_objs.npz   — 메시 보유 168종: master 2048(모델계)·직경·대칭 g(≤16)·
                       42뷰 sparse depth(u8,u8,f16 ≤6144px)·뷰 포즈(센터링 흡수)
  phase_a_{train,val}.npz — 장면 crop 포인트(카메라계 f16 ≤4096) + R(9) t(3) + obj 인덱스
                       분할: obj_id%5==0 → val (교집합 0 — 제로샷)

실행: python3 train/build_phase_a_data.py --shards 60 --out /mnt/samsung2tb/datasets/megapose/phase_a
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parents[1]))
from shape6d.onboarding.symmetry import detect_symmetry               # noqa: E402
from shape6d.onboarding.templates import (                            # noqa: E402
    CAM_DIST_FACTOR, N_VIEWS, TPL_FX, TPL_RES, icosphere42, lookat_poses, splat_depth,
)
from train.smoke_loader import iter_samples                           # noqa: E402

GSO_RAW = "/mnt/samsung2tb/datasets/megapose/gso_raw"
GSO_JSON = "/mnt/samsung2tb/datasets/megapose/MegaPose-GSO/gso_models.json"
PX_CAP = 6144          # 뷰당 sparse depth 저장 상한
PTS_CAP = 4096         # 장면 샘플 포인트 상한
SYM_MAX = 16           # g* 군 상한 (03 §9.3 — 연속축 12분할)


def _mesh_index() -> dict[int, str]:
    idx = {}
    for e in json.load(open(GSO_JSON)):
        p = f"{GSO_RAW}/{e['gso_id']}/meshes/model.obj"
        if os.path.exists(p):
            idx[int(e["obj_id"])] = p
    return idx


def _discretize_sym(sym, n_disc: int = 12) -> np.ndarray:
    """SymmetryResult → 회전군 [G,3,3] (연속축 n_disc 분할, G≤SYM_MAX, 항등 포함)."""
    rots = [np.eye(3)]
    for R in np.asarray(sym.sym_rots).reshape(-1, 3, 3):
        if np.linalg.norm(R - np.eye(3)) > 1e-6:
            rots.append(R)
    for ax in np.asarray(sym.sym_axes).reshape(-1, 3):
        a = ax / (np.linalg.norm(ax) + 1e-12)
        for k in range(1, n_disc):
            th = 2 * np.pi * k / n_disc
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            rots.append(np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)
    uniq = [rots[0]]
    for R in rots[1:]:
        if all(np.linalg.norm(R - U) > 1e-3 for U in uniq):
            uniq.append(R)
    return np.stack(uniq[:SYM_MAX]).astype(np.float32)


_CORR = None


def _load_corr(out_dir: Path):
    global _CORR
    # v2 우선 (다중 시작 + 양방향 잔차 — 구판은 붕괴 국소해로 61/168 오염 실증)
    p2 = out_dir / "frame_correction_a_v2.npz"
    p = p2 if p2.exists() else out_dir / "frame_correction.npz"
    if p.exists():
        z = np.load(p)
        bwd = z["res_bwd_rel"] if "res_bwd_rel" in z.files else np.zeros(len(z["s"]))
        _CORR = {int(o): (float(s), R, t, float(r), float(b)) for o, s, R, t, r, b in
                 zip(z["obj_id"], z["s"], z["R"], z["t"], z["res_rel"], bwd)}
        print(f"[corr] 렌더 프레임 보정 로드({p.name}): {len(_CORR)}종", flush=True)


def _pack_object(args):
    obj_id, path = args
    rng = np.random.default_rng(obj_id)
    m = trimesh.load(path, force="mesh")
    surf, _ = trimesh.sample.sample_surface(m, 40000, seed=obj_id)
    surf = np.asarray(surf, np.float64)
    # MegaPose 렌더 프레임 보정 (recover_gso_scale.py):
    # X_render = ((X_rawC − t_o) / s_o) @ R_o  — 미회수/고잔차 물체는 None 반환(제외)
    if _CORR is not None:
        e = _CORR.get(int(obj_id))
        if (e is None or not np.isfinite(e[0]) or not np.isfinite(e[3])
                or e[3] > 0.05 or e[4] > 0.05):     # 양방향 잔차 필터 (붕괴 검출)
            return None
        s_o, R_o, t_o, _, _ = e
        surf = ((surf - surf.mean(0) - t_o) / s_o) @ R_o
    c = surf.mean(0)
    r_b = float(np.linalg.norm(surf - c, axis=1).max())
    diam = float(np.linalg.norm(surf.max(0) - surf.min(0)))
    master = surf[rng.choice(len(surf), 2048, replace=False)].astype(np.float32)
    try:
        g = _discretize_sym(detect_symmetry(master[
            rng.choice(2048, 1024, replace=False)].astype(np.float64)))
    except Exception:
        g = np.eye(3, dtype=np.float32)[None]

    dirs = icosphere42()
    poses = lookat_poses(dirs, CAM_DIST_FACTOR * r_b).astype(np.float32)
    poses[:, :3, 3] -= np.einsum("vij,j->vi", poses[:, :3, :3], c)  # 센터링 흡수
    U = np.zeros((N_VIEWS, PX_CAP), np.uint8)
    V = np.zeros((N_VIEWS, PX_CAP), np.uint8)
    Z = np.zeros((N_VIEWS, PX_CAP), np.float16)
    NPX = np.zeros(N_VIEWS, np.int32)
    for vi in range(N_VIEWS):
        pc = surf @ poses[vi, :3, :3].T + poses[vi, :3, 3]
        d = splat_depth(pc, TPL_FX, TPL_FX, TPL_RES / 2, TPL_RES / 2,
                        (TPL_RES, TPL_RES), fill_iters=0)
        vs, us = np.nonzero(d > 0)
        if len(us) > PX_CAP:
            sel = rng.choice(len(us), PX_CAP, replace=False)
            us, vs = us[sel], vs[sel]
        n = len(us)
        U[vi, :n], V[vi, :n] = us.astype(np.uint8), vs.astype(np.uint8)
        Z[vi, :n] = d[vs, us].astype(np.float16)
        NPX[vi] = n
    return dict(obj_id=obj_id, master=(master - c).astype(np.float32), c=c.astype(np.float32),
                diam=diam, g=g, gn=len(g), pose=poses, u=U, v=V, z=Z, npx=NPX)


def build_objects(out: Path, workers: int = 8) -> dict[int, int]:
    idx = _mesh_index()
    t0 = time.time()
    with Pool(workers) as p:
        packs = [q for q in p.map(_pack_object, sorted(idx.items())) if q is not None]
    order = {q["obj_id"]: i for i, q in enumerate(packs)}
    G = np.zeros((len(packs), SYM_MAX, 3, 3), np.float32)
    G[:, :] = np.eye(3)
    for i, q in enumerate(packs):
        G[i, :q["gn"]] = q["g"]
    np.savez(out / f"{a.prefix}_objs.npz",
             obj_id=np.array([q["obj_id"] for q in packs], np.int32),
             master=np.stack([q["master"] for q in packs]),   # centroid 센터링된 모델계
             c=np.stack([q["c"] for q in packs]),             # 원 모델계 centroid (GT 병진 보정용)
             diam=np.array([q["diam"] for q in packs], np.float32),
             g=G, gn=np.array([q["gn"] for q in packs], np.int32),
             pose=np.stack([q["pose"] for q in packs]),
             u=np.stack([q["u"] for q in packs]), v=np.stack([q["v"] for q in packs]),
             z=np.stack([q["z"] for q in packs]),
             npx=np.stack([q["npx"] for q in packs]))
    gn = np.array([q["gn"] for q in packs])
    print(f"[objs] {len(packs)}종 {time.time()-t0:.0f}s · 비항등 대칭 보유 "
          f"{int((gn > 1).sum())}종 · 뷰당 px p50 "
          f"{int(np.median(np.concatenate([q['npx'] for q in packs])))}", flush=True)
    return order


def _scene_shard(args):
    shard, order, cap = args
    rng = np.random.default_rng(shard)
    P = np.zeros((cap, PTS_CAP, 3), np.float16)
    N = np.zeros(cap, np.int32)
    R = np.zeros((cap, 9), np.float32)
    T = np.zeros((cap, 3), np.float32)
    O = np.zeros(cap, np.int32)
    n = 0
    try:
        for s in iter_samples(shard, npt_range=(256, PTS_CAP), seed=shard):
            if n >= cap:
                break
            oi = order.get(int(s["obj_id"]))
            if oi is None or len(s["pts"]) < 256:
                continue
            pts = s["pts"]
            if len(pts) > PTS_CAP:
                pts = pts[rng.choice(len(pts), PTS_CAP, replace=False)]
            P[n, :len(pts)] = pts.astype(np.float16)
            N[n] = len(pts)
            R[n] = s["R"].astype(np.float32).ravel()
            T[n] = s["t"].astype(np.float32)
            O[n] = oi
            n += 1
    except Exception as e:  # 샤드 단위 격리
        print(f"shard {shard}: {type(e).__name__} {e}", flush=True)
    return dict(pts=P[:n], npt=N[:n], R=R[:n], t=T[:n], oi=O[:n])


def build_scenes(out: Path, order: dict[int, int], obj_ids: np.ndarray,
                 shards: int, cap: int, workers: int = 8):
    t0 = time.time()
    with Pool(workers) as p:
        parts = p.map(_scene_shard, [(s, order, cap) for s in range(shards)])
    cat = {k: np.concatenate([q[k] for q in parts if len(q[k])]) for k in parts[0]}
    is_val = (obj_ids[cat["oi"]] % 5 == 0)
    for split, sel in (("train", ~is_val), ("val", is_val)):
        np.savez(out / f"{a.prefix}_{split}.npz", **{k: v[sel] for k, v in cat.items()})
        no = len(np.unique(cat["oi"][sel]))
        print(f"[{split}] {int(sel.sum())}샘플 · 물체 {no}종 · 포인트 p50 "
              f"{int(np.median(cat['npt'][sel]))}", flush=True)
    print(f"[scenes] {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/samsung2tb/datasets/megapose/phase_a")
    ap.add_argument("--prefix", default="phase_a", help="출력 파일 접두 (재구축 시 phase_a2 등 — 구판 보존)")
    ap.add_argument("--shards", type=int, default=60)
    ap.add_argument("--cap", type=int, default=1200)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    _load_corr(out)
    order = build_objects(out)
    obj_ids = np.load(out / f"{a.prefix}_objs.npz")["obj_id"]
    build_scenes(out, order, obj_ids, a.shards, a.cap)
