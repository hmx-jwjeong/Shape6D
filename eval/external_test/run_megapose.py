"""MegaPose-GSO 샤드 일부(~100 샘플)에서 v0-geo 무학습 파이프라인 평가.

모드 A (GT 마스크 보조): S1을 우회해 신규 컨셉(S2 TDF 정합 → coarse → S4)을 절연 평가
모드 B (풀 파이프라인): S1 기하 클러스터 포함 — 클러터 한계 정량화

메시 전처리 규약(재투영 잔차로 실증 확정): center = AABB 중심, scale = 0.2m / 최장변.
실행: uv run python eval/external_test/run_megapose.py [--n 100]
"""
from __future__ import annotations

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from harness import (frame_from_points, onboard_mesh, overlay_png, run_frame,
                     sparsify_fixed_grid)

from shape6d.common.frame_bundle import CameraIntrinsics
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

DATA = Path(__file__).resolve().parents[2] / "Data" / "megapose_sample"
OUT = Path(__file__).parent / "megapose_results"
SIGMA = 0.008


def rle_to_mask(rle) -> np.ndarray:
    arr = np.zeros(int(np.prod(rle["size"])), dtype=bool)
    counts = rle["counts"]
    start = 0
    for i in range(len(counts) - 1):
        start += counts[i]
        arr[start:start + counts[i + 1]] = (i + 1) % 2
    return arr.reshape(*rle["size"], order="F")


def preprocess_mesh(name: str) -> trimesh.Trimesh:
    mesh = trimesh.load(DATA / "gso_raw" / name / "meshes" / "model.obj", force="mesh")
    c = (mesh.bounds[0] + mesh.bounds[1]) / 2
    s = 0.2 / (mesh.bounds[1] - mesh.bounds[0]).max()   # 최장변 → 0.2m (실증 규약)
    mesh.apply_translation(-c)
    mesh.apply_scale(s)
    return mesh


def eval_pose(res, R_gt, t_gt, sym_h, X_verify, diam):
    R_e, t_e = res.pose[:3, :3], res.pose[:3, 3]
    e_pos, e_rot = sym_h.sym_aware_error(R_e, t_e, R_gt, t_gt, X_verify)
    return dict(e_pos=float(e_pos), e_rot=float(e_rot), add_rel=float(e_pos / diam),
                verdict=res.verdict, p_conf=float(res.p_conf),
                stats={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in res.diag["stats"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--overlays", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    t = tarfile.open(DATA / "shard-000000.tar")
    picked = json.load(open(DATA / "picked.json"))
    gso = {e["obj_id"]: e["gso_id"] for e in json.load(open(DATA / "gso_models.json"))}

    onb_cache: dict[int, dict] = {}
    results = []
    n_done = 0
    t_start = time.time()

    for s0 in picked:
        if n_done >= args.n:
            break
        key, obj_id = s0["key"], s0["obj_id"]
        name = gso[obj_id]
        if not (DATA / "gso_raw" / name / "meshes" / "model.obj").exists():
            continue

        # --- 온보딩 (물체별 1회 캐시) ---
        if obj_id not in onb_cache:
            try:
                onb_cache[obj_id] = onboard_mesh(preprocess_mesh(name))
            except Exception as e:
                print(f"[skip] onboard {name}: {e}", flush=True)
                onb_cache[obj_id] = None
        onb = onb_cache[obj_id]
        if onb is None:
            continue

        # --- 프레임 로드 ---
        cam = json.load(t.extractfile(f"{key}.camera.json"))
        Km = np.array(cam["cam_K"]).reshape(3, 3)
        depth_m = np.array(Image.open(io.BytesIO(t.extractfile(f"{key}.depth.png").read()))
                           ).astype(np.float32) * cam["depth_scale"] / 1000.0
        rgb = np.array(Image.open(io.BytesIO(t.extractfile(f"{key}.rgb.jpg").read())))
        H, W = depth_m.shape
        K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2],
                             width=W, height=H)
        gts = json.load(t.extractfile(f"{key}.gt.json"))
        gis = json.load(t.extractfile(f"{key}.gt_info.json"))
        gt_idx = max((i for i, (g, gi) in enumerate(zip(gts, gis)) if g["obj_id"] == obj_id),
                     key=lambda i: gis[i]["px_count_visib"])
        R_gt = np.array(gts[gt_idx]["cam_R_m2c"]).reshape(3, 3)
        t_gt = np.array(gts[gt_idx]["cam_t_m2c"]) / 1000.0
        masks = json.load(t.extractfile(f"{key}.mask_visib.json"))
        mask = rle_to_mask(masks[str(gt_idx)])

        pts = sparsify_fixed_grid(depth_m, K)
        fb = frame_from_points(rgb[:, :, :3], pts, K)
        sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)

        rec = dict(key=key, obj_id=obj_id, name=name, visib=s0["visib"],
                   diam=onb["diam"], n_sym=int(len(onb["sym"].sym_rots)),
                   n_cont=int(len(onb["sym"].sym_axes)))

        # --- 모드 A: GT 마스크 보조 (S1 우회 — S2·coarse·S4 절연 평가) ---
        idx = fb.object_points(mask, erosion_px=2)
        rec["n_obj_pts"] = int(len(idx))
        if len(idx) >= 30:
            cand = Candidate(
                proposal=Proposal(mask=None, bbox=np.zeros(4), score=1.0, source="gt_mask",
                                  lidar_idx=idx, n_lidar=len(idx)),
                pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx])
            matcher = PointToTemplateMatcher(onb["tpl"]["tdf"], onb["tpl"]["tpl_center"],
                                             onb["diam"], top_views_pass2=5)
            m = matcher.match(cand.pts, k=5)
            cand.scores["depth"] = m.s_depth
            hyps = coarse_poses_from_match(m, onb["tpl"]["tpl_pose"], onb["tpl"]["tpl_center"])
            ver = Verifier(K, sym_h, sigma_lidar=SIGMA)
            vs, us = np.nonzero(fb.valid_mask)
            step = max(1, len(vs) // 20000)
            frame_obs = (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
                         fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))
            resA = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb["master"],
                       onb["master_n"], onb["X_verify"], onb["diam"],
                       s2_scores=cand.scores, frame_obs=frame_obs)
            rec["A"] = eval_pose(resA, R_gt, t_gt, sym_h, onb["X_verify"], onb["diam"])
            rec["A"]["s_depth"] = float(m.s_depth)
            if n_done < args.overlays:
                overlay_png(rgb[:, :, :3], onb["master"][::4], resA.pose, K,
                            str(OUT / f"overlay_A_{key}.png"), obs_pts=cand.pts)
        else:
            rec["A"] = None

        # --- 모드 B: 풀 파이프라인 (S1 포함) ---
        try:
            resB, diagB = run_frame(onb, fb, K, sigma=SIGMA, max_candidates=4)
            rec["B"] = (eval_pose(resB, R_gt, t_gt, sym_h, onb["X_verify"], onb["diam"])
                        if resB is not None else {"reason": diagB.get("reason")})
        except Exception as e:
            rec["B"] = {"reason": f"error: {e}"}

        results.append(rec)
        n_done += 1
        if n_done % 10 == 0:
            a_ok = sum(1 for r in results if r["A"] and r["A"]["add_rel"] < 0.1)
            print(f"{n_done}/{args.n}  A성공(ADD<0.1D) {a_ok}/{sum(1 for r in results if r['A'])} "
                  f" 경과 {time.time()-t_start:.0f}s", flush=True)

    json.dump(results, open(OUT / "results.json", "w"), indent=1)

    # --- 집계 ---
    A = [r["A"] for r in results if r["A"]]
    B = [r["B"] for r in results if isinstance(r.get("B"), dict) and "add_rel" in r["B"]]
    def agg(X, label):
        if not X:
            print(f"{label}: 없음"); return
        ok = [x for x in X if x["add_rel"] < 0.1]
        acc = [x for x in X if x["verdict"] == "ACCEPT"]
        acc_ok = [x for x in acc if x["add_rel"] < 0.1]
        fail = [x for x in X if x["add_rel"] >= 0.1]
        flips = sum(1 for x in fail if x["e_rot"] > 150)
        quarter = sum(1 for x in fail if 70 <= x["e_rot"] <= 110)
        line = (f"{label}: n={len(X)}  ADD<0.1D = {len(ok)}/{len(X)} ({100*len(ok)/len(X):.0f}%)"
                f"  | ACCEPT {len(acc)}건 중 정답 {len(acc_ok)} (오수락 {len(acc)-len(acc_ok)})"
                f"  | 실패 분류: 플립(>150°) {flips}, 90°급 {quarter}, 기타 {len(fail)-flips-quarter}")
        if ok:
            line += (f"  | 성공례 중앙값 pos {np.median([x['e_pos'] for x in ok])*1e3:.1f}mm"
                     f" rot {np.median([x['e_rot'] for x in ok]):.2f}°")
        print(line)
    agg(A, "모드 A (GT 마스크)")
    agg(B, "모드 B (풀 파이프라인)")
    print("저장:", OUT / "results.json")


if __name__ == "__main__":
    main()
