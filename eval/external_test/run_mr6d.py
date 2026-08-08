"""MR6D val 스플릿(실측 RealSense D435i)의 유로팔레트 프레임에서 v0-geo 평가.

MegaPose 평가(run_megapose.py)와 동일 프로토콜의 팔레트 도메인 실측판:
- 모드 A (GT 마스크 보조): S1 우회 — S2 TDF 정합 → coarse → S4 절연 평가
- 모드 B (풀 파이프라인): S1 기하 클러스터 포함
- dense depth → ML-X(80) 고정 각도 격자 희소화 (센서 자체 노이즈가 있으므로 추가 σ 없음)

데이터: Data/mr6d/ (HF anas-gouda/mr6d, BOP 포맷, 모델 단위 mm).
obj_id 1 = wooden Euro pallet (800×1200×144mm, AABB 중심 = 원점 → 스케일 0.001만 적용).
실행: uv run python eval/external_test/run_mr6d.py [--n 100]
"""
from __future__ import annotations

import argparse
import json
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

DATA = Path(__file__).resolve().parents[2] / "Data" / "mr6d"
OUT = Path(__file__).parent / "mr6d_results"
SCENES = ["000000", "000001", "000002", "000005", "000006"]
OBJ_ID = 1          # wooden Euro pallet
SIGMA = 0.008       # 검증기 σ_lidar (03 규약)
MIN_VISIB = 0.3


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
    ap.add_argument("--overlays", type=int, default=8)
    ap.add_argument("--upright", action="store_true",
                    help="직립 프리이어 ON: 아래에서 올려다보는 뷰 프루닝 (03 §1.4e, 04에서 플립 차단 실증)")
    ap.add_argument("--out", default=None, help="결과 json 파일명 (기본 results.json)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(DATA / "models" / "obj_000001.ply")
    mesh.apply_scale(0.001)                      # mm → m (AABB 중심은 이미 원점)
    onb = onboard_mesh(mesh)
    view_mask = None
    if args.upright:
        from shape6d.onboarding.templates import upright_view_mask
        view_mask = upright_view_mask()
        print(f"직립 프리이어 ON: 뷰 {int(view_mask.sum())}/42 유지", flush=True)
    sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)
    print(f"onboard: diam {onb['diam']:.3f}m, |S| {len(onb['sym'].sym_rots)}, "
          f"연속축 {len(onb['sym'].sym_axes)}", flush=True)

    # --- 프레임 목록 수집 (장면 균등 인터리브) ---
    per_scene = []
    for sc in SCENES:
        d = DATA / "val" / sc
        if not (d / "scene_gt.json").exists() or not (d / "rgb").exists():
            print(f"[skip] {sc}: 미완결 다운로드", flush=True)
            continue
        gt = json.load(open(d / "scene_gt.json"))
        gi = json.load(open(d / "scene_gt_info.json"))
        cam = json.load(open(d / "scene_camera.json"))
        items = []
        for im_id in sorted(gt, key=int):
            for gt_idx, o in enumerate(gt[im_id]):
                if o["obj_id"] != OBJ_ID:
                    continue
                info = gi[im_id][gt_idx]
                if info.get("visib_fract", 1.0) < MIN_VISIB:
                    continue
                items.append((sc, im_id, gt_idx, o, cam[im_id], info))
        per_scene.append(items)
    frames = [x for tup in __import__("itertools").zip_longest(*per_scene)
              for x in tup if x is not None][:args.n]
    print(f"평가 프레임 {len(frames)}건 (장면별 "
          f"{[len(s) for s in per_scene]} 중 인터리브)", flush=True)

    results = []
    t_start = time.time()
    for n_done, (sc, im_id, gt_idx, o, cam, info) in enumerate(frames, 1):
        d = DATA / "val" / sc
        Km = np.array(cam["cam_K"]).reshape(3, 3)
        depth_m = (np.array(Image.open(d / "depth" / f"{int(im_id):06d}.png"))
                   .astype(np.float32) * cam["depth_scale"] / 1000.0)
        rgb_p = d / "rgb" / f"{int(im_id):06d}.jpg"
        if not rgb_p.exists():
            rgb_p = rgb_p.with_suffix(".png")
        rgb = np.array(Image.open(rgb_p))[:, :, :3]
        H, W = depth_m.shape
        K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2],
                             width=W, height=H)
        R_gt = np.array(o["cam_R_m2c"]).reshape(3, 3)
        t_gt = np.array(o["cam_t_m2c"], dtype=np.float64) / 1000.0
        mask_p = d / "mask_visib" / f"{int(im_id):06d}_{gt_idx:06d}.png"
        if not mask_p.exists():
            continue
        mask = np.array(Image.open(mask_p)) > 0

        pts = sparsify_fixed_grid(depth_m, K)     # 실측 노이즈 그대로, 추가 σ 없음
        fb = frame_from_points(rgb, pts, K)

        rec = dict(scene=sc, im_id=im_id, gt_idx=gt_idx,
                   visib=float(info.get("visib_fract", 1.0)),
                   z_gt=float(t_gt[2]))

        # --- 모드 A: GT 마스크 보조 ---
        idx = fb.object_points(mask, erosion_px=2)
        rec["n_obj_pts"] = int(len(idx))
        if len(idx) >= 30:
            cand = Candidate(
                proposal=Proposal(mask=None, bbox=np.zeros(4), score=1.0, source="gt_mask",
                                  lidar_idx=idx, n_lidar=len(idx)),
                pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx])
            matcher = PointToTemplateMatcher(onb["tpl"]["tdf"], onb["tpl"]["tpl_center"],
                                             onb["diam"], top_views_pass2=5,
                                             view_mask=view_mask)
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
            if n_done <= args.overlays:
                overlay_png(rgb, onb["master"][::4], resA.pose, K,
                            str(OUT / f"overlay_A_{sc}_{int(im_id):06d}.png"),
                            obs_pts=cand.pts)
        else:
            rec["A"] = None

        # --- 모드 B: 풀 파이프라인 ---
        try:
            resB, diagB = run_frame(onb, fb, K, sigma=SIGMA, max_candidates=4)
            rec["B"] = (eval_pose(resB, R_gt, t_gt, sym_h, onb["X_verify"], onb["diam"])
                        if resB is not None else {"reason": diagB.get("reason")})
        except Exception as e:
            rec["B"] = {"reason": f"error: {e}"}

        results.append(rec)
        if n_done % 10 == 0:
            a_ok = sum(1 for r in results if r["A"] and r["A"]["add_rel"] < 0.1)
            print(f"{n_done}/{len(frames)}  A성공(ADD<0.1D) {a_ok}/"
                  f"{sum(1 for r in results if r['A'])}  경과 {time.time()-t_start:.0f}s",
                  flush=True)

    json.dump(results, open(OUT / (args.out or "results.json"), "w"), indent=1)

    # --- 집계 (run_megapose와 동일 포맷 + 스펙 게이트) ---
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
                f"  | 실패: 플립 {flips}, 90°급 {quarter}, 기타 {len(fail)-flips-quarter}")
        if ok:
            line += (f"  | 성공례 중앙값 pos {np.median([x['e_pos'] for x in ok])*1e3:.1f}mm"
                     f" rot {np.median([x['e_rot'] for x in ok]):.2f}°")
        print(line)
        spec = [x for x in ok if x["e_pos"] < 0.010 and x["e_rot"] < 1.0]
        if ok:
            print(f"   스펙(10mm/1°) 충족: {len(spec)}/{len(ok)} 성공례 중")

    agg(A, "모드 A (GT 마스크)")
    agg(B, "모드 B (풀 파이프라인)")
    print("저장:", OUT / (args.out or "results.json"))


if __name__ == "__main__":
    main()
