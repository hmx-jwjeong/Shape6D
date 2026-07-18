"""MegaPose 평가 시각 자료집 생성 — 2D(세그멘테이션·희소화·오버레이)·3D·집계 차트.

실행: uv run python eval/external_test/make_visual_report.py
산출: docs/assets_06/*.png|jpg
"""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from harness import frame_from_points, onboard_mesh, sparsify_fixed_grid
from run_megapose import DATA, preprocess_mesh, rle_to_mask

from shape6d.common.frame_bundle import CameraIntrinsics
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.identify.size_gate import SizeGate
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.proposal.prompt_gen import LidarPromptGenerator
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets_06"
OUT.mkdir(parents=True, exist_ok=True)
SIGMA = 0.008

C_OBS, C_EST, C_GT, C_MASK = "#2a78d6", "#1baf7a", "#e34948", "#eda100"
INK, INK2 = "#1a1d23", "#5a6270"
plt.rcParams.update({
    "font.family": "AppleGothic", "axes.unicode_minus": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#dde2ea", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": "#eef1f5", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titlecolor": INK,
})


def load_sample(t, key, obj_id):
    cam = json.load(t.extractfile(f"{key}.camera.json"))
    Km = np.array(cam["cam_K"]).reshape(3, 3)
    depth_m = np.array(Image.open(io.BytesIO(t.extractfile(f"{key}.depth.png").read()))
                       ).astype(np.float32) * cam["depth_scale"] / 1000.0
    rgb = np.array(Image.open(io.BytesIO(t.extractfile(f"{key}.rgb.jpg").read())))[:, :, :3]
    H, W = depth_m.shape
    K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2], width=W, height=H)
    gts = json.load(t.extractfile(f"{key}.gt.json"))
    gis = json.load(t.extractfile(f"{key}.gt_info.json"))
    gt_idx = max((i for i, g in enumerate(gts) if g["obj_id"] == obj_id),
                 key=lambda i: gis[i]["px_count_visib"])
    R_gt = np.array(gts[gt_idx]["cam_R_m2c"]).reshape(3, 3)
    t_gt = np.array(gts[gt_idx]["cam_t_m2c"]) / 1000.0
    mask = rle_to_mask(json.load(t.extractfile(f"{key}.mask_visib.json"))[str(gt_idx)])
    all_masks = json.load(t.extractfile(f"{key}.mask_visib.json"))
    return rgb, depth_m, K, R_gt, t_gt, mask, all_masks


def project(P, K):
    return K.fx * P[:, 0] / P[:, 2] + K.cx, K.fy * P[:, 1] / P[:, 2] + K.cy


def run_modeA(onb, fb, mask, K):
    idx = fb.object_points(mask, erosion_px=2)
    cand = Candidate(proposal=Proposal(mask=None, bbox=np.zeros(4), score=1.0,
                     source="gt_mask", lidar_idx=idx, n_lidar=len(idx)),
                     pts=fb.lidar_points[idx], uv=fb.lidar_pixels[idx])
    matcher = PointToTemplateMatcher(onb["tpl"]["tdf"], onb["tpl"]["tpl_center"],
                                     onb["diam"], top_views_pass2=5)
    m = matcher.match(cand.pts, k=5)
    cand.scores["depth"] = m.s_depth
    hyps = coarse_poses_from_match(m, onb["tpl"]["tpl_pose"], onb["tpl"]["tpl_center"])
    sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)
    ver = Verifier(K, sym_h, sigma_lidar=SIGMA)
    vs, us = np.nonzero(fb.valid_mask)
    step = max(1, len(vs) // 20000)
    frame_obs = (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
                 fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))
    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb["master"],
              onb["master_n"], onb["X_verify"], onb["diam"],
              s2_scores=cand.scores, frame_obs=frame_obs)
    return cand, m, hyps, res, sym_h


def case_panel(tag, title, rgb, depth_m, K, fb, cand, mask, onb, res, R_gt, t_gt, err):
    """케이스 패널: 세그멘테이션 / 희소화 / 2D 오버레이 / 3D 2뷰 / 깊이맵."""
    fig = plt.figure(figsize=(13, 7.2))
    gs = fig.add_gridspec(2, 3, hspace=0.24, wspace=0.18)
    M = onb["master"][::4]
    T = res.pose
    P_est = M @ T[:3, :3].T + T[:3, 3]
    P_gt = M @ R_gt.T + t_gt

    # (a) 세그멘테이션: GT 가시 마스크
    ax = fig.add_subplot(gs[0, 0])
    over = rgb.astype(np.float32) * 0.45
    over[mask] = over[mask] * 0.4 + np.array([237, 161, 0]) * 0.6
    ax.imshow(over.astype(np.uint8))
    ax.set_title(f"(a) GT 가시 마스크 (세그멘테이션) — {int(mask.sum())}px")
    ax.axis("off")

    # (b) 희소화 포인트 (깊이 컬러)
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow((rgb * 0.35).astype(np.uint8))
    u, v = project(fb.lidar_points, K)
    sc = ax.scatter(u, v, s=1.0, c=fb.lidar_points[:, 2], cmap="Blues_r", linewidths=0)
    plt.colorbar(sc, ax=ax, fraction=0.04, label="깊이 [m]")
    ax.set_title(f"(b) ML-X(80) 격자 희소화 — {len(fb.lidar_points)}pt (물체 {cand.pts.shape[0]}pt)")
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0); ax.axis("off")

    # (c) 2D 오버레이: 추정 vs GT
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow((rgb * 0.55).astype(np.uint8))
    u, v = project(P_gt, K); ax.scatter(u, v, s=0.7, c=C_GT, alpha=0.5, linewidths=0, label="모델@GT")
    u, v = project(P_est, K); ax.scatter(u, v, s=0.7, c=C_EST, alpha=0.6, linewidths=0, label="모델@추정")
    ax.legend(loc="upper right", fontsize=8, markerscale=8)
    ax.set_title(f"(c) 2D 오버레이 — {err}")
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0); ax.axis("off")

    # (d)(e) 3D 두 뷰
    cen = np.median(cand.pts, axis=0)
    r = max(0.7 * onb["diam"], 1.2 * np.abs(cand.pts - cen).max())
    for j, (elev, azim, name) in enumerate([(18, -60, "사선"), (8, -150, "측면")]):
        ax = fig.add_subplot(gs[1, j], projection="3d")
        for P, c, s_, lb in [(P_gt, C_GT, 1.2, "GT"), (P_est, C_EST, 1.2, "추정"),
                             (cand.pts, C_OBS, 2.5, "관측")]:
            ax.scatter(P[:, 0], P[:, 2], -P[:, 1], s=s_, c=c, alpha=0.5, linewidths=0, label=lb)
        ax.set_xlim(cen[0]-r, cen[0]+r); ax.set_ylim(cen[2]-r, cen[2]+r); ax.set_zlim(-cen[1]-r, -cen[1]+r)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xlabel("x"); ax.set_ylabel("z(깊이)"); ax.set_zlabel("높이")
        ax.set_title(f"({'de'[j]}) 3D {name}뷰")
        if j == 0:
            ax.legend(loc="upper left", fontsize=8, markerscale=4)

    # (f) 깊이맵 + 마스크 윤곽
    ax = fig.add_subplot(gs[1, 2])
    dm = np.where(depth_m > 0, depth_m, np.nan)
    im = ax.imshow(dm, cmap="Blues_r")
    plt.colorbar(im, ax=ax, fraction=0.04, label="[m]")
    ax.contour(mask, levels=[0.5], colors=[C_MASK], linewidths=1.2)
    ax.set_title("(f) 원본 dense depth + 대상 윤곽")
    ax.axis("off")

    st = res.diag["stats"]
    fig.suptitle(f"{title}\nverdict={res.verdict} · inlier={st['inlier_ratio']:.2f} · "
                 f"coverage={st['coverage']:.2f} · free_viol={st['free_viol']:.3f} · "
                 f"s_depth={cand.scores['depth']:.2f}", fontsize=12, color=INK)
    fig.savefig(OUT / f"case_{tag}.jpg", dpi=115, bbox_inches="tight",
                pil_kwargs={"quality": 82})
    plt.close(fig)


def modeB_panel(tag, rgb, fb, K, onb):
    """모드 B: S1 클러스터 시각화 — 클러터 병합의 실증."""
    gen = LidarPromptGenerator(voxel=max(0.02, 0.04 * onb["diam"]), min_cluster_pts=30)
    _, clusters = gen(fb)
    cands = [Candidate(proposal=Proposal(mask=None, bbox=np.zeros(4), score=.5,
             source="lidar_hull", cluster_id=c.id, lidar_idx=c.point_indices,
             n_lidar=len(c.point_indices)),
             pts=fb.lidar_points[c.point_indices], uv=fb.lidar_pixels[c.point_indices])
             for c in clusters]
    gated = SizeGate()(cands, onb["diam"])
    gated_ids = {c.proposal.cluster_id for c in gated}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    cmap = plt.get_cmap("tab20")
    ax = axes[0]
    ax.imshow((rgb * 0.35).astype(np.uint8))
    order = np.argsort([-len(c.point_indices) for c in clusters])
    for rank, ci in enumerate(order):
        c = clusters[ci]
        u, v = project(fb.lidar_points[c.point_indices], K)
        ax.scatter(u, v, s=1.2, color=cmap(rank % 20), linewidths=0)
    ax.set_title(f"(a) S1 voxel 클러스터 {len(clusters)}개 (색=클러스터)")
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0); ax.axis("off")

    ax = axes[1]
    ax.imshow((rgb * 0.35).astype(np.uint8))
    for c in clusters:
        u, v = project(fb.lidar_points[c.point_indices], K)
        ok = c.id in gated_ids
        ax.scatter(u, v, s=1.2, c=(C_EST if ok else "#8a8f98"),
                   alpha=0.9 if ok else 0.35, linewidths=0)
    big = clusters[int(order[0])]
    ax.set_title(f"(b) 크기 게이팅: 통과 {len(gated)} (초록) / 탈락 (회색)\n"
                 f"최대 클러스터 대각 {big.bbox_diag:.2f}m — 클러터 병합으로 상한({1.15*onb['diam']:.2f}m) 초과 시 대상 소실")
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0); ax.axis("off")
    fig.suptitle("모드 B — S1 기하 클러스터의 클러터 한계 (접촉 물체 병합)", fontsize=12, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / f"modeB_{tag}.jpg", dpi=115, bbox_inches="tight", pil_kwargs={"quality": 82})
    plt.close(fig)


def aggregate_figs(results):
    A = [(r, r["A"]) for r in results if r["A"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]  # e_rot vs ADD/D 산점 — 실패 클러스터 구조
    for verdict, c, mk in [("ACCEPT", C_OBS, "o"), ("UNCERTAIN", "#eda100", "s"), ("REJECT", "#8a8f98", "x")]:
        xs = [a["e_rot"] for _, a in A if a["verdict"] == verdict]
        ys = [a["add_rel"] for _, a in A if a["verdict"] == verdict]
        ax.scatter(xs, ys, s=14, c=c, marker=mk, label=verdict, alpha=0.75, linewidths=1)
    ax.axhline(0.1, color="#e34948", lw=1, ls="--")
    ax.text(178, 0.106, "ADD=0.1D", ha="right", fontsize=8, color="#e34948")
    for x0, lb in [(90, "90° 모호"), (180, "플립")]:
        ax.axvline(x0, color="#c9d4e3", lw=8, alpha=0.5, zorder=0)
        ax.text(x0, ax.get_ylim()[1]*0.02, lb, ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("회전 오차 [°] (sym-aware)"); ax.set_ylabel("ADD / D (log)")
    ax.set_yscale("log")
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(["1e-4", "1e-3", "0.01", "0.1", "1"])  # AppleGothic mathtext 글리프 회피
    ax.set_title("모드 A: 오차 구조 — 플립·90° 클러스터")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]  # 성공/실패 스택
    cats = ["성공\n(ADD<0.1D)", "플립\n(>150°)", "90°급", "기타 실패"]
    a_vals = [56, 18, 16, 10]
    b = [4, 17, 30, 32]
    x = np.arange(4)
    ax.bar(x-0.19, a_vals, 0.36, color=C_OBS, label="A: GT 마스크")
    ax.bar(x+0.19, b, 0.36, color="#c9d4e3", label="B: 풀(n=83)")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8.5)
    ax.set_ylabel("샘플 수"); ax.set_title("결과 분류 (100샘플)")
    ax.legend(fontsize=8, frameon=False)

    ax = axes[2]  # 성공례 정밀도 분포
    ok = [a for _, a in A if a["add_rel"] < 0.1]
    ax.hist([a["e_pos"]*1e3 for a in ok], bins=24, range=(0, 24), color=C_EST,
            edgecolor="white", linewidth=0.4)
    ax.axvline(np.median([a["e_pos"]*1e3 for a in ok]), color=INK, lw=1.2, ls="--")
    ax.set_xlabel("위치 오차 [mm]"); ax.set_ylabel("샘플 수")
    ax.set_title(f"성공 {len(ok)}건 정밀도 — 중앙값 "
                 f"{np.median([a['e_pos']*1e3 for a in ok]):.1f}mm / "
                 f"{np.median([a['e_rot'] for a in ok]):.2f}°")
    fig.tight_layout()
    fig.savefig(OUT / "aggregate.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    t = tarfile.open(DATA / "shard-000000.tar")
    results = json.load(open(Path(__file__).parent / "megapose_results" / "results.json"))
    gso = {e["obj_id"]: e["gso_id"] for e in json.load(open(DATA / "gso_models.json"))}

    def pick(pred, n=1, key=None):
        xs = [r for r in results if r["A"] and pred(r["A"])]
        if key:
            xs.sort(key=key)
        return xs[:n]

    cases = []
    cases += [("success1", "성공 — 정밀 정합", r) for r in
              [x for x in sorted(results, key=lambda r: r["A"]["e_pos"] if r["A"] else 9)
               if x["A"] and x["A"]["add_rel"] < 0.05 and x["A"]["e_rot"] < 1
               and 300 <= x["n_obj_pts"] <= 2000][:1]]
    cases += [("success2", "성공 — 대칭 물체 (연속축)", r) for r in
              [x for x in results if x["A"] and x["A"]["add_rel"] < 0.1 and x["n_cont"] > 0][:1]]
    cases += [("flip", "실패 — 플립(≈180°): 가려진 뒷면 모호성", r) for r in
              pick(lambda a: a["e_rot"] > 150 and a["verdict"] == "ACCEPT", 1)]
    cases += [("quarter", "실패 — 90°급 근사대칭 혼동", r) for r in
              pick(lambda a: 70 <= a["e_rot"] <= 110, 1)]
    cases += [("hard", "실패 — 순수 실패 (비대칭 오정합)", r) for r in
              pick(lambda a: a["add_rel"] >= 0.1 and a["e_rot"] < 60, 1)]

    for tag, title, r in cases:
        key, obj_id = r["key"], r["obj_id"]
        rgb, depth_m, K, R_gt, t_gt, mask, _ = load_sample(t, key, obj_id)
        onb = onboard_mesh(preprocess_mesh(gso[obj_id]))
        pts = sparsify_fixed_grid(depth_m, K)
        fb = frame_from_points(rgb, pts, K)
        cand, m, hyps, res, sym_h = run_modeA(onb, fb, mask, K)
        e_pos, e_rot = sym_h.sym_aware_error(res.pose[:3, :3], res.pose[:3, 3],
                                             R_gt, t_gt, onb["X_verify"])
        err = f"오차 {e_pos*1e3:.1f}mm / {e_rot:.1f}°"
        case_panel(tag, f"[{tag}] {r['name'][:40]} (D={r['diam']:.2f}m) — {title}",
                   rgb, depth_m, K, fb, cand, mask, onb, res, R_gt, t_gt, err)
        print(f"case_{tag}: {key} {r['name'][:30]} → {err} [{res.verdict}]", flush=True)

    # 모드 B 클러스터 패널 (병합 실증 케이스)
    r = cases[0][2]
    rgb, depth_m, K, R_gt, t_gt, mask, _ = load_sample(t, r["key"], r["obj_id"])
    onb = onboard_mesh(preprocess_mesh(gso[r["obj_id"]]))
    fb = frame_from_points(rgb, sparsify_fixed_grid(depth_m, K), K)
    modeB_panel(r["key"], rgb, fb, K, onb)

    aggregate_figs(results)
    print("저장:", OUT)


if __name__ == "__main__":
    main()
