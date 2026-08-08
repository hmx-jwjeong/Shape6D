"""docs/08 MR6D 평가 보고서 자산 생성 — 집계 차트 + 고선명 오버레이(추정 vs GT).

오버레이는 대표 4프레임의 모드-A 포즈를 결정론적으로 재계산해
추정(초록)·GT(빨강)·희소 관측(파랑)을 굵은 점으로 함께 그린다 —
겹치면 성공, 어긋나면 실패가 육안으로 즉시 구분된다.

산출: docs/assets_08/{fig_scatter.png, fig_scene.png, fig_visib.png, ov_*.jpg}
실행: uv run python eval/external_test/make_mr6d_report_assets.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))
from harness import frame_from_points, onboard_mesh, sparsify_fixed_grid

from shape6d.common.frame_bundle import CameraIntrinsics
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data" / "mr6d"
RES = Path(__file__).parent / "mr6d_results"
OUT = ROOT / "docs" / "assets_08"
OUT.mkdir(parents=True, exist_ok=True)
SIGMA = 0.008

rs = json.load(open(RES / "results.json"))
A = [r for r in rs if r["A"]]
ep = np.array([r["A"]["e_pos"] for r in A]) * 1e3
er = np.array([r["A"]["e_rot"] for r in A])
vis = np.array([r["visib"] for r in A])
acc = np.array([r["A"]["verdict"] == "ACCEPT" for r in A])
ok = ep < 145
scene = np.array([r["scene"] for r in A])

C_OK, C_BAD, C_ACC = "#0a7d43", "#b4232a", "#0b5fff"

# ---------------------------------------------------------------- 집계 차트
fig, ax = plt.subplots(figsize=(6.4, 4.2))
add_rel = ep / 1449.4
ax.scatter(add_rel[ok], er[ok], s=26, c=C_OK, alpha=0.75, label="success (ADD<0.1D)")
ax.scatter(add_rel[~ok & acc], er[~ok & acc], s=26, c=C_ACC, alpha=0.75,
           label="fail but ACCEPT (false accept)")
ax.scatter(add_rel[~ok & ~acc], er[~ok & ~acc], s=26, c=C_BAD, alpha=0.75,
           label="fail, REJECT")
for y in (90, 180):
    ax.axhline(y, color="#999", lw=0.7, ls="--")
ax.set_xlabel("ADD / D")
ax.set_ylabel("rotation error (deg, sym-aware)")
ax.set_title("MR6D mode A: rotation error vs ADD (n=150)")
ax.legend(fontsize=8, loc="center right")
ax.set_xlim(0, max(add_rel) * 1.05)
fig.tight_layout()
fig.savefig(OUT / "fig_scatter.png", dpi=130)

scenes = sorted(set(scene))
labels = {"000000": "000000\nstacked 2", "000001": "000001\nfloor",
          "000002": "000002\nfloor", "000005": "000005\nupright+occl",
          "000006": "000006\nLOADED"}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
w = 0.38
xs = np.arange(len(scenes))
n_ok = [int(ok[scene == s].sum()) for s in scenes]
n_all = [int((scene == s).sum()) for s in scenes]
n_flip = [int(((~ok) & (er > 150))[scene == s].sum()) for s in scenes]
ax.bar(xs - w / 2, [o / n for o, n in zip(n_ok, n_all)], w, color=C_OK, label="success rate")
ax.bar(xs + w / 2, [f / n for f, n in zip(n_flip, n_all)], w, color=C_BAD, label="flip rate")
ax.set_xticks(xs, [labels[s] for s in scenes], fontsize=8)
ax.set_ylabel("fraction")
ax.set_title("per-scene success / flip (mode A)")
for x, o, n in zip(xs, n_ok, n_all):
    ax.text(x - w / 2, o / n + 0.02, f"{o}/{n}", ha="center", fontsize=8)
ax.legend(fontsize=8)
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(OUT / "fig_scene.png", dpi=130)

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.scatter(vis[ok], ep[ok], s=24, c=C_OK, alpha=0.75, label="success")
ax.scatter(vis[~ok], np.clip(ep[~ok], 0, 600), s=24, c=C_BAD, alpha=0.7, label="fail (clip 600)")
ax.axhline(145, color="#999", lw=0.7, ls="--")
ax.text(0.31, 152, "ADD 0.1D = 145mm", fontsize=7.5, color="#666")
ax.axvline(0.8, color=C_ACC, lw=0.9, ls=":")
ax.text(0.805, 500, "visib 0.8\n-> 82% success\n24mm / 1.75deg", fontsize=8, color=C_ACC)
ax.set_xlabel("GT visible fraction")
ax.set_ylabel("position error (mm)")
ax.set_title("visibility governs accuracy")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "fig_visib.png", dpi=130)

# ---------------------------------------------------- 고선명 오버레이 (재계산)
mesh = trimesh.load(DATA / "models" / "obj_000001.ply")
mesh.apply_scale(0.001)
onb = onboard_mesh(mesh)
sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)
edges = mesh.copy()

PICKS = [  # (파일명, scene, im_id, gt_idx)
    ("ov_success.jpg", "000000", "5", 0),
    ("ov_grazing.jpg", "000000", "0", 0),
    ("ov_loaded.jpg", "000006", "0", 0),
    ("ov_upright.jpg", "000005", "0", 0),
]


def solve_mode_a(sc: str, im_id: str, gt_idx: int):
    d = DATA / "val" / sc
    cam = json.load(open(d / "scene_camera.json"))[im_id]
    o = json.load(open(d / "scene_gt.json"))[im_id][gt_idx]
    Km = np.array(cam["cam_K"]).reshape(3, 3)
    depth_m = (np.array(Image.open(d / "depth" / f"{int(im_id):06d}.png"))
               .astype(np.float32) * cam["depth_scale"] / 1000.0)
    rgb = np.array(Image.open(d / "rgb" / f"{int(im_id):06d}.jpg"))[:, :, :3]
    H, W = depth_m.shape
    K = CameraIntrinsics(fx=Km[0, 0], fy=Km[1, 1], cx=Km[0, 2], cy=Km[1, 2],
                         width=W, height=H)
    mask = np.array(Image.open(d / "mask_visib" / f"{int(im_id):06d}_{gt_idx:06d}.png")) > 0
    pts = sparsify_fixed_grid(depth_m, K)
    fb = frame_from_points(rgb, pts, K)
    idx = fb.object_points(mask, erosion_px=2)
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
    vs_, us_ = np.nonzero(fb.valid_mask)
    step = max(1, len(vs_) // 20000)
    frame_obs = (np.stack([us_[::step], vs_[::step]], 1).astype(np.float64),
                 fb.sparse_depth[vs_[::step], us_[::step]].astype(np.float64))
    res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb["master"],
              onb["master_n"], onb["X_verify"], onb["diam"],
              s2_scores=cand.scores, frame_obs=frame_obs)
    T_gt = np.eye(4)
    T_gt[:3, :3] = np.array(o["cam_R_m2c"]).reshape(3, 3)
    T_gt[:3, 3] = np.array(o["cam_t_m2c"], dtype=np.float64) / 1000.0
    return rgb, K, res.pose, T_gt, cand.pts


def project(P: np.ndarray, K: CameraIntrinsics):
    u = K.fx * P[:, 0] / P[:, 2] + K.cx
    v = K.fy * P[:, 1] / P[:, 2] + K.cy
    return u, v


def render(path: Path, rgb, K, T_est, T_gt, obs):
    model = onb["master"][::3]
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.imshow((rgb.astype(np.float32) * 0.55 + 255 * 0.28).clip(0, 255)
              .astype(np.uint8))                       # 배경을 밝게 눌러 점 대비 확보
    # GT (빨강) 먼저 — 성공 시 초록이 그 위를 덮어 "초록만 보임 = 정답"
    P = model @ T_gt[:3, :3].T + T_gt[:3, 3]
    u, v = project(P, K)
    ax.scatter(u, v, s=7, c="#e0342b", alpha=0.9, linewidths=0, label="model @ GT pose")
    P = model @ T_est[:3, :3].T + T_est[:3, 3]
    u, v = project(P, K)
    ax.scatter(u, v, s=7, c="#12d97c", alpha=0.85, linewidths=0, label="model @ estimate")
    u, v = project(obs, K)
    ax.scatter(u, v, s=5, c="#1b6bff", alpha=0.9, marker=".", linewidths=0,
               label="sparse obs (input)")
    leg = ax.legend(loc="upper right", fontsize=10, markerscale=2.2, framealpha=0.95)
    ax.set_xlim(0, K.width)
    ax.set_ylim(K.height, 0)
    ax.axis("off")
    fig.tight_layout(pad=0.1)
    fig.savefig(str(path).replace(".jpg", ".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    img = Image.open(str(path).replace(".jpg", ".png")).convert("RGB")
    img.save(path, quality=86)
    Path(str(path).replace(".jpg", ".png")).unlink()


for fname, sc, im_id, gi in PICKS:
    rgb, K, T_est, T_gt, obs = solve_mode_a(sc, im_id, gi)
    render(OUT / fname, rgb, K, T_est, T_gt, obs)
    print(fname, "done")

print("assets →", OUT)
