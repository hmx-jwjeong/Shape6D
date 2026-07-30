"""외부 RGB-D 데이터(BOP/MegaPose/데모)에서 v0-geo를 돌리는 공용 하네스.

- dense depth → ML-X(80) 고정 각도 격자 희소화 (05 간과점 #2의 dense→sparse 프로토콜 구현)
- 메시 온보딩(인메모리) → S1 → S2(전 후보) → coarse → S4, 최적 후보 선택
- 임의 자세 장면이므로 직립 뷰 프루닝은 기본 OFF
"""
from __future__ import annotations

import numpy as np
import trimesh

from shape6d.common.frame_bundle import CameraIntrinsics, build_frame_bundle
from shape6d.common.types import Candidate, Proposal
from shape6d.identify.depth_match import PointToTemplateMatcher
from shape6d.identify.size_gate import SizeGate
from shape6d.onboarding.sampling import fps_indices
from shape6d.onboarding.symmetry import detect_symmetry
from shape6d.onboarding.templates import build_view_templates
from shape6d.pose.template_init import coarse_poses_from_match
from shape6d.proposal.prompt_gen import LidarPromptGenerator
from shape6d.verify.symmetry_eval import SymmetryHandler
from shape6d.verify.verifier import Verifier

# 희소화 정본은 shape6d.common.sparsify — 여기서는 재수출 (기존 임포트 호환)
from shape6d.common.sparsify import (  # noqa: F401
    MLX80_DH_DEG, MLX80_DV_DEG, sparsify_fixed_grid,
)


def onboard_mesh(mesh: trimesh.Trimesh, n_surface: int = 60000, n_master: int = 8192,
                 seed: int = 0) -> dict:
    """인메모리 온보딩 (04 하네스와 동일 규약 — 단위: m)."""
    surf, fidx = trimesh.sample.sample_surface(mesh, n_surface, seed=seed)
    surf = np.asarray(surf)
    nrm = np.asarray(mesh.face_normals)[np.asarray(fidx)]
    diam = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    r_bs = float(np.linalg.norm(surf - surf.mean(0), axis=1).max())
    tpl = build_view_templates(surf.astype(np.float32), r_bs, diam)
    # mesh-거리 경로(trimesh signed_distance)는 대형 메시에서 스캔당 수 초 — 하네스는
    # 포인트 폴백 고정(노이즈 바닥 보정 임계, tests/test_symmetry.py로 검증된 경로)
    sym = detect_symmetry(surf[:4096], mesh=None)
    return dict(surf=surf, master=surf[:n_master], master_n=nrm[:n_master], diam=diam,
                tpl=tpl, sym=sym,
                X_verify=surf[:n_master][fps_indices(surf[:n_master].astype(np.float32), 2048)])


def frame_from_points(rgb: np.ndarray, pts: np.ndarray, K: CameraIntrinsics):
    return build_frame_bundle(rgb, pts, np.ones(len(pts), np.float32),
                              np.zeros(len(pts), np.float32), K, np.eye(4))


def run_frame(onb: dict, fb, K: CameraIntrinsics, sigma: float = 0.008,
              k: int = 5, view_mask=None, voxel: float | None = None,
              max_candidates: int = 5):
    """전 게이트 통과 후보를 각각 정합·검증하고 증거 최상 후보를 반환."""
    voxel = voxel or max(0.02, 0.04 * onb["diam"])
    gen = LidarPromptGenerator(voxel=voxel, min_cluster_pts=30)
    _, clusters = gen(fb)
    if not clusters:
        return None, {"reason": "no_clusters"}

    cands = [Candidate(
        proposal=Proposal(mask=None, bbox=np.zeros(4), score=.5, source="lidar_hull",
                          cluster_id=c.id, lidar_idx=c.point_indices, n_lidar=len(c.point_indices)),
        pts=fb.lidar_points[c.point_indices], uv=fb.lidar_pixels[c.point_indices])
        for c in clusters]
    gated = SizeGate()(cands, onb["diam"])
    if not gated:
        return None, {"reason": "gated_out", "n_clusters": len(clusters)}
    gated = sorted(gated, key=lambda c: -c.pts.shape[0])[:max_candidates]

    matcher = PointToTemplateMatcher(onb["tpl"]["tdf"], onb["tpl"]["tpl_center"],
                                     onb["diam"], view_mask=view_mask, top_views_pass2=k,
                                     tpl_pts=onb["tpl"].get("tpl_pts"),
                                     tpl_pose=onb["tpl"].get("tpl_pose"))
    sym_h = SymmetryHandler(onb["sym"].sym_rots, onb["sym"].sym_axes)
    ver = Verifier(K, sym_h, sigma_lidar=sigma)

    vs, us = np.nonzero(fb.valid_mask)
    step = max(1, len(vs) // 20000)
    frame_obs = (np.stack([us[::step], vs[::step]], 1).astype(np.float64),
                 fb.sparse_depth[vs[::step], us[::step]].astype(np.float64))

    best, best_diag = None, None
    for cand in gated:
        m = matcher.match(cand.pts, k=k)
        if m.s_depth < 0.25:
            continue
        cand.scores["depth"] = m.s_depth
        hyps = coarse_poses_from_match(m, onb["tpl"]["tpl_pose"], onb["tpl"]["tpl_center"])
        res = ver(hyps, cand.pts.astype(np.float64), cand.uv, onb["master"],
                  onb["master_n"], onb["X_verify"], onb["diam"],
                  s2_scores=cand.scores, frame_obs=frame_obs)
        q = res.diag["stats"]["inlier_ratio"] * res.diag["stats"]["coverage"] \
            - res.diag["stats"]["free_viol"]
        if best is None or q > best_diag["quality"]:
            best = res
            best_diag = {"quality": q, "s_depth": m.s_depth, "n_pts": cand.pts.shape[0],
                         "n_gated": len(gated), "sym": sym_h}
    if best is None:
        return None, {"reason": "no_match", "n_gated": len(gated)}
    return best, best_diag


def overlay_png(rgb: np.ndarray, pts_model: np.ndarray, T: np.ndarray,
                K: CameraIntrinsics, path: str, obs_pts: np.ndarray | None = None):
    """정성 확인용: 추정 포즈로 모델 포인트를 투영해 RGB 위에 표시."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    if obs_pts is not None and len(obs_pts):
        u = K.fx * obs_pts[:, 0] / obs_pts[:, 2] + K.cx
        v = K.fy * obs_pts[:, 1] / obs_pts[:, 2] + K.cy
        ax.scatter(u, v, s=1.5, c="#2a78d6", alpha=0.5, linewidths=0, label="관측(희소화)")
    P = pts_model @ T[:3, :3].T + T[:3, 3]
    u = K.fx * P[:, 0] / P[:, 2] + K.cx
    v = K.fy * P[:, 1] / P[:, 2] + K.cy
    ax.scatter(u, v, s=0.6, c="#1baf7a", alpha=0.55, linewidths=0, label="모델@추정 포즈")
    ax.legend(loc="upper right", fontsize=8, markerscale=6)
    ax.set_xlim(0, K.width); ax.set_ylim(K.height, 0); ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
