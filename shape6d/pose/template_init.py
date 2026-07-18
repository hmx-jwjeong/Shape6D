"""무학습 coarse 포즈 초기화 — S2 TDF 정합의 argmax(뷰, θ, δ)를 포즈로 복원.

v0-geo 경로의 S3 대체재이자, Shape6D-PEM(M2) 도입 후에도 비학습 폴백으로 유지.

유도 (03 §5.2 프레임 규약):
  정합식:  X_view − c_tpl[v] ≈ R_z(θ)·R_align·(p − c_obs) + δ
  템플릿:  X_view = R_v·X_model + t_v          (tpl_pose = T_obj2cam, 센터링 흡수됨)
  ⇒ p = R_alignᵀ·R_z(θ)ᵀ·(R_v·X_m + t_v − c_tpl − δ) + c_obs
  ⇒ R_est = R_alignᵀ·R_zᵀ·R_v,   t_est = R_alignᵀ·R_zᵀ·(t_v − c_tpl − δ) + c_obs
"""
from __future__ import annotations

import numpy as np

from ..common.types import PoseHypothesis
from ..identify.depth_match import MatchResult, _rotz


def coarse_poses_from_match(m: MatchResult, tpl_pose: np.ndarray,
                            tpl_center: np.ndarray) -> list[PoseHypothesis]:
    """MatchResult top-k → 카메라 좌표계 포즈 가설 리스트 (점수 내림차순)."""
    out = []
    for h in m.topk:
        R_v = tpl_pose[h.view, :3, :3]
        t_v = tpl_pose[h.view, :3, 3]
        A = m.R_align.T @ _rotz(h.theta).T
        R_est = A @ R_v
        t_est = A @ (t_v - tpl_center[h.view] - h.jitter) + m.centroid
        out.append(PoseHypothesis(R=R_est.astype(np.float64), t=t_est.astype(np.float64),
                                  score=h.score, refined=False))
    return out


def pose_to_T(h: PoseHypothesis) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = h.R
    T[:3, 3] = h.t
    return T
