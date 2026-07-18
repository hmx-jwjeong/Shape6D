"""S2-② point-to-template TDF 정합 (03 문서 §5.2 — 검증 [상-2]/[상-3] 수정판).

수정 반영:
- TDF 조회 전 centroid 시선 → z 정렬 회전 (화면 주변부 오프센터 보정)
- pass2: 상위 뷰 in-plane 15° 재탐색 + 병진 jitter ±2 voxel

numpy 레퍼런스 구현 — GPU/TRT 배치 gather 이식은 M1 (연산 구조 동일).
"""
from __future__ import annotations

import numpy as np

from ..onboarding.templates import TDF_HALF_EXTENT, TDF_RES, TDF_TRUNC, tdf_lookup


def _ray_align_rotation(centroid: np.ndarray) -> np.ndarray:
    """관측 centroid 시선 방향 → 카메라 z축 정렬 회전 (Rodrigues, 검증 [상-2])."""
    ray = centroid / (np.linalg.norm(centroid) + 1e-12)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(ray, z)
    s, c = np.linalg.norm(v), float(np.dot(ray, z))
    if s < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def _rotz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class PointToTemplateMatcher:
    def __init__(self, tdf: np.ndarray, tpl_center: np.ndarray, diameter: float,
                 k_inplane_pass1: int = 12, k_inplane_pass2: int = 24,
                 top_views_pass2: int = 3, jitter_vox: int = 2,
                 view_mask: np.ndarray | None = None):
        """tdf [42,48,48,48] f16, tpl_center [42,3] (뷰 프레임 median), view_mask [42] bool
        (팔레트류 수평 밴드 프루닝 옵션, 03 §1.4e)."""
        self.tdf = tdf
        self.tpl_center = tpl_center
        self.D = float(diameter)
        self.th1 = np.linspace(0, 2 * np.pi, k_inplane_pass1, endpoint=False)
        self.th2 = np.linspace(0, 2 * np.pi, k_inplane_pass2, endpoint=False)
        self.top_views = top_views_pass2
        vox = 2 * TDF_HALF_EXTENT * self.D / (TDF_RES - 1)
        j = np.arange(-jitter_vox, jitter_vox + 1) * vox
        self.jitters = np.stack(np.meshgrid(j, j, j, indexing="ij"), -1).reshape(-1, 3)
        self.n_views = tdf.shape[0]
        self.view_mask = view_mask if view_mask is not None else np.ones(self.n_views, bool)

    def _score_once(self, q: np.ndarray, v: int) -> float:
        d = tdf_lookup(self.tdf[v], q, self.D)
        trunc = TDF_TRUNC * self.D
        return float(np.mean(np.maximum(0.0, 1.0 - d / trunc)))

    def score(self, pts_cam: np.ndarray) -> tuple[float, dict]:
        """관측 포인트 [N,3] (카메라계) → S_depth ∈ [0,1] + 진단."""
        cen = np.median(pts_cam, axis=0)
        p = (pts_cam - cen) @ _ray_align_rotation(cen).T  # 뷰잉-레이 프레임 정렬

        # pass1: 전 뷰 × 거친 in-plane
        s1 = np.zeros((self.n_views, len(self.th1)), dtype=np.float32)
        for v in range(self.n_views):
            if not self.view_mask[v]:
                continue
            for j, th in enumerate(self.th1):
                s1[v, j] = self._score_once(p @ _rotz(th).T, v)
        top = np.argsort(s1.max(axis=1))[::-1][: self.top_views]

        # pass2: 상위 뷰 × 촘촘한 in-plane × 병진 jitter (가림 센터 편이 흡수)
        best, best_v = float(s1.max()), int(top[0])
        for v in top:
            if not self.view_mask[v]:
                continue
            coarse_best = self.th2[np.argmax([self._score_once(p @ _rotz(t).T, v) for t in self.th2])]
            q0 = p @ _rotz(coarse_best).T
            for dj in self.jitters:
                s = self._score_once(q0 + dj, v)
                if s > best:
                    best, best_v = s, int(v)
        return best, {"best_view": best_v, "pass1_max": float(s1.max())}


def smoke_test_score(matcher: PointToTemplateMatcher, tpl_pts_view: np.ndarray) -> float:
    """M2 착수 전 필수 게이트 (검증 [상-3]): 정답 포즈 주입 시 S_depth 분포 확인용 헬퍼 —
    템플릿 자기 자신의 가시 포인트를 관측으로 넣었을 때의 점수."""
    return matcher.score(tpl_pts_view + np.array([0, 0, 1.0]))[0]
