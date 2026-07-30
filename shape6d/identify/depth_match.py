"""S2-② point-to-template TDF 정합 (03 문서 §5.2 — 검증 [상-2]/[상-3] 수정판).

수정 반영:
- TDF 조회 전 centroid 시선 → z 정렬 회전 (화면 주변부 오프센터 보정)
- pass2: 상위 뷰 in-plane 15° 재탐색 + 병진 jitter ±2 voxel

argmax (뷰, θ, δ)는 곧 무학습 coarse 포즈의 파라미터다 → pose/template_init.py가 소비.
numpy 레퍼런스 구현 — GPU/TRT 배치 gather 이식은 M1 (연산 구조 동일).
"""
from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class MatchHypo:
    score: float                # 최종 점수 = s_oneway × coverage (coverage 미가용 시 s_oneway)
    view: int
    theta: float
    jitter: np.ndarray          # [3] 병진 jitter (템플릿 프레임)
    s_oneway: float = -1.0      # 관측→템플릿 편도 점수 (구 score, 진단용)
    coverage: float = 1.0       # 템플릿 측 재현율 (A-4 — tpl_pts 미제공 시 1.0)


@dataclass
class MatchResult:
    s_depth: float
    best: MatchHypo
    topk: list[MatchHypo]
    R_align: np.ndarray         # 시선축 정렬 회전
    centroid: np.ndarray        # 관측 median 센터
    per_view: np.ndarray = field(default=None)  # [V] pass1 최고점 (진단)


class PointToTemplateMatcher:
    def __init__(self, tdf: np.ndarray, tpl_center: np.ndarray, diameter: float,
                 k_inplane_pass1: int = 12, k_inplane_pass2: int = 24,
                 top_views_pass2: int = 3, jitter_vox: int = 2,
                 view_mask: np.ndarray | None = None,
                 tpl_pts: np.ndarray | None = None,
                 tpl_pose: np.ndarray | None = None,
                 coverage_tau_rel: float = TDF_TRUNC):
        """tdf [V,48³] f16, tpl_center [V,3] (뷰 프레임 median), view_mask [V] bool
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
        # A-4 coverage: 템플릿 측 재현율. tpl_pts(모델계 [V,P,3]) + tpl_pose([V,4,4],
        # 센터링 흡수됨)가 있으면 뷰별 "TDF 도메인 좌표" 포인트를 선계산해 둔다.
        # 미제공 시 coverage=1.0 (구 동작 유지 — 단방향 점수).
        self.tpl_view_pts: np.ndarray | None = None
        self.cov_tau = float(coverage_tau_rel) * self.D
        if tpl_pts is not None and tpl_pose is not None:
            Rv = tpl_pose[:, :3, :3]                                  # [V,3,3]
            tv = tpl_pose[:, :3, 3]                                   # [V,3]
            vp = np.einsum("vij,vpj->vpi", Rv, tpl_pts) + tv[:, None]  # 모델계→뷰계
            self.tpl_view_pts = (vp - tpl_center[:, None]).astype(np.float32)

    def _coverage(self, q: np.ndarray, v: int) -> float:
        """템플릿 뷰 v의 가시 포인트 중 관측 q(TDF 도메인)와 cov_tau 이내인 비율.

        단방향 s_oneway는 '작은 점 뭉치'에 만점을 주는 취약점이 있다(10 §5 A-4) —
        템플릿 재현율을 곱해 관측이 템플릿 표면을 얼마나 덮는지를 점수에 반영한다.
        """
        if self.tpl_view_pts is None:
            return 1.0
        T = self.tpl_view_pts[v]                                       # [P,3]
        # P(≤512)×N 거리 — N이 크면 청크 (순수 numpy, A1: 보간 없음)
        step = 4096
        dmin = np.full(len(T), np.inf, dtype=np.float32)
        qf = q.astype(np.float32)
        for s0 in range(0, len(qf), step):
            d = np.linalg.norm(T[:, None, :] - qf[None, s0:s0 + step, :], axis=-1)
            dmin = np.minimum(dmin, d.min(axis=1))
        return float((dmin < self.cov_tau).mean())

    def _score_once(self, q: np.ndarray, v: int) -> float:
        d = tdf_lookup(self.tdf[v], q, self.D)
        trunc = TDF_TRUNC * self.D
        return float(np.mean(np.maximum(0.0, 1.0 - d / trunc)))

    def match(self, pts_cam: np.ndarray, k: int = 3) -> MatchResult:
        cen = np.median(pts_cam, axis=0)
        R_a = _ray_align_rotation(cen)
        p = (pts_cam - cen) @ R_a.T  # 뷰잉-레이 프레임 정렬

        # pass1: 전 뷰 × 거친 in-plane (30°)
        s1 = np.full((self.n_views, len(self.th1)), -1.0, dtype=np.float32)
        for v in range(self.n_views):
            if not self.view_mask[v]:
                continue
            for j, th in enumerate(self.th1):
                s1[v, j] = self._score_once(p @ _rotz(th).T, v)
        per_view = s1.max(axis=1)
        top = np.argsort(per_view)[::-1][: self.top_views]

        # pass2: 상위 뷰 × 촘촘한 in-plane (15°) × 병진 jitter — 뷰별 최적 (θ, δ) 산출
        hypos: list[MatchHypo] = []
        for v in top:
            if not self.view_mask[v] or per_view[v] < 0:
                continue
            s_th = [self._score_once(p @ _rotz(t).T, int(v)) for t in self.th2]
            th_best = float(self.th2[int(np.argmax(s_th))])
            q0 = p @ _rotz(th_best).T
            best_s, best_d = -1.0, np.zeros(3)
            for dj in self.jitters:
                s = self._score_once(q0 + dj, int(v))
                if s > best_s:
                    best_s, best_d = s, dj
            cov = self._coverage(q0 + best_d, int(v))
            hypos.append(MatchHypo(score=best_s * cov, view=int(v), theta=th_best,
                                   jitter=best_d.copy(), s_oneway=best_s, coverage=cov))

        hypos.sort(key=lambda h: h.score, reverse=True)
        best = hypos[0] if hypos else MatchHypo(0.0, 0, 0.0, np.zeros(3))
        return MatchResult(s_depth=best.score, best=best, topk=hypos[:k],
                           R_align=R_a, centroid=cen, per_view=per_view)

    def score(self, pts_cam: np.ndarray) -> tuple[float, dict]:
        """관측 포인트 [N,3] → S_depth ∈ [0,1] + 진단 (기존 API 유지)."""
        m = self.match(pts_cam, k=1)
        return m.s_depth, {"best_view": m.best.view, "pass1_max": float(m.per_view.max())}
