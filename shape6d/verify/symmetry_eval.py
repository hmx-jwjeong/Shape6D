"""S4-③ 대칭군 등가 처리 (03 문서 §7.3): canonicalize·dedupe·sym-aware 오차.

정책 기본값: all_equivalent — 대칭 등가 포즈는 전부 정답 취급 (config symmetry.policy).
"""
from __future__ import annotations

import numpy as np


def _geodesic_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    c = (np.trace(R1 @ R2.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w < 1e-9:
        # 대각 지배 축 처리
        i = int(np.argmax(np.diag(R)))
        q = np.zeros(4)
        q[1 + i] = np.sqrt(max(0.0, 1 + 2 * R[i, i] - np.trace(R))) / 2
        return q
    x = (R[2, 1] - R[1, 2]) / (4 * w)
    y = (R[0, 2] - R[2, 0]) / (4 * w)
    z = (R[1, 0] - R[0, 1]) / (4 * w)
    return np.array([w, x, y, z])


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class SymmetryHandler:
    def __init__(self, sym_rots: np.ndarray, sym_axes: np.ndarray):
        """sym_rots [S,3,3] (항등 포함), sym_axes [A,3] — S0 캐시 정본 소비."""
        self.S = sym_rots.astype(np.float64)
        self.axes = sym_axes.astype(np.float64).reshape(-1, 3)

    def remove_twist(self, R: np.ndarray, axis_model: np.ndarray) -> np.ndarray:
        """연속 대칭축 twist 제거 (swing-twist 분해): R = R_swing·R_twist(axis)."""
        q = _quat_from_R(R)
        a = axis_model / np.linalg.norm(axis_model)
        proj = np.dot(q[1:], a)
        tw = np.array([q[0], *(proj * a)])
        nrm = np.linalg.norm(tw)
        if nrm < 1e-9:
            return R  # 180° 특이 — twist 부정, 그대로 둠
        R_twist = _R_from_quat(tw / nrm)
        return R @ R_twist.T

    def canonicalize(self, R: np.ndarray) -> np.ndarray:
        """연속축 twist 제거 후, 이산군에서 항등에 최근접한 대표 선택."""
        Rc = R.copy()
        for a in self.axes:
            Rc = self.remove_twist(Rc, a)
        if len(self.S) > 1:
            g = int(np.argmin([_geodesic_deg(Rc @ S, np.eye(3)) for S in self.S]))
            Rc = Rc @ self.S[g]
        return Rc

    def sym_distance_deg(self, R1: np.ndarray, R2: np.ndarray) -> float:
        """대칭 인식 회전 거리: min_g geodesic(R1·S_g, R2). 연속축은 twist 제거 후 비교."""
        A1, A2 = R1, R2
        for a in self.axes:
            A1 = self.remove_twist(A1, a)
            A2 = self.remove_twist(A2, a)
        return min(_geodesic_deg(A1 @ S, A2) for S in self.S)

    def dedupe(self, hyps: list, d_cad: float, ang_deg: float = 15.0,
               t_rel: float = 0.05) -> list:
        """가설 리스트(점수순) → 대칭 등가 중복 병합. refined 가설이 항상 생존 우선 (03 §7.1)."""
        ordered = sorted(hyps, key=lambda h: (not h.refined, -h.score))
        kept = []
        for h in ordered:
            dup = any(
                self.sym_distance_deg(h.R, k.R) < ang_deg
                and np.linalg.norm(h.t - k.t) < t_rel * d_cad
                for k in kept
            )
            if not dup:
                kept.append(h)
        return kept

    def sym_aware_error(self, R_est, t_est, R_gt, t_gt, X_m: np.ndarray) -> tuple[float, float]:
        """(평균 포인트 오차 [m], 대칭 인식 회전 오차 [deg]) — ADD-S/MSSD식 평가·라벨용."""
        gt = X_m @ R_gt.T + t_gt
        errs = [float(np.mean(np.linalg.norm(X_m @ (R_est @ S).T + t_est - gt, axis=1)))
                for S in self.S]
        # 연속축 무한군 근사: 12분할 추가 평가
        for a in self.axes:
            for k in range(1, 12):
                th = 2 * np.pi * k / 12
                c, s = np.cos(th), np.sin(th)
                K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
                Rk = np.eye(3) + s * K + (1 - c) * (K @ K)
                errs.append(float(np.mean(np.linalg.norm(X_m @ (R_est @ Rk).T + t_est - gt, axis=1))))
        return min(errs), self.sym_distance_deg(R_est, R_gt)
