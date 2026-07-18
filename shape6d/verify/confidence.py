"""S4-③ 신뢰도 보정 — 10-d 특징 로지스틱 (03 문서 §7.3).

캘리브레이션(M3-3) 전에는 휴리스틱 초기 가중치로 동작하되, fit()으로 교체 가능.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "inlier_ratio", "rmse_over_tau", "coverage", "free_viol",
    "s2_depth", "s2_sem", "s3_match", "log10_n_obs", "log10_cond_h", "border",
]

# 캘리브레이션 전 휴리스틱 초기값 (부호만 물리적으로 타당하게, M3-3에서 fit으로 교체)
_W0 = np.array([4.0, -2.0, 2.0, -8.0, 2.0, 0.5, 1.0, 0.5, 0.3, -0.3])
_B0 = -3.0


def make_features(scorer_diag: dict, icp_diag: dict, s2_scores: dict,
                  s3_match: float, n_obs: int, tau_z: float, border: bool) -> np.ndarray:
    return np.array([
        scorer_diag.get("inlier_ratio", 0.0),
        min(scorer_diag.get("rmse_inlier", np.inf) / tau_z, 2.0),
        scorer_diag.get("coverage", 0.0),
        scorer_diag.get("free_viol", 1.0),
        s2_scores.get("depth", 0.0),
        s2_scores.get("sem", 0.0),
        s3_match,
        np.log10(max(n_obs, 1)),
        np.log10(max(icp_diag.get("cond_H", 1e-30), 1e-30)) + 10.0,  # 대략 0 근방으로 시프트
        1.0 if border else 0.0,
    ], dtype=np.float64)


class ConfidenceCalibrator:
    def __init__(self, w: np.ndarray | None = None, b: float | None = None):
        self.w = _W0.copy() if w is None else np.asarray(w, np.float64)
        self.b = _B0 if b is None else float(b)
        self.calibrated = w is not None

    def __call__(self, x: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.w @ x + self.b))))

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> "ConfidenceCalibrator":
        """IRLS(Newton) 로지스틱 학습 — M3-3 캘리브레이션 CLI에서 사용."""
        n, d = X.shape
        Xa = np.concatenate([X, np.ones((n, 1))], axis=1)
        w = np.zeros(d + 1)
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(Xa @ w)))
            g = Xa.T @ (p - y) + l2 * w
            S = np.clip(p * (1 - p), 1e-6, None)
            H = (Xa * S[:, None]).T @ Xa + l2 * np.eye(d + 1)
            step = np.linalg.solve(H, g)
            w -= step
            if np.linalg.norm(step) < 1e-8:
                break
        return ConfidenceCalibrator(w=w[:-1], b=float(w[-1]))

    def save(self, path: str):
        np.savez(path, w=self.w, b=self.b)

    @staticmethod
    def load(path: str) -> "ConfidenceCalibrator":
        d = np.load(path)
        return ConfidenceCalibrator(w=d["w"], b=float(d["b"]))
