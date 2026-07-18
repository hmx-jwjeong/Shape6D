"""S4 오케스트레이터 (03 문서 §7): dedupe → 가설 선별 → ICP → 재스코어 → 판정.

하드 가드: inlier<25 → ACCEPT 금지 / degenerate → 금지 / free_viol>0.05 → REJECT.
UNCERTAIN이면 차순위 가설로 1회 재시도 (coarse 가설엔 ICP 예산 보상 — 검증 [M5]).
"""
from __future__ import annotations

import numpy as np

from ..common.frame_bundle import CameraIntrinsics
from ..common.types import PoseHypothesis, VerifyResult
from .confidence import ConfidenceCalibrator, make_features
from .icp import ProjectiveICP
from .scorer import HypothesisScorer
from .symmetry_eval import SymmetryHandler


class Verifier:
    def __init__(self, K: CameraIntrinsics, sym: SymmetryHandler,
                 sigma_lidar: float = 0.008, theta_acc: float = 0.7,
                 theta_rej: float = 0.3, n_inlier_min: int = 25,
                 free_viol_max: float = 0.05,
                 calibrator: ConfidenceCalibrator | None = None):
        self.K = K
        self.sym = sym
        self.tau_z = 3.0 * sigma_lidar          # 03 §2.6
        self.scorer = HypothesisScorer(K, tau_z=self.tau_z)
        self.icp = ProjectiveICP(K, huber_delta=1.5 * sigma_lidar)
        self.theta_acc = theta_acc
        self.theta_rej = theta_rej
        self.n_inlier_min = n_inlier_min
        self.free_viol_max = free_viol_max
        self.calib = calibrator or ConfidenceCalibrator()

    def _evaluate(self, h: PoseHypothesis, obs_pts, obs_uv, X_m, N_m, X_verify,
                  d_cad, s2_scores, border) -> VerifyResult:
        R, t, icp_diag = self.icp.refine(h.R, h.t, obs_pts, X_m, N_m, d_cad)
        _, sd = self.scorer(R, t, X_verify, obs_uv, obs_pts[:, 2])
        # 주 잔차 통계 = ICP p2pl (법선 투영 — grazing 불변). 스플랫은 free_viol 전용.
        r = icp_diag.get("r_p2pl", np.zeros(0))
        inl = np.abs(r) < self.tau_z
        stats = {
            "inlier_ratio": float(inl.mean()) if r.size else 0.0,
            "rmse_inlier": float(np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else np.inf,
            "coverage": icp_diag.get("coverage", 0.0),
            "free_viol": sd["free_viol"],
            "n_inlier": int(inl.sum()),
        }
        x = make_features(stats, icp_diag, s2_scores, h.score, obs_pts.shape[0],
                          self.tau_z, border)
        p = self.calib(x)
        # 하드 가드
        if stats["free_viol"] > self.free_viol_max:
            verdict = "REJECT"
        elif stats["n_inlier"] < self.n_inlier_min or icp_diag["degenerate"]:
            verdict = "UNCERTAIN" if p >= self.theta_rej else "REJECT"
        elif p >= self.theta_acc:
            verdict = "ACCEPT"
        elif p >= self.theta_rej:
            verdict = "UNCERTAIN"
        else:
            verdict = "REJECT"
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, t
        return VerifyResult(pose=T, p_conf=p, verdict=verdict,
                            diag={"stats": stats, "scorer": sd, "icp": icp_diag, "features": x})

    def __call__(self, hyps: list[PoseHypothesis], obs_pts: np.ndarray,
                 obs_uv: np.ndarray, X_m: np.ndarray, N_m: np.ndarray,
                 X_verify: np.ndarray, d_cad: float,
                 s2_scores: dict | None = None, border: bool = False) -> VerifyResult:
        s2_scores = s2_scores or {}
        hyps = self.sym.dedupe(hyps, d_cad)
        # 가설 사전 선별: 스플랫 잔차 점수
        pre = []
        for h in hyps:
            s, _ = self.scorer(h.R, h.t, X_verify, obs_uv, obs_pts[:, 2])
            pre.append((s, h))
        pre.sort(key=lambda x: -x[0])

        result = self._evaluate(pre[0][1], obs_pts, obs_uv, X_m, N_m, X_verify,
                                d_cad, s2_scores, border)
        if result.verdict != "ACCEPT" and len(pre) > 1:
            # 차순위 재시도 1회 — coarse 가설이므로 ICP 스케줄이 이미 광역 시작 (검증 [M5])
            retry = self._evaluate(pre[1][1], obs_pts, obs_uv, X_m, N_m, X_verify,
                                   d_cad, s2_scores, border)
            if retry.p_conf > result.p_conf:
                result = retry
            result.diag["retried"] = True
        return result
