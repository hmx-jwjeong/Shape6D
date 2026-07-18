"""S2-① 메트릭 크기 게이팅 (03 문서 §5.1) — 무료·RGB 무관 1차 판별."""
from __future__ import annotations

import numpy as np

from ..common.types import Candidate


class SizeGate:
    def __init__(self, tau_up: float = 0.15, beta_occ: float = 0.35,
                 beta_occ_relaxed: float = 0.25, n_min: int = 30):
        self.tau_up = tau_up
        self.beta_occ = beta_occ
        self.beta_occ_relaxed = beta_occ_relaxed
        self.n_min = n_min

    @staticmethod
    def robust_extent(pts: np.ndarray) -> float:
        """축별 5–95 percentile 트리밍 bbox 대각 (경계 bleed·플라잉 포인트 제거)."""
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        return float(np.linalg.norm(hi - lo))

    def __call__(self, cands: list[Candidate], d_cad: float) -> list[Candidate]:
        out = []
        sole = len(cands) == 1
        for c in cands:
            n = c.pts.shape[0]
            occluded = c.proposal.truncated or ("occluded" in c.flags)
            beta = self.beta_occ_relaxed if occluded else self.beta_occ
            if n < self.n_min:
                if sole:  # 유일 후보 구제 경로 (03 §2.5) — low_geo 강제
                    c.flags.add("low_geo")
                    c.scores["size"] = 0.0
                    out.append(c)
                continue
            d_obs = self.robust_extent(c.pts)
            if d_obs > (1 + self.tau_up) * d_cad:   # 상한 위반 = hard reject
                continue
            if d_obs < beta * d_cad:
                continue
            c.scores["size"] = float(np.clip(d_obs / d_cad, 0.0, 1.0))
            if c.proposal.truncated:
                c.flags.add("border")
            out.append(c)
        return out
