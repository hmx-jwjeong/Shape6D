"""S4-② projective point-to-plane ICP (03 문서 §7.2 — 검증 [중-5] 수정판).

- source = 관측 LiDAR(희소, EDGE_MIXED 제외), target = 포즈 적용 CAD 마스터
- 노멀은 CAD 측 (희소 관측 노멀 추정 불가·CAD 노멀은 정확 무료)
- 대응: projective association — 탐색 창을 τ_assoc에서 유도 (win = ceil(τ·fx/(Z·stride)))
- coarse-to-fine (τ_assoc, stride) 스케줄: 템플릿 양자화(~20°) 초기 오차의 수렴 반경 확보
"""
from __future__ import annotations

import numpy as np

from ..common.frame_bundle import CameraIntrinsics
from .scorer import splat_zbuffer


def _exp_so3(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    a = w / th
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


class ProjectiveICP:
    def __init__(self, K: CameraIntrinsics, huber_delta: float = 0.012,
                 schedule: list[tuple[float, int, int]] | None = None,
                 conv_rot: float = 1e-4, conv_trans: float = 5e-5,
                 max_win: int = 15, lam: float = 1e-9, w_pp: float = 0.0):
        """schedule: [(tau_assoc_m, stride, iters), ...] — 기본은 D_cad 기준 refine()에서 생성.
        huber_delta = 1.5σ_lidar (03 §2.6).
        w_pp: 접선 point-to-point 보조항 가중 (기본 0 = 비활성). E2E 실측(팔레트·σ5mm·
        master 8192): sliding 약구속의 실제 해법은 타깃 밀도 확보였고(yaw 1.04°→0.71°),
        p2p 항은 이산 샘플 스냅 바이어스로 오히려 위치를 해침(w_pp 0.25에서 pos 33mm).
        판재 단독 crop 등 진짜 sliding 퇴화 상황의 실험 옵션으로만 유지 — 기본 사용 금지."""
        self.K = K
        self.delta = huber_delta
        self.schedule = schedule
        self.conv_rot = conv_rot
        self.conv_trans = conv_trans
        self.max_win = max_win
        self.lam = lam
        self.w_pp = w_pp

    def _associate(self, P: np.ndarray, Xc: np.ndarray, tau: float, stride: int) -> np.ndarray:
        """관측점 → 모델점 인덱스 (없으면 -1). 투영 그리드 + win×win 이웃의 3D 최근접."""
        zbuf, gidx = splat_zbuffer(Xc, self.K, stride)
        Hg, Wg = gidx.shape
        z_med = float(np.median(P[:, 2]))
        win = int(np.clip(np.ceil(tau * self.K.fx / (max(z_med, 0.1) * stride)), 1, self.max_win))

        gu = np.clip((self.K.fx * P[:, 0] / P[:, 2] + self.K.cx) / stride, 0, Wg - 1).astype(np.int64)
        gv = np.clip((self.K.fy * P[:, 1] / P[:, 2] + self.K.cy) / stride, 0, Hg - 1).astype(np.int64)

        best_d2 = np.full(P.shape[0], tau * tau, np.float64)
        best_j = np.full(P.shape[0], -1, np.int64)
        for dy in range(-win, win + 1):
            for dx in range(-win, win + 1):
                yy = np.clip(gv + dy, 0, Hg - 1)
                xx = np.clip(gu + dx, 0, Wg - 1)
                j = gidx[yy, xx]
                ok = j >= 0
                if not ok.any():
                    continue
                d2 = np.full(P.shape[0], np.inf)
                d2[ok] = np.sum((P[ok] - Xc[j[ok]]) ** 2, axis=1)
                upd = d2 < best_d2
                best_d2[upd] = d2[upd]
                best_j[upd] = j[upd]
        return best_j

    def refine(self, R: np.ndarray, t: np.ndarray, P: np.ndarray,
               X_m: np.ndarray, N_m: np.ndarray, d_cad: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """반환: (R, t, diag{n_assoc, rmse, cond_H, degenerate, iters})."""
        sched = self.schedule or [
            # 템플릿 coarse 오차(뷰 17.8° + in-plane 7.5° → 변위 ~0.2D)의 수렴 반경 확보:
            # 1단계 τ는 0.25D — Huber+p2pl가 오대응을 감쇠하며 점진 축소
            (0.25 * d_cad, 4, 6),
            (0.10 * d_cad, 4, 4),
            (0.05 * d_cad, 2, 4),
            (max(0.02, 0.02 * d_cad), 2, 4),
        ]
        R, t = R.copy(), t.copy()
        diag = {"n_assoc": 0, "rmse": np.inf, "cond_H": 0.0, "degenerate": True, "iters": 0}
        H_last = None
        for tau, stride, iters in sched:
            for _ in range(iters):
                Xc = X_m @ R.T + t
                Nc = N_m @ R.T
                j = self._associate(P, Xc, tau, stride)
                ok = j >= 0
                if ok.sum() < 10:
                    break
                q, n = Xc[j[ok]], Nc[j[ok]]
                r = np.sum((P[ok] - q) * n, axis=1)
                w = np.where(np.abs(r) < self.delta, 1.0, self.delta / np.abs(r))
                J = np.concatenate([np.cross(q, n), n], axis=1)          # [M,6]
                Hm = (w[:, None, None] * J[:, :, None] * J[:, None, :]).sum(0)
                b = (w[:, None] * J * r[:, None]).sum(0)
                H_p2pl = Hm.copy()  # 퇴화 판정은 물리 관측성(p2pl)만으로 — p2p 정칙화 제외
                if self.w_pp > 0:
                    # point-to-point 보조항 (접선 성분만): 법선 성분은 p2pl과 중복이므로
                    # 제거하고, 평면 내 sliding DOF만 고정한다.
                    rv = P[ok] - q                                        # [M,3]
                    rv_t = rv - np.sum(rv * n, axis=1, keepdims=True) * n
                    rvn = np.linalg.norm(rv_t, axis=1)
                    wp = self.w_pp * np.where(rvn < 3 * self.delta, 1.0, 3 * self.delta / np.maximum(rvn, 1e-12))
                    Z = np.zeros_like(q[:, 0])
                    skew_q = np.stack([
                        np.stack([Z, -q[:, 2], q[:, 1]], -1),
                        np.stack([q[:, 2], Z, -q[:, 0]], -1),
                        np.stack([-q[:, 1], q[:, 0], Z], -1)], 1)         # [M,3,3]
                    Jpp = np.concatenate([-skew_q, np.tile(np.eye(3), (q.shape[0], 1, 1))], 2)  # [M,3,6]
                    Pt = np.eye(3)[None] - n[:, :, None] * n[:, None, :]  # 접선 사영
                    Jpp = np.einsum("mij,mjk->mik", Pt, Jpp)
                    Hm = Hm + np.einsum("m,mij,mik->jk", wp, Jpp, Jpp)
                    b = b + np.einsum("m,mij,mi->j", wp, Jpp, rv_t)
                xi = np.linalg.solve(Hm + self.lam * np.eye(6), b)
                # trust region: 소수·오염 대응에서 GN 과도 스텝 방지 (스케줄이 반복을 보장)
                s = min(1.0,
                        0.1 / max(np.linalg.norm(xi[:3]), 1e-12),        # 회전 ≤ ~5.7°/iter
                        0.5 * tau / max(np.linalg.norm(xi[3:]), 1e-12))  # 병진 ≤ τ/2 /iter
                xi = xi * s
                dR = _exp_so3(xi[:3])
                R, t = dR @ R, dR @ t + xi[3:]
                diag["iters"] += 1
                H_last = H_p2pl
                diag["n_assoc"] = int(ok.sum())
                diag["rmse"] = float(np.sqrt(np.mean(r ** 2)))
                if np.linalg.norm(xi[:3]) < self.conv_rot and np.linalg.norm(xi[3:]) < self.conv_trans:
                    break
        if H_last is not None:
            ev = np.linalg.eigvalsh(H_last)
            cond = float(ev[0] / max(ev[-1], 1e-30))
            diag["cond_H"] = cond
            diag["degenerate"] = cond < 1e-6   # 평면 조각 등 6-DoF 미구속 (03 §7.2)

        # 최종 p2pl 잔차 통계 — 검증(S4-③)의 주 신호. 깊이(z-차이) 잔차는 grazing
        # 표면에서 픽셀당 수십 mm가 정상이라 부적합(설계 수정: E2E 실측 근거).
        tau_f = sched[-1][0]
        Xc, Nc = X_m @ R.T + t, N_m @ R.T
        j = self._associate(P, Xc, tau_f, sched[-1][1])
        ok = j >= 0
        r_fin = np.sum((P[ok] - Xc[j[ok]]) * Nc[j[ok]], axis=1) if ok.any() else np.zeros(0)
        diag["r_p2pl"] = r_fin
        diag["coverage"] = float(ok.mean()) if len(ok) else 0.0
        diag["n_obs"] = int(len(P))
        return R, t, diag
