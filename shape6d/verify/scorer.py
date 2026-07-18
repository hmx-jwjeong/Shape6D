"""S4-① 가설 스코어러 — 투영 스플랫 잔차 (03 문서 §7.1, 렌더링 없음).

관측이 희소하므로 비교 지점은 LiDAR 유효 픽셀뿐. 부호 규약:
  r = obs_z − model_z.  r > +τ = free-space 위반(오포즈 강증거), r < −τ = 가림/오염(관대).
"""
from __future__ import annotations

import numpy as np

from ..common.frame_bundle import CameraIntrinsics


def splat_zbuffer(pts_cam: np.ndarray, K: CameraIntrinsics, stride: int):
    """모델 포인트 → stride 그리드 z-buffer. 반환 (zbuf [Hg,Wg] inf=미점유, gidx [Hg,Wg] i64 -1=미점유)."""
    Hg, Wg = K.height // stride, K.width // stride
    z = pts_cam[:, 2]
    ok = z > 1e-6
    u = (K.fx * pts_cam[ok, 0] / z[ok] + K.cx) / stride
    v = (K.fy * pts_cam[ok, 1] / z[ok] + K.cy) / stride
    ui, vi = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    inb = (ui >= 0) & (ui < Wg) & (vi >= 0) & (vi < Hg)
    src = np.nonzero(ok)[0][inb]
    pid = vi[inb] * Wg + ui[inb]
    order = np.lexsort((z[src], pid))
    pid_s, src_s = pid[order], src[order]
    first = np.ones(pid_s.size, bool)
    first[1:] = pid_s[1:] != pid_s[:-1]
    zbuf = np.full(Hg * Wg, np.inf, np.float32)
    gidx = np.full(Hg * Wg, -1, np.int64)
    zbuf[pid_s[first]] = z[src_s[first]]
    gidx[pid_s[first]] = src_s[first]
    return zbuf.reshape(Hg, Wg), gidx.reshape(Hg, Wg)


def _local_range3(zbuf: np.ndarray) -> np.ndarray:
    """점유 셀만의 3×3 이웃 깊이 범위(max−min) — grazing 표면 적응 임계용."""
    H, W = zbuf.shape
    fin = np.isfinite(zbuf)
    mn = np.where(fin, zbuf, np.inf).copy()
    mx = np.where(fin, zbuf, -np.inf).copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src = (slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx)))
            dst = (slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx)))
            np.minimum(mn[dst], np.where(fin, zbuf, np.inf)[src], out=mn[dst])
            np.maximum(mx[dst], np.where(fin, zbuf, -np.inf)[src], out=mx[dst])
    rng = mx - mn
    rng[~np.isfinite(rng)] = 0.0
    return np.clip(rng, 0.0, None)


def _erode1(m: np.ndarray) -> np.ndarray:
    """1셀 침식 (4-이웃)."""
    out = m.copy()
    out[1:, :] &= m[:-1, :]; out[:-1, :] &= m[1:, :]
    out[:, 1:] &= m[:, :-1]; out[:, :-1] &= m[:, 1:]
    out[0, :] = False; out[-1, :] = False; out[:, 0] = False; out[:, -1] = False
    return out


def _fill_holes3(zbuf: np.ndarray) -> np.ndarray:
    """빈 셀만 이웃 min으로 채움 — 점유 셀을 덮으면 실루엣 1셀 팽창으로
    경사면에서 가짜 free-space 위반이 생긴다 (교차 검증 [중-5] 부기 반영)."""
    H, W = zbuf.shape
    nb = np.full((H, W), np.inf, np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            src = (slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx)))
            dst = (slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx)))
            np.minimum(nb[dst], zbuf[src], out=nb[dst])
    out = zbuf.copy()
    empty = ~np.isfinite(zbuf)
    out[empty] = nb[empty]
    return out


class HypothesisScorer:
    def __init__(self, K: CameraIntrinsics, stride: int = 4, tau_z: float = 0.024,
                 gamma: float = 1.0):
        """stride 4 기본: X_verify ~1024pt가 대형 물체(팔레트급)에서도 그리드를 조밀
        점유해야 홀 채움發 가짜 free_viol이 없다 (셀 ≈ 13mm@3m — free-space 검사에 충분)."""
        self.K = K
        self.stride = stride
        self.tau_z = tau_z      # = 3σ_lidar (03 §2.6)
        self.gamma = gamma

    def __call__(self, R: np.ndarray, t: np.ndarray, X_verify: np.ndarray,
                 obs_uv: np.ndarray, obs_z: np.ndarray) -> tuple[float, dict]:
        Xc = X_verify @ R.T + t
        zbuf, _ = splat_zbuffer(Xc, self.K, self.stride)
        zbuf = _fill_holes3(zbuf)   # 스플랫 구멍 메움 (2048pt는 성김) — 빈 셀만
        # free-space 판정은 모델 점유의 1셀 침식 내부로 한정 (실루엣 스플랫 과점유 배제).
        # 주의: 다공성 물체(팔레트 포켓)는 틈새 너머 정당한 배경 관측이 수 %의 잔여
        # free_viol을 만든다 — 그래서 free_viol은 하드 REJECT가 아니라 소프트 특징이며
        # (임계는 M3-3 캘리브레이션 대상), 오포즈 판별의 주 무기는 가설 간 증거 비교다.
        interior = _erode1(np.isfinite(zbuf))

        # 셀별 적응 임계: 이웃 점유 셀의 깊이 범위(grazing 표면 = 큰 범위)를 관용에 반영.
        # z-차이 잔차는 grazing에서 픽셀당 수십 mm가 정상이므로 고정 τ_z만으로는
        # 정답 포즈도 대량 위반 판정된다 (E2E 실측 — 설계 수정 근거).
        rng_map = _local_range3(zbuf)

        gu = np.clip((obs_uv[:, 0] / self.stride).astype(np.int64), 0, zbuf.shape[1] - 1)
        gv = np.clip((obs_uv[:, 1] / self.stride).astype(np.int64), 0, zbuf.shape[0] - 1)
        zm = zbuf[gv, gu]
        covered = np.isfinite(zm)
        if not covered.any():
            return 0.0, {"coverage": 0.0, "free_viol": 0.0, "score": 0.0}
        r = obs_z[covered] - zm[covered]
        tau_eff = self.tau_z + rng_map[gv[covered], gu[covered]]
        cov = float(covered.mean())
        s_res = float(np.mean(np.maximum(0.0, 1.0 - np.abs(r) / np.maximum(tau_eff, 1e-9))))
        inner = interior[gv[covered], gu[covered]]
        free_viol = float(np.mean((r > tau_eff) & inner)) if inner.any() else 0.0
        return s_res * cov ** self.gamma, {
            "coverage": cov, "free_viol": free_viol, "score": s_res * cov ** self.gamma,
        }
