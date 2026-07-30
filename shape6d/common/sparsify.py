"""센서 스펙 기반 합성 희소화 (17 §5.5 — 학습 희소화의 정본 경로).

UAM 실기 마스크의 학습 사용이 금지되면서(테스트 셋 유출), 학습·파일럿의
희소화는 전부 이 모듈의 스펙 파라미터화 생성기로 통일한다. 실기 로그는
이 생성기의 통계적 '사후 검증'에만 쓴다.

레짐 2종 (10 권고 10 — N_obj ~ U[32,2000]이 한쪽 레짐만 덮는 문제):
- ML-X(80) 고정 각도 격자: δh 0.14° × δv 0.42° (03 §1.4b)
- Ruby128 계열 고밀도 스캔라인: 수평 조밀 · 수직 128채널 (UAM 실측 대비용)
"""
from __future__ import annotations

import numpy as np

from .frame_bundle import CameraIntrinsics

# SOS Lab ML-X(80) 각도 분해능 (03 §1.4b — D2-a 실측으로 교체 예정)
MLX80_DH_DEG = 0.14
MLX80_DV_DEG = 0.42
# RoboSense Ruby128 근사 (128ch / 수직 FOV 40° → δv ≈ 0.31°, 수평 0.2°@10Hz)
RUBY128_DH_DEG = 0.20
RUBY128_DV_DEG = 0.3125


def sparsify_fixed_grid(depth_m: np.ndarray, K: CameraIntrinsics,
                        dh_deg: float = MLX80_DH_DEG, dv_deg: float = MLX80_DV_DEG,
                        sigma: float = 0.0, seed: int = 0,
                        row_dropout: float = 0.0) -> np.ndarray:
    """dense depth → 고정 각도 격자 샘플 포인트 (카메라계 [N,3]).

    카메라 픽셀의 각도 피치(≈1/fx rad)를 LiDAR 격자(δh, δv)로 재샘플 —
    격자 방향에 가장 가까운 픽셀 1개만 유지 (실측치 선택, 보간 없음: A1).
    row_dropout: 스캔라인(행) 단위 결측 확률 — 실기에서 관측되는 행 뭉침
    결측(13 §3: 유효 픽셀이 줄무늬로 뭉침)의 합성 대응.
    """
    H, W = depth_m.shape
    rng = np.random.default_rng(seed)
    step_u = max(1, int(round(np.deg2rad(dh_deg) * K.fx)))
    step_v = max(1, int(round(np.deg2rad(dv_deg) * K.fy)))
    rows = np.arange(0, H, step_v)
    if row_dropout > 0:
        rows = rows[rng.random(len(rows)) >= row_dropout]
    vs, us = np.meshgrid(rows, np.arange(0, W, step_u), indexing="ij")
    z = depth_m[vs, us]
    ok = z > 1e-6
    u, v, z = us[ok].astype(np.float64), vs[ok].astype(np.float64), z[ok].astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    pts = np.stack([x, y, z], 1).astype(np.float32)
    if sigma > 0:
        pts[:, 2] += rng.normal(0, sigma, len(pts)).astype(np.float32)
    return pts


def subsample_to_target(pts: np.ndarray, n_range=(32, 2000), seed: int = 0) -> np.ndarray:
    """물체 위 포인트를 로그균등 목표 수로 랜덤 서브샘플 (03 §9 학습 분포)."""
    rng = np.random.default_rng(seed)
    n_tgt = int(np.exp(rng.uniform(*np.log(n_range))))
    if len(pts) <= n_tgt:
        return pts
    return pts[rng.choice(len(pts), n_tgt, replace=False)]


REGIMES = {
    "mlx80": dict(dh_deg=MLX80_DH_DEG, dv_deg=MLX80_DV_DEG),
    "ruby128": dict(dh_deg=RUBY128_DH_DEG, dv_deg=RUBY128_DV_DEG),
}


def sample_regime(depth_m: np.ndarray, K: CameraIntrinsics, sigma: float = 0.003,
                  seed: int = 0, p_ruby: float = 0.5,
                  row_dropout_range=(0.0, 0.3)) -> np.ndarray:
    """학습 증강용: 레짐 2종 + 행 결측을 무작위 샘플 (10 권고 10)."""
    rng = np.random.default_rng(seed)
    regime = "ruby128" if rng.random() < p_ruby else "mlx80"
    rd = float(rng.uniform(*row_dropout_range))
    return sparsify_fixed_grid(depth_m, K, sigma=sigma, row_dropout=rd,
                               seed=int(rng.integers(1 << 31)), **REGIMES[regime])
