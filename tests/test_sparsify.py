"""shape6d.common.sparsify — 센서 스펙 합성 희소화 (17 §5.5 정본 경로)."""
import numpy as np

from shape6d.common.frame_bundle import CameraIntrinsics
from shape6d.common.sparsify import (
    REGIMES, sample_regime, sparsify_fixed_grid, subsample_to_target,
)

K = CameraIntrinsics(fx=600.0, fy=600.0, cx=640.0, cy=400.0, width=1280, height=800)


def _depth(z=3.0):
    d = np.zeros((800, 1280), np.float32)
    d[200:600, 300:900] = z
    return d


def test_grid_density_ratio_between_regimes():
    d = _depth()
    n_mlx = len(sparsify_fixed_grid(d, K, **REGIMES["mlx80"]))
    n_ruby = len(sparsify_fixed_grid(d, K, **REGIMES["ruby128"]))
    assert n_mlx > 200 and n_ruby > 200
    # 각 피치 비 (0.20·0.3125)/(0.14·0.42) ≈ 0.94 → 격자 정수화 반영 넉넉한 대역
    assert 0.4 < n_ruby / n_mlx < 2.5


def test_row_dropout_reduces_rows_not_noise():
    d = _depth()
    p0 = sparsify_fixed_grid(d, K, seed=1, row_dropout=0.0)
    p1 = sparsify_fixed_grid(d, K, seed=1, row_dropout=0.5)
    rows0 = len(np.unique(np.round(p0[:, 1] / p0[:, 2] * K.fy).astype(int)))
    rows1 = len(np.unique(np.round(p1[:, 1] / p1[:, 2] * K.fy).astype(int)))
    assert rows1 < rows0  # 행 단위 결측 (스캔라인 뭉침 재현 — 13 §3)


def test_sigma_only_perturbs_z():
    d = _depth()
    a = sparsify_fixed_grid(d, K, sigma=0.0, seed=7)
    b = sparsify_fixed_grid(d, K, sigma=0.005, seed=7)
    assert np.allclose(a[:, :2], b[:, :2])
    assert not np.allclose(a[:, 2], b[:, 2])
    assert abs(float(np.std(b[:, 2] - a[:, 2])) - 0.005) < 0.001


def test_subsample_log_uniform_within_range():
    pts = np.random.default_rng(0).normal(size=(5000, 3)).astype(np.float32)
    ns = [len(subsample_to_target(pts, (32, 2000), seed=s)) for s in range(50)]
    assert min(ns) >= 32 and max(ns) <= 2000
    assert len(set(ns)) > 10  # 로그균등 샘플이 실제로 퍼짐


def test_sample_regime_runs_both():
    d = _depth()
    ns = {len(sample_regime(d, K, seed=s)) for s in range(8)}
    assert len(ns) >= 2  # 레짐/드롭아웃 변동이 실제 발생


def test_harness_reexport_compat():
    import sys
    sys.path.insert(0, "eval/external_test")
    from harness import sparsify_fixed_grid as h_sp  # noqa
    assert h_sp is sparsify_fixed_grid
