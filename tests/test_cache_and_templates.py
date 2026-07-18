"""S0 캐시 스키마(§2.4)·icosphere 42뷰·TDF·서브셋 인덱스 테스트."""
import numpy as np
import pytest

from shape6d.onboarding import cache
from shape6d.onboarding.sampling import fps_indices, make_subsets
from shape6d.onboarding.templates import CAM_DIST_FACTOR, TPL_FX, build_tdf, icosphere42, lookat_poses, tdf_lookup


def _minimal_arrays():
    return {
        "pts_master": np.zeros((16384, 3), np.float16),
        "nrm_master": np.zeros((16384, 3), np.float16),
        "idx_pem": np.zeros(2048, np.int32),
        "idx_sparse": np.zeros(196, np.int32),
        "idx_model": np.zeros(1024, np.int32),
        "idx_verify": np.zeros(2048, np.int32),
        "sym_rots": np.eye(3, dtype=np.float32)[None],
        "sym_axes": np.zeros((0, 3), np.float32),
        "diameter": np.float32(0.5),
        "bbox": np.zeros((2, 3), np.float32),
        "center_offset": np.zeros(3, np.float32),
        "radius": np.float32(0.25),
    }


def test_cache_roundtrip(tmp_path):
    p = cache.save(tmp_path, "obj_a", _minimal_arrays(), sym_summary="test")
    assert p.exists()
    c = cache.load(tmp_path, "obj_a")
    assert c.diameter == np.float32(0.5)
    assert c.manifest["obj_id"] == "obj_a"


def test_cache_validation_errors(tmp_path):
    arrays = _minimal_arrays()
    del arrays["diameter"]
    with pytest.raises(ValueError, match="diameter"):
        cache.save(tmp_path, "bad", arrays)
    arrays = _minimal_arrays()
    arrays["idx_sparse"] = np.zeros(100, np.int32)  # shape 위반
    with pytest.raises(ValueError, match="idx_sparse"):
        cache.save(tmp_path, "bad2", arrays)


def test_icosphere42():
    dirs = icosphere42()
    assert dirs.shape == (42, 3)
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0, atol=1e-5)
    # 스펙 상수 (03 §3.2 — 검증 [하-5] 반영값)
    assert abs(TPL_FX - 270.4) < 0.1
    assert CAM_DIST_FACTOR == 2.7


def test_lookat_poses():
    dirs = icosphere42()
    poses = lookat_poses(dirs, dist=2.7)
    origin_in_cam = poses[:, :3, 3]  # 모델 원점의 카메라 좌표
    assert np.allclose(np.linalg.norm(origin_in_cam, axis=1), 2.7, atol=1e-4)
    assert np.all(origin_in_cam[:, 2] > 2.69), "원점이 카메라 +z 전방에 있어야 함"


def test_tdf_surface_zero_far_trunc():
    pts = np.random.default_rng(0).uniform(-0.1, 0.1, (512, 3)).astype(np.float32)
    D = 0.35
    tdf = build_tdf(pts, D)
    d_surf = tdf_lookup(tdf, pts[:16], D)
    assert d_surf.max() < 0.02 * D  # 표면 위 ≈ 0 (voxel 양자화 허용)
    far = np.array([[0.55 * D, 0.55 * D, 0.55 * D]])
    assert tdf_lookup(tdf, far, D)[0] >= 0.099 * D  # 절단값


def test_subsets_nested():
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (2048, 3)).astype(np.float32)
    idx = fps_indices(pts, 196)
    assert len(np.unique(idx)) == 196
    sub = make_subsets(pts)
    assert set(sub["idx_sparse"]).issubset(set(sub["idx_pem"])), "idx_sparse ⊄ idx_pem (§2.4)"
