"""S0 템플릿 뷰·TDF (03 문서 §3.2·§5.2).

렌더 스펙 (검증 [하-5] 반영): d_cam = 2.7 · r_bsphere, fx = fy = 270.4 (FOV 45°, 224²).
"""
from __future__ import annotations

import numpy as np

N_VIEWS = 42
TPL_RES = 224
TPL_FX = 112.0 / np.tan(np.deg2rad(22.5))   # = 270.4
CAM_DIST_FACTOR = 2.7                        # ≥ r/sin(22.5°)=2.613r (구면 접선 조건)
TDF_RES = 48
TDF_HALF_EXTENT = 0.6                        # ×D_cad
TDF_TRUNC = 0.1                              # ×D_cad
TPL_PTS_PER_VIEW = 512


def icosphere42() -> np.ndarray:
    """icosphere L2 = 정이십면체 12정점 + 30모서리 중점 → 42 단위 방향 (SAM-6D 뷰 규약)."""
    phi = (1 + np.sqrt(5)) / 2
    v = []
    for a in (-1, 1):
        for b in (-phi, phi):
            v += [(0, a, b), (a, b, 0), (b, 0, a)]
    v = np.array(v, dtype=np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)

    # 모서리: 최근접 정점 쌍 (정이십면체 모서리 길이 = 2/sqrt(phi^2+1)·… → 거리로 판별)
    d = np.linalg.norm(v[:, None] - v[None], axis=2)
    edge_len = np.sort(np.unique(np.round(d, 6)))[1]
    mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(d[i, j] - edge_len) < 1e-6:
                m = (v[i] + v[j]) / 2
                mids.append(m / np.linalg.norm(m))
    dirs = np.vstack([v, np.array(mids)])
    assert dirs.shape == (42, 3), dirs.shape
    return dirs.astype(np.float32)


def lookat_poses(dirs: np.ndarray, dist: float) -> np.ndarray:
    """카메라를 dirs·dist 에 놓고 원점을 보는 T_obj2cam [V,4,4] (OpenCV 규약: +z가 전방)."""
    poses = []
    for dvec in dirs:
        z = -dvec / np.linalg.norm(dvec)          # 카메라 +z = 원점 방향
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, z)) > 0.99:
            up = np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z); x /= np.linalg.norm(x)
        y = np.cross(z, x)
        R_c2o = np.stack([x, y, z], axis=1)       # 카메라축 → 모델좌표
        T = np.eye(4)
        T[:3, :3] = R_c2o.T
        T[:3, 3] = -R_c2o.T @ (dvec * dist)
        poses.append(T)
    return np.stack(poses).astype(np.float32)


def build_tdf(tpl_pts_centered: np.ndarray, diameter: float) -> np.ndarray:
    """뷰별 가시면 포인트(센터링됨) → truncated distance field LUT [48³] fp16."""
    from scipy.spatial import cKDTree
    half = TDF_HALF_EXTENT * diameter
    trunc = TDF_TRUNC * diameter
    lin = np.linspace(-half, half, TDF_RES)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
    q = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)
    d, _ = cKDTree(tpl_pts_centered).query(q, k=1)
    return np.minimum(d, trunc).reshape(TDF_RES, TDF_RES, TDF_RES).astype(np.float16)


def tdf_lookup(tdf: np.ndarray, q: np.ndarray, diameter: float) -> np.ndarray:
    """포인트 배치 → TDF 값 (범위 밖 = trunc). 최근접 voxel gather (03 §5.2)."""
    half = TDF_HALF_EXTENT * diameter
    trunc = TDF_TRUNC * diameter
    vox = np.round((q + half) / (2 * half) * (TDF_RES - 1)).astype(np.int64)
    inside = np.all((vox >= 0) & (vox < TDF_RES), axis=1)
    out = np.full(q.shape[0], trunc, dtype=np.float32)
    vi = vox[inside]
    out[inside] = tdf[vi[:, 0], vi[:, 1], vi[:, 2]].astype(np.float32)
    return out


def render_depth_templates(mesh, dirs: np.ndarray, r_bsphere: float):
    """pyrender EGL depth 렌더 (Ubuntu/M1 단계 — lazy import).

    반환: depth [V,224,224] f32(m), poses [V,4,4]
    """
    try:
        import pyrender  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "pyrender 미설치 — 온보딩 렌더는 Ubuntu PC에서 `uv pip install -e '.[render]'` 후 실행 (M1)"
        ) from e
    raise NotImplementedError("M1: pyrender EGL 렌더 경로 구현 (03 §3.2 스펙 고정 완료)")
