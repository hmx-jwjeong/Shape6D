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


def splat_depth(pts_cam: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                res: tuple[int, int], fill_iters: int = 2) -> np.ndarray:
    """포인트 스플랫 z-buffer 깊이 렌더 (GL 무의존 — 크로스 플랫폼).

    S2 정합·TDF는 τ_trunc=0.1D의 관용을 갖고 있어 스플랫 근사로 충분하다.
    반환: depth [H,W] f32 (m), 0=배경.
    """
    H, W = res
    z = pts_cam[:, 2]
    ok = z > 1e-6
    u = np.round(fx * pts_cam[ok, 0] / z[ok] + cx).astype(np.int64)
    v = np.round(fy * pts_cam[ok, 1] / z[ok] + cy).astype(np.int64)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    pid = v[inb] * W + u[inb]
    zz = z[ok][inb]
    depth = np.full(H * W, np.inf, dtype=np.float32)
    np.minimum.at(depth, pid, zz)
    depth = depth.reshape(H, W)
    # 홀 메움: 빈 픽셀만 이웃 min으로 채움 (fill_iters회)
    for _ in range(fill_iters):
        empty = ~np.isfinite(depth)
        if not empty.any():
            break
        nb = np.full((H, W), np.inf, dtype=np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                src = (slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx)))
                dst = (slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx)))
                np.minimum(nb[dst], depth[src], out=nb[dst])
        depth[empty] = nb[empty]
    depth[~np.isfinite(depth)] = 0.0
    return depth


def build_view_templates(surface_pts: np.ndarray, r_bsphere: float, diameter: float,
                         center: np.ndarray | None = None, seed: int = 0) -> dict:
    """S0 템플릿 일체 생성 (03 §3.2): 42뷰 depth·가시 포인트·센터·TDF.

    surface_pts: 고밀도 표면 샘플 [M,3] (M≥5만 권장, 모델좌표계).
    반환 dict 키: tpl_depth [42,224,224] u16(mm), tpl_pose [42,4,4], tpl_K [3,3],
                  tpl_pts [42,512,3](모델계), tpl_center [42,3](뷰계 median), tdf [42,48³] f16
    """
    rng = np.random.default_rng(seed)
    c = surface_pts.mean(axis=0) if center is None else center
    P = surface_pts - c
    dirs = icosphere42()
    poses = lookat_poses(dirs, CAM_DIST_FACTOR * r_bsphere)
    # 센터링을 tpl_pose에 흡수: X_view = R·X_model + t_adj (t_adj = t − R·c)
    poses = poses.copy()
    for vi in range(N_VIEWS):
        poses[vi, :3, 3] = poses[vi, :3, 3] - poses[vi, :3, :3] @ c

    tpl_depth = np.zeros((N_VIEWS, TPL_RES, TPL_RES), np.uint16)
    tpl_pts = np.zeros((N_VIEWS, TPL_PTS_PER_VIEW, 3), np.float32)
    tpl_center = np.zeros((N_VIEWS, 3), np.float32)
    tdf = np.zeros((N_VIEWS, TDF_RES, TDF_RES, TDF_RES), np.float16)

    for vi in range(N_VIEWS):
        R, t = poses[vi, :3, :3], poses[vi, :3, 3]
        pc = surface_pts @ R.T + t  # t에 센터링 흡수됨
        depth = splat_depth(pc, TPL_FX, TPL_FX, TPL_RES / 2, TPL_RES / 2, (TPL_RES, TPL_RES))
        tpl_depth[vi] = np.round(depth * 1000.0).astype(np.uint16)

        # 가시 포인트: 자기 픽셀 z-buffer 승자(±5mm)만
        z = pc[:, 2]
        u = np.clip(np.round(TPL_FX * pc[:, 0] / z + TPL_RES / 2).astype(np.int64), 0, TPL_RES - 1)
        v = np.clip(np.round(TPL_FX * pc[:, 1] / z + TPL_RES / 2).astype(np.int64), 0, TPL_RES - 1)
        vis = np.abs(depth[v, u] - z) < 0.005 + 0.003 * diameter
        vis_idx = np.nonzero(vis)[0]
        sel = rng.permutation(vis_idx)[:TPL_PTS_PER_VIEW]
        if sel.size < TPL_PTS_PER_VIEW:  # 극단 뷰 보충
            sel = np.concatenate([sel, rng.choice(vis_idx, TPL_PTS_PER_VIEW - sel.size)])
        view_pts = pc[sel]
        tpl_center[vi] = np.median(view_pts, axis=0)
        tpl_pts[vi] = surface_pts[sel]  # 모델좌표계로 저장 (S4 소비 규약)
        tdf[vi] = build_tdf(view_pts - tpl_center[vi], diameter)

    K_tpl = np.array([[TPL_FX, 0, TPL_RES / 2], [0, TPL_FX, TPL_RES / 2], [0, 0, 1]], np.float32)
    return {"tpl_depth": tpl_depth, "tpl_pose": poses, "tpl_K": K_tpl,
            "tpl_pts": tpl_pts, "tpl_center": tpl_center, "tdf": tdf}
