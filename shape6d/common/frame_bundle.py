"""FrameBundle — 전 스테이지(S1~S4)가 소비하는 표준 입력 (03 문서 §2.2 정본).

좌표 규약: lidar_points는 생성 시점에 카메라 좌표계로 변환 완료(단위 m).
이미지 규약: 1280×800 (H=800, W=1280), RGB는 사전 undistort.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# 품질 플래그 비트 (03 §2.2)
EDGE_MIXED = 1        # 깊이 불연속 인접 — 빔 풋프린트 mixed pixel 의심
LOW_INTENSITY = 2     # 저반사 리턴 — 거리 노이즈 증가 의심
MULTI_RETURN = 4      # z-buffer 패배(동일 픽셀 후방) 등 다중 리턴 의심
NEAR_MASK_BOUNDARY = 8  # 인스턴스 마스크 경계 인접 (S1 이후 갱신)


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int = 1280
    height: int = 800

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64
        )


@dataclass
class FrameBundle:
    rgb: np.ndarray             # [H,W,3] uint8
    lidar_points: np.ndarray    # [N,3] f32, 카메라 좌표계
    lidar_intensity: np.ndarray  # [N] f32
    lidar_t_offset: np.ndarray  # [N] f32 (deskew 확장용, v1 미사용)
    lidar_pixels: np.ndarray    # [N,2] f32 서브픽셀 (u,v), 이미지 밖 = NaN
    sparse_depth: np.ndarray    # [H,W] f32, 0=무효
    valid_mask: np.ndarray      # [H,W] bool
    pix2pt: np.ndarray          # [H,W] i32, -1=무효
    point_quality: np.ndarray   # [N] u8
    K: CameraIntrinsics
    T_cam_lidar: np.ndarray     # [4,4] f64 (기록용)
    t_rgb: float = 0.0
    t_lidar: tuple = (0.0, 0.0)
    meta: dict = field(default_factory=dict)

    def object_points(
        self, mask: np.ndarray, erosion_px: int = 2, clean_only: bool = True
    ) -> np.ndarray:
        """마스크 내 LiDAR 포인트 인덱스 (03 §2.2 표준 절차: erosion 후 선별).

        clean_only=True면 quality==0 우선 — 부족하면(<30) EDGE_MIXED 외 허용으로 완화.
        반환: 포인트 인덱스 [M] int64
        """
        m = binary_erode(mask, erosion_px) if erosion_px > 0 else mask
        idx = self.pix2pt[m & self.valid_mask]
        idx = idx[idx >= 0]
        if clean_only:
            q = self.point_quality[idx]
            clean = idx[q == 0]
            if clean.size >= 30:
                return clean.astype(np.int64)
            relaxed = idx[(q & EDGE_MIXED) == 0]
            return relaxed.astype(np.int64)
        return idx.astype(np.int64)


def project_points(pts_cam: np.ndarray, K: CameraIntrinsics, z_near: float = 0.05):
    """카메라 좌표 포인트 → 픽셀 (u,v). 이미지 밖/후방은 NaN.

    반환: uv [N,2] f32, z [N] f32, in_img [N] bool
    """
    z = pts_cam[:, 2].astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K.fx * pts_cam[:, 0] / z + K.cx
        v = K.fy * pts_cam[:, 1] / z + K.cy
    uv = np.stack([u, v], axis=1).astype(np.float32)
    in_img = (
        (z > z_near)
        & (uv[:, 0] >= 0) & (uv[:, 0] < K.width)
        & (uv[:, 1] >= 0) & (uv[:, 1] < K.height)
    )
    uv[~in_img] = np.nan
    return uv, z, in_img


def rasterize(uv: np.ndarray, z: np.ndarray, hw: tuple[int, int]):
    """z-buffer 래스터화. 반환: sparse_depth [H,W], valid [H,W] bool, pix2pt [H,W] i32.

    동일 픽셀에 여러 포인트가 오면 최근접(z 최소)이 승리.
    """
    H, W = hw
    ok = np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & (z > 0)
    idx = np.nonzero(ok)[0]
    u = np.clip(np.round(uv[idx, 0]).astype(np.int64), 0, W - 1)
    v = np.clip(np.round(uv[idx, 1]).astype(np.int64), 0, H - 1)
    pid = v * W + u
    order = np.lexsort((z[idx], pid))  # 픽셀별로 정렬 후 z 오름차순
    pid_s, idx_s = pid[order], idx[order]
    first = np.ones(pid_s.size, dtype=bool)
    first[1:] = pid_s[1:] != pid_s[:-1]

    depth = np.zeros(H * W, dtype=np.float32)
    p2p = np.full(H * W, -1, dtype=np.int32)
    depth[pid_s[first]] = z[idx_s[first]]
    p2p[pid_s[first]] = idx_s[first]
    valid = depth > 0
    losers = idx_s[~first]  # z-buffer 패배 포인트 (후방 다중 리턴 의심)
    return depth.reshape(H, W), valid.reshape(H, W), p2p.reshape(H, W), losers


def _neighbor_minmax(depth: np.ndarray, valid: np.ndarray, r: int):
    """유효 픽셀만 고려한 이웃(반경 r, 자기 제외) min/max 깊이 맵."""
    H, W = depth.shape
    mn_src = np.where(valid, depth, np.inf)
    mx_src = np.where(valid, depth, -np.inf)
    out_mn = np.full_like(mn_src, np.inf)
    out_mx = np.full_like(mx_src, -np.inf)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            src = (slice(max(0, dy), H + min(0, dy)), slice(max(0, dx), W + min(0, dx)))
            dst = (slice(max(0, -dy), H + min(0, -dy)), slice(max(0, -dx), W + min(0, -dx)))
            np.minimum(out_mn[dst], mn_src[src], out=out_mn[dst])
            np.maximum(out_mx[dst], mx_src[src], out=out_mx[dst])
    return out_mn, out_mx


def flag_quality(
    pts_cam: np.ndarray,
    uv: np.ndarray,
    sparse_depth: np.ndarray,
    valid: np.ndarray,
    intensity: np.ndarray,
    losers: np.ndarray,
    delta_edge: float = 0.03,
    r_edge_px: int = 3,
    low_intensity_pct: float = 10.0,
) -> np.ndarray:
    """포인트별 품질 플래그 (03 §2.2). 반환: [N] uint8."""
    n = pts_cam.shape[0]
    flags = np.zeros(n, dtype=np.uint8)

    # EDGE_MIXED: 자기 픽셀 이웃의 유효 깊이 범위가 delta_edge 초과
    mn, mx = _neighbor_minmax(sparse_depth, valid, r_edge_px)
    ok = np.isfinite(uv[:, 0])
    ui = np.clip(np.round(uv[ok, 0]).astype(np.int64), 0, sparse_depth.shape[1] - 1)
    vi = np.clip(np.round(uv[ok, 1]).astype(np.int64), 0, sparse_depth.shape[0] - 1)
    has_nb = np.isfinite(mn[vi, ui])
    z_self = pts_cam[ok, 2]
    rng = np.maximum(mx[vi, ui], z_self) - np.minimum(mn[vi, ui], z_self)
    edge = np.zeros(ok.sum(), dtype=bool)
    edge[has_nb] = rng[has_nb] > delta_edge
    idx_ok = np.nonzero(ok)[0]
    flags[idx_ok[edge]] |= EDGE_MIXED

    # LOW_INTENSITY: 하위 percentile
    if intensity is not None and intensity.size == n and np.any(intensity > 0):
        thr = np.percentile(intensity[intensity > 0], low_intensity_pct)
        flags[intensity < thr] |= LOW_INTENSITY

    # z-buffer 패배 = 후방 다중 리턴 의심
    flags[losers] |= MULTI_RETURN
    return flags


def build_frame_bundle(
    rgb: np.ndarray,
    lidar_pts_lidar_frame: np.ndarray,
    intensity: np.ndarray,
    t_offsets: np.ndarray,
    K: CameraIntrinsics,
    T_cam_lidar: np.ndarray,
    delta_edge: float = 0.03,
    r_edge_px: int = 3,
) -> FrameBundle:
    """원시 관측 → FrameBundle (03 §2.2·§4.2 파이프라인, <5ms 예산)."""
    R, t = T_cam_lidar[:3, :3], T_cam_lidar[:3, 3]
    pts_cam = (lidar_pts_lidar_frame @ R.T + t).astype(np.float32)
    uv, z, _ = project_points(pts_cam, K)
    depth, valid, p2p, losers = rasterize(uv, z, (K.height, K.width))
    flags = flag_quality(pts_cam, uv, depth, valid, intensity, losers,
                         delta_edge=delta_edge, r_edge_px=r_edge_px)
    return FrameBundle(
        rgb=rgb, lidar_points=pts_cam,
        lidar_intensity=np.asarray(intensity, dtype=np.float32),
        lidar_t_offset=np.asarray(t_offsets, dtype=np.float32),
        lidar_pixels=uv, sparse_depth=depth, valid_mask=valid, pix2pt=p2p,
        point_quality=flags, K=K, T_cam_lidar=np.asarray(T_cam_lidar, dtype=np.float64),
    )


def binary_erode(mask: np.ndarray, r: int) -> np.ndarray:
    """정사각 구조요소 침식 (scipy 무의존, r ≤ 수 px 용도)."""
    out = mask.copy()
    H, W = mask.shape
    for _ in range(r):
        nxt = out.copy()
        nxt[1:, :] &= out[:-1, :]
        nxt[:-1, :] &= out[1:, :]
        nxt[:, 1:] &= out[:, :-1]
        nxt[:, :-1] &= out[:, 1:]
        # 경계는 보수적으로 침식
        nxt[0, :] = False; nxt[-1, :] = False; nxt[:, 0] = False; nxt[:, -1] = False
        out = nxt
    return out
