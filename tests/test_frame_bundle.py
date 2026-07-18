"""FrameBundle 정본(03 §2.2) 단위테스트: 투영·z-buffer·품질 플래그·erosion 추출."""
import numpy as np

from shape6d.common.frame_bundle import (
    EDGE_MIXED,
    MULTI_RETURN,
    CameraIntrinsics,
    binary_erode,
    build_frame_bundle,
)

K = CameraIntrinsics(fx=200.0, fy=200.0, cx=64.0, cy=64.0, width=128, height=128)
T_ID = np.eye(4)


def _fb(pts, intensity=None):
    n = len(pts)
    rgb = np.zeros((128, 128, 3), np.uint8)
    inten = np.ones(n, np.float32) if intensity is None else intensity
    return build_frame_bundle(rgb, np.asarray(pts, np.float32), inten,
                              np.zeros(n, np.float32), K, T_ID)


def test_projection_and_rasterize():
    pts = [[0, 0, 1.0], [0.1, 0, 1.0], [0, 0.1, 1.0]]
    fb = _fb(pts)
    assert fb.valid_mask.sum() == 3
    u, v = fb.lidar_pixels[0]
    assert abs(u - 64) < 0.5 and abs(v - 64) < 0.5
    assert fb.sparse_depth[64, 64] == 1.0
    assert fb.pix2pt[64, 64] == 0


def test_zbuffer_winner_and_multi_return():
    pts = [[0, 0, 1.0], [0, 0, 1.5]]  # 동일 픽셀 — 근접점 승리
    fb = _fb(pts)
    assert fb.pix2pt[64, 64] == 0
    assert fb.point_quality[1] & MULTI_RETURN
    assert not (fb.point_quality[0] & MULTI_RETURN)


def test_edge_mixed_flag():
    # 인접 픽셀(3px 이내)에 깊이 1.0 vs 2.0 → 양쪽 EDGE_MIXED
    pts = [[0, 0, 1.0], [2 * 2.0 / 200.0, 0, 2.0]]  # 두 번째: u=cx+2px @z=2
    fb = _fb(pts)
    assert fb.point_quality[0] & EDGE_MIXED
    assert fb.point_quality[1] & EDGE_MIXED
    # 멀리 떨어진 점(>3px)은 클린
    pts2 = [[0, 0, 1.0], [0.2, 0, 1.0]]  # 40px 떨어짐, 동일 깊이
    fb2 = _fb(pts2)
    assert fb2.point_quality[0] == 0 and fb2.point_quality[1] == 0


def test_object_points_erosion():
    # 5×5 픽셀 블록에 점 배치, 마스크 = 그 블록 → erosion 2면 중심부만
    pts = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            pts.append([dx * 1.0 / 200.0, dy * 1.0 / 200.0, 1.0])
    fb = _fb(pts)
    mask = np.zeros((128, 128), bool)
    mask[62:67, 62:67] = True  # 포인트 블록(픽셀 62..66)과 정확히 일치
    idx_e0 = fb.object_points(mask, erosion_px=0)
    idx_e2 = fb.object_points(mask, erosion_px=2)
    assert len(idx_e0) == 25
    assert len(idx_e2) == 1  # erosion 2 → 중심 1픽셀만 잔존


def test_binary_erode():
    m = np.zeros((10, 10), bool)
    m[2:8, 2:8] = True
    e = binary_erode(m, 1)
    assert e.sum() == 16  # 6×6 → 4×4
