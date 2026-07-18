"""S1 프롬프트 생성: 평면 제거 + voxel 클러스터링 (03 §4.2)."""
import numpy as np

from shape6d.common.frame_bundle import CameraIntrinsics, build_frame_bundle
from shape6d.proposal.prompt_gen import LidarPromptGenerator

K = CameraIntrinsics(fx=400.0, fy=400.0, cx=640.0, cy=400.0)


def _scene():
    rng = np.random.default_rng(0)
    # 바닥 평면 (카메라 아래쪽, 기울어진 채 전방)
    xs = np.linspace(-0.6, 0.6, 40)
    zs = np.linspace(1.0, 2.5, 40)
    X, Z = np.meshgrid(xs, zs)
    floor = np.stack([X.ravel(), np.full(X.size, 0.45), Z.ravel()], 1)
    # 물체 2개 (바닥 위)
    box1 = rng.normal(0, 0.04, (250, 3)) + np.array([0.25, 0.30, 1.4])
    box2 = rng.normal(0, 0.04, (250, 3)) + np.array([-0.30, 0.30, 1.8])
    pts = np.vstack([floor, box1, box2]).astype(np.float32)
    rgb = np.zeros((800, 1280, 3), np.uint8)
    return build_frame_bundle(rgb, pts, np.ones(len(pts), np.float32),
                              np.zeros(len(pts), np.float32), K, np.eye(4))


def test_two_clusters_after_plane_removal():
    fb = _scene()
    gen = LidarPromptGenerator(voxel=0.05, min_cluster_pts=30)
    prompts, clusters = gen(fb)
    assert len(clusters) == 2, f"클러스터 {len(clusters)} != 2"
    assert len(prompts) == 2
    for p in prompts:
        assert np.isfinite(p.points_uv).all()
        assert 1 <= p.points_uv.shape[0] <= 3
        assert p.lidar_idx.size >= 30


def test_empty_scene():
    rgb = np.zeros((800, 1280, 3), np.uint8)
    pts = np.zeros((0, 3), np.float32)
    fb = build_frame_bundle(rgb, pts, np.zeros(0, np.float32), np.zeros(0, np.float32), K, np.eye(4))
    prompts, clusters = LidarPromptGenerator()(fb)
    assert prompts == [] and clusters == []
