"""파이프라인 오케스트레이션·타이머 스모크 테스트."""
import numpy as np

from shape6d.common.frame_bundle import CameraIntrinsics, build_frame_bundle
from shape6d.pipeline import Shape6DPipeline


def test_not_ready_stub():
    K = CameraIntrinsics(fx=400, fy=400, cx=640, cy=400)
    fb = build_frame_bundle(np.zeros((800, 1280, 3), np.uint8), np.zeros((0, 3), np.float32),
                            np.zeros(0, np.float32), np.zeros(0, np.float32), K, np.eye(4))
    r = Shape6DPipeline()(fb)
    assert r.status.startswith("not_ready")
    assert "S1_proposal" in r.timing


def test_config_loads():
    import pathlib

    import yaml
    cfg = yaml.safe_load((pathlib.Path(__file__).parent.parent / "shape6d/config/pipeline.yaml").read_text())
    # §2.5 하한 소유권 표와 §2.6 σ 단일 주입의 존재 확인
    assert cfg["thresholds"]["s2_n_min"] == 30
    assert cfg["thresholds"]["s4_min_inlier"] == 25
    assert cfg["sensor"]["sigma_lidar_m"] == 0.008
    assert cfg["symmetry"]["policy"] == "all_equivalent"
