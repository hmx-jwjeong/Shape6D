"""파이프라인 공용 타입 (03 문서 §4·§5·§7 인터페이스)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Cluster:
    id: int
    point_indices: np.ndarray   # FrameBundle 포인트 인덱스 [n]
    centroid: np.ndarray        # [3] (최근접 실측점으로 스냅됨)
    bbox_diag: float


@dataclass
class PromptSet:
    cluster_id: int
    points_uv: np.ndarray       # [K,2] full-res 픽셀 좌표
    labels: np.ndarray          # [K] 1=positive
    lidar_idx: np.ndarray


@dataclass
class Proposal:
    mask: np.ndarray            # [800,1280] bool
    bbox: np.ndarray            # [4] xyxy full-res
    score: float                # SAM predicted_iou 또는 0.30 (hull 경로)
    source: str                 # "sam" | "lidar_hull"
    cluster_id: int = -1
    lidar_idx: np.ndarray = None
    n_lidar: int = 0
    area: int = 0
    median_depth: float = float("nan")
    truncated: bool = False     # 이미지 경계 절단


@dataclass
class Candidate:
    """S2 입력/출력 단위 (Proposal → Candidate 변환은 S2 진입부가 수행, 03 §5.1)."""
    proposal: Proposal
    pts: np.ndarray             # [N_j,3] erosion+quality 필터 후 포인트 (카메라계, m)
    uv: np.ndarray              # [N_j,2]
    scores: dict = field(default_factory=dict)   # {"size","depth","sem","fused"}
    flags: set = field(default_factory=set)      # {"low_geo","border","occluded"}


@dataclass
class PoseHypothesis:
    R: np.ndarray               # [3,3]
    t: np.ndarray               # [3]
    score: float
    refined: bool = False       # True = fine 정련품(hyp0), False = coarse 원본 (03 §7.1)


@dataclass
class VerifyResult:
    pose: np.ndarray            # [4,4] T_cam_obj
    p_conf: float
    verdict: str                # "ACCEPT" | "UNCERTAIN" | "REJECT"
    diag: dict = field(default_factory=dict)
