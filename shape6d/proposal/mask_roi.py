"""S1 v2 — 선택식 ROI (C-9, 20 v1.1 §2).

삭제식(평면 제거→클러스터)이 아니라: 거친 클러스터(평면 제거 없음)를
프롬프트 시드로만 쓰고, RGB 분할 마스크(침식 δpx) ∩ LiDAR 로 ROI를 "고른다".
유효 마스크가 없으면 클러스터 ROI 직행(기하 폴백 — 기하 우선 원칙의 보험).

분할기: 하네스 A/B 용도로 HF `facebook/sam-vit-base`(Apache-2.0) 사용.
배포 경로는 EfficientViT-SAM-L0(Apache)로 교체 예정 — A/B의 판정 대상은
세그멘터 종류가 아니라 ROI 전략(C-9)이므로 하네스 대체를 허용한다.
"""
from __future__ import annotations

import numpy as np

from ..common.frame_bundle import FrameBundle
from ..common.types import Cluster
from .prompt_gen import LidarPromptGenerator


def _erode(mask: np.ndarray, r: int) -> np.ndarray:
    """3×3 침식 r회 — 외부보정 오차에 의한 경계 점 오귀속 방지 (C-9 δpx)."""
    m = mask.astype(bool)
    for _ in range(r):
        p = np.pad(m, 1, mode="constant")
        m = (p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]
             & p[1:-1, :-2] & p[1:-1, 2:])
    return m


class MaskROIGenerator:
    """클러스터 시드 → SAM 마스크 → ROI 선택. 반환 형식은 v1과 동일한 클러스터 목록."""

    def __init__(self, erode_px: int = 3, min_roi_pts: int = 15,
                 containment_min: float = 0.5, expand_box: float = 0.12,
                 model_name: str = "facebook/sam-vit-base", device: str | None = None,
                 seed_cfg: dict | None = None):
        self.erode_px = erode_px
        self.min_roi_pts = min_roi_pts
        self.containment_min = containment_min
        self.expand_box = expand_box
        self.model_name = model_name
        self.device = device
        # 시드용 클러스터 — 평면 제거 폐지(max_planes=0)가 v2의 핵심
        cfg = dict(max_planes=0, min_cluster_pts=15)
        cfg.update(seed_cfg or {})
        cfg["max_planes"] = 0
        self.seeder = LidarPromptGenerator(**cfg)
        self._model = None
        self._proc = None

    def _lazy_load(self):
        if self._model is not None:
            return
        import torch
        from transformers import SamModel, SamProcessor
        dev = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._proc = SamProcessor.from_pretrained(self.model_name)
        self._model = SamModel.from_pretrained(self.model_name).to(dev).eval()
        self._dev = dev

    def _predict_mask(self, rgb: np.ndarray, pts_uv: np.ndarray, box: np.ndarray):
        import torch
        inputs = self._proc(rgb, input_points=[[pts_uv.tolist()]],
                            input_boxes=[[box.tolist()]],
                            return_tensors="pt").to(self._dev)
        with torch.no_grad():
            out = self._model(**inputs, multimask_output=True)
        masks = self._proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu())[0][0].numpy()   # [3,H,W]
        scores = out.iou_scores.cpu().numpy().reshape(-1)          # [3]
        return masks, scores

    def __call__(self, rgb: np.ndarray, fb: FrameBundle
                 ) -> tuple[list[Cluster], dict]:
        """반환: (ROI 클러스터 목록, 진단). 각 Cluster.point_indices = 선택된 LiDAR 점."""
        self._lazy_load()
        _, seeds = self.seeder(fb)
        H, W = rgb.shape[:2]
        out, diag = [], {"n_seeds": len(seeds), "fallback": 0, "mask_ok": 0}
        for c in seeds:
            uv_all = fb.lidar_pixels[c.point_indices]
            ok = np.isfinite(uv_all[:, 0])
            if not ok.any():
                continue
            uv = uv_all[ok]
            reps_uv = uv[np.linspace(0, len(uv) - 1, min(3, len(uv))).astype(int)]
            lo, hi = uv.min(0), uv.max(0)
            m = self.expand_box * max(hi[0] - lo[0], hi[1] - lo[1], 20.0)
            box = np.array([max(lo[0] - m, 0), max(lo[1] - m, 0),
                            min(hi[0] + m, W - 1), min(hi[1] + m, H - 1)])
            try:
                masks, scores = self._predict_mask(rgb, reps_uv, box)
            except Exception:
                masks = None
            sel = None
            if masks is not None:
                ui, vi = uv[:, 0].round().astype(int), uv[:, 1].round().astype(int)
                inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                best_q = -1.0
                for mk, sc in zip(masks, scores):
                    cont = float(mk[vi[inb], ui[inb]].mean()) if inb.any() else 0.0
                    q = cont * float(sc)
                    if cont >= self.containment_min and q > best_q:
                        best_q, sel = q, mk
            if sel is not None:
                mk = _erode(sel, self.erode_px)
                puv = fb.lidar_pixels
                fin = np.isfinite(puv[:, 0])
                ui = np.zeros(len(puv), int); vi = np.zeros(len(puv), int)
                ui[fin] = puv[fin, 0].round().astype(int).clip(0, W - 1)
                vi[fin] = puv[fin, 1].round().astype(int).clip(0, H - 1)
                idx = np.nonzero(fin & mk[vi, ui])[0]
                if len(idx) >= self.min_roi_pts:
                    P = fb.lidar_points[idx]
                    out.append(Cluster(id=c.id, point_indices=idx,
                                       centroid=P.mean(0),
                                       bbox_diag=float(np.linalg.norm(P.max(0) - P.min(0)))))
                    diag["mask_ok"] += 1
                    continue
            # 폴백: 유효 마스크 없음 → 클러스터 ROI 직행 (C-9)
            out.append(c)
            diag["fallback"] += 1
        return out, diag
