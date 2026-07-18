"""S1 LiDAR 프롬프트 생성 — 평면 제거 + voxel-hash 클러스터링 (03 문서 §4.2).

kNN/KD-tree 불사용(A4): 26-연결 voxel BFS. EViT-SAM 래퍼는 M1에서 결합.
"""
from __future__ import annotations

import numpy as np

from ..common.frame_bundle import FrameBundle
from ..common.types import Cluster, PromptSet


class LidarPromptGenerator:
    def __init__(self, voxel: float = 0.02, ransac_dist: float = 0.008,
                 ransac_iters: int = 200, max_planes: int = 2,
                 plane_min_ratio: float = 0.15, min_cluster_pts: int = 15,
                 k_prompts: int = 3, size_gate: tuple[float, float] | None = None,
                 seed: int = 0):
        self.voxel = voxel
        self.ransac_dist = ransac_dist
        self.ransac_iters = ransac_iters
        self.max_planes = max_planes
        self.plane_min_ratio = plane_min_ratio
        self.min_cluster_pts = min_cluster_pts
        self.k_prompts = k_prompts
        self.size_gate = size_gate  # (min_m, max_m) — 느슨하게, 리콜 우선
        self.rng = np.random.default_rng(seed)

    def __call__(self, fb: FrameBundle) -> tuple[list[PromptSet], list[Cluster]]:
        # 검증 [M3]: 카메라 FOV 밖(NaN uv) 포인트는 프롬프트 경로에서 제외
        in_img = np.isfinite(fb.lidar_pixels[:, 0])
        pts = fb.lidar_points
        keep = in_img.copy()

        # 지지 평면 제거 (반복 RANSAC)
        for _ in range(self.max_planes):
            idx = np.nonzero(keep)[0]
            if idx.size < 100:
                break
            inliers = self._ransac_plane(pts[idx])
            if inliers.size < self.plane_min_ratio * in_img.sum():
                break
            keep[idx[inliers]] = False

        clusters = self._voxel_clusters(pts, np.nonzero(keep)[0])
        prompts = []
        for c in clusters:
            reps = self._representatives(pts, c)
            uv = fb.lidar_pixels[reps]
            good = np.isfinite(uv[:, 0])
            if not good.any():
                continue
            prompts.append(PromptSet(cluster_id=c.id, points_uv=uv[good],
                                     labels=np.ones(int(good.sum()), dtype=np.int64),
                                     lidar_idx=c.point_indices))
        return prompts, clusters

    # -- internals -----------------------------------------------------------
    def _ransac_plane(self, P: np.ndarray) -> np.ndarray:
        best: np.ndarray = np.empty(0, dtype=np.int64)
        n = P.shape[0]
        for _ in range(self.ransac_iters):
            i = self.rng.choice(n, 3, replace=False)
            a, b, c = P[i]
            nrm = np.cross(b - a, c - a)
            norm = np.linalg.norm(nrm)
            if norm < 1e-12:
                continue
            nrm /= norm
            d = np.abs((P - a) @ nrm)
            inl = np.nonzero(d < self.ransac_dist)[0]
            if inl.size > best.size:
                best = inl
        return best

    def _voxel_clusters(self, pts: np.ndarray, idx: np.ndarray) -> list[Cluster]:
        if idx.size == 0:
            return []
        key = np.floor(pts[idx] / self.voxel).astype(np.int64)
        vox_map: dict[tuple, list[int]] = {}
        for i, k in zip(idx, map(tuple, key)):
            vox_map.setdefault(k, []).append(int(i))

        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx, dy, dz) != (0, 0, 0)]
        seen: set[tuple] = set()
        clusters: list[Cluster] = []
        cid = 0
        for start in vox_map:
            if start in seen:
                continue
            comp, stack = [], [start]
            seen.add(start)
            while stack:
                v = stack.pop()
                comp.append(v)
                for o in offsets:
                    nb = (v[0] + o[0], v[1] + o[1], v[2] + o[2])
                    if nb in vox_map and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            pidx = np.array([i for v in comp for i in vox_map[v]], dtype=np.int64)
            if pidx.size < self.min_cluster_pts:
                continue
            P = pts[pidx]
            diag = float(np.linalg.norm(P.max(0) - P.min(0)))
            if self.size_gate and not (self.size_gate[0] <= diag <= self.size_gate[1]):
                if diag > self.size_gate[1]:
                    split = self._try_split(pts, pidx)
                    if split is not None:
                        for sp in split:
                            clusters.append(self._make_cluster(pts, sp, cid)); cid += 1
                        continue
                    # 분할 실패 → 원본 통과 (리콜 우선, 03 §4.3)
                else:
                    continue
            clusters.append(self._make_cluster(pts, pidx, cid)); cid += 1
        return clusters

    def _make_cluster(self, pts: np.ndarray, pidx: np.ndarray, cid: int) -> Cluster:
        P = pts[pidx]
        cen = P.mean(axis=0)
        snap = pidx[int(np.argmin(np.linalg.norm(P - cen, axis=1)))]  # 실측점 스냅
        return Cluster(id=cid, point_indices=pidx, centroid=pts[snap].copy(),
                       bbox_diag=float(np.linalg.norm(P.max(0) - P.min(0))))

    def _try_split(self, pts: np.ndarray, pidx: np.ndarray):
        """크기 상한 위반 클러스터를 제1주축 투영 히스토그램 최소밀도 지점에서 2-분할."""
        P = pts[pidx]
        c = P - P.mean(0)
        ax = np.linalg.eigh(np.cov(c.T))[1][:, -1]
        proj = c @ ax
        hist, edges = np.histogram(proj, bins=24)
        inner = hist[4:-4]
        if inner.size == 0 or inner.min() > 0.3 * hist.mean():
            return None
        cut = edges[4 + int(np.argmin(inner)) + 1]
        left, right = pidx[proj <= cut], pidx[proj > cut]
        if min(left.size, right.size) < self.min_cluster_pts:
            return None
        return [left, right]

    def _representatives(self, pts: np.ndarray, c: Cluster) -> np.ndarray:
        """centroid 스냅점 + 제1주축 양끝 실측점 (합성점 금지, 03 §4.2)."""
        P = pts[c.point_indices]
        cen_i = c.point_indices[int(np.argmin(np.linalg.norm(P - c.centroid, axis=1)))]
        ax = np.linalg.eigh(np.cov((P - P.mean(0)).T))[1][:, -1]
        proj = (P - P.mean(0)) @ ax
        reps = [cen_i, c.point_indices[int(np.argmin(proj))], c.point_indices[int(np.argmax(proj))]]
        return np.unique(np.array(reps[: self.k_prompts], dtype=np.int64))
