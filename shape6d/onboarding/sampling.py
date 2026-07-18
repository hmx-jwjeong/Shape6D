"""S0 표면 샘플링 + 서브셋 인덱스 (03 문서 §2.4 마스터+인덱스 방식)."""
from __future__ import annotations

import numpy as np

N_MASTER = 16384
N_PEM = 2048
N_SPARSE = 196
N_MODEL = 1024
N_VERIFY = 2048


def sample_master(mesh, n: int = N_MASTER, seed: int = 0):
    """Poisson disk 근사 샘플 + face 노멀. 반환 (pts [n,3] f32, nrm [n,3] f32)."""
    import trimesh

    pts, face_idx = trimesh.sample.sample_surface_even(mesh, count=n * 2, seed=seed)
    if len(pts) < n:  # sample_surface_even은 개수 미보장 — 부족분은 uniform 보충
        extra, extra_f = trimesh.sample.sample_surface(mesh, count=n - len(pts), seed=seed + 1)
        pts = np.vstack([pts, extra]); face_idx = np.concatenate([face_idx, extra_f])
    sel = _reject_to_count(np.asarray(pts), n, seed)
    pts = np.asarray(pts)[sel]
    nrm = np.asarray(mesh.face_normals)[np.asarray(face_idx)[sel]]
    return pts.astype(np.float32), nrm.astype(np.float32)


def _reject_to_count(pts: np.ndarray, n: int, seed: int) -> np.ndarray:
    """초과 샘플 → voxel-hash rejection으로 정확히 n개 (균일성 유지)."""
    rng = np.random.default_rng(seed)
    if len(pts) <= n:
        return np.arange(len(pts))
    diag = np.linalg.norm(pts.max(0) - pts.min(0))
    vox = diag / np.sqrt(n) / 1.2
    key = np.floor((pts - pts.min(0)) / max(vox, 1e-9)).astype(np.int64)
    _, first_idx = np.unique(key, axis=0, return_index=True)
    sel = first_idx
    if len(sel) >= n:
        return rng.permutation(sel)[:n]
    rest = np.setdiff1d(np.arange(len(pts)), sel)
    fill = rng.permutation(rest)[: n - len(sel)]
    return np.concatenate([sel, fill])


def fps_indices(pts: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """farthest point sampling 인덱스 (O(N·n) numpy — 오프라인 전용, A4의 온라인 금지와 무관)."""
    rng = np.random.default_rng(seed)
    N = pts.shape[0]
    idx = np.empty(n, dtype=np.int64)
    idx[0] = rng.integers(N)
    d = np.linalg.norm(pts - pts[idx[0]], axis=1)
    for i in range(1, n):
        idx[i] = int(np.argmax(d))
        d = np.minimum(d, np.linalg.norm(pts - pts[idx[i]], axis=1))
    return idx


def make_subsets(pts_master: np.ndarray, seed: int = 0) -> dict[str, np.ndarray]:
    """마스터 → 이름 있는 서브셋 인덱스 (03 §2.4): idx_sparse ⊂ idx_pem ⊂ master."""
    idx_pem = fps_indices(pts_master, N_PEM, seed)
    sparse_in_pem = fps_indices(pts_master[idx_pem], N_SPARSE, seed + 1)
    idx_sparse = idx_pem[sparse_in_pem]
    idx_model = fps_indices(pts_master, N_MODEL, seed + 2)
    idx_verify = fps_indices(pts_master, N_VERIFY, seed + 3)
    return {
        "idx_pem": idx_pem.astype(np.int32),
        "idx_sparse": idx_sparse.astype(np.int32),
        "idx_model": idx_model.astype(np.int32),
        "idx_verify": idx_verify.astype(np.int32),
    }
